"""Main Orchestrator - recursive multi-agent coordination.

Any agent can spawn its own sub-swarm of agents using the same pattern:
stateless agents, memento notes, grunt work to Llama. This is recursive -
a sub-swarm can spawn sub-sub-swarms, limited by depth.

The whole thing is callable from CLI with a few words.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.agents.base import AgentMessage, AgentResponse, BaseAgent
from src.agents.claude_agent import ClaudeAgent
from src.agents.codex_agent import CodexAgent
from src.agents.gemini_agent import GeminiAgent
from src.agents.llama_agent import LlamaAgent
from src.memento.memento import Memento
from src.routing.router import MessageRouter
from src.rules.loader import RulesLoader


# Simple tasks that should go to Llama
_GRUNT_KEYWORDS = [
    "format", "convert", "list", "sort", "extract", "template",
    "boilerplate", "rename", "reorder", "cleanup", "prettify",
]


class Orchestrator:
    """Recursive multi-agent orchestrator.

    Each orchestrator manages a team of stateless agents. Any orchestrator
    can spawn child orchestrators - sub-swarms that handle subtasks
    independently with their own memento and fresh agents.

        User: "build me a REST API"
            |
        Orchestrator (depth=0)
            |
          Claude: "complex task, break it down"
            |
        ┌───┼────────┐
        |   |        |
      Codex swarm  Claude swarm  Llama (grunt)
      (depth=1)    (depth=1)
      |   |   |    |   |
     sub-agents   sub-agents
      (fresh)      (fresh)

    Each level has its own Memento. Parent notes are inherited as
    context so sub-swarms know the bigger picture.
    """

    def __init__(
        self,
        rules_path: str | None = None,
        memento_path: str | None = None,
        claude_model: str = "claude-sonnet-4-20250514",
        codex_model: str = "o3-mini",
        gemini_model: str = "gemini-2.0-flash",
        llama_model: str = "llama3.2",
        llama_base_url: str | None = None,
        depth: int = 0,
        max_depth: int = 3,
    ):
        self.depth = depth
        self.max_depth = max_depth

        # Config stored for spawning children
        self._config = {
            "rules_path": rules_path,
            "claude_model": claude_model,
            "codex_model": codex_model,
            "gemini_model": gemini_model,
            "llama_model": llama_model,
            "llama_base_url": llama_base_url,
        }

        self.memento = Memento(persist_path=memento_path)
        self.rules = RulesLoader(rules_path=rules_path)
        self.rules.load()

        self.claude = ClaudeAgent(model_id=claude_model)
        self.codex = CodexAgent(model_id=codex_model)
        self.gemini = GeminiAgent(model_id=gemini_model)
        self.llama = LlamaAgent(model_id=llama_model, base_url=llama_base_url)

        self._agents: dict[str, BaseAgent] = {
            "claude": self.claude,
            "codex": self.codex,
            "gemini": self.gemini,
            "llama": self.llama,
        }

        self.router = MessageRouter(
            agents=self._agents,
            rules_loader=self.rules,
            memento=self.memento,
        )

    def spawn(self, inherit_notes: bool = True) -> Orchestrator:
        """Spawn a child orchestrator (sub-swarm).

        The child gets its own fresh agents and memento. If inherit_notes
        is True, the parent's memento briefing is copied as a high-priority
        note so the sub-swarm knows the bigger picture.
        """
        if self.depth >= self.max_depth:
            raise RecursionError(
                f"Max swarm depth ({self.max_depth}) reached. "
                f"Task may be too complex to decompose further."
            )

        child = Orchestrator(
            **self._config,
            depth=self.depth + 1,
            max_depth=self.max_depth,
        )

        if inherit_notes:
            briefing = self.memento.briefing()
            if briefing:
                child.memento.note("parent_ctx", briefing, priority=3)

        return child

    async def run(self, task: str) -> str:
        """Run a task. May spawn sub-swarms for complex work."""
        self.memento.note("goal", task[:150], priority=3)

        # Grunt work -> Llama directly
        if self._is_grunt_work(task):
            self.memento.note("route", "llama:grunt", priority=1)
            response = await self.router.delegate_grunt_work(task)
            if response.success:
                self.memento.note("result", response.content[:150], priority=2)
                return response.content

        # Ask Claude to break the task down
        subtasks = await self._decompose(task)

        if subtasks and len(subtasks) > 1 and self.depth < self.max_depth:
            # Complex task: spawn sub-swarms for each subtask
            return await self._run_swarm(subtasks)

        # Simple enough for a single agent
        primary = self.rules.get_best_agent_for(task) or "claude"
        self.memento.note("route", f"{primary}:primary", priority=1)

        agents_to_use = [primary]
        task_lower = task.lower()
        if any(kw in task_lower for kw in ["code", "program", "function", "implement", "debug"]):
            if "codex" not in agents_to_use:
                agents_to_use.append("codex")
        if any(kw in task_lower for kw in ["image", "picture", "photo", "audio", "sound", "video"]):
            if "gemini" not in agents_to_use:
                agents_to_use.append("gemini")

        if len(agents_to_use) == 1:
            msg = AgentMessage(source="orchestrator", target=primary, content=task)
            response = await self.router.route(msg)
            if response.success:
                self.memento.note("result", response.content[:150], priority=2)
            return response.content if response.success else f"Failed: {response.error}"

        msg = AgentMessage(source="orchestrator", target="broadcast", content=task)
        results = await self.router.broadcast(msg, targets=agents_to_use)
        successful = [r for r in results.values() if r.success]

        if not successful:
            return "All agents failed. Check health and API keys."

        if len(successful) == 1:
            final = successful[0].content
        else:
            synthesis = await self.claude.synthesize(
                successful, memento=self.memento.briefing()
            )
            final = synthesis.content

        self.memento.note("result", final[:150], priority=2)
        return final

    async def _decompose(self, task: str) -> list[str]:
        """Ask Claude to break a complex task into independent subtasks."""
        msg = AgentMessage(
            source="orchestrator",
            target="claude",
            memento=self.memento.briefing(),
            content=(
                "Break this task into independent subtasks that can run in parallel. "
                "Return ONLY a numbered list, one subtask per line. "
                "If the task is simple enough for one agent, return just the task itself.\n\n"
                f"Task: {task}"
            ),
        )
        response = await self.claude.send(msg)
        if not response.success:
            return [task]

        # Parse numbered list
        lines = response.content.strip().split("\n")
        subtasks = []
        for line in lines:
            line = line.strip()
            # Strip numbering like "1.", "1)", "- "
            for prefix in [".", ")", "- ", "* "]:
                idx = line.find(prefix)
                if idx != -1 and idx < 4:
                    line = line[idx + len(prefix):].strip()
                    break
            if line:
                subtasks.append(line)

        return subtasks if subtasks else [task]

    async def _run_swarm(self, subtasks: list[str]) -> str:
        """Spawn sub-swarms and run subtasks in parallel."""
        self.memento.note(
            "swarm",
            f"spawning {len(subtasks)} sub-swarms at depth {self.depth + 1}",
            priority=2,
        )

        async def run_subtask(subtask: str) -> str:
            child = self.spawn(inherit_notes=True)
            return await child.run(subtask)

        results = await asyncio.gather(
            *(run_subtask(st) for st in subtasks),
            return_exceptions=True,
        )

        # Collect results
        parts = []
        for subtask, result in zip(subtasks, results):
            if isinstance(result, Exception):
                parts.append(f"[FAILED: {subtask}]: {result}")
            else:
                parts.append(f"[{subtask[:50]}]: {result}")

        # Synthesize all sub-swarm results
        synthesis_responses = [
            AgentResponse(
                agent_name=f"swarm_{i}",
                content=str(r) if not isinstance(r, Exception) else "",
                success=not isinstance(r, Exception),
                error=str(r) if isinstance(r, Exception) else None,
            )
            for i, r in enumerate(results)
        ]

        successful = [r for r in synthesis_responses if r.success]
        if not successful:
            return "All sub-swarms failed."

        if len(successful) == 1:
            final = successful[0].content
        else:
            synthesis = await self.claude.synthesize(
                successful, memento=self.memento.briefing()
            )
            final = synthesis.content

        self.memento.note("result", final[:150], priority=2)
        return final

    async def send_to(self, agent_name: str, content: str) -> AgentResponse:
        """Send a message directly to a specific agent."""
        msg = AgentMessage(source="user", target=agent_name, content=content)
        return await self.router.route(msg)

    async def grunt(self, task: str) -> str:
        """Send grunt work directly to Llama."""
        return await self.llama.do_grunt_work(task)

    async def health_check(self) -> dict[str, bool]:
        """Check the health of all agents."""
        return await self.router.health_check_all()

    def notes(self) -> str:
        """Get the current memento briefing."""
        return self.memento.briefing()

    def _is_grunt_work(self, task: str) -> bool:
        task_lower = task.lower()
        return any(kw in task_lower for kw in _GRUNT_KEYWORDS)
