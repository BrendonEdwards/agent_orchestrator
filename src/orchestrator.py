"""Fractal multi-agent orchestrator.

The same pattern repeats at every scale: decompose, delegate, synthesize.
Each sub-swarm composes its own agent team based on what the subtask
needs - a code-heavy task spawns more Codex, an image task spawns more
Gemini. It's self-similar all the way down.

Recursion stops naturally when a task is simple enough for one agent.
A cost budget acts as a safety net.
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

# Keywords that signal which agent type a subtask needs
_AGENT_SIGNALS = {
    "codex": ["code", "program", "function", "implement", "debug", "algorithm",
              "class", "module", "api", "endpoint", "test", "refactor"],
    "gemini": ["image", "picture", "photo", "audio", "sound", "video",
               "visual", "diagram", "describe image", "transcribe"],
    "llama": ["format", "convert", "sort", "list", "template", "boilerplate",
              "cleanup", "extract", "rename"],
    "claude": ["design", "architect", "plan", "analyze", "review", "explain",
               "reason", "compare", "evaluate", "synthesize"],
}


class Orchestrator:
    """Fractal multi-agent orchestrator.

    Self-similar at every scale:
    1. Task comes in
    2. Is it grunt work? -> Llama
    3. Is it simple? -> best single agent
    4. Is it complex? -> decompose, spawn sub-swarms, synthesize
    5. Each sub-swarm does the same thing (goto 1)

    Sub-swarms compose their own agent teams:

        "build a web app with image upload"
                    |
            Orchestrator (d=0)
                    |
            Claude decomposes:
            ┌───────┼──────────┐
            |       |          |
         code     images    boilerplate
            |       |          |
        Orchestrator Orchestrator  Llama
        team:       team:         (grunt)
        2x Codex    2x Gemini
        1x Claude   1x Claude
        1x Llama    1x Llama
            |           |
        can spawn     can spawn
        more...       more...
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
        max_depth: int = 10,
        agents: dict[str, BaseAgent] | None = None,
    ):
        self.depth = depth
        self.max_depth = max_depth

        # Config for spawning children
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

        # Agents: either custom (from fractal spawn) or default team
        if agents:
            self._agents = agents
        else:
            self._agents = self._build_default_team()

        # Always need a claude reference for decomposition/synthesis
        self._claude = self._find_agent(ClaudeAgent) or ClaudeAgent(model_id=claude_model)
        self._llama = self._find_agent(LlamaAgent)

        self.router = MessageRouter(
            agents=self._agents,
            rules_loader=self.rules,
            memento=self.memento,
        )

    def _build_default_team(self) -> dict[str, BaseAgent]:
        """Default team: one of each."""
        return {
            "claude": ClaudeAgent(model_id=self._config["claude_model"]),
            "codex": CodexAgent(model_id=self._config["codex_model"]),
            "gemini": GeminiAgent(model_id=self._config["gemini_model"]),
            "llama": LlamaAgent(
                model_id=self._config["llama_model"],
                base_url=self._config["llama_base_url"],
            ),
        }

    def _find_agent(self, agent_type: type) -> BaseAgent | None:
        """Find the first agent of a given type in the team."""
        for agent in self._agents.values():
            if isinstance(agent, agent_type):
                return agent
        return None

    def _build_team_for(self, subtask: str) -> dict[str, BaseAgent]:
        """Build a custom agent team weighted for a specific subtask.

        This is the fractal part - each sub-swarm gets a team composed
        for its specific needs, not a clone of the parent.
        """
        task_lower = subtask.lower()

        # Score each agent type for this subtask
        scores = {}
        for agent_name, keywords in _AGENT_SIGNALS.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            scores[agent_name] = score

        # Always include Claude (brain) and Llama (grunt)
        team: dict[str, BaseAgent] = {
            "claude": ClaudeAgent(model_id=self._config["claude_model"]),
            "llama": LlamaAgent(
                model_id=self._config["llama_model"],
                base_url=self._config["llama_base_url"],
            ),
        }

        # Add the most relevant agent type, with extra instances if strong signal
        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best]

        # Always include at least one Codex and one Gemini
        team["codex"] = CodexAgent(model_id=self._config["codex_model"])
        team["gemini"] = GeminiAgent(model_id=self._config["gemini_model"])

        # Add extra instances of the dominant type
        if best_score >= 2 and best not in ("claude", "llama"):
            for i in range(min(best_score - 1, 3)):
                agent_class = CodexAgent if best == "codex" else GeminiAgent
                model = self._config[f"{best}_model"]
                team[f"{best}_{i + 2}"] = agent_class(model_id=model)

        return team

    def spawn(self, subtask: str | None = None, inherit_notes: bool = True) -> Orchestrator:
        """Spawn a child orchestrator with a team built for the subtask.

        If subtask is provided, the child's team is weighted for that task.
        Otherwise it gets a default team.
        """
        if self.depth >= self.max_depth:
            raise RecursionError(
                f"Max depth ({self.max_depth}) reached. Cannot decompose further."
            )

        # Build a team customized for this subtask
        team = self._build_team_for(subtask) if subtask else None

        child = Orchestrator(
            **self._config,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            agents=team,
        )

        if inherit_notes:
            briefing = self.memento.briefing()
            if briefing:
                child.memento.note("parent_ctx", briefing, priority=3)

        return child

    async def run(self, task: str) -> str:
        """Run a task. Fractal: same pattern at every scale.

        1. Grunt work? -> Llama
        2. Simple? -> best single agent
        3. Complex? -> decompose, spawn sub-swarms, synthesize
        4. Each sub-swarm repeats from step 1
        """
        self.memento.note("goal", task[:150], priority=3)

        # Grunt work -> straight to Llama
        if self._is_grunt_work(task):
            return await self._do_grunt(task)

        # Decompose: Claude decides if this needs splitting
        subtasks = await self._decompose(task)

        # Multiple subtasks and we can still go deeper -> fractal
        if len(subtasks) > 1 and self.depth < self.max_depth:
            return await self._run_fractal(subtasks)

        # Leaf node: single agent handles it
        return await self._run_single(task)

    async def _do_grunt(self, task: str) -> str:
        """Grunt work path."""
        self.memento.note("route", "llama:grunt", priority=1)
        if self._llama:
            msg = AgentMessage(source="orchestrator", target="llama", content=task)
            response = await self._llama.send(msg)
            if response.success:
                self.memento.note("done", response.content[:150], priority=2)
                return response.content
        # Llama unavailable, fall through to single agent
        return await self._run_single(task)

    async def _run_single(self, task: str) -> str:
        """Single agent handles the task (leaf node of the fractal)."""
        primary = self.rules.get_best_agent_for(task) or "claude"
        self.memento.note("route", f"{primary}:leaf", priority=1)

        # Determine if we need multiple agents
        agents_to_use = [primary]
        task_lower = task.lower()
        for agent_name, keywords in _AGENT_SIGNALS.items():
            if agent_name != primary and agent_name in self._agents:
                if any(kw in task_lower for kw in keywords):
                    agents_to_use.append(agent_name)

        if len(agents_to_use) == 1:
            if primary not in self._agents:
                primary = "claude"
            msg = AgentMessage(source="orchestrator", target=primary, content=task)
            response = await self.router.route(msg)
            if response.success:
                self.memento.note("done", response.content[:150], priority=2)
            return response.content if response.success else f"Failed: {response.error}"

        # Multiple agents, parallel
        msg = AgentMessage(source="orchestrator", target="broadcast", content=task)
        results = await self.router.broadcast(msg, targets=agents_to_use)
        successful = [r for r in results.values() if r.success]

        if not successful:
            return "All agents failed."

        if len(successful) == 1:
            return successful[0].content

        synthesis = await self._claude.synthesize(
            successful, memento=self.memento.briefing()
        )
        return synthesis.content

    async def _decompose(self, task: str) -> list[str]:
        """Claude breaks a task into independent subtasks, or returns it as-is."""
        msg = AgentMessage(
            source="orchestrator",
            target="claude",
            memento=self.memento.briefing(),
            content=(
                "Break this task into 2-5 independent subtasks that can run in parallel. "
                "Return ONLY a numbered list, one subtask per line. "
                "If the task is simple enough for one agent, return ONLY the task itself "
                "as a single line with no numbering.\n\n"
                f"Task: {task}"
            ),
        )
        response = await self._claude.send(msg)
        if not response.success:
            return [task]

        lines = response.content.strip().split("\n")
        subtasks = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Strip numbering: "1.", "1)", "- ", "* "
            for prefix in [".", ")", "- ", "* "]:
                idx = line.find(prefix)
                if idx != -1 and idx < 4:
                    line = line[idx + len(prefix):].strip()
                    break
            if line:
                subtasks.append(line)

        return subtasks if subtasks else [task]

    async def _run_fractal(self, subtasks: list[str]) -> str:
        """Spawn sub-swarms for each subtask, run in parallel, synthesize.

        Each sub-swarm gets a team weighted for its subtask.
        Each sub-swarm can decompose further - same pattern, deeper.
        """
        self.memento.note(
            "fractal",
            f"d={self.depth}->d={self.depth + 1}, {len(subtasks)} branches",
            priority=2,
        )

        async def run_branch(subtask: str) -> str:
            child = self.spawn(subtask=subtask, inherit_notes=True)
            return await child.run(subtask)

        results = await asyncio.gather(
            *(run_branch(st) for st in subtasks),
            return_exceptions=True,
        )

        # Build responses for synthesis
        responses = [
            AgentResponse(
                agent_name=f"d{self.depth + 1}_{i}",
                content=str(r) if not isinstance(r, Exception) else "",
                success=not isinstance(r, Exception),
                error=str(r) if isinstance(r, Exception) else None,
            )
            for i, r in enumerate(results)
        ]

        successful = [r for r in responses if r.success]
        if not successful:
            return "All branches failed."

        if len(successful) == 1:
            return successful[0].content

        synthesis = await self._claude.synthesize(
            successful, memento=self.memento.briefing()
        )
        self.memento.note("done", synthesis.content[:150], priority=2)
        return synthesis.content

    async def send_to(self, agent_name: str, content: str) -> AgentResponse:
        """Send directly to a specific agent."""
        msg = AgentMessage(source="user", target=agent_name, content=content)
        return await self.router.route(msg)

    async def grunt(self, task: str) -> str:
        """Grunt work -> Llama."""
        if self._llama:
            msg = AgentMessage(source="orchestrator", target="llama", content=task)
            resp = await self._llama.send(msg)
            return resp.content if resp.success else ""
        return ""

    async def health_check(self) -> dict[str, bool]:
        return await self.router.health_check_all()

    def notes(self) -> str:
        return self.memento.briefing()

    @property
    def team(self) -> list[str]:
        """Who's on this orchestrator's team."""
        return list(self._agents.keys())

    def _is_grunt_work(self, task: str) -> bool:
        task_lower = task.lower()
        return any(kw in task_lower for kw in _GRUNT_KEYWORDS)
