"""Main Orchestrator - coordinates the Claude Agent Team.

The orchestrator sits at the top of the architecture, managing the agent
team with Claude as the central hub. It handles task intake, analysis,
delegation, and synthesis.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent
from src.agents.claude_agent import ClaudeAgent
from src.agents.codex_agent import CodexAgent
from src.agents.gemini_agent import GeminiAgent
from src.agents.llama_agent import LlamaAgent
from src.memento.memento import Memento
from src.routing.router import MessageRouter
from src.rules.loader import RulesLoader


class Orchestrator:
    """Top-level orchestrator for the multi-agent AI system.

    Architecture (matching whiteboard):
        Orchestrator
            |
          Claude  <-- central hub
         /  |  \\
      Codex Gemini Llama
         \\  |  /
          (mesh)

    Claude analyzes tasks and delegates to specialized agents:
    - Codex: code generation, language translation, context compression
    - Gemini: multimodal tasks (language, imagery, sound)
    - Llama: local/private inference, fast iteration

    The Memento system records all decisions and routing for auditability.
    Rules.md defines model strengths and weaknesses for routing.
    """

    def __init__(
        self,
        rules_path: str | None = None,
        memento_dir: str | None = None,
        claude_model: str = "claude-sonnet-4-20250514",
        codex_model: str = "o3-mini",
        gemini_model: str = "gemini-2.0-flash",
        llama_model: str = "llama3.2",
        llama_base_url: str | None = None,
    ):
        # Initialize the Memento audit system
        self.memento = Memento(persist_dir=memento_dir)

        # Initialize the rules loader
        self.rules = RulesLoader(rules_path=rules_path)
        self.rules.load()

        # Initialize agents
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

        # Initialize the message router
        self.router = MessageRouter(
            agents=self._agents,
            rules_loader=self.rules,
            memento=self.memento,
            codex_agent=self.codex,
        )

        self.memento.record(
            event_type="init",
            source="orchestrator",
            content="Orchestrator initialized with agents: " + ", ".join(self._agents.keys()),
        )

    async def run(self, task: str) -> str:
        """Run a task through the full orchestration pipeline.

        Steps:
        1. Record the incoming task
        2. Have Claude analyze and create a routing plan
        3. Delegate subtasks to appropriate agents
        4. Compress inter-agent context via Codex when needed
        5. Have Claude synthesize the final response
        6. Record the full decision chain in Memento
        """
        # Step 1: Record the task
        self.memento.record(
            event_type="task_received",
            source="user",
            target="orchestrator",
            content=task,
        )

        # Step 2: Claude analyzes the task
        self.memento.record(
            event_type="analysis",
            source="orchestrator",
            target="claude",
            content="Requesting task analysis",
        )

        analysis = await self.claude.analyze_task(task)

        self.memento.record(
            event_type="analysis",
            source="claude",
            target="orchestrator",
            content=f"Analysis complete: {analysis.get('raw_analysis', '')[:200]}",
        )

        # Step 3: Determine which agents to involve based on rules + analysis
        primary = self.rules.get_best_agent_for(task)
        if primary is None:
            primary = "claude"

        agents_to_use = [primary]
        # Add supporting agents based on task characteristics
        task_lower = task.lower()
        if any(kw in task_lower for kw in ["code", "program", "function", "implement", "debug"]):
            if "codex" not in agents_to_use:
                agents_to_use.append("codex")
        if any(kw in task_lower for kw in ["image", "picture", "photo", "audio", "sound", "video"]):
            if "gemini" not in agents_to_use:
                agents_to_use.append("gemini")
        if any(kw in task_lower for kw in ["private", "local", "offline", "sensitive"]):
            if "llama" not in agents_to_use:
                agents_to_use.append("llama")

        self.memento.record(
            event_type="delegation",
            source="orchestrator",
            content=f"Delegating to agents: {', '.join(agents_to_use)}",
            metadata={"primary": primary, "agents": agents_to_use},
        )

        # Step 4: Send task to selected agents
        responses: list[AgentResponse] = []
        if len(agents_to_use) == 1:
            msg = AgentMessage(source="orchestrator", target=agents_to_use[0], content=task)
            resp = await self.router.route(msg)
            responses.append(resp)
        else:
            msg = AgentMessage(source="orchestrator", target="broadcast", content=task)
            results = await self.router.broadcast(msg, targets=agents_to_use)
            responses.extend(results.values())

        # Step 5: Synthesize if multiple agents responded
        successful = [r for r in responses if r.success]

        if len(successful) == 0:
            self.memento.record(
                event_type="error",
                source="orchestrator",
                content="All agents failed",
                metadata={"errors": [r.error for r in responses]},
            )
            return "All agents failed to process the task. Check agent health and API keys."

        if len(successful) == 1:
            final = successful[0].content
        else:
            self.memento.record(
                event_type="synthesis",
                source="orchestrator",
                target="claude",
                content=f"Synthesizing {len(successful)} agent responses",
            )
            synthesis = await self.claude.synthesize(successful)
            final = synthesis.content

        # Step 6: Record completion
        self.memento.record(
            event_type="complete",
            source="orchestrator",
            content=f"Task complete. Response length: {len(final)} chars",
            metadata={"agents_used": agents_to_use},
        )

        return final

    async def send_to(self, agent_name: str, content: str) -> AgentResponse:
        """Send a message directly to a specific agent."""
        msg = AgentMessage(source="user", target=agent_name, content=content)
        return await self.router.route(msg)

    async def health_check(self) -> dict[str, bool]:
        """Check the health of all agents."""
        results = await self.router.health_check_all()
        self.memento.record(
            event_type="health_check",
            source="orchestrator",
            content=f"Health check results: {results}",
        )
        return results

    def get_audit_trail(self, last_n: int | None = None) -> str:
        """Get the Memento audit trail as a human-readable string."""
        return self.memento.get_trail_summary(last_n or 50)

    def save_memento(self, filename: str | None = None) -> str | None:
        """Persist the Memento trail to disk."""
        path = self.memento.save(filename)
        return str(path) if path else None
