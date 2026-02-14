"""Main Orchestrator - coordinates the Claude Agent Team.

The orchestrator manages a team of AI agents:
- Claude: central hub, complex reasoning, task analysis, synthesis
- Codex: code generation and technical work
- Gemini: multimodal tasks (imagery, sound, language)
- Llama: dogs body, grunt work via ollama (formatting, boilerplate, etc.)

Memento fights context rot by keeping ultra-concise survival notes.
Agents communicate using a compact protocol, not English.
"""

from __future__ import annotations

from src.agents.base import AgentMessage, AgentResponse, BaseAgent
from src.agents.claude_agent import ClaudeAgent
from src.agents.codex_agent import CodexAgent
from src.agents.gemini_agent import GeminiAgent
from src.agents.llama_agent import LlamaAgent
from src.memento.memento import Memento
from src.routing.router import MessageRouter
from src.rules.loader import RulesLoader


# Simple tasks that should go to Llama instead of wasting thinking agents
_GRUNT_KEYWORDS = [
    "format", "convert", "list", "sort", "extract", "template",
    "boilerplate", "rename", "reorder", "cleanup", "prettify",
]


class Orchestrator:
    """Top-level orchestrator for the multi-agent AI system.

    Architecture:
        Orchestrator
            |
          Claude  <-- central hub (thinking)
         /  |  \\
      Codex Gemini Llama (grunt work)

    Memento: concise survival notes against context rot
    Protocol: agents talk in compact format, not English
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
    ):
        # Memento: concise notes to fight context rot
        self.memento = Memento(persist_path=memento_path)

        # Rules: model strengths and weaknesses
        self.rules = RulesLoader(rules_path=rules_path)
        self.rules.load()

        # Agents
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

        # Router: handles inter-agent comms with compact protocol
        self.router = MessageRouter(
            agents=self._agents,
            rules_loader=self.rules,
            memento=self.memento,
        )

    async def run(self, task: str) -> str:
        """Run a task through the orchestration pipeline.

        1. Note the task goal in memento (concise!)
        2. Check if it's grunt work (send to Llama) or thinking work
        3. For thinking work: Claude analyzes, delegates, synthesizes
        4. Memento records key outcomes as survival notes
        """
        # Memento: remember what we're doing
        self.memento.note("goal", task[:150], priority=3)

        # Is this grunt work? Send to Llama directly
        if self._is_grunt_work(task):
            self.memento.note("route", "llama:grunt", priority=1)
            response = await self.router.delegate_grunt_work(task)
            if response.success:
                self.memento.note("result", response.content[:150], priority=2)
                return response.content
            # Llama failed, fall through to thinking agents

        # Thinking work: figure out who should handle it
        primary = self.rules.get_best_agent_for(task) or "claude"
        self.memento.note("route", f"{primary}:primary", priority=1)

        # Determine supporting agents
        agents_to_use = [primary]
        task_lower = task.lower()
        if any(kw in task_lower for kw in ["code", "program", "function", "implement", "debug"]):
            if "codex" not in agents_to_use:
                agents_to_use.append("codex")
        if any(kw in task_lower for kw in ["image", "picture", "photo", "audio", "sound", "video"]):
            if "gemini" not in agents_to_use:
                agents_to_use.append("gemini")

        # Single agent path
        if len(agents_to_use) == 1:
            msg = AgentMessage(source="orchestrator", target=primary, content=task)
            response = await self.router.route(msg)
            if response.success:
                self.memento.note("result", response.content[:150], priority=2)
            return response.content if response.success else f"Failed: {response.error}"

        # Multi-agent path: parallel execution then synthesis
        msg = AgentMessage(source="orchestrator", target="broadcast", content=task)
        results = await self.router.broadcast(msg, targets=agents_to_use)
        successful = [r for r in results.values() if r.success]

        if not successful:
            return "All agents failed. Check health and API keys."

        if len(successful) == 1:
            final = successful[0].content
        else:
            synthesis = await self.claude.synthesize(successful, memento=self.memento.briefing())
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
        """Get the current memento briefing (survival notes)."""
        return self.memento.briefing()

    def _is_grunt_work(self, task: str) -> bool:
        """Determine if a task is simple enough for Llama."""
        task_lower = task.lower()
        return any(kw in task_lower for kw in _GRUNT_KEYWORDS)
