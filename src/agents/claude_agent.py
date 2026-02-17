"""Claude agent - the brain. Runs via Claude Code CLI.

Uses your existing Claude Pro subscription. No API keys needed.
Each call spawns: claude -p "prompt" --output-format text
"""

from __future__ import annotations

import shutil

from src.agents._subprocess import run_cli
from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent
from src.config import DEFAULT_CONFIG, OrchestratorConfig


class ClaudeAgent(BaseAgent):
    """Claude via the Claude Code CLI.

    Spawns `claude -p "prompt"` as a subprocess. Uses your Pro
    subscription - no API key, no per-token billing.
    """

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-20250514",
        cli_path: str | None = None,
        config: OrchestratorConfig = DEFAULT_CONFIG,
    ):
        super().__init__(
            name="claude",
            model_id=model_id,
            capabilities=[
                AgentCapability.TEXT_GENERATION,
                AgentCapability.CODE_GENERATION,
                AgentCapability.CODE_REVIEW,
                AgentCapability.REASONING,
                AgentCapability.MATH,
                AgentCapability.SUMMARIZATION,
            ],
        )
        self._cli = cli_path or shutil.which("claude") or "claude"
        self._config = config

    async def send(self, message: AgentMessage) -> AgentResponse:
        """Spawn claude CLI, capture output."""
        prompt = message.content
        if message.memento:
            prompt = f"[Survival notes: {message.memento}]\n\n{prompt}"

        return await run_cli(
            self.name,
            [self._cli, "-p", prompt, "--output-format", "text"],
            config=self._config,
            metadata={"cli": self._cli, "model": self.model_id},
        )

    async def synthesize(self, results: list[AgentResponse], memento: str = "") -> AgentResponse:
        """Synthesize results from multiple agents."""
        parts = []
        for r in results:
            if r.success:
                parts.append(f"[{r.agent_name}]: {r.content}")
            else:
                parts.append(f"[{r.agent_name}]: (failed) {r.error}")

        msg = AgentMessage(
            source="orchestrator",
            target="claude",
            content="Synthesize these agent responses into one coherent answer.\n\n"
            + "\n\n".join(parts),
            memento=memento,
        )
        return await self.send(msg)

    async def health_check(self) -> bool:
        try:
            resp = await run_cli(
                self.name,
                [self._cli, "-p", "respond with ok", "--output-format", "text"],
                config=OrchestratorConfig(
                    agent_timeout=self._config.health_check_timeout,
                    agent_retries=0,
                ),
            )
            return resp.success and len(resp.content) > 0
        except Exception:
            return False
