"""Codex agent: code generation and implementation via OpenAI's Codex CLI.

This wrapper is intentionally CLI-first. It uses the user's authenticated
Codex CLI session rather than API keys.
"""

from __future__ import annotations

import shutil

from src.agents._subprocess import run_cli
from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent
from src.config import DEFAULT_CONFIG, OrchestratorConfig


class CodexAgent(BaseAgent):
    """Codex via the OpenAI Codex CLI.

    The model label is metadata only. Actual model selection is controlled by
    the installed Codex CLI and the user's account settings.
    """

    def __init__(
        self,
        model_id: str = "gpt-5.2-codex",
        cli_path: str | None = None,
        config: OrchestratorConfig = DEFAULT_CONFIG,
    ):
        super().__init__(
            name="codex",
            model_id=model_id,
            capabilities=[
                AgentCapability.CODE_GENERATION,
                AgentCapability.CODE_REVIEW,
                AgentCapability.TRANSLATION,
                AgentCapability.REASONING,
                AgentCapability.MATH,
            ],
        )
        self._cli = cli_path or shutil.which("codex") or "codex"
        self._config = config

    async def send(self, message: AgentMessage) -> AgentResponse:
        """Spawn codex CLI, capture output."""
        prompt = message.content
        if message.memento:
            prompt = f"[Survival notes: {message.memento}]\n\n{prompt}"

        return await run_cli(
            self.name,
            [self._cli, "-q", prompt],
            config=self._config,
            metadata={"cli": self._cli, "model": self.model_id},
        )

    async def health_check(self) -> bool:
        """Run a real prompt so auth and command syntax are checked."""
        try:
            resp = await run_cli(
                self.name,
                [self._cli, "-q", "respond with ok"],
                config=OrchestratorConfig(
                    agent_timeout=self._config.health_check_timeout,
                    agent_retries=0,
                ),
            )
            return resp.success and len(resp.content) > 0
        except Exception:
            return False
