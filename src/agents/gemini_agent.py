"""Gemini agent: multimodal and fast work via the Gemini CLI.

This wrapper is intentionally CLI-first. It uses the user's authenticated
Gemini CLI session rather than API keys.
"""

from __future__ import annotations

import shutil

from src.agents._subprocess import run_cli
from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent
from src.config import DEFAULT_CONFIG, OrchestratorConfig


class GeminiAgent(BaseAgent):
    """Gemini via the Gemini CLI.

    The model label is metadata only. Actual model selection is controlled by
    the installed Gemini CLI and the user's account settings.
    """

    def __init__(
        self,
        model_id: str = "gemini-3-pro-preview",
        cli_path: str | None = None,
        config: OrchestratorConfig = DEFAULT_CONFIG,
    ):
        super().__init__(
            name="gemini",
            model_id=model_id,
            capabilities=[
                AgentCapability.TEXT_GENERATION,
                AgentCapability.IMAGE_UNDERSTANDING,
                AgentCapability.IMAGE_GENERATION,
                AgentCapability.AUDIO_UNDERSTANDING,
                AgentCapability.TRANSLATION,
                AgentCapability.REASONING,
            ],
        )
        self._cli = cli_path or shutil.which("gemini") or "gemini"
        self._config = config

    async def send(self, message: AgentMessage) -> AgentResponse:
        """Spawn gemini CLI, capture output."""
        prompt = message.content
        if message.memento:
            prompt = f"[Survival notes: {message.memento}]\n\n{prompt}"

        return await run_cli(
            self.name,
            [self._cli, "-p", prompt],
            config=self._config,
            metadata={"cli": self._cli, "model": self.model_id},
        )

    async def health_check(self) -> bool:
        """Run a real prompt so auth and command syntax are checked."""
        try:
            resp = await run_cli(
                self.name,
                [self._cli, "-p", "respond with ok"],
                config=OrchestratorConfig(
                    agent_timeout=self._config.health_check_timeout,
                    agent_retries=0,
                ),
            )
            return resp.success and len(resp.content) > 0
        except Exception:
            return False
