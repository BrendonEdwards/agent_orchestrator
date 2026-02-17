"""Gemini agent - multimodal + grunt work. Runs via Gemini CLI.

Uses your existing Gemini Advanced subscription. No API keys needed.
Each call spawns: gemini -p "prompt"
"""

from __future__ import annotations

import shutil

from src.agents._subprocess import run_cli
from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent
from src.config import DEFAULT_CONFIG, OrchestratorConfig


class GeminiAgent(BaseAgent):
    """Gemini via the Gemini CLI.

    Spawns `gemini` as a subprocess. Uses your existing subscription.
    Handles both MULTIMODAL and FAST capabilities.
    """

    def __init__(
        self,
        model_id: str = "gemini-2.0-flash",
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
        return shutil.which(self._cli) is not None
