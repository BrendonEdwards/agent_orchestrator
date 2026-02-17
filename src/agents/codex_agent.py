"""Codex agent - code generation. Runs via OpenAI Codex CLI.

Uses your existing ChatGPT Plus/Pro subscription. No API keys needed.
Install: npm install -g @openai/codex
Each call spawns: codex -q "prompt"
"""

from __future__ import annotations

import shutil

from src.agents._subprocess import run_cli
from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent
from src.config import DEFAULT_CONFIG, OrchestratorConfig


class CodexAgent(BaseAgent):
    """Codex via the OpenAI Codex CLI.

    Spawns `codex` as a subprocess. Uses your existing subscription.
    Install with: npm install -g @openai/codex
    """

    def __init__(
        self,
        model_id: str = "o3-mini",
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
        return shutil.which(self._cli) is not None
