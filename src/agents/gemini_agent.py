"""Gemini agent - multimodal. Runs via Gemini CLI.

Uses your existing Gemini Advanced subscription. No API keys needed.
Install: npm install -g @anthropic-ai/claude-code  (or Google's gemini CLI)
Each call spawns: gemini -p "prompt"
"""

from __future__ import annotations

import asyncio
import shutil

from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent


class GeminiAgent(BaseAgent):
    """Gemini via the Gemini CLI.

    Spawns `gemini` as a subprocess. Uses your existing subscription.
    Install with: npm install -g @anthropic-ai/claude-code  (TODO: real gemini CLI)
    """

    def __init__(self, model_id: str = "gemini-2.0-flash", cli_path: str | None = None):
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

    async def send(self, message: AgentMessage) -> AgentResponse:
        """Spawn gemini CLI, capture output."""
        prompt = message.content
        if message.memento:
            prompt = f"[Survival notes: {message.memento}]\n\n{prompt}"

        try:
            proc = await asyncio.create_subprocess_exec(
                self._cli, "-p", prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)

            if proc.returncode != 0:
                return AgentResponse(
                    agent_name=self.name, content="",
                    success=False, error=stderr.decode().strip(),
                )

            return AgentResponse(
                agent_name=self.name,
                content=stdout.decode().strip(),
                metadata={"cli": self._cli, "model": self.model_id},
            )
        except asyncio.TimeoutError:
            return AgentResponse(
                agent_name=self.name, content="",
                success=False, error="CLI timed out after 300s",
            )
        except Exception as e:
            return AgentResponse(
                agent_name=self.name, content="",
                success=False, error=str(e),
            )

    async def health_check(self) -> bool:
        return shutil.which(self._cli) is not None
