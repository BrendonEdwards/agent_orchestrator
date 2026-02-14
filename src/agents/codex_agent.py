"""Codex agent - code generation. Runs via OpenAI Codex CLI.

Uses your existing ChatGPT Plus/Pro subscription. No API keys needed.
Install: npm install -g @openai/codex
Short prompts: codex -q "prompt"
Long prompts: piped via stdin
"""

from __future__ import annotations

import asyncio
import shutil

from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent

_STDIN_THRESHOLD = 32_000


class CodexAgent(BaseAgent):
    """Codex via the OpenAI Codex CLI.

    Spawns `codex` as a subprocess. Uses your existing subscription.
    Install with: npm install -g @openai/codex
    Prompts over 32KB are piped via stdin.
    """

    def __init__(self, model_id: str = "o3-mini", cli_path: str | None = None):
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

    async def send(self, message: AgentMessage) -> AgentResponse:
        """Spawn codex CLI, capture output."""
        prompt = message.content
        if message.memento:
            prompt = f"[Survival notes: {message.memento}]\n\n{prompt}"

        try:
            if len(prompt) > _STDIN_THRESHOLD:
                proc = await asyncio.create_subprocess_exec(
                    self._cli, "-q", "-",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=prompt.encode()), timeout=300,
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    self._cli, "-q", prompt,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=300,
                )

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
