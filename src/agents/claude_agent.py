"""Claude agent - the brain. Runs via Claude Code CLI.

Uses your existing Claude Pro subscription. No API keys needed.
Short prompts: claude -p "prompt" --output-format text
Long prompts: echo "prompt" | claude -p - --output-format text
"""

from __future__ import annotations

import asyncio
import shutil

from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent

# OS arg limit varies (128KB-2MB). Stay safe with a 32KB threshold.
_STDIN_THRESHOLD = 32_000


class ClaudeAgent(BaseAgent):
    """Claude via the Claude Code CLI.

    Spawns `claude -p "prompt"` as a subprocess. Uses your Pro
    subscription - no API key, no per-token billing.

    Prompts over 32KB are piped via stdin to avoid OS arg length limits.
    """

    def __init__(self, model_id: str = "claude-sonnet-4-20250514", cli_path: str | None = None):
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

    async def send(self, message: AgentMessage) -> AgentResponse:
        """Spawn claude CLI, capture output."""
        prompt = message.content
        if message.memento:
            prompt = f"[Survival notes: {message.memento}]\n\n{prompt}"

        try:
            if len(prompt) > _STDIN_THRESHOLD:
                # Long prompt: pipe via stdin to avoid OS arg length limits
                proc = await asyncio.create_subprocess_exec(
                    self._cli, "-p", "-",
                    "--output-format", "text",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=prompt.encode()), timeout=300,
                )
            else:
                # Short prompt: pass as arg
                proc = await asyncio.create_subprocess_exec(
                    self._cli, "-p", prompt,
                    "--output-format", "text",
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
            proc = await asyncio.create_subprocess_exec(
                self._cli, "-p", "respond with ok",
                "--output-format", "text",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            return proc.returncode == 0 and len(stdout) > 0
        except Exception:
            return False
