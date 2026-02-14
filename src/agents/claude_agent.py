"""Claude agent - the brain. Direct Anthropic API.

Fast path: direct HTTP to api.anthropic.com, no CLI subprocess overhead.
Requires: ANTHROPIC_API_KEY env var
pip install anthropic
"""

from __future__ import annotations

import os
from typing import Any

from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent


class ClaudeAgent(BaseAgent):
    """Claude via the Anthropic API.

    Direct API call - no subprocess, no CLI bootstrap, structured errors.
    """

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 4096,
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
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._max_tokens = max_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def send(self, message: AgentMessage) -> AgentResponse:
        """Direct API call to Anthropic."""
        prompt = message.content
        if message.memento:
            prompt = f"[Survival notes: {message.memento}]\n\n{prompt}"

        try:
            client = self._get_client()
            response = await client.messages.create(
                model=self.model_id,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text if response.content else ""
            return AgentResponse(
                agent_name=self.name,
                content=content,
                token_usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                metadata={"model": self.model_id},
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
            client = self._get_client()
            response = await client.messages.create(
                model=self.model_id,
                max_tokens=10,
                messages=[{"role": "user", "content": "respond with ok"}],
            )
            return len(response.content) > 0
        except Exception:
            return False
