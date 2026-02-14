"""Codex agent - code generation. Direct OpenAI API.

Fast path: direct HTTP to api.openai.com, no CLI subprocess overhead.
Requires: OPENAI_API_KEY env var
pip install openai
"""

from __future__ import annotations

import os
from typing import Any

from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent


class CodexAgent(BaseAgent):
    """Codex via the OpenAI API.

    Direct API call - no subprocess, structured errors, token tracking.
    """

    def __init__(
        self,
        model_id: str = "o3-mini",
        api_key: str | None = None,
        max_tokens: int = 4096,
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
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._max_tokens = max_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import openai
            self._client = openai.AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def send(self, message: AgentMessage) -> AgentResponse:
        """Direct API call to OpenAI."""
        prompt = message.content
        if message.memento:
            prompt = f"[Survival notes: {message.memento}]\n\n{prompt}"

        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are a code-focused AI. Be precise and concise."},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=self._max_tokens,
            )
            content = response.choices[0].message.content or ""
            return AgentResponse(
                agent_name=self.name,
                content=content,
                token_usage={
                    "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "output_tokens": response.usage.completion_tokens if response.usage else 0,
                },
                metadata={"model": self.model_id},
            )
        except Exception as e:
            return AgentResponse(
                agent_name=self.name, content="",
                success=False, error=str(e),
            )

    async def health_check(self) -> bool:
        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": "1"}],
                max_completion_tokens=5,
            )
            return len(response.choices) > 0
        except Exception:
            return False
