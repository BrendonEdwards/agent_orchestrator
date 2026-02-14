"""Codex agent - code generation and technical work.

Stateless. Every call is fresh. Memento notes are the only context.
"""

from __future__ import annotations

import os
from typing import Any

from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent


class CodexAgent(BaseAgent):
    """Codex: code generation, code translation, technical tasks.

    Stateless - no conversation history. Gets memento survival notes
    and the current task, nothing else.
    """

    def __init__(
        self,
        model_id: str = "o3-mini",
        api_key: str | None = None,
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
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import openai

            self._client = openai.AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def send(self, message: AgentMessage) -> AgentResponse:
        """Fresh call to Codex. No history, just memento notes + task."""
        try:
            client = self._get_client()

            system_content = (
                "You are a code-focused agent. No memory of prior interactions. "
                "Be concise - output code and minimal explanation."
            )
            if message.memento:
                system_content += f"\n\nSurvival notes:\n{message.memento}"

            response = await client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": message.content},
                ],
                max_completion_tokens=4096,
            )

            result_text = response.choices[0].message.content or ""

            return AgentResponse(
                agent_name=self.name,
                content=result_text,
                token_usage={
                    "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "output_tokens": response.usage.completion_tokens if response.usage else 0,
                },
                metadata={"model": self.model_id},
            )
        except Exception as e:
            return AgentResponse(
                agent_name=self.name, content="", success=False, error=str(e),
            )

    async def health_check(self) -> bool:
        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": "ping"}],
                max_completion_tokens=10,
            )
            return len(response.choices) > 0
        except Exception:
            return False
