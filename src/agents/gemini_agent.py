"""Gemini agent - multimodal. Direct Google Generative AI API.

Fast path: direct HTTP to generativelanguage.googleapis.com.
Requires: GEMINI_API_KEY env var (free tier available at aistudio.google.com)
pip install google-genai
"""

from __future__ import annotations

import os
from typing import Any

from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent


class GeminiAgent(BaseAgent):
    """Gemini via the Google Generative AI API.

    Direct API call - no subprocess, structured errors, token tracking.
    Free tier available at aistudio.google.com.
    """

    def __init__(
        self,
        model_id: str = "gemini-2.0-flash",
        api_key: str | None = None,
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
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def send(self, message: AgentMessage) -> AgentResponse:
        """Direct API call to Google Generative AI."""
        prompt = message.content
        if message.memento:
            prompt = f"[Survival notes: {message.memento}]\n\n{prompt}"

        try:
            client = self._get_client()
            response = await client.aio.models.generate_content(
                model=self.model_id,
                contents=prompt,
            )
            content = response.text or ""
            usage = {}
            if response.usage_metadata:
                usage = {
                    "input_tokens": response.usage_metadata.prompt_token_count or 0,
                    "output_tokens": response.usage_metadata.candidates_token_count or 0,
                }
            return AgentResponse(
                agent_name=self.name,
                content=content,
                token_usage=usage,
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
            response = await client.aio.models.generate_content(
                model=self.model_id,
                contents="respond with ok",
            )
            return bool(response.text)
        except Exception:
            return False
