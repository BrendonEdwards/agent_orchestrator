"""Gemini agent - multimodal: language, imagery, sound.

Stateless. Every call is fresh. Memento notes are the only context.
"""

from __future__ import annotations

import os
from typing import Any

from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent


class GeminiAgent(BaseAgent):
    """Gemini: images, audio, cross-modal tasks.

    Stateless - no conversation history. Gets memento survival notes
    and the current task, nothing else.
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
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def send(self, message: AgentMessage) -> AgentResponse:
        """Fresh call to Gemini. No history, just memento notes + task."""
        try:
            client = self._get_client()

            prompt = ""
            if message.memento:
                prompt = f"[Survival notes: {message.memento}]\n\n"
            prompt += message.content

            # Handle multimodal content
            contents: list[Any] = []
            if "image_data" in message.metadata:
                contents.append(message.metadata["image_data"])
            if "audio_data" in message.metadata:
                contents.append(message.metadata["audio_data"])
            contents.append(prompt)

            response = await client.aio.models.generate_content(
                model=self.model_id,
                contents=contents,
            )

            result_text = response.text or ""
            token_usage = {}
            if response.usage_metadata:
                token_usage = {
                    "input_tokens": response.usage_metadata.prompt_token_count or 0,
                    "output_tokens": response.usage_metadata.candidates_token_count or 0,
                }

            return AgentResponse(
                agent_name=self.name,
                content=result_text,
                token_usage=token_usage,
                metadata={"model": self.model_id},
            )
        except Exception as e:
            return AgentResponse(
                agent_name=self.name, content="", success=False, error=str(e),
            )

    async def health_check(self) -> bool:
        try:
            client = self._get_client()
            response = await client.aio.models.generate_content(
                model=self.model_id,
                contents="ping",
            )
            return bool(response.text)
        except Exception:
            return False
