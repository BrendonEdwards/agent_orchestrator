"""Gemini agent - specialized for multimodal tasks.

As noted on the whiteboard: "Language / Imagery / Sound."
Gemini handles tasks involving images, audio, and cross-modal understanding.
"""

from __future__ import annotations

import os
from typing import Any

from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent


class GeminiAgent(BaseAgent):
    """Gemini agent for multimodal processing.

    Key responsibilities:
    - Image understanding and generation
    - Audio/sound processing
    - Cross-modal tasks (e.g., describing images, transcribing audio)
    - Language translation with multimodal context
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
                AgentCapability.CONVERSATION,
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
        """Send a message to Gemini and return the response."""
        try:
            client = self._get_client()

            context_parts = []
            for ctx in self._context:
                context_parts.append(f"[{ctx['role']}]: {ctx['content']}")

            full_prompt = ""
            if context_parts:
                full_prompt = "\n".join(context_parts) + "\n\n"
            full_prompt += message.content

            # Handle multimodal content if present in metadata
            contents: list[Any] = []
            if "image_data" in message.metadata:
                contents.append(message.metadata["image_data"])
            if "audio_data" in message.metadata:
                contents.append(message.metadata["audio_data"])
            contents.append(full_prompt)

            response = await client.aio.models.generate_content(
                model=self.model_id,
                contents=contents,
            )

            result_text = response.text or ""
            self.add_to_context("user", message.content)
            self.add_to_context("assistant", result_text)

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
                metadata={
                    "model": self.model_id,
                    "source_message_id": message.id,
                    "multimodal": bool(message.metadata.get("image_data") or message.metadata.get("audio_data")),
                },
            )
        except Exception as e:
            return AgentResponse(
                agent_name=self.name,
                content="",
                success=False,
                error=str(e),
            )

    async def process_image(self, prompt: str, image_data: Any) -> AgentResponse:
        """Process an image with a text prompt."""
        msg = AgentMessage(
            source="orchestrator",
            target="gemini",
            content=prompt,
            metadata={"image_data": image_data},
        )
        return await self.send(msg)

    async def process_audio(self, prompt: str, audio_data: Any) -> AgentResponse:
        """Process audio with a text prompt."""
        msg = AgentMessage(
            source="orchestrator",
            target="gemini",
            content=prompt,
            metadata={"audio_data": audio_data},
        )
        return await self.send(msg)

    async def health_check(self) -> bool:
        """Check if the Gemini API is reachable."""
        try:
            client = self._get_client()
            response = await client.aio.models.generate_content(
                model=self.model_id,
                contents="ping",
            )
            return bool(response.text)
        except Exception:
            return False
