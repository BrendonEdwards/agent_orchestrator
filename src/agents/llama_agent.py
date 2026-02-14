"""Llama Local agent - runs Meta's Llama model locally.

Llama Local provides fast, private inference without sending data
to external APIs. Useful for sensitive tasks and rapid iteration.
"""

from __future__ import annotations

import os
from typing import Any

from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent


class LlamaAgent(BaseAgent):
    """Llama agent running locally for private, fast inference.

    Key responsibilities:
    - Privacy-sensitive tasks (data never leaves the machine)
    - Fast local inference for rapid iteration
    - Offline operation capability
    - Cost-free inference for high-volume tasks

    Supports two backends:
    - llama-cpp-python: Direct local model loading
    - OpenAI-compatible API: For local servers (e.g., ollama, vllm, llama.cpp server)
    """

    def __init__(
        self,
        model_id: str = "llama3.2",
        base_url: str | None = None,
        model_path: str | None = None,
    ):
        super().__init__(
            name="llama",
            model_id=model_id,
            capabilities=[
                AgentCapability.TEXT_GENERATION,
                AgentCapability.CODE_GENERATION,
                AgentCapability.REASONING,
                AgentCapability.SUMMARIZATION,
                AgentCapability.CONVERSATION,
                AgentCapability.LOCAL_EXECUTION,
            ],
        )
        self._base_url = base_url or os.environ.get(
            "LLAMA_BASE_URL", "http://localhost:11434/v1"
        )
        self._model_path = model_path
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get an OpenAI-compatible client pointing at the local Llama server."""
        if self._client is None:
            import openai

            self._client = openai.AsyncOpenAI(
                base_url=self._base_url,
                api_key="not-needed",
            )
        return self._client

    async def send(self, message: AgentMessage) -> AgentResponse:
        """Send a message to the local Llama instance and return the response."""
        try:
            client = self._get_client()

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a local AI agent within a multi-agent system. "
                        "You handle tasks that require privacy, fast inference, "
                        "or offline operation. Be concise and accurate."
                    ),
                },
            ]
            for ctx in self._context:
                messages.append({"role": ctx["role"], "content": ctx["content"]})
            messages.append({"role": "user", "content": message.content})

            response = await client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_completion_tokens=2048,
            )

            result_text = response.choices[0].message.content or ""
            self.add_to_context("user", message.content)
            self.add_to_context("assistant", result_text)

            return AgentResponse(
                agent_name=self.name,
                content=result_text,
                token_usage={
                    "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "output_tokens": response.usage.completion_tokens if response.usage else 0,
                },
                metadata={
                    "model": self.model_id,
                    "source_message_id": message.id,
                    "local": True,
                },
            )
        except Exception as e:
            return AgentResponse(
                agent_name=self.name,
                content="",
                success=False,
                error=str(e),
            )

    async def health_check(self) -> bool:
        """Check if the local Llama server is running and reachable."""
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
