"""Llama agent - the dogs body. Grunt work via cloud inference.

Stateless (obviously - it's doing grunt work, not thinking).
No memento notes needed for simple tasks.

Runs on cloud providers (no local compute):
- Groq (default): GROQ_API_KEY=your-key   (free tier available, very fast)
- Together AI:    TOGETHER_API_KEY=your-key
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent

# Cloud providers: domain -> (env var for key, base URL)
_PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "key_env": "TOGETHER_API_KEY",
    },
}


class LlamaAgent(BaseAgent):
    """Llama: grunt work via cloud inference. No local compute.

    Stateless. Runs via Groq (default) or Together AI.
    Both have free tiers. Both are fast.
    """

    def __init__(
        self,
        model_id: str = "llama-3.2-3b-preview",
        base_url: str | None = None,
        provider: str = "groq",
    ):
        super().__init__(
            name="llama",
            model_id=model_id,
            capabilities=[
                AgentCapability.TEXT_GENERATION,
                AgentCapability.SUMMARIZATION,
            ],
        )
        # Resolve provider config
        if base_url:
            self._base_url = base_url
            self._api_key = self._detect_key(base_url)
        else:
            prov = _PROVIDERS.get(provider, _PROVIDERS["groq"])
            self._base_url = prov["base_url"]
            self._api_key = os.environ.get(prov["key_env"], "")

        self._client: Any = None

    def _detect_key(self, url: str) -> str:
        """Detect the right API key based on URL."""
        if "groq.com" in url:
            return os.environ.get("GROQ_API_KEY", "")
        if "together.xyz" in url:
            return os.environ.get("TOGETHER_API_KEY", "")
        return os.environ.get("LLAMA_API_KEY", "")

    def _get_client(self) -> Any:
        if self._client is None:
            import openai

            self._client = openai.AsyncOpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
            )
        return self._client

    async def send(self, message: AgentMessage) -> AgentResponse:
        """Fresh call to cloud Llama. No history, no context needed."""
        try:
            client = self._get_client()

            response = await client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "system",
                        "content": "Do the task. Output only what's asked for.",
                    },
                    {"role": "user", "content": message.content},
                ],
                max_completion_tokens=2048,
            )

            return AgentResponse(
                agent_name=self.name,
                content=response.choices[0].message.content or "",
                token_usage={
                    "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "output_tokens": response.usage.completion_tokens if response.usage else 0,
                },
                metadata={"model": self.model_id, "provider": self._base_url},
            )
        except Exception as e:
            return AgentResponse(
                agent_name=self.name, content="", success=False, error=str(e),
            )

    async def do_grunt_work(self, task: str) -> str:
        """Simple fire-and-forget task."""
        msg = AgentMessage(source="orchestrator", target="llama", content=task)
        resp = await self.send(msg)
        return resp.content if resp.success else ""

    async def batch(self, tasks: list[str]) -> list[str]:
        """Run multiple simple tasks in parallel."""
        msgs = [AgentMessage(source="orchestrator", target="llama", content=t) for t in tasks]
        responses = await asyncio.gather(*(self.send(m) for m in msgs))
        return [r.content if r.success else "" for r in responses]

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
