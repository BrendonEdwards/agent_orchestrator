"""Llama agent - the dogs body. Grunt work via Groq CLI or API.

For grunt work we still use the Groq API (free tier) since there's
no Llama Pro subscription. Groq's free tier is generous enough.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent

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
    """Llama: grunt work via Groq (free tier) or Together AI.

    This is the one agent that still uses an API - but Groq's free tier
    is generous and Llama only handles simple formatting/boilerplate tasks.
    """

    def __init__(
        self,
        model_id: str = "llama-3.2-3b-preview",
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
        prov = _PROVIDERS.get(provider, _PROVIDERS["groq"])
        self._base_url = prov["base_url"]
        self._api_key = os.environ.get(prov["key_env"], "")
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import openai
            self._client = openai.AsyncOpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
            )
        return self._client

    async def send(self, message: AgentMessage) -> AgentResponse:
        """Call Groq/Together API for grunt work."""
        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "Do the task. Output only what's asked for."},
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
                agent_name=self.name, content="",
                success=False, error=str(e),
            )

    async def do_grunt_work(self, task: str) -> str:
        msg = AgentMessage(source="orchestrator", target="llama", content=task)
        resp = await self.send(msg)
        return resp.content if resp.success else ""

    async def batch(self, tasks: list[str]) -> list[str]:
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
