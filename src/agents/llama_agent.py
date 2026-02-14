"""Llama agent - the dogs body. Grunt work.

Stateless (obviously - it's doing grunt work, not thinking).
No memento notes needed for simple tasks.

Runs on:
- Ollama locally: LLAMA_BASE_URL=http://localhost:11434/v1 (default)
- Groq cloud:     LLAMA_BASE_URL=https://api.groq.com/openai/v1
                  GROQ_API_KEY=your-key
- Together cloud: LLAMA_BASE_URL=https://api.together.xyz/v1
                  TOGETHER_API_KEY=your-key

Set the base URL to a cloud provider and you never run anything locally.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent

# Known cloud providers that need API keys
_CLOUD_PROVIDERS = {
    "groq.com": "GROQ_API_KEY",
    "together.xyz": "TOGETHER_API_KEY",
}


class LlamaAgent(BaseAgent):
    """Llama: grunt work. Formatting, boilerplate, simple tasks.

    Stateless. Runs via any OpenAI-compatible endpoint:
    - Ollama (local, default)
    - Groq (cloud, fast)
    - Together AI (cloud)
    - Any OpenAI-compatible server
    """

    def __init__(
        self,
        model_id: str = "llama3.2",
        base_url: str | None = None,
    ):
        super().__init__(
            name="llama",
            model_id=model_id,
            capabilities=[
                AgentCapability.TEXT_GENERATION,
                AgentCapability.SUMMARIZATION,
                AgentCapability.LOCAL_EXECUTION,
            ],
        )
        self._base_url = base_url or os.environ.get(
            "LLAMA_BASE_URL", "http://localhost:11434/v1"
        )
        self._client: Any = None

    def _resolve_api_key(self) -> str:
        """Resolve API key based on the base URL.

        Local Ollama doesn't need one. Cloud providers do.
        """
        for domain, env_var in _CLOUD_PROVIDERS.items():
            if domain in self._base_url:
                return os.environ.get(env_var, "")
        return "not-needed"

    def _get_client(self) -> Any:
        if self._client is None:
            import openai

            self._client = openai.AsyncOpenAI(
                base_url=self._base_url,
                api_key=self._resolve_api_key(),
            )
        return self._client

    async def send(self, message: AgentMessage) -> AgentResponse:
        """Fresh call to ollama. No history, no context needed."""
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
                metadata={"model": self.model_id, "local": True},
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
