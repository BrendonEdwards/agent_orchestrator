"""Llama Local agent - the dogs body of the orchestrator.

Llama via ollama handles the grunt work: simple, repetitive tasks that
don't require complex reasoning. It's fast, free, local, and always
available. The thinking agents (Claude, Codex, Gemini) delegate their
menial subtasks here to conserve their own context and API costs.
"""

from __future__ import annotations

import os
from typing import Any

from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent


class LlamaAgent(BaseAgent):
    """Llama agent for grunt work via ollama.

    This is the dogs body - it handles simple tasks so the thinking
    agents don't waste their context windows on:
    - Formatting and cleanup
    - Simple text extraction or transformation
    - Boilerplate generation
    - Data munging and conversion
    - Repetitive batch operations
    - Anything that needs doing but not thinking
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

    def _get_client(self) -> Any:
        """Get an OpenAI-compatible client pointing at ollama."""
        if self._client is None:
            import openai

            self._client = openai.AsyncOpenAI(
                base_url=self._base_url,
                api_key="not-needed",
            )
        return self._client

    async def send(self, message: AgentMessage) -> AgentResponse:
        """Send a simple task to the local Llama instance."""
        try:
            client = self._get_client()

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You do grunt work. Simple tasks, no overthinking. "
                        "Be direct, output only what's asked for, nothing extra."
                    ),
                },
                {"role": "user", "content": message.content},
            ]

            response = await client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_completion_tokens=2048,
            )

            result_text = response.choices[0].message.content or ""

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

    async def do_grunt_work(self, task: str) -> str:
        """Convenience method for simple fire-and-forget tasks."""
        msg = AgentMessage(source="orchestrator", target="llama", content=task)
        resp = await self.send(msg)
        return resp.content if resp.success else ""

    async def batch(self, tasks: list[str]) -> list[str]:
        """Run multiple simple tasks. Returns results in order."""
        import asyncio

        msgs = [AgentMessage(source="orchestrator", target="llama", content=t) for t in tasks]
        responses = await asyncio.gather(*(self.send(m) for m in msgs))
        return [r.content if r.success else "" for r in responses]

    async def health_check(self) -> bool:
        """Check if ollama is running."""
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
