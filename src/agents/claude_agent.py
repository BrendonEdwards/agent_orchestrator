"""Claude agent - the brain of the orchestrator.

Stateless. Every call is fresh. Memento notes are the only context.
"""

from __future__ import annotations

import os
from typing import Any

from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent


class ClaudeAgent(BaseAgent):
    """Claude: task analysis, complex reasoning, synthesis.

    Stateless - no conversation history. Gets memento survival notes
    and the current task, nothing else.
    """

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
    ):
        super().__init__(
            name="claude",
            model_id=model_id,
            capabilities=[
                AgentCapability.TEXT_GENERATION,
                AgentCapability.CODE_GENERATION,
                AgentCapability.CODE_REVIEW,
                AgentCapability.REASONING,
                AgentCapability.MATH,
                AgentCapability.SUMMARIZATION,
            ],
        )
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def send(self, message: AgentMessage) -> AgentResponse:
        """Fresh call to Claude. No history, just memento notes + task."""
        try:
            client = self._get_client()

            system = (
                "You are the brain of a multi-agent system. "
                "You have no memory of previous interactions. "
                "Your survival notes below are all you know about prior context."
            )
            if message.memento:
                system += f"\n\nSurvival notes:\n{message.memento}"

            response = await client.messages.create(
                model=self.model_id,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": message.content}],
            )

            return AgentResponse(
                agent_name=self.name,
                content=response.content[0].text,
                token_usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                metadata={"model": self.model_id},
            )
        except Exception as e:
            return AgentResponse(
                agent_name=self.name, content="", success=False, error=str(e),
            )

    async def synthesize(self, results: list[AgentResponse], memento: str = "") -> AgentResponse:
        """Synthesize results from multiple agents. Fresh call."""
        parts = []
        for r in results:
            if r.success:
                parts.append(f"[{r.agent_name}]: {r.content}")
            else:
                parts.append(f"[{r.agent_name}]: (failed) {r.error}")

        msg = AgentMessage(
            source="orchestrator",
            target="claude",
            content="Synthesize these agent responses into one coherent answer.\n\n"
            + "\n\n".join(parts),
            memento=memento,
        )
        return await self.send(msg)

    async def health_check(self) -> bool:
        try:
            client = self._get_client()
            response = await client.messages.create(
                model=self.model_id,
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            return len(response.content) > 0
        except Exception:
            return False
