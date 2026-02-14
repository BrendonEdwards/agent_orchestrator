"""Codex agent - specialized for code generation and translation.

As noted on the whiteboard: "Immediate language translation to conserve context."
Codex compresses and translates between programming languages and natural language
representations to minimize token usage across the agent network.
"""

from __future__ import annotations

import os
from typing import Any

from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent


class CodexAgent(BaseAgent):
    """Codex agent for code generation and context-conserving translation.

    Key responsibilities:
    - Code generation and completion
    - Language-to-language translation (both programming and natural languages)
    - Context compression: translating verbose descriptions into compact
      representations before passing to other agents
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
        """Send a message to Codex and return the response."""
        try:
            client = self._get_client()

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a code-focused AI agent within a multi-agent system. "
                        "You specialize in code generation, code review, and translating "
                        "between programming languages. When communicating results back, "
                        "be concise to conserve context window space for other agents."
                    ),
                },
            ]
            for ctx in self._context:
                messages.append({"role": ctx["role"], "content": ctx["content"]})
            messages.append({"role": "user", "content": message.content})

            response = await client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_completion_tokens=4096,
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
                metadata={"model": self.model_id, "source_message_id": message.id},
            )
        except Exception as e:
            return AgentResponse(
                agent_name=self.name,
                content="",
                success=False,
                error=str(e),
            )

    async def translate_for_context(self, content: str, target_format: str = "compact") -> str:
        """Translate content into a compact form to conserve context.

        This implements the whiteboard's "Immediate language translation
        to conserve context" concept. It compresses verbose outputs from
        one agent into a compact representation before routing to another.
        """
        prompt = (
            f"Compress the following content into a {target_format} representation "
            f"that preserves all key information but uses minimal tokens. "
            f"Use abbreviations, remove filler words, and structure as key-value pairs "
            f"where possible.\n\n"
            f"Content:\n{content}"
        )

        msg = AgentMessage(source="router", target="codex", content=prompt)
        response = await self.send(msg)
        return response.content if response.success else content

    async def translate_code(self, code: str, source_lang: str, target_lang: str) -> AgentResponse:
        """Translate code from one programming language to another."""
        prompt = (
            f"Translate the following {source_lang} code to {target_lang}. "
            f"Preserve functionality exactly. Only output the translated code.\n\n"
            f"```{source_lang}\n{code}\n```"
        )
        msg = AgentMessage(source="orchestrator", target="codex", content=prompt)
        return await self.send(msg)

    async def health_check(self) -> bool:
        """Check if the OpenAI API is reachable."""
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
