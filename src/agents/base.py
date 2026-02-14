"""Base agent interface for all AI model agents."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentCapability(str, Enum):
    """Capabilities that an agent can support."""

    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    TRANSLATION = "translation"
    IMAGE_UNDERSTANDING = "image_understanding"
    IMAGE_GENERATION = "image_generation"
    AUDIO_UNDERSTANDING = "audio_understanding"
    REASONING = "reasoning"
    MATH = "math"
    SUMMARIZATION = "summarization"
    CONVERSATION = "conversation"
    FUNCTION_CALLING = "function_calling"
    LOCAL_EXECUTION = "local_execution"


class AgentMessage(BaseModel):
    """A message passed between agents."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    source: str
    target: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    parent_message_id: str | None = None


class AgentResponse(BaseModel):
    """Response from an agent after processing a message."""

    agent_name: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_usage: dict[str, int] = Field(default_factory=dict)
    success: bool = True
    error: str | None = None


class BaseAgent(ABC):
    """Abstract base class for all AI model agents."""

    def __init__(self, name: str, model_id: str, capabilities: list[AgentCapability]):
        self.name = name
        self.model_id = model_id
        self.capabilities = capabilities
        self._context: list[dict[str, str]] = []
        self._max_context_messages = 50

    @abstractmethod
    async def send(self, message: AgentMessage) -> AgentResponse:
        """Send a message to this agent and get a response."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if this agent's backing service is reachable."""

    def supports(self, capability: AgentCapability) -> bool:
        """Check if this agent supports a given capability."""
        return capability in self.capabilities

    def add_to_context(self, role: str, content: str) -> None:
        """Add a message to the agent's conversation context."""
        self._context.append({"role": role, "content": content})
        if len(self._context) > self._max_context_messages:
            self._context = self._context[-self._max_context_messages :]

    def clear_context(self) -> None:
        """Clear the agent's conversation context."""
        self._context.clear()

    def get_context_summary(self) -> str:
        """Return a compressed summary of context to conserve tokens."""
        if not self._context:
            return ""
        messages = [f"[{m['role']}]: {m['content'][:200]}" for m in self._context[-5:]]
        return "\n".join(messages)

    def __repr__(self) -> str:
        caps = ", ".join(c.value for c in self.capabilities)
        return f"{self.__class__.__name__}(name={self.name!r}, model={self.model_id!r}, caps=[{caps}])"
