"""Base agent interface for all AI model agents.

Agents are STATELESS. Every call is a fresh spawn with zero prior context.
The only "memory" comes from memento notes injected into the message.
This is by design - context rot is the enemy. A fresh agent with good
notes outperforms a stale agent with a full context window.
"""

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
    LOCAL_EXECUTION = "local_execution"


class AgentMessage(BaseModel):
    """A message to send to an agent. Includes memento notes as context."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    source: str
    target: str
    content: str
    memento: str = ""  # Survival notes - the only context the agent gets
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentResponse(BaseModel):
    """Response from an agent after processing a message."""

    agent_name: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_usage: dict[str, int] = Field(default_factory=dict)
    success: bool = True
    error: str | None = None


class BaseAgent(ABC):
    """Abstract base class for all AI model agents.

    Agents are stateless. No conversation history, no accumulated context.
    Each send() is a fresh call. The memento field on AgentMessage is
    the only context an agent receives - concise survival notes that
    let it function without needing to remember anything.
    """

    def __init__(self, name: str, model_id: str, capabilities: list[AgentCapability]):
        self.name = name
        self.model_id = model_id
        self.capabilities = capabilities

    @abstractmethod
    async def send(self, message: AgentMessage) -> AgentResponse:
        """Send a message to this agent and get a response.

        Each call is a fresh spawn. The message.memento field contains
        concise survival notes - the only context provided.
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if this agent's backing service is reachable."""

    def supports(self, capability: AgentCapability) -> bool:
        """Check if this agent supports a given capability."""
        return capability in self.capabilities

    def __repr__(self) -> str:
        caps = ", ".join(c.value for c in self.capabilities)
        return f"{self.__class__.__name__}(name={self.name!r}, model={self.model_id!r}, caps=[{caps}])"
