"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest

from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent
from src.config import OrchestratorConfig


class FakeAgent(BaseAgent):
    """In-memory agent that returns canned responses. No subprocess needed."""

    def __init__(
        self,
        name: str = "fake",
        *,
        response: str = "ok",
        success: bool = True,
        token_usage: dict | None = None,
    ):
        super().__init__(
            name=name,
            model_id="fake-1.0",
            capabilities=[
                AgentCapability.TEXT_GENERATION,
                AgentCapability.CODE_GENERATION,
                AgentCapability.REASONING,
            ],
        )
        self._response = response
        self._success = success
        self._token_usage = token_usage or {"input_tokens": 10, "output_tokens": 20}
        self.calls: list[AgentMessage] = []

    async def send(self, message: AgentMessage) -> AgentResponse:
        self.calls.append(message)
        return AgentResponse(
            agent_name=self.name,
            content=self._response,
            success=self._success,
            token_usage=self._token_usage,
            error=None if self._success else "fake error",
        )

    async def health_check(self) -> bool:
        return self._success


@pytest.fixture
def fast_config() -> OrchestratorConfig:
    """Config with very short timeouts for testing."""
    return OrchestratorConfig(
        agent_timeout=5,
        health_check_timeout=2,
        stall_timeout=3,
        agent_retries=0,
    )


@pytest.fixture
def fake_claude() -> FakeAgent:
    return FakeAgent(name="claude", response="synthesized result")


@pytest.fixture
def fake_codex() -> FakeAgent:
    return FakeAgent(name="codex", response="def hello(): pass")


@pytest.fixture
def fake_gemini() -> FakeAgent:
    return FakeAgent(name="gemini", response="formatted output")


@pytest.fixture
def fake_team(fake_claude, fake_codex, fake_gemini) -> dict[str, FakeAgent]:
    return {"claude": fake_claude, "codex": fake_codex, "gemini": fake_gemini}
