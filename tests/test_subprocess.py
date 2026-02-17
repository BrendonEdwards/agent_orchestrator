"""Tests for the shared subprocess runner with retry logic."""

import pytest

from src.agents._subprocess import run_cli
from src.config import OrchestratorConfig


@pytest.mark.asyncio
async def test_successful_command():
    """Running a real command that succeeds."""
    config = OrchestratorConfig(agent_timeout=5, agent_retries=0)
    resp = await run_cli("test", ["echo", "hello world"], config=config)
    assert resp.success
    assert resp.content == "hello world"


@pytest.mark.asyncio
async def test_failed_command():
    """Non-zero exit code returns failure."""
    config = OrchestratorConfig(agent_timeout=5, agent_retries=0)
    resp = await run_cli("test", ["false"], config=config)
    assert not resp.success


@pytest.mark.asyncio
async def test_missing_binary():
    """Missing binary returns failure immediately (no retry)."""
    config = OrchestratorConfig(agent_timeout=5, agent_retries=2)
    resp = await run_cli("test", ["nonexistent_binary_xyz"], config=config)
    assert not resp.success
    assert "not found" in resp.error.lower()


@pytest.mark.asyncio
async def test_timeout():
    """Command that exceeds timeout is killed."""
    config = OrchestratorConfig(agent_timeout=0.5, agent_retries=0)
    resp = await run_cli("test", ["sleep", "10"], config=config)
    assert not resp.success
    assert "timed out" in resp.error.lower()


@pytest.mark.asyncio
async def test_retry_on_failure():
    """Retries the configured number of times before giving up."""
    config = OrchestratorConfig(agent_timeout=5, agent_retries=1, retry_base_delay=0.01)
    resp = await run_cli("test", ["false"], config=config)
    assert not resp.success
    assert "2 attempts" in resp.error  # 1 original + 1 retry


@pytest.mark.asyncio
async def test_metadata_passed_through():
    """Metadata makes it into the response."""
    config = OrchestratorConfig(agent_timeout=5, agent_retries=0)
    resp = await run_cli(
        "test", ["echo", "hi"],
        config=config,
        metadata={"cli": "/usr/bin/echo", "model": "test-1"},
    )
    assert resp.metadata["cli"] == "/usr/bin/echo"
