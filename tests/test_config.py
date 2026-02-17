"""Tests for the centralized config module."""

import os

from src.config import DEFAULT_CONFIG, OrchestratorConfig


def test_defaults():
    cfg = OrchestratorConfig()
    assert cfg.agent_timeout == 300
    assert cfg.health_check_timeout == 30
    assert cfg.stall_timeout == 120
    assert cfg.qa_pass_score == 7
    assert cfg.max_qa_retries == 2
    assert cfg.max_depth == 10
    assert cfg.max_memento_notes == 20
    assert cfg.max_note_length == 200
    assert cfg.agent_retries == 2
    assert cfg.retry_base_delay == 1.0


def test_override():
    cfg = OrchestratorConfig(agent_timeout=60, qa_pass_score=9)
    assert cfg.agent_timeout == 60
    assert cfg.qa_pass_score == 9
    # Others stay default
    assert cfg.max_depth == 10


def test_frozen():
    cfg = OrchestratorConfig()
    try:
        cfg.agent_timeout = 999  # type: ignore[misc]
        assert False, "Should have raised"
    except AttributeError:
        pass


def test_default_config_singleton():
    assert isinstance(DEFAULT_CONFIG, OrchestratorConfig)
    assert DEFAULT_CONFIG.agent_timeout == 300
