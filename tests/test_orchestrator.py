"""Tests for orchestrator logic using fake agents (no subprocesses)."""

import pytest

from src.agents.base import AgentMessage
from src.memento.memento import Memento
from src.orchestrator import (
    CAPABILITY_PROVIDERS,
    Orchestrator,
    _VALID_CAPABILITIES,
)


class TestCapabilityRouting:
    def test_all_capabilities_have_providers(self):
        for cap in _VALID_CAPABILITIES:
            assert cap in CAPABILITY_PROVIDERS

    def test_resolve_provider(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch._agents = {}
        assert orch._resolve_provider("REASONING") == "claude"
        assert orch._resolve_provider("CODE") == "codex"
        assert orch._resolve_provider("MULTIMODAL") == "gemini"
        assert orch._resolve_provider("FAST") == "gemini"
        # Unknown defaults to reasoning
        assert orch._resolve_provider("UNKNOWN") == "claude"


class TestGruntWorkDetection:
    def test_detects_grunt_keywords(self):
        orch = Orchestrator.__new__(Orchestrator)
        assert orch._is_grunt_work("format this json")
        assert orch._is_grunt_work("Convert CSV to JSON")
        assert orch._is_grunt_work("sort this list alphabetically")

    def test_not_grunt_for_complex(self):
        orch = Orchestrator.__new__(Orchestrator)
        assert not orch._is_grunt_work("design a microservice architecture")
        assert not orch._is_grunt_work("implement binary search tree")


class TestSubtaskParsing:
    def test_parse_tagged_subtasks(self):
        orch = Orchestrator.__new__(Orchestrator)
        content = (
            "1. [CODE] Implement the REST API\n"
            "2. [REASONING] Design the database schema\n"
            "3. [FAST] Generate boilerplate configs\n"
        )
        result = orch._parse_tagged_subtasks(content)
        assert len(result) == 3
        assert result[0] == ("Implement the REST API", "CODE")
        assert result[1] == ("Design the database schema", "REASONING")
        assert result[2] == ("Generate boilerplate configs", "FAST")

    def test_parse_no_numbering(self):
        orch = Orchestrator.__new__(Orchestrator)
        content = "[CODE] Build the widget"
        result = orch._parse_tagged_subtasks(content)
        assert len(result) == 1
        assert result[0] == ("Build the widget", "CODE")

    def test_parse_dash_prefix(self):
        orch = Orchestrator.__new__(Orchestrator)
        content = "- [MULTIMODAL] Process the images\n- [REASONING] Analyze results"
        result = orch._parse_tagged_subtasks(content)
        assert len(result) == 2

    def test_parse_untagged_defaults_reasoning(self):
        orch = Orchestrator.__new__(Orchestrator)
        content = "Just do the thing"
        result = orch._parse_tagged_subtasks(content)
        assert len(result) == 1
        assert result[0][1] == "REASONING"

    def test_parse_empty(self):
        orch = Orchestrator.__new__(Orchestrator)
        assert orch._parse_tagged_subtasks("") == []
        assert orch._parse_tagged_subtasks("   \n   \n") == []


class TestSpawn:
    def test_spawn_increments_depth(self):
        orch = Orchestrator(qa_enabled=False)
        child = orch.spawn("test subtask")
        assert child.depth == orch.depth + 1

    def test_spawn_respects_max_depth(self):
        orch = Orchestrator(max_depth=1, depth=1, qa_enabled=False)
        with pytest.raises(RecursionError):
            orch.spawn("too deep")

    def test_spawn_inherits_config(self):
        orch = Orchestrator(qa_enabled=False, max_qa_retries=5, qa_pass_score=9)
        child = orch.spawn("test")
        assert child.max_qa_retries == 5
        assert child.qa_pass_score == 9
