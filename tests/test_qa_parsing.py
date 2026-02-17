"""Tests for resilient QA response parsing.

Exercises various formatting deviations that LLMs commonly produce
instead of the exact ITEM_1: PASS/FAIL template.
"""

import pytest

from src.orchestrator import Orchestrator


@pytest.fixture
def orch():
    """Minimal orchestrator for testing parse logic only."""
    return Orchestrator.__new__(Orchestrator)


CHECKLIST = [
    "Function handles empty input",
    "Return type is correct",
    "Edge cases covered",
]


class TestExactFormat:
    """Tests with the exact expected format."""

    def test_all_pass(self, orch):
        content = (
            "ITEM_1: PASS\n"
            "ITEM_2: PASS\n"
            "ITEM_3: PASS\n"
            "SCORE: 9/10\n"
            "FEEDBACK: none"
        )
        orch.qa_pass_score = 7
        result = orch._parse_qa_response(content, CHECKLIST, "codex")
        assert result.passed is True
        assert result.score == 9
        assert len(result.checklist_scores) == 3
        assert all(result.checklist_scores.values())

    def test_some_fail(self, orch):
        content = (
            "ITEM_1: PASS\n"
            "ITEM_2: FAIL\n"
            "ITEM_3: PASS\n"
            "SCORE: 5/10\n"
            "FEEDBACK: Return type is wrong"
        )
        orch.qa_pass_score = 7
        result = orch._parse_qa_response(content, CHECKLIST, "codex")
        assert result.passed is False
        assert result.score == 5
        assert result.checklist_scores[CHECKLIST[1]] is False


class TestCommonDeviations:
    """Tests with formatting variations LLMs commonly produce."""

    def test_markdown_bold(self, orch):
        """Reviewer wraps labels in **bold**."""
        content = (
            "**ITEM_1**: PASS\n"
            "**ITEM_2**: FAIL\n"
            "**ITEM_3**: PASS\n"
            "**SCORE**: 6/10\n"
            "**FEEDBACK**: Fix the return type"
        )
        orch.qa_pass_score = 7
        result = orch._parse_qa_response(content, CHECKLIST, "claude")
        assert result.score == 6
        assert result.checklist_scores[CHECKLIST[0]] is True
        assert result.checklist_scores[CHECKLIST[1]] is False

    def test_no_underscore(self, orch):
        """Reviewer writes 'Item 1' instead of 'ITEM_1'."""
        content = (
            "Item 1: PASS\n"
            "Item 2: PASS\n"
            "Item 3: FAIL\n"
            "Score: 6/10\n"
            "Feedback: Edge cases missing"
        )
        orch.qa_pass_score = 7
        result = orch._parse_qa_response(content, CHECKLIST, "claude")
        assert result.score == 6
        assert result.checklist_scores[CHECKLIST[2]] is False

    def test_bare_numbers(self, orch):
        """Reviewer writes '1: PASS' instead of 'ITEM_1: PASS'."""
        content = (
            "1: PASS\n"
            "2: PASS\n"
            "3: PASS\n"
            "SCORE: 8/10\n"
            "FEEDBACK: none"
        )
        orch.qa_pass_score = 7
        result = orch._parse_qa_response(content, CHECKLIST, "gemini")
        assert result.passed is True
        assert len(result.checklist_scores) == 3

    def test_score_without_denominator(self, orch):
        """Reviewer writes 'Score: 8' instead of 'Score: 8/10'."""
        content = (
            "ITEM_1: PASS\n"
            "SCORE: 8\n"
            "FEEDBACK: looks good"
        )
        orch.qa_pass_score = 7
        result = orch._parse_qa_response(content, CHECKLIST, "codex")
        assert result.score == 8
        assert result.passed is True

    def test_multiline_feedback(self, orch):
        """Feedback spans multiple lines."""
        content = (
            "ITEM_1: FAIL\n"
            "SCORE: 4/10\n"
            "FEEDBACK: Several issues found:\n"
            "- Missing error handling\n"
            "- No input validation"
        )
        orch.qa_pass_score = 7
        result = orch._parse_qa_response(content, CHECKLIST, "codex")
        assert "Missing error handling" in result.feedback
        assert "No input validation" in result.feedback

    def test_score_clamped(self, orch):
        """Score > 10 gets clamped."""
        content = "SCORE: 15/10\nFEEDBACK: perfect"
        orch.qa_pass_score = 7
        result = orch._parse_qa_response(content, CHECKLIST, "codex")
        assert result.score == 10

    def test_empty_response_defaults(self, orch):
        """Completely unparseable response defaults gracefully."""
        content = "I think this looks great!"
        orch.qa_pass_score = 7
        result = orch._parse_qa_response(content, CHECKLIST, "codex")
        # Defaults: score=10, passed=True (fail-open so we don't block on bad parsing)
        assert result.score == 10
        assert result.passed is True
        assert result.checklist_scores == {}

    def test_overall_keyword(self, orch):
        """Reviewer uses 'Overall:' instead of 'Score:'."""
        content = (
            "ITEM_1: PASS\n"
            "Overall: 7/10\n"
            "Feedback: acceptable"
        )
        orch.qa_pass_score = 7
        result = orch._parse_qa_response(content, CHECKLIST, "codex")
        assert result.score == 7
        assert result.passed is True
