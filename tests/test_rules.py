"""Tests for the rules loader."""

import tempfile
from pathlib import Path

from src.rules.loader import RulesLoader


def test_defaults_when_no_file():
    loader = RulesLoader()
    models = loader.load()
    assert "claude" in models
    assert "codex" in models
    assert "gemini" in models
    assert "llama" in models


def test_get_best_agent_defaults():
    loader = RulesLoader()
    loader.load()
    # "reasoning" should favour claude
    result = loader.get_best_agent_for("complex reasoning task")
    assert result == "claude"


def test_parse_markdown(tmp_path):
    md = tmp_path / "rules.md"
    md.write_text(
        "## Claude\n"
        "### Strengths\n"
        "- Deep reasoning\n"
        "- Code review\n"
        "### Weaknesses\n"
        "- Image generation\n"
        "## Codex\n"
        "### Strengths\n"
        "- Code generation\n"
    )
    loader = RulesLoader(rules_path=str(md))
    models = loader.load()
    assert "claude" in models
    assert "Deep reasoning" in models["claude"].strengths
    assert "Image generation" in models["claude"].weaknesses
    assert "codex" in models
    assert "Code generation" in models["codex"].strengths


def test_routing_context():
    loader = RulesLoader()
    loader.load()
    ctx = loader.get_routing_context()
    assert "claude" in ctx
    assert "strengths=" in ctx
