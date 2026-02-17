"""Tests for the memento memory system."""

import json
import tempfile
from pathlib import Path

from src.memento.memento import Memento, Note


def test_note_and_recall():
    m = Memento()
    m.note("goal", "build api")
    assert m.recall("goal") == "build api"
    assert m.count == 1


def test_overwrite():
    m = Memento()
    m.note("x", "first")
    m.note("x", "second")
    assert m.recall("x") == "second"
    assert m.count == 1


def test_forget():
    m = Memento()
    m.note("a", "1")
    m.forget("a")
    assert m.recall("a") is None
    assert m.count == 0


def test_briefing_format():
    m = Memento()
    m.note("a", "1", priority=2)
    m.note("b", "2", priority=1)
    briefing = m.briefing()
    # Higher priority first
    assert briefing.startswith("a=1")
    assert "b=2" in briefing
    assert " | " in briefing


def test_briefing_for_prefix():
    m = Memento()
    m.note("ctx:a", "1")
    m.note("ctx:b", "2")
    m.note("goal", "3")
    result = m.briefing_for("ctx:")
    assert "ctx:a=1" in result
    assert "ctx:b=2" in result
    assert "goal" not in result


def test_compaction():
    m = Memento(max_notes=3)
    m.note("a", "1", priority=1)
    m.note("b", "2", priority=3)
    m.note("c", "3", priority=2)
    m.note("d", "4", priority=1)  # Should trigger compaction
    assert m.count == 3
    # Lowest priority ("a") should have been dropped
    assert m.recall("a") is None
    assert m.recall("b") == "2"


def test_long_value_truncated():
    m = Memento()
    long = "x" * 300
    m.note("k", long)
    val = m.recall("k")
    assert len(val) <= 200
    assert val.endswith("...")


def test_save_load(tmp_path):
    path = str(tmp_path / "memento.json")
    m = Memento(persist_path=path)
    m.note("goal", "test save")
    m.save()

    m2 = Memento(persist_path=path)
    m2.load()
    assert m2.recall("goal") == "test save"


def test_clear():
    m = Memento()
    m.note("a", "1")
    m.note("b", "2")
    m.clear()
    assert m.count == 0
    assert m.briefing() == ""


def test_all_notes():
    m = Memento()
    m.note("a", "1")
    m.note("b", "2")
    assert m.all_notes == {"a": "1", "b": "2"}
