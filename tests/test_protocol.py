"""Tests for the compact inter-agent protocol."""

from src.routing.protocol import (
    decode,
    encode,
    is_protocol_message,
    result_message,
    task_message,
    to_human,
)


def test_encode_decode_roundtrip():
    fields = {"task_type": "codegen", "language": "py", "name": "sum_evens"}
    encoded = encode(fields)
    decoded = decode(encoded)
    assert decoded == fields


def test_encode_format():
    encoded = encode({"task_type": "review", "language": "go"})
    assert "T:review" in encoded
    assert "L:go" in encoded
    assert "|" in encoded


def test_decode_basic():
    result = decode("T:codegen|L:py|N:sum_evens")
    assert result["task_type"] == "codegen"
    assert result["language"] == "py"
    assert result["name"] == "sum_evens"


def test_escaped_separator():
    fields = {"description": "a|b|c"}
    encoded = encode(fields)
    decoded = decode(encoded)
    assert decoded["description"] == "a|b|c"


def test_escaped_newline():
    fields = {"description": "line1\nline2"}
    encoded = encode(fields)
    assert "\\n" in encoded
    decoded = decode(encoded)
    assert decoded["description"] == "line1\nline2"


def test_task_message():
    msg = task_message("codegen", "sum evens", language="py", name="sum_evens")
    decoded = decode(msg)
    assert decoded["task_type"] == "codegen"
    assert decoded["language"] == "py"
    assert decoded["name"] == "sum_evens"


def test_result_message():
    msg = result_message("codex", "def hello(): pass")
    decoded = decode(msg)
    assert decoded["source"] == "codex"
    assert decoded["result"] == "def hello(): pass"


def test_result_message_with_error():
    msg = result_message("codex", "", error="timeout")
    decoded = decode(msg)
    assert decoded["error"] == "timeout"


def test_is_protocol_message():
    assert is_protocol_message("T:codegen|L:py|N:hello")
    assert not is_protocol_message("Hello, please write a function")
    assert not is_protocol_message("just a string")


def test_to_human():
    msg = "T:codegen|L:py"
    human = to_human(msg)
    assert "task_type: codegen" in human
    assert "language: py" in human
