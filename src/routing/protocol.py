"""Compact inter-agent communication protocol.

Agents don't need English to talk to each other. Code is written in
English so humans can read it, but when Claude tells Codex to generate
a function, or Gemini sends results back to Claude, the messages don't
need natural language prose. They need structured, minimal tokens.

This module defines a compact wire format for inter-agent messages that
strips away all the human-friendly padding and uses terse notation.

Example English: "Please write a Python function that takes a list of
integers and returns the sum of all even numbers. The function should
be named sum_evens and handle empty lists by returning 0."

Same thing in protocol:
    T:codegen|L:py|N:sum_evens|I:list[int]|O:int|D:sum even vals,0 if empty

Savings: ~180 chars -> ~60 chars. That's 2/3 fewer tokens burned on
inter-agent chatter.
"""

from __future__ import annotations

from typing import Any


# Field separator
_SEP = "|"
# Key-value separator
_KV = ":"

# Standard field keys (kept to 1-2 chars)
_FIELDS = {
    "T": "task_type",     # codegen, review, translate, summarize, format, extract, describe
    "L": "language",      # py, js, ts, go, rs, etc.
    "N": "name",          # function/class/variable name
    "I": "input",         # input type or description
    "O": "output",        # output type or description
    "D": "description",   # terse task description
    "S": "source",        # source agent
    "R": "result",        # result data
    "E": "error",         # error info
    "C": "context",       # memento briefing
    "P": "priority",      # 1-5
    "F": "format",        # json, csv, text, code, etc.
    "X": "extra",         # overflow for anything else
}

# Inverse mapping for decoding
_FIELD_NAMES = {v: k for k, v in _FIELDS.items()}


def encode(fields: dict[str, str]) -> str:
    """Encode a dict into compact protocol format.

    >>> encode({"task_type": "codegen", "language": "py", "name": "sum_evens"})
    'T:codegen|L:py|N:sum_evens'
    """
    parts = []
    for full_name, value in fields.items():
        key = _FIELD_NAMES.get(full_name, full_name)
        # Escape separators in values
        safe_value = value.replace(_SEP, "\\|").replace("\n", "\\n")
        parts.append(f"{key}{_KV}{safe_value}")
    return _SEP.join(parts)


def decode(message: str) -> dict[str, str]:
    """Decode a protocol message back into a dict with full field names.

    >>> decode('T:codegen|L:py|N:sum_evens')
    {'task_type': 'codegen', 'language': 'py', 'name': 'sum_evens'}
    """
    result = {}
    # Split on unescaped separators
    parts = _split_escaped(message, _SEP)
    for part in parts:
        if _KV not in part:
            continue
        key, value = part.split(_KV, 1)
        key = key.strip()
        value = value.replace("\\|", _SEP).replace("\\n", "\n")
        full_name = _FIELDS.get(key, key)
        result[full_name] = value
    return result


def _split_escaped(text: str, sep: str) -> list[str]:
    """Split on separator, respecting backslash escapes."""
    parts = []
    current = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text) and text[i + 1] == sep:
            current.append(sep)
            i += 2
        elif text[i] == sep:
            parts.append("".join(current))
            current = []
            i += 1
        else:
            current.append(text[i])
            i += 1
    parts.append("".join(current))
    return parts


def task_message(
    task_type: str,
    description: str,
    language: str | None = None,
    name: str | None = None,
    input_spec: str | None = None,
    output_spec: str | None = None,
    context: str | None = None,
) -> str:
    """Build a compact task message."""
    fields = {"task_type": task_type, "description": description}
    if language:
        fields["language"] = language
    if name:
        fields["name"] = name
    if input_spec:
        fields["input"] = input_spec
    if output_spec:
        fields["output"] = output_spec
    if context:
        fields["context"] = context
    return encode(fields)


def result_message(source: str, result: str, error: str | None = None) -> str:
    """Build a compact result message."""
    fields = {"source": source, "result": result}
    if error:
        fields["error"] = error
    return encode(fields)


def is_protocol_message(text: str) -> bool:
    """Check if a string looks like a protocol message vs natural language."""
    if _SEP not in text:
        return False
    parts = text.split(_SEP)
    # At least 2 fields, and most should contain ':'
    if len(parts) < 2:
        return False
    kv_count = sum(1 for p in parts if _KV in p and len(p.split(_KV, 1)[0].strip()) <= 2)
    return kv_count >= len(parts) * 0.5


def to_human(message: str) -> str:
    """Convert a protocol message to human-readable form (for debugging)."""
    fields = decode(message)
    lines = [f"  {name}: {value}" for name, value in fields.items()]
    return "Message:\n" + "\n".join(lines)
