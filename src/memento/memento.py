"""Memento - anti-context-rot memory system for the agent orchestrator.

Inspired by the film Memento: the protagonist has no long-term memory but
survives by keeping extremely concise notes. Similarly, as LLM context
windows fill up, reasoning quality degrades ("context rot"). Memento
combats this by maintaining a small set of ultra-concise survival notes
that let agents keep functioning even as earlier context is lost.

The key insight: you don't need to remember everything. You need the
right tiny notes to reconstruct what matters.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.config import DEFAULT_CONFIG


class Note(BaseModel):
    """A single concise survival note, like a tattoo in the film."""

    key: str  # Short label, e.g. "goal", "usr_wants", "ctx:codex_result"
    value: str  # The concise note itself - must be brief
    priority: int = 1  # Higher = more important to keep during compaction
    created: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __str__(self) -> str:
        return f"{self.key}={self.value}"


class Memento:
    """Anti-context-rot memory system.

    Like the film: no long-term memory, just concise notes that let you
    function. When the note space fills up, low-priority notes are
    compacted or dropped to make room - just like the protagonist
    choosing which notes matter most.

    Usage:
        mem = Memento()
        mem.note("goal", "build REST API for users")
        mem.note("constraint", "must use PostgreSQL")
        mem.note("done:auth", "JWT auth implemented, tested")

        # Later, when context is getting long, inject the notes
        context = mem.briefing()
        # -> "goal=build REST API for users | constraint=must use PostgreSQL | ..."
    """

    def __init__(self, persist_path: str | None = None, max_notes: int = DEFAULT_CONFIG.max_memento_notes):
        self._notes: dict[str, Note] = {}
        self._max_notes = max_notes
        self._persist_path = Path(persist_path) if persist_path else None

    def note(self, key: str, value: str, priority: int = 1) -> None:
        """Write a concise note. Overwrites any existing note with the same key.

        Keep values SHORT. This is the whole point - if you can't say it
        in ~200 chars, you're remembering too much detail.
        """
        max_len = DEFAULT_CONFIG.max_note_length
        if len(value) > max_len:
            value = value[:max_len - 3] + "..."

        self._notes[key] = Note(key=key, value=value, priority=priority)

        # If we've exceeded capacity, compact
        if len(self._notes) > self._max_notes:
            self._compact()

    def forget(self, key: str) -> None:
        """Explicitly forget a note (task done, no longer relevant)."""
        self._notes.pop(key, None)

    def recall(self, key: str) -> str | None:
        """Recall a specific note by key."""
        note = self._notes.get(key)
        return note.value if note else None

    def briefing(self) -> str:
        """Get all notes as a compact briefing string.

        This is what gets injected into agent prompts to fight context rot.
        Format is deliberately terse - every token counts.
        """
        if not self._notes:
            return ""
        # Sort by priority (high first), then by creation time
        sorted_notes = sorted(
            self._notes.values(),
            key=lambda n: (-n.priority, n.created),
        )
        return " | ".join(str(n) for n in sorted_notes)

    def briefing_for(self, prefix: str) -> str:
        """Get notes matching a key prefix (e.g. 'ctx:' for context notes)."""
        matching = [n for k, n in self._notes.items() if k.startswith(prefix)]
        matching.sort(key=lambda n: (-n.priority, n.created))
        return " | ".join(str(n) for n in matching)

    def _compact(self) -> None:
        """Drop lowest-priority notes to stay within capacity.

        Like the film: when you run out of skin to write on, you have
        to decide what's worth remembering.
        """
        if len(self._notes) <= self._max_notes:
            return

        sorted_keys = sorted(
            self._notes.keys(),
            key=lambda k: (self._notes[k].priority, self._notes[k].created),
        )
        # Drop the least important, oldest notes
        to_drop = len(self._notes) - self._max_notes
        for key in sorted_keys[:to_drop]:
            del self._notes[key]

    def save(self) -> Path | None:
        """Persist notes to disk."""
        if not self._persist_path:
            return None
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: n.model_dump(mode="json") for k, n in self._notes.items()}
        self._persist_path.write_text(json.dumps(data, indent=2, default=str))
        return self._persist_path

    def load(self) -> None:
        """Load notes from disk."""
        if not self._persist_path or not self._persist_path.exists():
            return
        data = json.loads(self._persist_path.read_text())
        for key, item in data.items():
            self._notes[key] = Note(**item)

    def clear(self) -> None:
        """Wipe all notes."""
        self._notes.clear()

    @property
    def count(self) -> int:
        return len(self._notes)

    @property
    def all_notes(self) -> dict[str, str]:
        """All notes as a simple key->value dict."""
        return {k: n.value for k, n in self._notes.items()}
