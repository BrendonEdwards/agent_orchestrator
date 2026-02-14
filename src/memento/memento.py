"""Memento - audit and memory system for the agent orchestrator.

Inspired by the whiteboard note: "Memento - how did we write the notes?"
Memento tracks every decision, delegation, and result in the orchestration
pipeline, providing a full audit trail of how the agents arrived at their answers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class MementoEntry(BaseModel):
    """A single entry in the Memento audit trail."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: str  # e.g., "task_received", "delegation", "response", "synthesis"
    source: str  # which agent or component generated this entry
    target: str | None = None  # which agent or component received
    content: str  # description of what happened
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_compact(self) -> str:
        """Return a compact single-line representation."""
        ts = self.timestamp.strftime("%H:%M:%S")
        target_str = f" -> {self.target}" if self.target else ""
        return f"[{ts}] {self.event_type}: {self.source}{target_str} | {self.content[:120]}"


class Memento:
    """Audit and memory system that records how the agents produced their outputs.

    The Memento system answers the question "how did we write the notes?" by
    maintaining a chronological log of all orchestration events including:
    - Task analysis and routing decisions
    - Agent delegations and their results
    - Context translations and compressions
    - Final synthesis steps

    Entries can be persisted to disk for long-term audit trails.
    """

    def __init__(self, persist_dir: str | None = None):
        self._entries: list[MementoEntry] = []
        self._persist_dir = Path(persist_dir) if persist_dir else None
        if self._persist_dir:
            self._persist_dir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        event_type: str,
        source: str,
        content: str,
        target: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MementoEntry:
        """Record a new event in the Memento trail."""
        entry = MementoEntry(
            event_type=event_type,
            source=source,
            target=target,
            content=content,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        return entry

    def get_trail(self, last_n: int | None = None) -> list[MementoEntry]:
        """Get the audit trail, optionally limited to the last N entries."""
        if last_n is not None:
            return self._entries[-last_n:]
        return list(self._entries)

    def get_trail_for_agent(self, agent_name: str) -> list[MementoEntry]:
        """Get all entries involving a specific agent."""
        return [
            e
            for e in self._entries
            if e.source == agent_name or e.target == agent_name
        ]

    def get_trail_summary(self, last_n: int = 20) -> str:
        """Get a human-readable summary of the recent audit trail."""
        entries = self.get_trail(last_n)
        if not entries:
            return "No entries in memento trail."
        return "\n".join(e.to_compact() for e in entries)

    def get_decision_chain(self) -> list[MementoEntry]:
        """Get only the decision/routing entries to understand the reasoning chain."""
        decision_types = {"task_received", "analysis", "delegation", "synthesis", "routing"}
        return [e for e in self._entries if e.event_type in decision_types]

    def save(self, filename: str | None = None) -> Path | None:
        """Persist the current trail to a JSON file."""
        if not self._persist_dir:
            return None
        if filename is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"memento_{ts}.json"
        path = self._persist_dir / filename
        data = [e.model_dump(mode="json") for e in self._entries]
        path.write_text(json.dumps(data, indent=2, default=str))
        return path

    def load(self, path: str | Path) -> None:
        """Load a previously saved trail."""
        path = Path(path)
        data = json.loads(path.read_text())
        for item in data:
            self._entries.append(MementoEntry(**item))

    def clear(self) -> None:
        """Clear all entries from the trail."""
        self._entries.clear()

    @property
    def entry_count(self) -> int:
        return len(self._entries)
