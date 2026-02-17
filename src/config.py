"""Central configuration for tunable constants.

All magic numbers live here instead of scattered across modules.
Override via constructor args or environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class OrchestratorConfig:
    """All tunable constants in one place."""

    # Agent subprocess timeout (seconds)
    agent_timeout: float = float(os.environ.get("ORCH_AGENT_TIMEOUT", "300"))

    # Health-check timeout (seconds)
    health_check_timeout: float = float(os.environ.get("ORCH_HEALTH_TIMEOUT", "30"))

    # Stall detection threshold (seconds)
    stall_timeout: float = float(os.environ.get("ORCH_STALL_TIMEOUT", "120"))

    # QA pass threshold (out of 10)
    qa_pass_score: int = int(os.environ.get("ORCH_QA_PASS_SCORE", "7"))

    # Max QA retry attempts before accepting best effort
    max_qa_retries: int = int(os.environ.get("ORCH_MAX_QA_RETRIES", "2"))

    # Max fractal recursion depth
    max_depth: int = int(os.environ.get("ORCH_MAX_DEPTH", "10"))

    # Memento: max notes before compaction
    max_memento_notes: int = int(os.environ.get("ORCH_MAX_NOTES", "20"))

    # Memento: max characters per note
    max_note_length: int = int(os.environ.get("ORCH_MAX_NOTE_LEN", "200"))

    # Agent subprocess retry attempts on transient failures
    agent_retries: int = int(os.environ.get("ORCH_AGENT_RETRIES", "2"))

    # Base delay for exponential backoff (seconds)
    retry_base_delay: float = float(os.environ.get("ORCH_RETRY_BASE_DELAY", "1.0"))


# Module-level default instance
DEFAULT_CONFIG = OrchestratorConfig()
