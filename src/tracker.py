"""SwarmTracker - live visibility into the fractal swarm.

Shared across all orchestrator depths. Tracks:
- Token usage per agent, per depth, rolled up to totals
- Active agent calls with elapsed time
- Stuck detection: flags agents that haven't responded in too long
- Progress events emitted to stderr so stdout stays clean for results

The point is NOT to cap cost. $20k is fine if the work is real.
The point is to catch agents that are stuck/idle/hung so you're
not burning wall-clock time waiting on nothing.
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class AgentCall:
    """A single in-flight agent call."""

    call_id: str
    agent_name: str
    depth: int
    task_preview: str  # First ~80 chars of the task
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    success: bool | None = None  # None = still running
    stall_warned: bool = False


@dataclass
class DepthStats:
    """Accumulated stats for one depth level."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    failures: int = 0
    total_wall_secs: float = 0.0


class SwarmTracker:
    """Live swarm observability. Shared across all depth levels.

    Usage:
        tracker = SwarmTracker(stall_timeout=120)
        orch = Orchestrator(..., tracker=tracker)
        result = await orch.run(task)
        tracker.print_summary()
    """

    def __init__(
        self,
        stall_timeout: float = 120.0,
        quiet: bool = False,
        on_event: Callable[[str], None] | None = None,
    ):
        self.stall_timeout = stall_timeout
        self.quiet = quiet
        self._on_event = on_event

        # All calls ever made (completed and in-flight)
        self._calls: list[AgentCall] = []
        # Currently in-flight calls, keyed by call_id
        self._active: dict[str, AgentCall] = {}
        # Stats per depth level
        self._depth_stats: dict[int, DepthStats] = {}
        # Global counters
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_calls: int = 0
        self.total_failures: int = 0

        # Stall detection
        self._stall_checker: asyncio.Task[None] | None = None
        self._running = False

        # Wall clock
        self._started_at: float | None = None

        # Lock for thread safety in asyncio
        self._lock = asyncio.Lock()

        # Call counter for IDs
        self._counter = 0

    # ── Progress output ──────────────────────────────────────────────

    def _emit(self, msg: str) -> None:
        """Emit a progress line to stderr."""
        if self.quiet:
            return
        line = f"  [{self._elapsed():>7.1f}s] {msg}"
        if self._on_event:
            self._on_event(line)
        else:
            print(line, file=sys.stderr, flush=True)

    def _elapsed(self) -> float:
        if self._started_at is None:
            return 0.0
        return time.monotonic() - self._started_at

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        """Call once before the first orchestrator.run()."""
        self._started_at = time.monotonic()
        self._running = True
        self._emit("swarm started")

    def stop(self) -> None:
        """Call after orchestrator.run() completes."""
        self._running = False
        if self._stall_checker and not self._stall_checker.done():
            self._stall_checker.cancel()

    def start_stall_checker(self) -> None:
        """Launch the background stall detection loop."""
        if self._stall_checker is None or self._stall_checker.done():
            self._stall_checker = asyncio.ensure_future(self._check_stalls())

    async def _check_stalls(self) -> None:
        """Periodically check for stalled agent calls."""
        try:
            while self._running:
                await asyncio.sleep(min(self.stall_timeout / 4, 15.0))
                now = time.monotonic()
                async with self._lock:
                    for call in self._active.values():
                        elapsed = now - call.started_at
                        if elapsed >= self.stall_timeout and not call.stall_warned:
                            call.stall_warned = True
                            self._emit(
                                f"⚠ STALL d={call.depth} {call.agent_name} "
                                f"has been running for {elapsed:.0f}s: "
                                f"{call.task_preview}"
                            )
        except asyncio.CancelledError:
            pass

    # ── Agent call tracking ──────────────────────────────────────────

    async def begin_call(
        self, agent_name: str, depth: int, task: str, label: str = ""
    ) -> str:
        """Record the start of an agent call. Returns a call_id."""
        async with self._lock:
            self._counter += 1
            call_id = f"c{self._counter}"
            preview = task[:80].replace("\n", " ")
            call = AgentCall(
                call_id=call_id,
                agent_name=agent_name,
                depth=depth,
                task_preview=preview,
            )
            self._calls.append(call)
            self._active[call_id] = call
            self.total_calls += 1

            tag = f"d={depth}"
            if label:
                tag += f"/{label}"
            self._emit(f"[{tag}] {agent_name} working... ({preview})")

        return call_id

    async def end_call(
        self,
        call_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        success: bool = True,
    ) -> None:
        """Record the end of an agent call."""
        async with self._lock:
            call = self._active.pop(call_id, None)
            if call is None:
                return

            call.finished_at = time.monotonic()
            call.input_tokens = input_tokens
            call.output_tokens = output_tokens
            call.success = success
            wall = call.finished_at - call.started_at

            # Update depth stats
            ds = self._depth_stats.setdefault(call.depth, DepthStats())
            ds.calls += 1
            ds.input_tokens += input_tokens
            ds.output_tokens += output_tokens
            ds.total_wall_secs += wall
            if not success:
                ds.failures += 1

            # Update global counters
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            if not success:
                self.total_failures += 1

            status = "done" if success else "FAILED"
            tok = input_tokens + output_tokens
            self._emit(
                f"[d={call.depth}] {call.agent_name} {status} "
                f"({wall:.1f}s, {tok:,} tok)"
            )

    # ── Orchestrator-level events ────────────────────────────────────

    def event(self, depth: int, msg: str) -> None:
        """Emit a structured orchestrator event."""
        self._emit(f"[d={depth}] {msg}")

    def qa_result(
        self, depth: int, worker: str, reviewer: str, score: int, passed: bool, attempt: int
    ) -> None:
        """Emit a QA result event."""
        verdict = "PASS" if passed else "FAIL"
        self._emit(
            f"[d={depth}] QA: {reviewer} reviewed {worker}'s work: "
            f"{score}/10 {verdict} (attempt {attempt})"
        )

    def decomposed(self, depth: int, subtasks: list[str]) -> None:
        """Emit decomposition event."""
        self._emit(f"[d={depth}] decomposed into {len(subtasks)} subtasks:")
        for i, st in enumerate(subtasks, 1):
            preview = st[:100].replace("\n", " ")
            self._emit(f"  [{i}] {preview}")

    def synthesizing(self, depth: int, n_branches: int) -> None:
        """Emit synthesis event."""
        self._emit(f"[d={depth}] synthesizing {n_branches} branch results...")

    # ── Summary ──────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Print a full summary to stderr."""
        total_wall = self._elapsed()
        total_tok = self.total_input_tokens + self.total_output_tokens

        lines = [
            "",
            "─── swarm summary ───────────────────────────────────",
            f"  wall time:     {total_wall:>10.1f}s",
            f"  total calls:   {self.total_calls:>10,}",
            f"  failures:      {self.total_failures:>10,}",
            f"  input tokens:  {self.total_input_tokens:>10,}",
            f"  output tokens: {self.total_output_tokens:>10,}",
            f"  total tokens:  {total_tok:>10,}",
            "",
        ]

        # Per-depth breakdown
        if self._depth_stats:
            lines.append("  by depth:")
            for depth in sorted(self._depth_stats):
                ds = self._depth_stats[depth]
                tok = ds.input_tokens + ds.output_tokens
                lines.append(
                    f"    d={depth}: {ds.calls} calls, "
                    f"{tok:,} tok, "
                    f"{ds.total_wall_secs:.1f}s agent time, "
                    f"{ds.failures} failures"
                )
            lines.append("")

        # Per-agent breakdown
        agent_stats: dict[str, dict[str, int]] = {}
        for call in self._calls:
            if call.finished_at is not None:
                a = agent_stats.setdefault(
                    call.agent_name, {"calls": 0, "in": 0, "out": 0, "fails": 0}
                )
                a["calls"] += 1
                a["in"] += call.input_tokens
                a["out"] += call.output_tokens
                if call.success is False:
                    a["fails"] += 1

        if agent_stats:
            lines.append("  by agent:")
            for name in sorted(agent_stats):
                a = agent_stats[name]
                tok = a["in"] + a["out"]
                fail_str = f", {a['fails']} failed" if a["fails"] else ""
                lines.append(
                    f"    {name}: {a['calls']} calls, {tok:,} tok{fail_str}"
                )
            lines.append("")

        # Stall warnings
        stalled = [c for c in self._calls if c.stall_warned]
        if stalled:
            lines.append(f"  ⚠ {len(stalled)} stall warning(s) detected")
            lines.append("")

        # Still active (shouldn't happen at summary time, but just in case)
        if self._active:
            lines.append(f"  ⚠ {len(self._active)} call(s) still active:")
            now = time.monotonic()
            for call in self._active.values():
                elapsed = now - call.started_at
                lines.append(
                    f"    {call.agent_name} d={call.depth} "
                    f"running for {elapsed:.0f}s: {call.task_preview}"
                )
            lines.append("")

        lines.append("─────────────────────────────────────────────────────")

        print("\n".join(lines), file=sys.stderr, flush=True)

    # ── Snapshot for programmatic access ─────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Return a dict snapshot of current state (for APIs/testing)."""
        return {
            "elapsed_secs": self._elapsed(),
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "active_calls": len(self._active),
            "stall_warnings": sum(1 for c in self._calls if c.stall_warned),
            "by_depth": {
                d: {
                    "calls": ds.calls,
                    "tokens": ds.input_tokens + ds.output_tokens,
                    "failures": ds.failures,
                }
                for d, ds in self._depth_stats.items()
            },
        }
