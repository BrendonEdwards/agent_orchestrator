"""Tests for the SwarmTracker."""

import asyncio

import pytest

from src.tracker import SwarmTracker


@pytest.mark.asyncio
async def test_begin_end_call():
    tracker = SwarmTracker(quiet=True)
    tracker.start()

    call_id = await tracker.begin_call("claude", depth=0, task="test task")
    assert tracker.total_calls == 1
    assert len(tracker._active) == 1

    await tracker.end_call(call_id, input_tokens=100, output_tokens=50, success=True)
    assert len(tracker._active) == 0
    assert tracker.total_input_tokens == 100
    assert tracker.total_output_tokens == 50
    assert tracker.total_failures == 0

    tracker.stop()


@pytest.mark.asyncio
async def test_failure_tracking():
    tracker = SwarmTracker(quiet=True)
    tracker.start()

    call_id = await tracker.begin_call("codex", depth=1, task="bad task")
    await tracker.end_call(call_id, success=False)
    assert tracker.total_failures == 1

    tracker.stop()


@pytest.mark.asyncio
async def test_depth_stats():
    tracker = SwarmTracker(quiet=True)
    tracker.start()

    c1 = await tracker.begin_call("claude", depth=0, task="t1")
    await tracker.end_call(c1, input_tokens=10, output_tokens=5)

    c2 = await tracker.begin_call("codex", depth=1, task="t2")
    await tracker.end_call(c2, input_tokens=20, output_tokens=10)

    snap = tracker.snapshot()
    assert snap["by_depth"][0]["calls"] == 1
    assert snap["by_depth"][1]["tokens"] == 30

    tracker.stop()


@pytest.mark.asyncio
async def test_snapshot():
    tracker = SwarmTracker(quiet=True)
    tracker.start()

    snap = tracker.snapshot()
    assert snap["total_calls"] == 0
    assert snap["total_tokens"] == 0
    assert snap["active_calls"] == 0

    tracker.stop()


@pytest.mark.asyncio
async def test_event_callback():
    events = []
    tracker = SwarmTracker(quiet=False, on_event=events.append)
    tracker.start()

    tracker.event(0, "test event")
    assert any("test event" in e for e in events)

    tracker.stop()


def test_print_summary_no_crash():
    """Summary should not crash even with no data."""
    tracker = SwarmTracker(quiet=True)
    tracker.start()
    tracker.print_summary()
    tracker.stop()
