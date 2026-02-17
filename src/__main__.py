"""CLI entry point. Call with a few words, get a result.

Usage:
    python -m src "build a REST API for users"
    python -m src "translate this Python to Go" --agent codex
    python -m src "format this JSON: {a:1}" --grunt
    python -m src --health
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from src.orchestrator import Orchestrator
from src.tracker import SwarmTracker


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agent-orchestrator",
        description="Fractal multi-model AI agent swarm with cross-model QA.",
    )
    parser.add_argument(
        "task",
        nargs="*",
        help="What you want done (just say it in a few words)",
    )
    parser.add_argument(
        "--agent", "-a",
        help="Send directly to a specific agent (claude, codex, gemini)",
    )
    parser.add_argument(
        "--grunt", "-g",
        action="store_true",
        help="Force task to Gemini (grunt work, no QA)",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Check which agents are alive",
    )
    parser.add_argument(
        "--notes", "-n",
        action="store_true",
        help="Show current memento survival notes",
    )
    parser.add_argument(
        "--rules", "-r",
        default="rules.md",
        help="Path to rules.md (default: rules.md)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Max fractal depth (default: 10, stops naturally when tasks are simple)",
    )
    parser.add_argument(
        "--team",
        action="store_true",
        help="Show the agent team composition",
    )
    parser.add_argument(
        "--no-qa",
        action="store_true",
        help="Disable cross-model QA (faster, less reliable)",
    )
    parser.add_argument(
        "--qa-retries",
        type=int,
        default=2,
        help="Max QA retry attempts before accepting (default: 2)",
    )
    parser.add_argument(
        "--qa-threshold",
        type=int,
        default=7,
        help="QA pass score out of 10 (default: 7)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress live progress output (still prints summary)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable all progress output and summary",
    )
    parser.add_argument(
        "--stall-timeout",
        type=float,
        default=120.0,
        help="Seconds before flagging an agent call as stalled (default: 120)",
    )

    args = parser.parse_args()
    task = " ".join(args.task) if args.task else ""

    if not task and not args.health and not args.notes and not args.team:
        parser.print_help()
        sys.exit(1)

    # Set up tracker for live visibility
    use_tracker = not args.no_progress
    tracker = SwarmTracker(
        stall_timeout=args.stall_timeout,
        quiet=args.quiet,
    ) if use_tracker else None

    orch = Orchestrator(
        rules_path=args.rules,
        max_depth=args.max_depth,
        qa_enabled=not args.no_qa,
        max_qa_retries=args.qa_retries,
        qa_pass_score=args.qa_threshold,
        tracker=tracker,
    )

    if args.team:
        print(f"Team: {', '.join(orch.team)}")
        return

    if args.health:
        results = asyncio.run(orch.health_check())
        for name, alive in results.items():
            status = "UP" if alive else "DOWN"
            print(f"  {name}: {status}")
        return

    if args.notes:
        notes = orch.notes()
        print(notes if notes else "(no notes)")
        return

    # Run with tracker lifecycle
    async def _run() -> str:
        if tracker:
            tracker.start()
            tracker.start_stall_checker()
        try:
            if args.grunt:
                return await orch.grunt(task)
            elif args.agent:
                response = await orch.send_to(args.agent, task)
                return response.content if response.success else f"Error: {response.error}"
            else:
                return await orch.run(task)
        finally:
            if tracker:
                tracker.stop()

    result = asyncio.run(_run())

    # Summary goes to stderr, result to stdout
    if tracker:
        tracker.print_summary()

    print(result)


if __name__ == "__main__":
    main()
