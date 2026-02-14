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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agent-orchestrator",
        description="Multi-model AI agent swarm. Just say what you need.",
    )
    parser.add_argument(
        "task",
        nargs="*",
        help="What you want done (just say it in a few words)",
    )
    parser.add_argument(
        "--agent", "-a",
        help="Send directly to a specific agent (claude, codex, gemini, llama)",
    )
    parser.add_argument(
        "--grunt", "-g",
        action="store_true",
        help="Force task to Llama (grunt work)",
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
        default=3,
        help="Max swarm recursion depth (default: 3)",
    )

    args = parser.parse_args()
    task = " ".join(args.task) if args.task else ""

    if not task and not args.health and not args.notes:
        parser.print_help()
        sys.exit(1)

    orch = Orchestrator(rules_path=args.rules, max_depth=args.max_depth)

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

    if args.grunt:
        result = asyncio.run(orch.grunt(task))
    elif args.agent:
        response = asyncio.run(orch.send_to(args.agent, task))
        result = response.content if response.success else f"Error: {response.error}"
    else:
        result = asyncio.run(orch.run(task))

    print(result)


if __name__ == "__main__":
    main()
