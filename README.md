# Agent Orchestrator

A CLI-first, fractal multi-model AI agent swarm with memento memory and cross-model quality assurance.

The project uses authenticated local CLIs for Claude, Codex and Gemini. It deliberately avoids SDK/API-key dependencies in the default install.

```
              "build a web app with image upload"
                            |
                    Orchestrator (d=0)
                            |
                Claude decomposes + tags:
                ┌───────────┼──────────────┐
                |           |              |
          [CODE]       [MULTIMODAL]     [FAST]
         "API code"   "image pipeline"  "HTML templates"
                |           |              |
             Codex       Gemini          Gemini
            (works)      (works)       (no QA needed)
                |           |
             Claude      Claude          <- different model reviews
            reviews      reviews
                |           |
           score: 5/10  score: 8/10
                |           |
             FAIL        PASS
           sent back
          w/ feedback
                |
           Codex fixes
                |
           Claude reviews
           score: 9/10
                |
             PASS
```

## Why

LLMs have blind spots. Claude is strong at reasoning and synthesis. Codex is strong at long-horizon code work. Gemini is strong at multimodal and fast transformation tasks. No single model catches everything.

This orchestrator makes them check each other's work. A different model reviews each non-trivial output against a checklist generated before work begins. Failed items go back to the worker with specific feedback.

The fractal part means complex tasks decompose recursively: sub-swarms spawn sub-swarms, each with its own memento notes and QA cycle, until tasks are simple enough for one agent.

## Key design decisions

### Capability routing, not model routing

Claude judges what a task needs: `REASONING`, `CODE`, `MULTIMODAL`, or `FAST`. A static config table maps that capability to a provider. When the model landscape changes, update the table and wrapper metadata rather than rewriting orchestration logic.

```python
CAPABILITY_PROVIDERS = {
    "REASONING":  "claude",
    "CODE":       "codex",
    "MULTIMODAL": "gemini",
    "FAST":       "gemini",
}
```

### Stateless agents plus memento memory

Every agent call is a fresh subprocess. No conversation history is carried between calls. Instead, the orchestrator injects compact survival notes: goals, constraints, routes, checklist summaries, QA scores and parent context.

Fresh agent plus concise notes is the core anti-context-rot pattern.

### CLI subprocesses, not SDK calls

Claude, Codex and Gemini run as local CLI tools. Each `send()` spawns a subprocess and captures stdout. The default package dependencies are intentionally small.

### Cross-model QA pairings

| Worker capability | Reviewer capability | Why |
|---|---|---|
| `REASONING` | `CODE` | Catches implementation gaps in plans and analysis |
| `CODE` | `REASONING` | Catches architecture, edge case and specification issues |
| `MULTIMODAL` | `REASONING` | Validates interpretation and output claims |
| `FAST` | none | Grunt work is not worth a full QA pass |

## Current model guidance

The wrappers store model labels as metadata only. The actual model used is controlled by the installed CLI and account settings.

Recommended current posture:

| Agent | Role | Metadata/default stance |
|---|---|---|
| Claude | Reasoning, decomposition, synthesis and QA | Use the Claude Code `sonnet` alias for daily coding/reasoning, or `opus`/`opusplan` for complex planning if available |
| Codex | Code generation, refactoring, migrations and implementation | GPT-5.2-Codex is the preferred coding model where the Codex CLI supports it |
| Gemini | Multimodal analysis and fast transformation work | Gemini 3 Pro Preview is the preferred high-capability multimodal target where the Gemini CLI supports it |

## Setup

### Prerequisites

Python 3.10+ and the CLI tools:

```bash
npm install -g @anthropic-ai/claude-code
npm install -g @openai/codex
# Install Gemini CLI using Google's current instructions

claude login
codex auth
gemini auth
```

### Install

```bash
git clone https://github.com/BrendonEdwards/agent_orchestrator.git
cd agent_orchestrator
pip install -e .
```

### Development install

```bash
pip install -e ".[dev]"
pytest
```

## Usage

```bash
python -m src "build a REST API for user management"
python -m src "write a binary search" --agent codex
python -m src "format this JSON: {a:1,b:2}" --grunt
python -m src "quick prototype of a login form" --no-qa
python -m src "implement auth middleware" --qa-threshold 9 --qa-retries 4
python -m src --health
python -m src --team
```

## Python API

```python
import asyncio
from src import Orchestrator

async def main():
    orch = Orchestrator(
        qa_enabled=True,
        qa_pass_score=7,
        max_qa_retries=2,
    )

    result = await orch.run("Build a web app with image upload")
    print(result)
    print(orch.notes())

asyncio.run(main())
```

## How it works

1. Claude decomposes complex tasks into tagged subtasks.
2. The capability table routes each subtask to the right provider.
3. Claude generates a checklist before work begins.
4. The worker sees the checklist and produces output.
5. A different model reviews the work.
6. Failed work is sent back with specific feedback until it passes or retries are exhausted.
7. Branch results are synthesised by Claude.

## Architecture

```
src/
├── __main__.py             # CLI entry point
├── orchestrator.py         # Fractal orchestrator + QA loop
├── tracker.py              # Live observability
├── agents/
│   ├── base.py             # Stateless agent interface
│   ├── claude_agent.py     # Claude CLI wrapper
│   ├── codex_agent.py      # Codex CLI wrapper
│   └── gemini_agent.py     # Gemini CLI wrapper
├── memento/
│   └── memento.py          # Anti-context-rot survival notes
├── routing/
│   ├── router.py           # Message routing + memento injection
│   └── protocol.py         # Experimental compact wire protocol
└── rules/
    └── loader.py           # Model strengths/weaknesses notes
```

## Memento system

Every orchestrator maintains a small set of concise notes. Low-priority notes are dropped as capacity fills.

```text
goal=Build REST API | route=codex:leaf:CODE | checklist=CRUD endpoints | qa_d0=r=claude s=9/10 PASS
```

This keeps each agent call focused without carrying a full and potentially degraded conversation history.

## SwarmTracker

The tracker records:

* Per-agent and per-depth call counts
* Wall-clock timing
* Failure counts
* Stall warnings for long-running calls
* Token totals when the underlying agent response provides usage metadata

CLI wrappers do not always expose token usage, so timing and stall detection are currently more reliable than token accounting.

## Known hardening backlog

* Convert decomposition, checklist and QA responses to strict JSON schemas.
* Decide whether unparseable QA should fail closed by default or be configurable.
* Add a durable run workspace containing task manifests, artefacts, QA ledgers and final reports.
* Make rules fully drive routing, or keep them clearly documented as advisory.
* Add end-to-end tests with fake Claude, Codex and Gemini CLI binaries.

## Licence

Apache 2.0. See `LICENSE` for details.
