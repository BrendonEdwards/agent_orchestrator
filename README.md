# Agent Orchestrator

A fractal multi-model AI agent swarm with cross-model quality assurance.

Uses your existing Pro subscriptions (Claude, ChatGPT, Gemini) via their CLIs. No API keys needed.

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
             Claude      Codex           <- different model reviews
            reviews     reviews
                |           |
           score: 5/10  score: 8/10
                |           |
             FAIL!       PASS
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

LLMs have blind spots. Claude rationalizes past logical gaps. Codex ignores architecture. Gemini struggles with multi-step reasoning. No single model catches everything.

This orchestrator makes them check each other's work. A different model reviews every piece of output against a strict checklist generated before work begins. Failed items get sent back with specific feedback.

The fractal part means complex tasks decompose recursively - sub-swarms spawn sub-swarms, each with their own QA cycle, until tasks are simple enough for a single agent.

## Key Design Decisions

**Capability routing, not model routing.** Claude judges what a task *needs* (REASONING, CODE, MULTIMODAL, FAST). A static config table maps that to a provider. When a better model drops, update one line in the table - not the orchestration logic.

```python
CAPABILITY_PROVIDERS = {
    "REASONING":  "claude",    # Deep analysis, planning, architecture
    "CODE":       "codex",     # Code generation, debugging, refactoring
    "MULTIMODAL": "gemini",    # Images, audio, video, diagrams
    "FAST":       "gemini",    # Formatting, boilerplate, grunt work
}
```

**Stateless agents + memento memory.** Every agent call is a fresh subprocess. No conversation history. No context rot. Instead, a compact "memento" system injects survival notes into every call - goals, constraints, QA scores, parent context. Fresh agent + concise notes beats stale agent + full context.

**CLI subprocesses, not API calls.** Claude, Codex, and Gemini run as CLI tools you already have installed. Each `send()` spawns a subprocess, captures stdout. Uses your existing Pro subscriptions. No API keys needed.

**Cross-model QA pairings.** Workers are reviewed by a model with different strengths:

| Worker Capability | Reviewer Capability | Why |
|-------------------|---------------------|-----|
| REASONING (Claude) | CODE (Codex) | Catches logical gaps Claude rationalizes past |
| CODE (Codex) | REASONING (Claude) | Catches architectural issues Codex ignores |
| MULTIMODAL (Gemini) | REASONING (Claude) | Validates multimodal output descriptions |
| FAST (Gemini) | None | Grunt work - not worth QA cost |

## Setup

### Prerequisites

Python 3.10+ and the CLI tools:

```bash
# Install CLIs
npm install -g @anthropic-ai/claude-code   # Claude
npm install -g @openai/codex               # Codex
# Gemini CLI - install per Google's instructions

# Authenticate (one-time, uses your Pro subscriptions)
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

### Optional

```bash
# Dev tools (pytest, ruff)
pip install -e ".[dev]"
```

## Usage

### CLI

```bash
# Describe what you need - orchestrator handles the rest
python -m src "build a REST API for user management"

# Direct to a specific agent (no decomposition, no QA)
python -m src "write a binary search" --agent codex

# Grunt work -> Gemini (fast, no QA)
python -m src "format this JSON: {a:1,b:2}" --grunt

# Disable QA for speed
python -m src "quick prototype of a login form" --no-qa

# Strict QA (higher threshold, more retries)
python -m src "implement auth middleware" --qa-threshold 9 --qa-retries 4

# Check which agents are reachable
python -m src --health

# Show team composition
python -m src --team
```

### Python API

```python
import asyncio
from src import Orchestrator

async def main():
    orch = Orchestrator(
        qa_enabled=True,
        qa_pass_score=7,       # Score out of 10 to pass QA
        max_qa_retries=2,      # How many times to send work back
    )

    # Complex task -> fractal decomposition + cross-model QA
    result = await orch.run("Build a web app with image upload")
    print(result)

    # Grunt work -> Gemini (fast, no QA)
    await orch.grunt("sort alphabetically: zebra, apple, mango")

    # Direct agent send
    resp = await orch.send_to("codex", "implement quicksort in Rust")

    # Check survival notes (QA scores, routing decisions, etc.)
    print(orch.notes())

asyncio.run(main())
```

## How It Works

### 1. Decomposition

Claude breaks complex tasks into independent subtasks and tags each with the capability it needs:

```
Task: "Build an e-commerce site with product images"

Claude outputs:
1. [CODE] Implement product CRUD API with database models
2. [CODE] Build shopping cart and checkout flow
3. [MULTIMODAL] Create image upload and thumbnail pipeline
4. [FAST] Generate project scaffolding and config files
```

### 2. Routing

The `CAPABILITY_PROVIDERS` table maps each tag to a provider. The orchestrator never picks models - it picks capabilities.

### 3. Checklist Generation

Before any work begins, Claude generates a strict pass/fail checklist:

```
Task: "Implement product CRUD API"

Checklist:
1. All CRUD endpoints (GET, POST, PUT, DELETE) are implemented
2. Input validation rejects malformed data with proper error codes
3. Database queries use parameterized statements
4. Response format is consistent JSON with status codes
5. Edge cases handled: empty results, duplicate IDs, not found
```

The worker sees this checklist. The reviewer scores against it.

### 4. Cross-Model QA

A different model reviews the work:

```
Worker: Codex (CODE capability)
Reviewer: Claude (REASONING capability)

ITEM_1: PASS
ITEM_2: PASS
ITEM_3: FAIL - Uses string concatenation for SQL queries
ITEM_4: PASS
ITEM_5: FAIL - Returns 200 for not-found instead of 404
SCORE: 6/10
FEEDBACK: Fix SQL injection vulnerability and HTTP status codes
```

Score < 7? Work goes back to the worker with the specific failed items and feedback.

### 5. Fractal Recursion

Sub-swarms can decompose further. Each child orchestrator gets its own team, memento, and QA cycle. This continues until tasks are simple enough for a single agent, up to a configurable max depth (default 10).

### 6. Synthesis

Results from all branches are synthesized by Claude into a single coherent response.

## Architecture

```
src/
├── __main__.py             # CLI entry point
├── orchestrator.py         # Fractal orchestrator + QA loop
├── tracker.py              # Live observability (tokens, timing, stalls)
├── agents/
│   ├── base.py             # Stateless agent interface
│   ├── claude_agent.py     # Claude  (subprocess: claude -p)
│   ├── codex_agent.py      # Codex   (subprocess: codex -q)
│   ├── gemini_agent.py     # Gemini  (subprocess: gemini -p)
│   └── llama_agent.py      # Llama   (optional, Groq API)
├── memento/
│   └── memento.py          # Anti-context-rot survival notes
├── routing/
│   ├── router.py           # Message routing + memento injection
│   └── protocol.py         # Compact wire protocol (saves ~2/3 tokens)
└── rules/
    └── loader.py           # Model strengths/weaknesses config
```

### Memento System

Inspired by the film. Every orchestrator maintains up to 20 ultra-concise notes (max 150 chars each) with priority levels. Low-priority notes get dropped as capacity fills. Every agent call receives these notes as context - no more, no less.

```
goal=Build REST API | route=codex:leaf:CODE | checklist=CRUD endpoints | ...
```

This beats long conversation histories. Agents don't drift, don't hallucinate about prior context, and don't waste tokens on stale information.

### SwarmTracker

Live observability shared across all depths:
- Per-agent and per-depth token usage
- Wall-clock timing for every call
- Stall detection (configurable timeout, default 120s)
- Progress events emitted to stderr (stdout stays clean for results)

### Compact Wire Protocol

Inter-agent messages use a token-efficient format when human readability isn't needed:

```
English:  "Write a Python function called sum_evens that takes a list of
           integers and returns the sum of even values"  (180 chars)

Protocol: T:codegen|L:py|N:sum_evens|I:list[int]|O:int|D:sum even vals
           (60 chars, saves ~2/3 of tokens)
```

## Configuration

### Environment Variables

No environment variables required. All three agents (Claude, Codex, Gemini) authenticate through their own CLI login mechanisms.

If you want to use the optional Llama agent via Groq, set `GROQ_API_KEY`.

### Rules File

`rules.md` defines model strengths, weaknesses, and routing preferences. The orchestrator reads this at startup to inform routing decisions. Edit this file to tune which models handle what.

### CLI Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--agent NAME` | - | Send directly to a specific agent |
| `--grunt` | - | Force Gemini for grunt work (no QA) |
| `--no-qa` | - | Disable QA for speed |
| `--qa-threshold N` | `7` | QA pass score (1-10) |
| `--qa-retries N` | `2` | Max QA retry attempts |
| `--max-depth N` | `10` | Max fractal decomposition depth |
| `--rules PATH` | `rules.md` | Path to rules file |
| `--health` | - | Check which agents are alive |
| `--team` | - | Show agent team composition |
| `--notes` | - | Show memento survival notes |
| `--quiet` | - | Suppress progress output |
| `--stall-timeout N` | `120` | Seconds before flagging an agent as stalled |

## Swapping Models

The whole point of capability-based routing: when a better model drops, you update the config table, not the orchestration logic.

```python
# Hypothetical: GPT-5 is the new reasoning king
CAPABILITY_PROVIDERS = {
    "REASONING":  "codex",     # <- changed from "claude"
    "CODE":       "codex",
    "MULTIMODAL": "gemini",
    "FAST":       "gemini",
}

# Hypothetical: Claude gets great at code too
CAPABILITY_PROVIDERS = {
    "REASONING":  "claude",
    "CODE":       "claude",    # <- changed from "codex"
    "MULTIMODAL": "gemini",
    "FAST":       "gemini",
}
```

The reasoning model (Claude) doesn't need to know what models exist. It judges what a task *needs*. The table handles the rest.

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
