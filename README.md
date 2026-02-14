# Agent Orchestrator

Fractal multi-model AI agent swarm with cross-model QA. Uses your existing Pro subscriptions via their CLIs. No API keys needed (except Groq free tier for grunt work).

```bash
python -m src "build me a web app with image upload"
```

## How it works

Same pattern at every scale: **decompose, checklist, delegate, QA, synthesize**.

Each agent is a CLI subprocess:

```
python orchestrator
    |
    ├─> claude -p "decompose this task" --output-format text
    |       (spawns Claude Code CLI, captures stdout)
    |
    ├─> codex -q "implement the API endpoints"
    |       (spawns Codex CLI, captures stdout)
    |
    ├─> gemini -p "process these images"
    |       (spawns Gemini CLI, captures stdout)
    |
    └─> Groq API -> Llama (grunt work, free tier)
```

The fractal with QA:

```
    "build a web app with image upload"
                |
        Orchestrator (d=0)
        team: claude, codex, gemini, llama
                |
        Claude decomposes into 3 subtasks
        Claude generates a checklist for each
                |
    ┌───────────┼───────────┐
    |           |           |
  "backend     "image      "HTML
   API code"    processing"  templates"
    |           |           |
 Orch (d=1)  Orch (d=1)    Llama
 team:       team:         (grunt, no QA)
 2x Codex    2x Gemini
 1x Claude   1x Claude
 1x Gemini   1x Codex
 1x Llama    1x Llama
    |           |
 Codex does  Gemini does
 the work    the work
    |           |
 Claude QAs  Codex QAs     <- DIFFERENT model reviews
 score: 5/10 score: 8/10
    |           |
 FAIL!       PASS
 sent back   done
 w/ feedback
    |
 Codex fixes
    |
 Claude QAs
 score: 9/10
    |
 PASS
```

## Core Ideas

**Fractal** - Self-similar at every scale. Decompose, delegate to sub-swarms, sub-swarms can decompose further. Stops when tasks are simple enough for one agent.

**Cross-model QA** - Every piece of work is reviewed by a DIFFERENT model type. Different models have different blind spots - that's the point.

**Checklist-driven** - Before work begins, Claude generates strict pass/fail criteria. Worker sees the checklist. Reviewer scores against it. Failed items get sent back.

**CLI subprocesses** - Agents are CLI tools you already have installed. Each `send()` spawns a subprocess, captures stdout. Uses your Pro subscriptions, not API credits.

**Stateless + Memento** - Every agent call is a fresh subprocess. No context rot. Memento survival notes are the only context.

## Where compute runs

**Nothing runs on your machine** except the orchestrator script itself. Each CLI subprocess talks to its provider's servers.

| Agent | How it runs | Uses |
|-------|-------------|------|
| Claude | `claude -p "prompt" --output-format text` | Claude Pro subscription |
| Codex | `codex -q "prompt"` | ChatGPT Plus/Pro subscription |
| Gemini | `gemini -p "prompt"` | Gemini Advanced subscription |
| Llama | Groq API (free tier) | Free Groq account |

**No API keys needed for Claude, Codex, or Gemini** - the CLIs handle auth through your existing subscription login.

## Setup

```bash
# 1. Install the orchestrator
pip install -e .

# 2. Install the CLIs (if not already)
npm install -g @anthropic-ai/claude-code   # claude CLI
npm install -g @openai/codex               # codex CLI
npm install -g @anthropic-ai/claude-code   # gemini CLI (TODO: actual package)

# 3. Log into each CLI (one-time, uses your Pro subscriptions)
claude login
codex auth
gemini auth

# 4. Only API key needed: Groq for Llama grunt work (free tier)
export GROQ_API_KEY="gsk_..."              # console.groq.com (free)

# 5. Run it
python -m src "build a REST API with auth and tests"
```

## CLI

```bash
# Just say what you need
python -m src "build a REST API for user management"

# Direct to a specific agent
python -m src "write a binary search" --agent codex

# Grunt work (straight to Llama on Groq, no QA)
python -m src "format this JSON: {a:1,b:2}" --grunt

# Disable QA for speed
python -m src "quick prototype of a login form" --no-qa

# Strict QA (higher threshold, more retries)
python -m src "implement auth middleware" --qa-threshold 9 --qa-retries 4

# Check what's alive
python -m src --health

# See the team
python -m src --team

# See survival notes
python -m src --notes
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

    # Complex task -> fractal decomposition + QA
    result = await orch.run("Build a web app with image upload")
    print(result)

    # Grunt work -> Llama on Groq (free), no QA
    await orch.grunt("sort alphabetically: zebra, apple, mango")

    # Manual fractal spawn with custom team
    child = orch.spawn(subtask="implement the image processing pipeline")
    print(child.team)  # ['claude', 'gemini', 'gemini_2', 'codex', 'llama']

    # Check survival notes (includes QA scores)
    print(orch.notes())

asyncio.run(main())
```

## QA Pairings

| Worker | Reviewer | Why |
|--------|----------|-----|
| Claude | Codex (OpenAI) | Codex catches logical gaps Claude might rationalize past |
| Codex | Claude | Claude catches architectural issues Codex might ignore |
| Gemini | Claude | Claude validates multimodal output descriptions |
| Llama | None | Grunt work, not worth QA cost |

## Project Structure

```
src/
├── __main__.py             # CLI - just say what you need
├── orchestrator.py         # Fractal orchestrator + QA loop
├── agents/
│   ├── base.py             # Stateless agent interface
│   ├── claude_agent.py     # Claude (subprocess: claude -p)
│   ├── codex_agent.py      # Codex (subprocess: codex -q)
│   ├── gemini_agent.py     # Gemini (subprocess: gemini -p)
│   └── llama_agent.py      # Llama (Groq API, free tier)
├── memento/
│   └── memento.py          # Anti-context-rot survival notes
├── routing/
│   ├── router.py           # Inter-agent message routing
│   └── protocol.py         # Compact wire format (not English)
└── rules/
    └── loader.py           # Rules.md parser
```

## License

Apache 2.0
