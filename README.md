# Agent Orchestrator

Multi-model AI agent swarm with cross-model QA. Direct API calls - fast, structured errors, token tracking.

```bash
python -m src "build me a web app with image upload"
```

## How it works

Default mode: **shallow fan-out** (depth=1). Decompose once, delegate in parallel, QA, synthesize. Fast and practical.

```
python orchestrator
    |
    ├─> Anthropic API  -> Claude decomposes task
    |
    ├─> OpenAI API     -> Codex implements code      (parallel)
    ├─> Google API     -> Gemini processes images     (parallel)
    ├─> Groq API       -> Llama formats boilerplate   (parallel)
    |
    ├─> Cross-model QA (different model reviews each piece)
    |
    └─> Claude synthesizes final result
```

Use `--deep` for full fractal recursion (depth=3):

```
    "build a web app with image upload"
                |
        Orchestrator (d=0)
                |
        Claude decomposes + generates checklists
                |
    ┌───────────┼───────────┐
    |           |           |
  "backend"  "images"    "templates"
    |           |           |
  Codex       Gemini      Llama
  does it     does it     (no QA)
    |           |
  Claude QAs  Codex QAs     <- DIFFERENT model reviews
  score: 5/10 score: 8/10
    |           |
  FAIL!       PASS
  sent back
  w/ feedback
    |
  Codex fixes -> Claude QAs -> 9/10 -> PASS
```

## Core Ideas

**Shallow by default** - One level of decomposition + QA handles most tasks and finishes in seconds, not minutes.

**Cross-model QA** - Every piece of work is reviewed by a DIFFERENT model type. Different models have different blind spots.

**Checklist-driven** - Claude generates strict pass/fail criteria before work begins. Reviewer scores against it.

**Direct API** - No CLI subprocess overhead. Direct HTTP to each provider. Structured errors, token tracking, ~2-5s per call.

**Stateless + Memento** - Every agent call is fresh. No context rot. Memento survival notes are the only context.

## Where compute runs

All compute runs on provider servers. Your machine just orchestrates.

| Agent | API | Cost |
|-------|-----|------|
| Claude | Anthropic Messages API | Pay-per-token |
| Codex | OpenAI Chat API | Pay-per-token |
| Gemini | Google GenAI API | Free tier available |
| Llama | Groq API | Free tier |

## Setup

```bash
# 1. Install
pip install -e .
pip install anthropic openai google-genai

# 2. Set API keys
export ANTHROPIC_API_KEY="sk-ant-..."    # console.anthropic.com
export OPENAI_API_KEY="sk-..."           # platform.openai.com
export GEMINI_API_KEY="AI..."            # aistudio.google.com (free)
export GROQ_API_KEY="gsk_..."            # console.groq.com (free)

# 3. Run
python -m src "build a REST API with auth and tests"
```

## CLI

```bash
# Default: shallow fan-out (fast, depth=1)
python -m src "build a REST API for user management"

# Deep mode: full fractal recursion (depth=3, thorough)
python -m src "build a complete web app" --deep

# Direct to a specific agent
python -m src "write a binary search" --agent codex

# Grunt work (Llama on Groq, no QA)
python -m src "format this JSON: {a:1,b:2}" --grunt

# No QA (fastest)
python -m src "quick prototype" --no-qa

# Strict QA
python -m src "implement auth middleware" --qa-threshold 9 --qa-retries 4

# Check what's alive
python -m src --health

# See the team
python -m src --team
```

## Python API

```python
import asyncio
from src import Orchestrator

async def main():
    # Default: shallow (depth=1), fast
    orch = Orchestrator()
    result = await orch.run("Build a web app with image upload")
    print(result)

    # Deep mode
    deep = Orchestrator(max_depth=3)
    result = await deep.run("Build a complete e-commerce platform")

    # Grunt work -> Llama on Groq (free)
    await orch.grunt("sort alphabetically: zebra, apple, mango")

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
├── __main__.py             # CLI entry point
├── orchestrator.py         # Orchestrator + QA loop
├── agents/
│   ├── base.py             # Stateless agent interface
│   ├── claude_agent.py     # Claude (Anthropic API)
│   ├── codex_agent.py      # Codex (OpenAI API)
│   ├── gemini_agent.py     # Gemini (Google GenAI API)
│   └── llama_agent.py      # Llama (Groq API, free tier)
├── memento/
│   └── memento.py          # Anti-context-rot survival notes
├── routing/
│   ├── router.py           # Inter-agent message routing
│   └── protocol.py         # Compact wire format
└── rules/
    └── loader.py           # Rules.md parser
```

## Shallow vs Deep

| Mode | Flag | Depth | Speed | Best for |
|------|------|-------|-------|----------|
| Shallow | (default) | 1 | ~10-20s | Most tasks, prototyping |
| Deep | `--deep` | 3 | ~60-120s | Complex multi-part projects |
| Direct | `--agent X` | 0 | ~3-5s | Single-agent tasks |
| Grunt | `--grunt` | 0 | ~1-2s | Formatting, boilerplate |

## License

Apache 2.0
