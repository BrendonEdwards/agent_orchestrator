# Agent Orchestrator

Fractal multi-model AI agent swarm with cross-model QA. Direct API calls. Goes as deep as the task demands.

```bash
python -m src "build me a C compiler from scratch"
```

## How it works

Same pattern at every scale: **decompose, checklist, delegate, QA, synthesize**.

Depth is automatic. "What's 2+2?" stays at level 0. "Build a C compiler" spawns sub-swarms that spawn sub-swarms until every piece is small enough for one agent to handle.

```
    "build a C compiler from scratch"
                    |
            Orchestrator (d=0)
            Claude decomposes:
            ┌───────┼──────────┬──────────┐
            |       |          |          |
         "lexer"  "parser"  "codegen"  "tests"
            |       |          |          |
        Orch(d=1)  Orch(d=1)  Orch(d=1)  Llama
        Claude      Claude     Claude     (grunt)
        decomposes: decomposes: decomposes:
        ┌──┼──┐    ┌──┼──┐    ┌──┼──┐
        |  |  |    |  |  |    |  |  |
      tok reg lit  AST prec  IR  x86 opt
       |   |   |    ...       ...
     Codex Codex Codex
     does  does  does
       |   |   |
     Claude Claude Claude  <- different model QAs each
     scores scores scores
       |   |   |
     pass/ pass/ pass/
     fail  fail  fail
```

Meanwhile "what's 2+2?" hits `_decompose`, Claude returns it as a single task, one agent answers directly. No recursion.

## Core Ideas

**Fractal** - Self-similar at every scale. The orchestrator at depth 5 works identically to depth 0. Decompose, delegate, QA, synthesize. Stops naturally when tasks are simple enough for one agent.

**Cross-model QA** - Every piece of work is reviewed by a DIFFERENT model type. Different models have different blind spots - that's the whole point.

**Checklist-driven** - Before work begins, Claude generates strict pass/fail criteria. The worker sees the checklist. The reviewer scores against it. Failed items get sent back with specific feedback.

**Direct API** - No CLI subprocess overhead. Direct HTTP to each provider. Structured errors, token tracking.

**Stateless + Memento** - Every agent call is fresh. No context rot. Memento survival notes are the only context carried forward.

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
# It goes as deep as it needs to
python -m src "build a REST API for user management"
python -m src "build a C compiler from scratch"
python -m src "what is 2+2"

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
```

## Python API

```python
import asyncio
from src import Orchestrator

async def main():
    orch = Orchestrator()

    # Simple -> stays shallow
    result = await orch.run("What is 2+2?")

    # Complex -> goes deep automatically
    result = await orch.run("Build a C compiler from scratch")
    print(result)

    # Grunt work -> Llama on Groq (free), no QA
    await orch.grunt("sort alphabetically: zebra, apple, mango")

    # Manual fractal spawn with custom team
    child = orch.spawn(subtask="implement the image processing pipeline")
    print(child.team)  # ['claude', 'gemini', 'gemini_2', 'codex', 'llama']

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
│   ├── claude_agent.py     # Claude (Anthropic API)
│   ├── codex_agent.py      # Codex (OpenAI API)
│   ├── gemini_agent.py     # Gemini (Google GenAI API)
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
