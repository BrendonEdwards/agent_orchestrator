# Agent Orchestrator

Fractal multi-model AI agent swarm with cross-model QA. Just say what you need.

```bash
python -m src "build me a web app with image upload"
```

## How it works

Same pattern at every scale: **decompose, checklist, delegate, QA, synthesize**.

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

**Fractal** - Self-similar at every scale. An orchestrator decomposes, delegates to sub-swarms, each sub-swarm can decompose further. No fixed depth - stops when tasks are simple enough for one agent.

**Cross-model QA** - Every piece of work is reviewed by a DIFFERENT model type. Claude's work reviewed by Codex. Codex's work reviewed by Claude. Gemini's work reviewed by Claude. Different models have different blind spots - that's the point.

**Checklist-driven** - Before any work begins, Claude generates a strict checklist of verifiable pass/fail criteria. The worker sees the checklist. The reviewer scores against it. Failed items get sent back with specific feedback.

**Retry loop** - If work doesn't pass QA (default threshold: 7/10), it goes back to the original agent with the reviewer's feedback and the specific failed checklist items. Max 2 retries, then accepts best effort.

**Stateless + Memento** - Every agent call is a fresh spawn. No conversation history, no context rot. The only memory is ~20 ultra-concise survival notes. Parent notes flow down to sub-swarms.

**Llama as dogs body** - Grunt work goes straight to Llama, no QA needed.

## Where compute runs

**Nothing runs on your machine** (unless you want it to).

| Agent | Provider | Compute location |
|-------|----------|-----------------|
| Claude | Anthropic API | Anthropic's servers |
| Codex | OpenAI API | OpenAI's servers |
| Gemini | Google AI API | Google's servers |
| Llama | Configurable | See below |

Llama is the only one that *can* run locally (via Ollama), but doesn't have to:

```bash
# Local (default) - runs on your machine
export LLAMA_BASE_URL="http://localhost:11434/v1"

# Groq cloud - runs on Groq's servers, very fast
export LLAMA_BASE_URL="https://api.groq.com/openai/v1"
export GROQ_API_KEY="your-key"

# Together AI cloud - runs on Together's servers
export LLAMA_BASE_URL="https://api.together.xyz/v1"
export TOGETHER_API_KEY="your-key"
```

Point `LLAMA_BASE_URL` at any OpenAI-compatible endpoint and all compute is remote.

## CLI

```bash
# Just say what you need
python -m src "build a REST API for user management"

# Direct to a specific agent (bypasses decomposition)
python -m src "write a binary search" --agent codex

# Grunt work (straight to Llama, no QA)
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

### How to invoke from Claude Code CLI

You're already in a Claude Code session. Here's exactly how to run it:

```bash
# 1. Make sure you're in the project directory
cd /path/to/agent_orchestrator

# 2. Install dependencies
pip install -e .

# 3. Set your API keys
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AI..."

# 4. Optional: point Llama at a cloud provider so nothing runs locally
export LLAMA_BASE_URL="https://api.groq.com/openai/v1"
export GROQ_API_KEY="gsk_..."

# 5. Run it - just say what you need
python -m src "build a REST API with auth and tests"

# 6. Or from Python directly
python -c "
import asyncio
from src import Orchestrator

async def main():
    orch = Orchestrator()
    result = await orch.run('build a REST API with auth and tests')
    print(result)

asyncio.run(main())
"
```

That's it. The orchestrator will:
1. Ask Claude to decompose the task
2. Generate checklists for each subtask
3. Spawn fractal sub-swarms with weighted teams
4. Agents do the work (on provider servers)
5. Different models QA each other's work
6. Failed work gets sent back with feedback
7. Results synthesized and returned

## Python API

```python
import asyncio
from src import Orchestrator

async def main():
    orch = Orchestrator(
        rules_path="rules.md",
        qa_enabled=True,       # cross-model QA on
        qa_pass_score=7,       # minimum score to pass
        max_qa_retries=2,      # retries before accepting
    )

    # Complex task -> fractal decomposition + QA at every level
    result = await orch.run("Build a web app with image upload")
    print(result)

    # Grunt work -> straight to Llama, no QA
    await orch.grunt("sort alphabetically: zebra, apple, mango")

    # Manual fractal spawn with custom team
    child = orch.spawn(subtask="implement the image processing pipeline")
    print(child.team)  # ['claude', 'gemini', 'gemini_2', 'codex', 'llama']

    # Disable QA for a quick prototype
    fast_orch = Orchestrator(qa_enabled=False)
    await fast_orch.run("quick prototype of a login page")

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

## Setup

```bash
pip install -e .
```

```bash
export ANTHROPIC_API_KEY="your-key"                # Claude (Anthropic servers)
export OPENAI_API_KEY="your-key"                   # Codex (OpenAI servers)
export GOOGLE_API_KEY="your-key"                   # Gemini (Google servers)

# Pick ONE for Llama:
export LLAMA_BASE_URL="http://localhost:11434/v1"   # Ollama (local)
export LLAMA_BASE_URL="https://api.groq.com/openai/v1"  # Groq (cloud)
export GROQ_API_KEY="your-key"
```

## Project Structure

```
src/
├── __main__.py             # CLI - just say what you need
├── orchestrator.py         # Fractal orchestrator + QA loop
├── agents/
│   ├── base.py             # Stateless agent interface
│   ├── claude_agent.py     # Claude (brain, on Anthropic servers)
│   ├── codex_agent.py      # Codex (code, on OpenAI servers)
│   ├── gemini_agent.py     # Gemini (multimodal, on Google servers)
│   └── llama_agent.py      # Llama (grunt, local or cloud)
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
