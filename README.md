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

**Llama as dogs body** - Grunt work goes straight to Llama via Groq, no QA needed.

## Where compute runs

**Nothing runs on your machine.** All inference happens on provider servers.

| Agent | Provider | Compute | Cost |
|-------|----------|---------|------|
| Claude | Anthropic API | Anthropic's servers | Pay-per-token |
| Codex | OpenAI API | OpenAI's servers | Pay-per-token |
| Gemini | Google AI Studio | Google's servers | Free tier available |
| Llama | Groq (default) | Groq's servers | Free tier available |

### API keys vs Pro subscriptions

**Pro/Plus subscriptions DO NOT give API access.** They are completely separate products:

| Subscription | What you get | API access? |
|-------------|-------------|-------------|
| Claude Pro ($20/mo) | claude.ai web chat | No |
| ChatGPT Plus ($20/mo) | chatgpt.com web chat | No |
| Gemini Advanced ($20/mo) | gemini.google.com web chat | No |

You need **API keys** (separate billing, pay-per-token). The cheapest path:

| Provider | How to get started |
|----------|--------------------|
| Anthropic | console.anthropic.com - add credits, get API key |
| OpenAI | platform.openai.com - add credits, get API key |
| Google AI | aistudio.google.com - free API key, generous free tier |
| Groq | console.groq.com - free API key, generous free tier |

**Gemini and Groq both have free tiers**, so 2 of 4 agents cost nothing to start.

## CLI

```bash
# Just say what you need
python -m src "build a REST API for user management"

# Direct to a specific agent (bypasses decomposition)
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

### How to invoke from Claude Code CLI

```bash
# 1. Install
cd /path/to/agent_orchestrator
pip install -e .

# 2. Set API keys (Gemini and Groq have free tiers)
export ANTHROPIC_API_KEY="sk-ant-..."    # console.anthropic.com
export OPENAI_API_KEY="sk-..."           # platform.openai.com
export GOOGLE_API_KEY="AI..."            # aistudio.google.com (free)
export GROQ_API_KEY="gsk_..."            # console.groq.com (free)

# 3. Run it
python -m src "build a REST API with auth and tests"
```

That's it. The orchestrator will:
1. Ask Claude to decompose the task
2. Generate checklists for each subtask
3. Spawn fractal sub-swarms with weighted teams
4. Agents do the work (on provider servers, not yours)
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

    # Grunt work -> straight to Llama on Groq, no QA
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
# Required (pay-per-token)
export ANTHROPIC_API_KEY="your-key"     # Claude
export OPENAI_API_KEY="your-key"        # Codex

# Free tiers available
export GOOGLE_API_KEY="your-key"        # Gemini (aistudio.google.com)
export GROQ_API_KEY="your-key"          # Llama (console.groq.com)
```

## Project Structure

```
src/
├── __main__.py             # CLI - just say what you need
├── orchestrator.py         # Fractal orchestrator + QA loop
├── agents/
│   ├── base.py             # Stateless agent interface
│   ├── claude_agent.py     # Claude (Anthropic servers)
│   ├── codex_agent.py      # Codex (OpenAI servers)
│   ├── gemini_agent.py     # Gemini (Google servers)
│   └── llama_agent.py      # Llama (Groq servers)
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
