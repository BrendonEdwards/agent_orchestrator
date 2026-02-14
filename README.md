# Agent Orchestrator

Fractal multi-model AI agent swarm. Just say what you need.

```bash
python -m src "build me a web app with image upload"
```

## How it works

Same pattern at every scale: **decompose, delegate, synthesize**.

```
    "build a web app with image upload"
                |
        Orchestrator (d=0)
        team: claude, codex, gemini, llama
                |
        Claude: "3 subtasks"
                |
    ┌───────────┼───────────┐
    |           |           |
  "backend     "image      "HTML
   API code"    processing"  templates"
    |           |           |
 Orch (d=1)  Orch (d=1)    Llama
 team:       team:         (grunt)
 2x Codex    2x Gemini
 1x Claude   1x Claude
 1x Gemini   1x Codex
 1x Llama    1x Llama
    |           |
 could       could
 decompose   decompose
 further...  further...
```

Each sub-swarm builds its own team weighted for the subtask. A code-heavy
branch gets extra Codex instances. An image branch gets extra Gemini.
Recursion stops naturally when Claude decides a task is simple enough
for one agent.

## Core Ideas

**Fractal** - The same structure repeats at every scale. An orchestrator decomposes a task into subtasks. Each subtask gets its own orchestrator with a custom-composed team. Those can decompose further. Self-similar all the way down. No fixed depth limit - it stops when tasks become simple.

**Stateless + Memento** - Like the film. Every agent call is a fresh spawn. No conversation history, no context rot. The only memory is ~20 ultra-concise survival notes stamped onto each message. Parent notes are inherited by children so sub-swarms know the bigger picture.

**Llama as dogs body** - Ollama handles grunt work that doesn't need thinking. Doesn't get memento notes - just the task.

**Compact protocol** - Agents don't need English to talk to each other:

```
English:  "Please write a Python function called sum_evens..."
Protocol: T:codegen|L:py|N:sum_evens|I:list[int]|O:int|D:sum even vals
```

## CLI

```bash
# Just say what you need - it decomposes and swarms automatically
python -m src "build a REST API for user management"

# Direct to a specific agent
python -m src "write a binary search" --agent codex

# Grunt work (straight to Llama)
python -m src "format this JSON: {a:1,b:2}" --grunt

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
    orch = Orchestrator(rules_path="rules.md")

    # Complex task -> fractal decomposition
    result = await orch.run("Build a web app with image upload")
    print(result)

    # Grunt work -> straight to Llama
    await orch.grunt("sort alphabetically: zebra, apple, mango")

    # Manual fractal spawn with custom team for subtask
    child = orch.spawn(subtask="implement the image processing pipeline")
    print(child.team)  # -> ['claude', 'gemini', 'gemini_2', 'codex', 'llama']
    result = await child.run("implement the image processing pipeline")

    # Check survival notes
    print(orch.notes())

asyncio.run(main())
```

## Setup

```bash
pip install -e .
```

```bash
export ANTHROPIC_API_KEY="your-key"                # Claude
export OPENAI_API_KEY="your-key"                   # Codex
export GOOGLE_API_KEY="your-key"                   # Gemini
export LLAMA_BASE_URL="http://localhost:11434/v1"   # Llama (ollama)
```

## Project Structure

```
src/
├── __main__.py             # CLI - just say what you need
├── orchestrator.py         # Fractal swarm orchestrator
├── agents/
│   ├── base.py             # Stateless agent interface
│   ├── claude_agent.py     # Claude (brain)
│   ├── codex_agent.py      # Codex (code)
│   ├── gemini_agent.py     # Gemini (multimodal)
│   └── llama_agent.py      # Llama (grunt work)
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
