# Agent Orchestrator

Multi-model AI agent swarm. Just say what you need.

```bash
python -m src "build me a REST API for users"
```

## Architecture

```
    "build me a REST API"
            |
    Orchestrator (depth=0)
            |
        Claude: "complex - break it down"
            |
    ┌───────┼──────────┐
    |       |          |
  Codex   Claude    Llama
  swarm   swarm    (grunt)
  (d=1)   (d=1)
  |  |    |  |
 fresh   fresh     Each sub-swarm has its own
 agents  agents    memento, agents, and can
                   spawn further sub-swarms

Memento: concise survival notes (fights context rot)
Protocol: agents talk in compact format, not English
```

## Core Ideas

**Recursive swarms** - Any agent that gets a complex task can spawn its own sub-swarm of agents, using the same pattern all the way down. Claude breaks the task into subtasks, each subtask gets its own orchestrator with fresh agents and its own memento. Sub-swarms can spawn sub-sub-swarms. Depth-limited to prevent runaway recursion.

**Stateless agents + Memento** - Like the film. Every agent call is a fresh spawn with zero prior context. No conversation history accumulates, so no context rot. The only "memory" is Memento: ~20 ultra-concise survival notes stamped onto each message. A fresh agent with good notes beats a stale agent with a bloated context window.

**Llama as dogs body** - Ollama handles the grunt work that doesn't need thinking: formatting, boilerplate, cleanup, data conversion. Doesn't even get memento notes - just the task and nothing else.

**Compact protocol** - Agents don't need English to talk to each other. Inter-agent messages use a terse key-value format:

```
English:  "Please write a Python function called sum_evens that takes
           a list of integers and returns the sum of even numbers"

Protocol: T:codegen|L:py|N:sum_evens|I:list[int]|O:int|D:sum even vals
```

## CLI

```bash
# Just say what you need
python -m src "build a REST API for user management"

# Send to a specific agent
python -m src "write a binary search" --agent codex

# Grunt work (straight to Llama)
python -m src "format this JSON: {a:1,b:2}" --grunt

# Check what's alive
python -m src --health

# See survival notes
python -m src --notes

# Control swarm depth
python -m src "redesign the whole system" --max-depth 4
```

## Python API

```python
import asyncio
from src import Orchestrator

async def main():
    orch = Orchestrator(rules_path="rules.md")

    # Complex task -> auto-decomposes, spawns sub-swarms
    result = await orch.run("Design a REST API for user management")
    print(result)

    # Grunt work -> straight to Llama
    cleaned = await orch.grunt("sort alphabetically: zebra, apple, mango")

    # Manual sub-swarm spawning
    child = orch.spawn()
    result = await child.run("handle just this subtask")

    # Check survival notes
    print(orch.notes())

asyncio.run(main())
```

## Setup

```bash
pip install -e .
```

### Environment Variables

```bash
export ANTHROPIC_API_KEY="your-key"                # Claude
export OPENAI_API_KEY="your-key"                   # Codex
export GOOGLE_API_KEY="your-key"                   # Gemini
export LLAMA_BASE_URL="http://localhost:11434/v1"   # Llama (ollama)
```

## Project Structure

```
src/
├── __main__.py             # CLI entry point
├── orchestrator.py         # Recursive swarm orchestrator
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
