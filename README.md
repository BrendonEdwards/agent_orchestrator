# Agent Orchestrator

Multi-model AI agent orchestration system with Claude as the central hub.

## Architecture

```
         ┌─────────────┐     ┌──────────────────────────────┐
         │ Orchestrator │────▶│ Rules.md: strengths/weaknesses│
         └──────┬───────┘     └──────────────────────────────┘
                │
          ┌─────▼─────┐
          │   Claude   │  ◀── The brain (thinking, synthesis)
          └─────┬──────┘
          ╱     │     ╲
    ┌────▼─┐ ┌─▼────┐ ┌▼─────┐
    │ Codex│ │Gemini │ │Llama │ ◀── The dogs body (grunt work)
    └──────┘ └──────┘ └──────┘

    Memento: concise survival notes (fights context rot)
    Protocol: agents talk in compact format, not English
```

## Core Ideas

**Stateless agents + Memento** - Like the film. Every agent call is a fresh spawn with zero prior context. No conversation history accumulates, so no context rot. The only "memory" is Memento: ~20 ultra-concise survival notes that get stamped onto each message. When note space fills up, low-priority notes are dropped - like choosing which tattoo matters most. A fresh agent with good notes beats a stale agent with a bloated context window.

**Llama as dogs body** - Ollama handles the grunt work that doesn't need thinking: formatting, boilerplate, cleanup, data conversion. Doesn't even get memento notes - just the task and nothing else.

**Compact protocol** - Agents don't need English to talk to each other. Code is in English for humans, but inter-agent messages use a terse key-value format that cuts token usage by ~60%:

```
English:  "Please write a Python function called sum_evens that takes
           a list of integers and returns the sum of even numbers"

Protocol: T:codegen|L:py|N:sum_evens|I:list[int]|O:int|D:sum even vals
```

## Agents

| Agent | Role | When to use |
|-------|------|-------------|
| **Claude** | Brain | Complex reasoning, synthesis, coordination |
| **Codex** | Coder | Code generation, technical translation |
| **Gemini** | Eyes/Ears | Images, audio, sound, multimodal |
| **Llama** | Dogs body | Formatting, boilerplate, simple grunt work |

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

## Usage

```python
import asyncio
from src import Orchestrator

async def main():
    orch = Orchestrator(rules_path="rules.md")

    # Complex task -> Claude analyzes, delegates, synthesizes
    result = await orch.run("Design a REST API for user management")
    print(result)

    # Grunt work -> goes straight to Llama
    result = await orch.run("Format this JSON: {a:1,b:2}")
    print(result)

    # Direct grunt work
    cleaned = await orch.grunt("Sort these alphabetically: zebra, apple, mango")
    print(cleaned)

    # Check survival notes (memento)
    print(orch.notes())

    # Check what's alive
    health = await orch.health_check()
    print(health)

asyncio.run(main())
```

## Project Structure

```
src/
├── orchestrator.py          # Main orchestrator
├── agents/
│   ├── base.py              # Base agent interface
│   ├── claude_agent.py      # Claude (brain)
│   ├── codex_agent.py       # Codex (code)
│   ├── gemini_agent.py      # Gemini (multimodal)
│   └── llama_agent.py       # Llama (grunt work)
├── memento/
│   └── memento.py           # Anti-context-rot survival notes
├── routing/
│   ├── router.py            # Inter-agent message routing
│   └── protocol.py          # Compact wire format (not English)
└── rules/
    └── loader.py            # Rules.md parser
```

## License

Apache 2.0
