# Agent Orchestrator

Multi-model AI agent orchestration system with Claude as the central hub.

## Architecture

```
         ┌─────────────┐     ┌─────────────────────────────────┐
         │ Orchestrator │────▶│ Rules.md: Model strengths +     │
         └──────┬───────┘     │           weaknesses            │
                │              └─────────────────────────────────┘
                ▼
           ┌─────────┐
           │  Claude  │  ◀── Central hub / coordinator
           └────┬────┘
          ╱     │     ╲
    ┌────▼─┐ ┌─▼───┐ ┌▼─────┐
    │ Codex│ │Gemini│ │Llama │
    └──┬───┘ └──┬──┘ └──┬───┘
       └────────┼────────┘
            (mesh)

    Memento: audit trail of all decisions
```

## Agents

| Agent | Provider | Specialization |
|-------|----------|----------------|
| **Claude** | Anthropic | Central hub - reasoning, synthesis, coordination |
| **Codex** | OpenAI | Code generation, language translation, context compression |
| **Gemini** | Google | Multimodal - language, imagery, sound |
| **Llama** | Local | Private inference, fast iteration, offline operation |

## Key Concepts

- **Orchestrator**: Top-level coordinator that manages the agent team
- **Claude as Hub**: Claude analyzes tasks, delegates to specialists, and synthesizes results
- **Context Compression**: Codex translates verbose outputs into compact representations before routing between agents (conserves context windows)
- **Rules.md**: Configurable model strengths/weaknesses that inform routing decisions
- **Memento**: Audit trail answering "how did we write the notes?" - records every decision, delegation, and result
- **Message Router**: Handles inter-agent communication with full mesh support

## Setup

```bash
pip install -e .
```

### Environment Variables

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"     # For Claude
export OPENAI_API_KEY="your-openai-key"           # For Codex
export GOOGLE_API_KEY="your-google-key"            # For Gemini
export LLAMA_BASE_URL="http://localhost:11434/v1"  # For Llama (ollama default)
```

## Usage

```python
import asyncio
from src import Orchestrator

async def main():
    orch = Orchestrator(rules_path="rules.md", memento_dir="memento_logs")

    # Run a task through the full pipeline
    result = await orch.run("Write a Python function to parse CSV files")
    print(result)

    # Send directly to a specific agent
    response = await orch.send_to("gemini", "Describe this image")
    print(response.content)

    # Check agent health
    health = await orch.health_check()
    print(health)

    # View the audit trail
    print(orch.get_audit_trail())

    # Save the memento log
    orch.save_memento()

asyncio.run(main())
```

## Project Structure

```
agent_orchestrator/
├── src/
│   ├── orchestrator.py          # Main orchestrator
│   ├── agents/
│   │   ├── base.py              # Base agent interface
│   │   ├── claude_agent.py      # Claude (central hub)
│   │   ├── codex_agent.py       # Codex (code + translation)
│   │   ├── gemini_agent.py      # Gemini (multimodal)
│   │   └── llama_agent.py       # Llama (local inference)
│   ├── memento/
│   │   └── memento.py           # Audit trail system
│   ├── routing/
│   │   └── router.py            # Inter-agent message routing
│   └── rules/
│       └── loader.py            # Rules.md parser
├── rules.md                     # Model strengths & weaknesses
├── pyproject.toml
└── requirements.txt
```

## License

Apache 2.0
