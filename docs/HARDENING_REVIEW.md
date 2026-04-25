# Hardening review notes

This branch applies the safe hardening changes from the review and records the remaining orchestrator-level changes that should be made with a local checkout.

## Applied in this branch

- Removed unused SDK/API dependencies from `pyproject.toml` and `requirements.txt` so the default install matches the CLI-first architecture.
- Removed `src/agents/llama_agent.py` and the public Llama export.
- Updated default rules and tests to use the three-agent team: Claude, Codex and Gemini.
- Fixed CLI timeout handling so timed-out subprocesses are terminated before retrying.
- Added real CLI health checks for Codex and Gemini instead of only checking whether the executable exists.
- Updated wrapper metadata and docs for the current model posture:
  - Claude uses the Claude Code `sonnet` alias by default.
  - Codex metadata points to GPT-5.2-Codex for agentic coding where supported.
  - Gemini metadata points to Gemini 3 Pro Preview for high-capability multimodal work where supported.
- Rewrote the README to describe the CLI-only architecture, the Memento memory pattern, cross-model QA, and the known hardening backlog.

## Remaining recommended changes

### 1. Make QA structured and fail safer

`src/orchestrator.py` currently parses free-form QA responses with regex and defaults to pass when a reviewer response cannot be parsed. That avoids deadlocks, but it weakens the core promise of strict cross-model QA.

Recommended change:

- Ask reviewers for strict JSON.
- Parse with a Pydantic model.
- Re-ask the reviewer once if parsing fails.
- Add a config option such as `qa_fail_open: bool = False`.
- Default unparseable QA to fail closed or return `needs_human_review`.

Suggested schema:

```json
{
  "items": [
    {"id": 1, "passed": true, "reason": "Meets the endpoint requirement"},
    {"id": 2, "passed": false, "reason": "Missing duplicate ID handling"}
  ],
  "score": 6,
  "passed": false,
  "feedback": "Add duplicate ID handling and retry."
}
```

### 2. Persist memento automatically when configured

`Memento` supports `save()` and `load()`, but the orchestrator should call `load()` when a `memento_path` is supplied and `save()` after material note changes.

### 3. Add a durable run workspace

For serious autonomous work, text-only stdout is not enough. Add a run folder such as:

```text
runs/2026-04-26T190000Z/
├── manifest.json
├── memento.json
├── subtasks.json
├── qa-ledger.json
├── artefacts/
└── final-report.md
```

### 4. Add end-to-end fake CLI tests

Unit tests use fake agents, which is good. Add an integration test that creates temporary fake `claude`, `codex` and `gemini` executables on PATH, then runs the real CLI entry point.

### 5. Decide whether rules are advisory or authoritative

At the moment, `rules.md` documents model behaviour, while `CAPABILITY_PROVIDERS` performs the main runtime mapping. Either keep this clearly advisory, or move routing configuration fully into a structured config file.
