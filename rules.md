# Model Rules: Strengths & Weaknesses

Routing rules for the orchestrator. These rules document the intended model roles.
The main runtime mapping still lives in `CAPABILITY_PROVIDERS` inside `src/orchestrator.py`.

## Claude

### Strengths
- Complex reasoning and multi-step analysis
- Task decomposition and plan synthesis
- Code review and architectural design
- Nuanced instruction following
- Long-context understanding when configured through Claude Code aliases

### Weaknesses
- Direct image generation
- Audio/video processing as a primary task
- Simple repetitive formatting tasks where a faster model is enough

### Best For
- Task analysis and routing decisions
- Response synthesis from multiple agents
- Complex reasoning chains
- Code architecture and review
- QA adjudication when another model produced the work

### Avoid
- Direct image generation
- Audio processing as the primary output

## Codex

### Strengths
- Long-horizon agentic coding
- Code generation across many languages
- Codebase refactoring and migrations
- Test creation and debugging
- Terminal-oriented software engineering workflows

### Weaknesses
- Multimodal understanding as the primary task
- Narrative or brand-heavy creative writing
- Purely visual judgement without code context

### Best For
- Writing new code from specifications
- Translating code between languages
- Debugging and code optimisation
- Implementing fixes after QA feedback
- Reviewing reasoning-heavy plans for implementation gaps

### Avoid
- Image or audio processing as the primary task
- Simple formatting tasks where Gemini is sufficient

## Gemini

### Strengths
- Multimodal understanding across text, images, video, audio and PDFs
- Large-context document and media analysis
- Fast formatting and transformation work
- Visual reasoning and UI/screenshot interpretation
- Broad agentic tasks when multimodal context is central

### Weaknesses
- Specialist code generation compared with Codex
- Acting as the final authority on complex architecture without cross-checking

### Best For
- Image, video, audio and PDF analysis
- Visual QA and screenshot interpretation
- Translation and cultural context
- Formatting, extraction, sorting and boilerplate where QA is not worth the cost

### Avoid
- Complex production code changes where Codex should lead
- Final review of subtle reasoning without Claude or Codex cross-checking
