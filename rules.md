# Model Rules: Strengths & Weaknesses

Routing rules for the orchestrator. Determines which agent gets which work.

## Claude

### Strengths
- Complex reasoning and multi-step analysis
- Synthesizing information from multiple sources
- Code review and architectural design
- Nuanced conversation and instruction following
- Long-context understanding

### Weaknesses
- Image generation
- Audio/video processing

### Best For
- Task analysis and routing decisions
- Response synthesis from multiple agents
- Complex reasoning chains
- Code architecture and review

### Avoid
- Image generation
- Audio processing
- Simple formatting tasks (use Llama)

## Codex

### Strengths
- Code generation across many languages
- Code translation between programming languages
- Mathematical reasoning
- Technical problem solving

### Weaknesses
- Creative or narrative writing
- Multimodal understanding

### Best For
- Writing new code from specifications
- Translating code between languages
- Technical and mathematical tasks
- Debugging and code optimization

### Avoid
- Image or audio processing
- Simple formatting tasks (use Llama)

## Gemini

### Strengths
- Multimodal understanding (text, images, audio, video)
- Image analysis and description
- Audio and sound processing
- Language translation with cultural context

### Weaknesses
- Complex multi-step logical reasoning
- Specialized code generation

### Best For
- Image understanding and captioning
- Audio transcription and analysis
- Multimodal tasks combining text with media
- Language translation

### Avoid
- Pure code generation
- Complex mathematical proofs
- Simple formatting tasks (use Llama)

## Llama

### Strengths
- Fast (local, no network latency)
- Free (no API cost)
- Always available (offline capable)
- Good enough for simple tasks

### Weaknesses
- Cannot do complex reasoning
- No multimodal capabilities
- Smaller context window

### Best For
- Formatting and cleanup
- Boilerplate generation
- Simple text extraction and transformation
- Data conversion
- Repetitive batch operations
- Any grunt work that doesn't need thinking

### Avoid
- Complex reasoning
- Multimodal tasks
- Tasks requiring high accuracy
- Anything that needs real thought
