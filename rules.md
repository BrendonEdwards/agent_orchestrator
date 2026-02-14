# Model Rules: Strengths & Weaknesses

Configuration for the Agent Orchestrator's routing decisions. Each model's
strengths and weaknesses determine which tasks are delegated to it.

## Claude

### Strengths
- Complex reasoning and multi-step analysis
- Nuanced conversation and instruction following
- Code review and architectural design
- Synthesizing information from multiple sources
- Long-context understanding
- Safety and alignment

### Weaknesses
- Image generation
- Audio/video processing
- Real-time data access

### Best For
- Task analysis and routing decisions
- Response synthesis from multiple agents
- Complex reasoning chains
- Code architecture and review
- Conversational interactions

### Avoid
- Direct image generation tasks
- Audio transcription or generation

## Codex

### Strengths
- Code generation across many languages
- Code translation between programming languages
- Context compression and summarization
- Mathematical reasoning
- Technical problem solving

### Weaknesses
- Creative or narrative writing
- Multimodal understanding
- Conversational nuance

### Best For
- Writing new code from specifications
- Translating code between languages
- Compressing verbose context into compact form
- Technical and mathematical tasks
- Debugging and code optimization

### Avoid
- Image or audio processing
- Open-ended creative tasks

## Gemini

### Strengths
- Multimodal understanding (text, images, audio, video)
- Image analysis and description
- Audio and sound processing
- Language translation with cultural context
- Cross-modal reasoning

### Weaknesses
- Complex multi-step logical reasoning
- Specialized code generation
- Very long context processing

### Best For
- Image understanding and captioning
- Audio transcription and analysis
- Multimodal tasks combining text with media
- Language translation
- Visual question answering

### Avoid
- Pure code generation tasks
- Complex mathematical proofs
- Tasks requiring very long context windows

## Llama

### Strengths
- Fast local inference (no network latency)
- Complete data privacy (nothing leaves the machine)
- Zero API cost for high-volume tasks
- Offline operation capability
- Customizable through fine-tuning

### Weaknesses
- Smaller context window than cloud models
- Less capable on very complex reasoning
- No multimodal capabilities
- Requires local GPU resources

### Best For
- Privacy-sensitive data processing
- High-volume repetitive tasks
- Offline or air-gapped environments
- Rapid prototyping and iteration
- Cost-sensitive operations

### Avoid
- Tasks requiring very large context windows
- Complex multi-step reasoning chains
- Multimodal tasks
- Tasks requiring the highest possible quality
