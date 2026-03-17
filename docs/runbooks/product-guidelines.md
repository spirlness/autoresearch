# Product Guidelines: autoresearch

## Writing Style & Tone
- **Concise & Data-Driven**: Documentation and logs should prioritize hard data, metrics, and technical rationale over narrative descriptions.
- **Formal but Practical**: Use professional, objective language suitable for technical research and software engineering.
- **Action-Oriented**: Focus on clear outcomes of experiments and specific recommendations for architectural changes.

## Branding & Visual Identity
- **Functional Aesthetics**: The "UI" (logs, terminal output, charts) should prioritize information density and clarity.
- **Minimalist**: Avoid decorative elements; focus on making training progress, metrics, and error logs the primary focus.

## User Experience (UX) Principles
- **Agent-First Design**: Ensure that all logs, configuration files, and metrics are easy for an LLM agent to parse, analyze, and modify.
- **Fast Feedback Loops**: Maximize the speed at which a user (or agent) can understand the result of an experiment and decide on the next step.
- **Clear Failure Modes**: Error messages must be precise, providing clear information on what failed (e.g., OOM, compilation error, configuration error).

## Documentation Guidelines
- **Self-Documenting Code**: Code should be readable and use descriptive naming, with comments reserved for explaining "why" rather than "what."
- **Traceable History**: Every significant experiment or architectural change must be documented with its context, the hypothesized result, and the actual outcome.
