# JARVIS AI Assistant

Welcome to the JARVIS documentation. JARVIS is a local-first AI assistant for macOS with iMessage integration, optimized for Apple Silicon.

## Key Features

- **Local-First:** All AI inference happens on your machine using MLX.
- **Privacy:** Your messages never leave your device.
- **Adaptive Memory:** Learns your style and preferences over time.
- **Smart Routing:** Intelligently decides when to generate, clarify, or acknowledge.

## Documentation Sections

- [Architecture](ARCHITECTURE.md): The technical design of JARVIS.
- [How it Works](HOW_IT_WORKS.md): Deep dive into the pipeline.
- [API Reference](api_reference.md): Automatically generated documentation from the source code.
- [Testing Guidelines](TESTING_GUIDELINES.md): How we ensure quality.

## Getting Started

To build the documentation locally:

```bash
mkdocs serve
```

To build for deployment:

```bash
mkdocs build
```
