# Jarvis AI Assistant

Local-first AI assistant for macOS with email and iMessage integration.

## Features

- Intelligent email management via Gmail API
- iMessage integration (read-only, local database)
- MLX-based language models running on Apple Silicon
- No cloud data transmission - fully local inference
- Template-first generation with semantic matching

## Requirements

- macOS on Apple Silicon (M1/M2/M3)
- Python 3.11+
- 8-16GB RAM recommended
- [uv](https://docs.astral.sh/uv/) package manager

## Development Setup

```bash
# Clone and setup
git clone <repo-url>
cd jarvis-ai-assistant
make setup  # Installs deps + enables git hooks

# Verify everything works
make verify
```

## Common Commands

```bash
make test          # Run tests (results in test_results.txt)
make check         # Run all linters
make verify        # Full verification (lint + test)
make health        # Project health summary
make help          # List all available commands
```

## Workflow

1. Create feature branch: `git checkout -b feature/my-thing`
2. Make changes
3. Run `make verify` before committing
4. Push and create PR

For parallel work, use worktrees - see [CLAUDE.md](CLAUDE.md) for details.

## Project Structure

```
jarvis-ai-assistant/
├── benchmarks/      # Validation gates (memory, hallucination, coverage, latency)
├── contracts/       # Python Protocol interfaces for all modules
├── core/            # Infrastructure (memory controller, health monitoring)
├── integrations/    # External services (Gmail, iMessage)
├── models/          # MLX model loading and inference
├── tests/           # Test suite
└── scripts/         # Utility scripts
```

## Documentation

- [CLAUDE.md](CLAUDE.md) - Development workflow, architecture, and coding guidelines
- [contracts/](contracts/) - Interface definitions and contracts

## License

[Add license here]
