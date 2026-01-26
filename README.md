# JARVIS AI Assistant

Local-first AI assistant for macOS with intelligent iMessage management using MLX-based language models. Runs entirely on Apple Silicon with no cloud data transmission.

## Project Status

| Component | Status | Notes |
|-----------|--------|-------|
| Contracts/Interfaces | Complete | All 9 protocol definitions |
| Model Generator (WS8) | Complete | MLX loader, template fallback, RAG support |
| iMessage Reader (WS10) | Complete | Schema detection, attachments, reactions |
| Memory Profiler (WS1) | Complete | MLX memory profiling with model unload |
| HHEM Benchmark (WS2) | Complete | Vectara HHEM model evaluation |
| Latency Benchmark (WS4) | Complete | Cold/warm/hot start scenarios |
| Memory Controller (WS5) | Complete | Three-tier modes (FULL/LITE/MINIMAL) |
| Degradation Controller (WS6) | Complete | Circuit breaker pattern |
| Setup Wizard | Complete | Environment validation, config init |
| CLI Entry Point | Complete | Chat, search, health, benchmark commands |
| FastAPI Layer | Complete | REST API for Tauri frontend integration |
| Config System | Complete | Nested sections with migration support |

**Default Model**: Qwen2.5-0.5B-Instruct-4bit

## Features

### Implemented

- **iMessage Integration** - Read-only local database access with schema auto-detection (v14/v15)
- **MLX Model Generation** - Local inference on Apple Silicon with memory-aware loading
- **Template-First Generation** - Semantic matching against 25 iMessage scenario templates (0.7 threshold)
- **Memory Controller** - Three-tier modes based on available RAM (FULL/LITE/MINIMAL)
- **Graceful Degradation** - Circuit breaker pattern for feature failures
- **HHEM Validation** - Post-generation hallucination scoring via Vectara model
- **Setup Wizard** - Guided first-time setup with permission and environment validation
- **CLI Interface** - Interactive chat, message search with filters, health monitoring, benchmarks
- **FastAPI Layer** - REST API for Tauri desktop frontend integration
- **Config System** - Nested configuration with automatic migration between versions

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- 8GB RAM minimum (16GB recommended)
- Full Disk Access permission (for iMessage)
- [uv](https://docs.astral.sh/uv/) package manager

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd jarvis-ai-assistant
make setup  # Installs deps + enables git hooks

# Run setup wizard (validates environment)
python -m jarvis.setup

# Verify everything works
make verify
```

## CLI Usage

```bash
# Interactive chat mode
jarvis chat

# Search iMessage conversations
jarvis search-messages "meeting tomorrow"
jarvis search-messages "dinner" --limit 50
jarvis search-messages "project" --sender "John" --after 2024-01-01

# Show system health status
jarvis health

# Run benchmarks
jarvis benchmark memory
jarvis benchmark latency
jarvis benchmark hhem

# Version information
jarvis --version
```

## Development Commands

```bash
make test          # Run tests (results in test_results.txt)
make test-fast     # Stop at first failure
make check         # Run all linters (ruff, mypy)
make verify        # Full verification (lint + test)
make health        # Project health summary
make help          # List all available commands
```

## Running Benchmarks

All benchmarks are implemented and functional:

```bash
# Full overnight evaluation (sequential, memory-safe)
./scripts/overnight_eval.sh

# Quick mode for testing
./scripts/overnight_eval.sh --quick

# Individual benchmarks
python -m benchmarks.memory.run --output results/memory.json
python -m benchmarks.hallucination.run --output results/hhem.json
python -m benchmarks.latency.run --output results/latency.json

# Check gate pass/fail status
python scripts/check_gates.py results/latest/
```

### Validation Gates

| Gate | Metric | Pass | Conditional | Fail |
|------|--------|------|-------------|------|
| G1 | Model stack memory | <5.5GB | 5.5-6.5GB | >6.5GB |
| G2 | Mean HHEM score | >=0.5 | 0.4-0.5 | <0.4 |
| G3 | Warm-start latency | <3s | 3-5s | >5s |
| G4 | Cold-start latency | <15s | 15-20s | >20s |

## Project Structure

```
jarvis-ai-assistant/
├── jarvis/          # CLI entry point, setup wizard, and config
├── api/             # FastAPI REST layer for Tauri frontend
├── benchmarks/      # Validation gate implementations
│   ├── memory/      # MLX memory profiler
│   ├── hallucination/  # HHEM benchmark
│   └── latency/     # Latency benchmark
├── contracts/       # Python Protocol interfaces (9 protocols)
├── core/            # Infrastructure
│   ├── health/      # Circuit breaker, degradation, permissions
│   └── memory/      # Memory controller and monitoring
├── integrations/
│   └── imessage/    # iMessage reader
├── models/          # MLX model loading and inference
├── tests/           # Test suite (854 tests)
├── scripts/         # Benchmark and reporting utilities
├── desktop/         # Tauri desktop app (Svelte frontend)
└── docs/            # Design docs and audit report
```

## Workflow

1. Create feature branch: `git checkout -b feature/my-thing`
2. Make changes
3. Run `make verify` before committing
4. Push and create PR

For parallel work, use git worktrees - see [CLAUDE.md](CLAUDE.md) for details.

## Documentation

- [CLAUDE.md](CLAUDE.md) - Development workflow, architecture, and coding guidelines
- [docs/CODEBASE_AUDIT_REPORT.md](docs/CODEBASE_AUDIT_REPORT.md) - Full codebase audit
- [docs/JARVIS-v1-Design-Document.md](docs/JARVIS-v1-Design-Document.md) - Architecture design
- [docs/JARVIS-v1-Development-Guide.md](docs/JARVIS-v1-Development-Guide.md) - Development guide

## License

MIT License - see pyproject.toml
