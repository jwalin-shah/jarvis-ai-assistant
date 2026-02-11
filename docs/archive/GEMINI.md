# GEMINI.md

Instructions for Gemini when working with this repository. See `CLAUDE.md` for the full reference; this file captures the critical rules that apply to all LLMs.

## Project Overview

JARVIS is a local-first AI assistant for macOS providing intelligent iMessage management using MLX-based language models on Apple Silicon (8GB RAM).

## Build & Test

```bash
make setup        # First-time setup
make test         # Run tests (output -> test_results.txt)
make verify       # Full verification (lint + typecheck + test)
make lint         # Run ruff
make format       # Auto-format
```

- Always use `uv run` for Python commands
- Always use `pnpm` for Node.js
- Always read `test_results.txt` after running tests

## Five Core Principles

1. **Think Before Coding** - State assumptions. ASK when uncertain.
2. **Simplicity First** - Minimum code. No speculation. No overengineering.
3. **Surgical Changes** - Touch ONLY what's needed. No drive-by refactoring.
4. **Goal-Driven** - Define verifiable success criteria. Loop until verified.
5. **Performance by Default** - Always parallelize, batch, cache.

## Performance Checklist (MANDATORY)

- [ ] **Batch Everything**: `embedder.encode(list)` not loop with `embedder.encode(x)`
- [ ] **Batch DB Operations**: `executemany()` not loops with `execute()`
- [ ] **Binary for Embeddings**: `embedding.tobytes()` / `np.frombuffer()`, never JSON float lists
- [ ] **Singleton Models**: Load once via double-check locking, never per-request
- [ ] **Memory Limits**: Set MLX memory limits before model operations
- [ ] **Vectorized Operations**: Prefer NumPy/pandas over Python loops
- [ ] **Cache Expensive Ops**: Embeddings, classification results, DB queries with TTL
- [ ] **Reuse Connections**: Don't open/close DB connections per operation
- [ ] **No N+1 Queries**: Never query in a loop. Use CTEs, JOINs, batch fetch.
- [ ] **Visible Progress**: All scripts >30s need progress bars, ETAs, and logging

## Memory Constraints (8GB System)

- Keep working set under 500MB for batch operations
- Use `n_jobs=1` for datasets >100MB (parallel = swap thrashing)
- Stream large datasets to disk with `np.memmap()`
- One model loaded at a time

## Code Style

- Line length: 100 characters
- Python 3.11+ with strict type hints
- Linting: ruff (E, F, I, N, W, UP)
- All prompts in `jarvis/prompts.py`
- Errors inherit from `JarvisError` in `jarvis/errors.py`

## Key Architecture

- Socket server: `jarvis/socket_server.py` (Unix socket + WebSocket)
- Reply pipeline: `jarvis/reply_service.py` -> `jarvis/prompts.py` -> `models/`
- Classifiers: `jarvis/classifiers/` (LightGBM + cascade)
- Search: `jarvis/search/` (semantic + vector)
- Desktop: `desktop/` (Tauri + Svelte)
- iMessage: `integrations/imessage/` (read-only SQLite)
