# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

JARVIS is a local-first AI assistant for macOS providing intelligent iMessage management using MLX-based language models on Apple Silicon.

**Default Model**: LFM-2.5-1.2B-Instruct-4bit
**Docs**: `docs/ARCHITECTURE.md` (detailed), `docs/CLI_GUIDE.md` (CLI usage)

## Quick Reference

```bash
make setup    # First-time setup
make test     # Run tests (ALWAYS use this, never raw pytest)
make verify   # Full verification before PR
make health   # Check project status

uv run python -m jarvis.setup  # Setup wizard
```

## Build Commands

```bash
make install        # Install dependencies
make test           # Run tests (output → test_results.txt)
make test-fast      # Stop at first failure
make lint           # Run ruff
make format         # Auto-format
make typecheck      # Run mypy
make verify         # Full verification (lint + typecheck + test)
```

---

## Behavioral Hooks (MANDATORY)

**Five Core Principles:**

1. **Think Before Coding** - Don't assume, ASK. Present options with tradeoffs.
2. **Simplicity First** - Minimum code. No speculation. No overengineering.
3. **Surgical Changes** - Touch ONLY what's needed. No drive-by refactoring.
4. **Goal-Driven** - Define verifiable success criteria. Loop until verified.
5. **Performance by Default** - Always parallelize, batch, cache.

---

### Think Before Coding
- State assumptions explicitly - if uncertain, ASK
- Present multiple approaches when ambiguity exists
- Stop and clarify when confused

### Simplicity First
- No features beyond what was requested
- No abstractions for single-use code
- If 200 lines could be 50, rewrite it

### Surgical Changes
- Don't "improve" adjacent code
- Match existing style
- Every changed line should trace to user's request

### Goal-Driven
- Transform vague → verifiable: "Add validation" → "Tests pass for empty/null/invalid"
- Loop until criteria met (don't say "should work" without verifying)

### Performance by Default
- **Always parallelize**: `n_jobs=-1` for scikit-learn (GridSearchCV, cross_val_score)
- **Always batch**: Process lists together, not one-at-a-time loops
- **Always cache**: Expensive computations (embeddings, model loads)
- **Vectorized ops**: NumPy/pandas over Python loops
- A slow script is NOT acceptable - performance is a correctness requirement

---

### Shell Commands
- **Always use `uv run`** for Python commands
- **Always use `rm -f`** to avoid interactive prompts

### Test Rules (MANDATORY)
- **ALWAYS** use `make test` - never raw pytest
- **AFTER** tests, **ALWAYS** read `test_results.txt`
- If tests fail, **quote the ACTUAL error** from `test_results.txt`

### Before Saying "Done"
1. Run `make verify`
2. Read `test_results.txt` - confirm all pass
3. Run `git diff` - verify surgical changes
4. Report with **specific evidence** ("All 47 tests pass")

### When Tests Fail
1. Read FULL error from `test_results.txt`
2. Identify ROOT CAUSE, not symptom
3. Fix ONE issue at a time
4. Re-run tests after each fix

---

## Code Style

- Line length: 100 characters
- Python 3.11+ with strict type hints
- Linting: ruff (E, F, I, N, W, UP)
- Run `make format` before committing

## Key Constraints

- **Memory Budget**: 8GB minimum, sequential model loading
- **Read-Only DB**: iMessage chat.db uses `file:...?mode=ro`
- **No Fine-Tuning**: Use RAG + few-shot instead
- **Prompts**: All in `jarvis/prompts.py` - nowhere else
- **Errors**: Inherit from `JarvisError` in `jarvis/errors.py`
- **Batching**: Use `classify_batch()`, `embedder.encode(list)` - never loop
- **Parallelization**: `n_jobs=-1` for all sklearn operations
