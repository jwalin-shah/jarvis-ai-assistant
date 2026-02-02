# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

JARVIS is a local-first AI assistant for macOS providing intelligent iMessage management using MLX-based language models on Apple Silicon.

**Default Model**: LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit

**Documentation:**
- `docs/DESIGN.md` - Comprehensive design doc with rationale and decisions (start here)
- `docs/ARCHITECTURE.md` - Technical implementation status
- `docs/ARCHITECTURE_V2.md` - Unix socket + direct SQLite optimizations
- `docs/CLI_GUIDE.md` - CLI usage

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
- **Always use `uv run`** for Python commands (not pip, python directly)
- **Always use `pnpm`** for Node.js (not npm, yarn)
- **Always use `rm -f`** to avoid interactive prompts

### Use Efficient Tools (NOT Bash equivalents)
- **Grep tool** for searching - NEVER `grep -r` or `rg` via Bash (slow on large dirs like node_modules)
- **Glob tool** for finding files - NEVER `find` via Bash
- **Read tool** for file contents - NEVER `cat`, `head`, `tail` via Bash
- **Edit tool** for modifications - NEVER `sed`, `awk` via Bash
- Bash is for: git, make, npm, uv, and actual shell operations only

### Test Rules (MANDATORY)
- **ALWAYS** use `make test` - never raw pytest
- **AFTER** tests, **ALWAYS** read `test_results.txt`
- If tests fail, **quote the ACTUAL error** from `test_results.txt`

### When to Run Tests
- **DO** run tests after code changes (`.py` files)
- **DON'T** run tests after doc-only changes (`.md` files) - wastes tokens
- **DON'T** re-run tests if `test_results.txt` already has the info you need

### Running Tests Efficiently
1. Run `make test` (outputs to `test_results.txt`)
2. If tests hang or take too long, read `test_results.txt` to check progress
3. Use `tail -20 test_results.txt` to check recent output without re-running
4. **NEVER** re-run tests just to see results - read the file instead

### Before Saying "Done"
1. Run `make verify` (only after code changes)
2. Read `test_results.txt` - confirm all pass
3. Run `git diff` - verify surgical changes
4. **Commit your changes** - don't leave work uncommitted
5. Report with **specific evidence** ("All 47 tests pass")

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
