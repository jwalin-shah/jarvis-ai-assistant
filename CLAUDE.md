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

#### Never Load Twice
- **NEVER load data/models/files twice** - this is a critical efficiency bug
- If you need metadata (e.g., checking if a key exists), read ONLY the metadata:
  - Safetensors: read 8-byte header size + JSON header, not full weights
  - JSON: if checking structure, don't parse the whole file twice
  - DB: query once, store result, don't re-query
- Before writing any load/read code, ask: "Will this data be needed again? Am I loading it elsewhere?"
- Common anti-pattern to AVOID:
  ```python
  # BAD: loads weights twice
  weights = load(path)
  has_key = "pooler" in weights.keys()
  del weights  # doesn't actually free memory immediately
  model.load_weights(path)  # loads AGAIN

  # GOOD: load once, use for both
  weights = load(path)
  has_key = "pooler" in weights.keys()
  model.load_weights(weights)  # reuse same data
  ```

#### Memory-Constrained Systems (8GB)
This system has **8GB RAM**. Large data processing MUST account for this:

- **Stream to disk**: Never accumulate large arrays in RAM
  - Use `np.memmap()` for embeddings, write as you go
  - Process in chunks, don't load everything first
  - 100k embeddings × 384 dims × 4 bytes = 150MB (acceptable)
  - 500k embeddings × 768 dims × 4 bytes = 1.5GB (too much, stream to disk)

- **Binary over JSON**: JSON serialization of float arrays is extremely slow
  - 500 embeddings × 384 floats as JSON = ~2MB of text to parse
  - Same as base64 binary = ~300KB
  - Always use `binary=True` for embedding server requests
  - Return base64-encoded numpy bytes, not JSON lists

- **Reuse connections**: Don't open/close sockets per batch
  - Open once, send all batches, close at end
  - Socket connect/disconnect overhead adds up over 1000s of batches

- **Adaptive batch sizes by model**:
  ```python
  if model in ("bge-large", "arctic-l"):    # 1.3GB weights
      batch_size = 64
  elif model in ("bge-base", "arctic-m"):   # 500MB weights
      batch_size = 128
  else:                                      # <200MB weights
      batch_size = 256
  ```

- **Watch for swap**: If Activity Monitor shows >500MB swap, you're memory-bound
  - 0% CPU + high RAM = swapping to disk = 10-100x slower
  - Reduce batch size or stream to disk

- **MLX memory management**:
  - `mx.clear_cache()` forces GPU sync - NEVER call per-batch (kills throughput)
  - Call cache clear only: when switching models, or every ~10 batches max
  - `del array` doesn't free MLX memory immediately - GC timing is unpredictable
  - If memory grows unbounded, clear cache periodically, not per-operation

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

- **Memory Budget**: 8GB total system RAM - this is TIGHT
  - Never hold >500MB of data in RAM at once
  - Stream large datasets to disk (memmap)
  - One model loaded at a time, unload before loading next
- **Read-Only DB**: iMessage chat.db uses `file:...?mode=ro`
- **No Fine-Tuning**: Use RAG + few-shot instead
- **Prompts**: All in `jarvis/prompts.py` - nowhere else
- **Errors**: Inherit from `JarvisError` in `jarvis/errors.py`
- **Batching**: Use `classify_batch()`, `embedder.encode(list)` - never loop
- **Parallelization**: `n_jobs=-1` for all sklearn operations
- **IPC Protocol**: Use binary encoding (base64) not JSON for large arrays
