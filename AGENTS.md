# Repository Guidelines

## Project Structure
- `jarvis/`: Core Python library (CLI, prompts, retrieval, response generation).
- `api/`: FastAPI server.
- `desktop/`: Tauri + Svelte desktop app (see `desktop/README.md`).
- `core/`, `models/`, `integrations/`, `contracts/`: Shared infrastructure and ML/runtime pieces.
- `benchmarks/`: Evaluation and validation gates.
- `tests/`: Unit and integration tests.
- `scripts/`: Utilities for benchmarks and reporting.
- `docs/`: Design and architecture docs (see `docs/DESIGN.md`).

## Build, Test, and Development Commands
- `make setup`: Install dependencies and configure git hooks.
- `make api-dev`: Run the API server locally on port 8742.
- `make desktop-setup`: Install desktop app dependencies.
- `cd desktop && npm run tauri dev`: Launch the desktop app (API must be running).
- `make test`: Run the full test suite (writes `test_results.txt`).
- `make test-fast`: Stop on first failure.
- `make check`: Run lint + format-check + typecheck.
- `make verify`: Run full verification (checks + tests).
- `make help`: List all available commands.

## Coding Style & Naming Conventions
- Python is formatted with Ruff (`make format`) and linted with Ruff (`make lint`).
- Line length is 100 characters; target Python version is 3.11.
- Lint rules include `E,F,I,N,W,UP` with `E741` ignored; ML scripts have specific per-file ignores.
- Type checking uses strict `mypy` (`make typecheck`).
- All LLM prompts must live in `jarvis/prompts.py`.

## Testing Guidelines
- Tests run with `pytest` via `make test`; coverage includes `jarvis/`, `api/`, `models/`, `core/`, `integrations/`, `contracts/`, `benchmarks/`.
- Test files are named `test_*.py` and test functions use `test_*`.
- Use descriptive names like `test_expand_slang_preserves_capitalization`.

## Commit & Pull Request Guidelines
- Commit format:
  - `<type>: <short description>` with optional body.
  - Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.
- PR flow: branch from `main`, run `make verify`, push, open PR, ensure CI passes, request review.

## Performance Checklist for New Code

The codebase targets Apple Silicon with 8GB RAM. All code must respect these constraints:

### Critical Performance Rules

- [ ] **Batch Everything**: Use `embedder.encode(list)` not `for x: embedder.encode(x)`
- [ ] **Batch DB Operations**: Use `executemany()` not loops with `execute()`
- [ ] **Binary for Embeddings**: Use `embedding.tobytes()` / `np.frombuffer()`, never JSON float lists
- [ ] **Singleton Models**: Load models once via double-check locking, never per-request
- [ ] **Memory Limits**: Set MLX memory limits (`mx.set_memory_limit()`) before model operations
- [ ] **Stream Large Data**: Use `np.memmap` for datasets >500MB
- [ ] **Lazy Imports**: Import heavy modules (sklearn, torch, mlx) inside functions, not at module level
- [ ] **Vectorized Operations**: Prefer NumPy/pandas over Python loops
- [ ] **Cache Expensive Ops**: Cache embeddings, classification results, DB queries with TTL
- [ ] **Reuse Connections**: Don't open/close DB connections per operation

### Anti-Patterns to Avoid

```python
# ❌ BAD: Unbatched embedding
for text in texts:
    emb = embedder.encode(text)  # N round trips

# ✅ GOOD: Batched embedding
embeddings = embedder.encode(texts)  # 1 round trip

# ❌ BAD: JSON for embeddings
json.dumps(embedding.tolist())  # 2MB for 500 embeddings

# ✅ GOOD: Binary for embeddings
embedding.astype(np.float32).tobytes()  # 300KB for 500 embeddings

# ❌ BAD: Row-by-row DB inserts
for row in rows:
    conn.execute("INSERT ...", row)

# ✅ GOOD: Bulk insert
conn.executemany("INSERT ...", rows)

# ❌ BAD: Module-level heavy imports
import torch  # Slows startup even if unused

# ✅ GOOD: Lazy imports
def process():
    import torch  # Only when needed
```

### Memory Constraints (8GB System)

- Keep working set under 500MB for batch operations
- Use `n_jobs=1` for datasets >100MB (parallel = swap thrashing)
- Clear MLX cache after large operations: `mx.clear_cache()`
- Monitor swap: 0% CPU + high RAM = swapping = 10-100x slower

### Performance Review Checklist

Before submitting PRs with data processing or ML code:

1. Is anything loaded/computed twice?
2. Are there loops that could be batched/vectorized?
3. Is JSON being used where binary would be faster?
4. Are connections/resources reused or reopened each time?
5. Could this blow past 500MB RAM?
6. Is there visible progress for long operations?
