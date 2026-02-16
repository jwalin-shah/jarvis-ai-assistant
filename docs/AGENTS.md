# Repository Guidelines

## Secrets and Environment Variables

**CRITICAL**: The `.env` file contains API keys and must NEVER be committed.

- `.env` is gitignored (see root `.gitignore` line 81)
- `.env.example` contains template values for reference
- **Required for DSPy/Evals:** set `CEREBRAS_API_KEY` in local `.env` before running judge-backed evals.

---

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
- All LLM prompts must live in `jarvis/prompts/`.

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

---

## Test Flake Prevention Guidelines

All tests must be deterministic and reliable. Flaky tests waste CI resources and undermine confidence in the test suite.

### Flake Prevention Checklist

Before submitting tests, verify:

- [ ] **No Arbitrary Sleeps**: Use event-driven synchronization, not `time.sleep()`
- [ ] **Seeded Randomness**: Use `seeded_random` fixture for any random operations
- [ ] **Frozen Time**: Use `frozen_time` fixture for time-dependent tests
- [ ] **Isolated State**: Each test cleans up after itself (files, DB, caches)
- [ ] **Hardware Declared**: Use `@hardware_required()` for Apple Silicon dependencies
- [ ] **Resource Budgets**: Use `ResourceBudget` for memory-intensive operations

### Critical Anti-Patterns

```python
# ❌ BAD: Arbitrary sleep (timing flake)
def test_async():
    task = asyncio.create_task(op())
    time.sleep(0.1)  # May be too short!
    assert task.done()

# ✅ GOOD: Event-driven synchronization
def test_async():
    task = asyncio.create_task(op())
    await asyncio.wait_for(task, timeout=5.0)
    assert task.done()

# ❌ BAD: Unseeded random (randomness flake)
def test_random():
    result = np.random.randn(384)  # Different every run
    assert result[0] > 0  # Sometimes fails!

# ✅ GOOD: Seeded random
def test_random(seeded_random):
    rng = seeded_random["numpy"]
    result = rng.randn(384)  # Deterministic
    assert result[0] == 1.4967  # Always passes

# ❌ BAD: Real time (timing flake)
def test_timestamp():
    before = datetime.now()
    operation()
    after = datetime.now()
    assert (after - before).seconds < 1

# ✅ GOOD: Frozen time
def test_timestamp(frozen_time):
    with frozen_time.freeze():
        before = datetime.now()
        operation()
        after = datetime.now()
        # Time only advances when we say so
        assert before == after  # Unless frozen_time.advance() called

# ❌ BAD: Shared state (state flake)
_GLOBAL_CACHE = {}

def test_caching():
    _GLOBAL_CACHE["key"] = "value"
    # Leaks to other tests!

# ✅ GOOD: Isolated fixtures
@pytest.fixture
def isolated_cache():
    cache = {}
    yield cache
    cache.clear()  # Guaranteed cleanup

# ❌ BAD: Undeclared hardware dependency
def test_mlx():
    import mlx.core as mx  # Fails on Linux!
    mx.array([1, 2, 3])

# ✅ GOOD: Hardware requirement declared
@hardware_required(requires_apple_silicon=True)
def test_mlx():
    import mlx.core as mx
    mx.array([1, 2, 3])
```

### Hardened Fixture Reference

| Need               | Import From                        | Use                                      |
| ------------------ | ---------------------------------- | ---------------------------------------- |
| Deterministic time | `tests.utils.time_mocking`         | `freeze_time("2024-01-01")`              |
| Seeded RNG         | `tests.fixtures.isolated_fixtures` | `seeded_random["numpy"]`                 |
| Isolated env       | `tests.fixtures.isolated_fixtures` | `isolated_env["VAR"]`                    |
| Temp workspace     | `tests.fixtures.isolated_fixtures` | `temp_workspace / "file"`                |
| Resource budget    | `tests.utils.resource_mocking`     | `ResourceBudget(max_memory_mb=500)`      |
| Async control      | `tests.utils.async_determinism`    | `controlled_timeout()`                   |
| No network         | `tests.fixtures.isolated_fixtures` | `@pytest.mark.usefixtures("no_network")` |

### Running with Quarantine

```bash
# Record flake data
uv run pytest --flake-detection --flake-db=.flake_history.db

# Generate report
uv run python -m tests.ci.flake_report --output=report.html

# Auto-quarantine flaky tests
uv run python -m tests.ci.auto_quarantine --threshold=0.5

# Skip quarantined tests
uv run pytest --skip-quarantined
```

### See Also

- Testing guidelines: `docs/TESTING_GUIDELINES.md`

---

## Domain Expert Skills

Skill definitions live in `.claude/skills/` and provide domain-specific context for code review and implementation. When working on a particular area, load the relevant skill:

| Domain         | Skill              | Key Files                                                          |
| -------------- | ------------------ | ------------------------------------------------------------------ |
| Backend/Server | backend-expert     | `jarvis/socket_server.py`, `jarvis/prefetch/`, `jarvis/watcher.py` |
| LLM/Generation | ai-llm-expert      | `jarvis/reply_service.py`, `models/`, `jarvis/prompts/`            |
| Data/Search    | data-expert        | `jarvis/search/`, `jarvis/contacts/`, `jarvis/db/`                 |
| ML/Classifiers | ml-expert          | `jarvis/classifiers/`, `jarvis/features/`, `scripts/train_*`       |
| Frontend       | frontend-expert    | `desktop/src/**/*.svelte`, `desktop/src/**/*.ts`                   |
| Tauri/Rust     | tauri-expert       | `desktop/src-tauri/**/*.rs`                                        |
| Testing        | testing-expert     | `tests/`, `test_*.py`                                              |
| Performance    | performance-expert | `scripts/`, `models/`, any I/O or encoding code                    |
| Security       | security-expert    | `jarvis/socket_server.py`, `jarvis/db/`, `api/routers/`            |
| Docs           | docs-expert        | `docs/`, architectural changes                                     |
| Refactoring    | refactor-expert    | Any file with functions >50 lines or files >300 lines              |

**Cross-cutting:** Always apply performance-expert and security-expert alongside domain skills when relevant.
