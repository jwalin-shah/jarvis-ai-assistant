# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

JARVIS is a local-first AI assistant for macOS providing intelligent iMessage management using MLX-based language models on Apple Silicon.

**Default Model**: lfm-1.2b-ft (fine-tuned for iMessage)
**Extraction Model**: lfm-0.7b (V4 instruction-based; models/lfm-0.7b-4bit)

**Documentation:**

- `docs/HOW_IT_WORKS.md` - How JARVIS works end-to-end (start here)
- `docs/ARCHITECTURE.md` - Technical implementation status
- `docs/design/V2_ARCHITECTURE.md` - Unix socket + direct SQLite optimizations
- `docs/design/PIPELINE.md` - Classification & routing pipeline
- `docs/SCHEMA.md` - Database schema

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

- **Always parallelize**: `n_jobs=1` for large datasets (>100MB), `n_jobs=2` only for small datasets - constrained by 8GB RAM
- **Always batch**: Process lists together, not one-at-a-time loops
- **Always cache**: Expensive computations (embeddings, model loads)
- **Vectorized ops**: NumPy/pandas over Python loops
- A slow script is NOT acceptable - performance is a correctness requirement

#### Visible Progress (MANDATORY)

**ALL scripts/operations MUST have visible, real-time progress.** Zero visibility is unacceptable.

- **Python stdout buffering**: Use `print(..., flush=True)` or `python -u` (unbuffered mode)
- **Long operations (>30s)** require ALL of:
  1. Progress indicator (bar, counter, percentage)
  2. ETA or time estimate
  3. Current step description (what's happening NOW)
  4. Log to file in real-time (use `logging` with `FileHandler`)
- **Training/GridSearch**: Always use `verbose=2` or higher AND log to file
- **Background processes**: Provide a way to check status without killing/restarting
- **macOS memory tracking**: Use `jarvis/utils/memory.py` to show real pressure, not just swap

Examples of REQUIRED visibility:

```python
# GridSearchCV
search = GridSearchCV(..., verbose=2)  # Shows per-fold progress
logging.basicConfig(handlers=[FileHandler("progress.log"), StreamHandler()])

# Long loops
for i, item in enumerate(items):
    print(f"Processing {i+1}/{len(items)}: {item.name}", flush=True)

# Data processing
with tqdm(total=len(data), desc="Encoding") as pbar:
    for batch in batches:
        process(batch)
        pbar.update(len(batch))
```

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
  - Use `mx.set_memory_limit()` and `mx.set_cache_limit()` (not deprecated `mx.metal.*` variants)
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

### Writing Tests (CRITICAL)

- **NEVER write tests that mirror the implementation** - tests must verify _behavior_, not parrot code structure
- Tests should be written from the _spec_, not from reading the code. If you read the code first, you'll just restate it in test form.
- A good test fails when behavior breaks. A bad test fails when you refactor internals.
- **Test real scenarios**: use actual inputs/outputs, not mocked-to-the-gills stubs that just confirm "the mock was called"
- **Integration > unit for new modules**: if a module talks to DB or embeddings, write at least one integration test with a real (in-memory) DB
- **Never assert on mocks alone**: `mock.assert_called_once()` proves nothing about correctness. Assert on _output_.
- Ask: "If I replaced the implementation with something totally different but correct, would this test still pass?" If no, the test is too coupled.

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
- **Prompts**: All in `jarvis/prompts/` - nowhere else
- **Errors**: Inherit from `JarvisError` in `jarvis/errors.py`
- **Batching**: Use `classify_batch()`, `embedder.encode(list)` - never loop
- **Parallelization**: `n_jobs=1` for large datasets (>100MB), `n_jobs=2` only for small datasets (memory-constrained)
- **IPC Protocol**: Use binary encoding (base64) not JSON for large arrays

### CRITICAL: N+1 Query Anti-Pattern (NEVER DO THIS)

**N+1 queries are a systemic performance killer.** With 400k messages in iMessage DB, naive code patterns became 1400ms delays. This MUST be caught in code review.

#### What is N+1?

```python
# BAD: N+1 pattern - 1 query + N subqueries
messages = db.query("SELECT * FROM message LIMIT 100")  # 1 query
for msg in messages:
    attachments = db.query(f"SELECT * FROM attachment WHERE message_id = {msg.id}")  # 100 queries
    # Total: 101 queries instead of 1
```

#### The Same Pattern in Different Forms

**SQL Correlated Subqueries** (scales terribly):

```sql
-- BAD: Subquery runs for EVERY chat
SELECT chat.id,
  (SELECT COUNT(*) FROM message WHERE chat_id = chat.id),  -- N subqueries
  (SELECT MAX(date) FROM message WHERE chat_id = chat.id),  -- N subqueries
  (SELECT text FROM message WHERE chat_id = chat.id ORDER BY date DESC LIMIT 1)  -- N subqueries
FROM chat
```

**With 400k messages across 1000 chats: 3000+ subquery executions**

**TypeScript/Python Loops**:

```typescript
// BAD: N+1 loop
const messages = await db.query('SELECT * FROM message LIMIT 100');
for (const msg of messages) {
  const attachments = await db.query(`...WHERE message_id = ${msg.id}`); // 100 queries
  const reactions = await db.query(`...WHERE message_id = ${msg.id}`); // 100 queries
  // Total: 201 queries instead of 3
}
```

**Individual INSERTs**:

```python
# BAD: N individual INSERTs
for fact in facts:
    db.execute("INSERT INTO fact VALUES (...)", ...)  # 50 inserts instead of 1
```

#### The Fix: Always Batch

**SQL: Use CTEs + JOINs instead of subqueries**:

```sql
-- GOOD: Single pass
WITH chat_stats AS (
  SELECT chat_id, COUNT(*) as msg_count, MAX(date) as last_date
  FROM message
  GROUP BY chat_id
)
SELECT chat.*, chat_stats.msg_count, chat_stats.last_date
FROM chat
JOIN chat_stats ON chat.id = chat_stats.chat_id
```

**TypeScript/Python: Batch fetch then map**:

```typescript
// GOOD: 3 queries instead of 201
const messages = await db.query('SELECT * FROM message LIMIT 100');
const messageIds = messages.map((m) => m.id);
const attachmentsMap = await getAttachmentsByIds(messageIds); // 1 query
const reactionsMap = await getReactionsByIds(messageIds); // 1 query

for (const msg of messages) {
  msg.attachments = attachmentsMap.get(msg.id); // O(1) lookup
  msg.reactions = reactionsMap.get(msg.id); // O(1) lookup
}
```

**Python INSERTs: Use executemany()**:

```python
# GOOD: 1 batch insert instead of 50
batch_data = [(field1, field2, ...) for fact in facts]
db.executemany("INSERT INTO fact VALUES (?, ?, ...)", batch_data)
```

#### Code Review Checklist

Before merging ANY code that touches data access:

- [ ] **Loops that query database?** → Flag as potential N+1
- [ ] **Subqueries in SELECT without GROUP BY aggregation?** → Likely correlated subquery
- [ ] **Individual INSERTs/UPDATEs in loop?** → Use batch operations
- [ ] **Multiple queries for same entity?** → Join or prefetch
- [ ] **Post-query filtering in code?** → Push into SQL WHERE clause
- [ ] **Timing: Does operation complete in <100ms?** → If not, profile it

#### Real Examples Fixed in This Codebase

| Pattern                       | File                 | Before                         | After            | Impact      |
| ----------------------------- | -------------------- | ------------------------------ | ---------------- | ----------- |
| Message attachments/reactions | `direct.ts`          | 201 queries                    | 3 queries        | 30x faster  |
| Fact extraction INSERTs       | `fact_storage.py`    | 50 INSERTs                     | 1 batch          | 50x faster  |
| Last message text             | `queries.sql`        | Correlated subquery            | CTE JOIN         | 100x faster |
| Search filtering              | `semantic_search.py` | Fetch 1000, filter 200 in code | Fetch 200 in SQL | 5x faster   |
| Graph building                | `knowledge_graph.py` | 1100 add_node() calls          | 3 batch calls    | 6x faster   |

#### Performance Testing Rule

ANY code touching database or large data structures must include a performance test:

```python
def test_get_conversations_performance():
    """Verify conversations load in <100ms with 400k messages."""
    start = time.time()
    convos = get_conversations(limit=50)
    elapsed_ms = (time.time() - start) * 1000

    assert elapsed_ms < 100, f"Too slow: {elapsed_ms:.1f}ms (should be <100ms)"
    assert len(convos) <= 50
```

**This is MANDATORY for**:

- Database queries (SELECT, JOIN, WHERE)
- Loops over large datasets
- Batch operations (INSERT, UPDATE)
- Graph building, embeddings, ML inference

**Rule: If you can't benchmark it, don't merge it.**

---

## Environment Variables & Secrets

**Critical:** The `.env` file contains API keys and secrets. It is:
- Auto-ignored in `.gitignore` (line 81)
- Listed in `.gitignore` as: `.env` and `.env.*`
- **NEVER commit to git**

**Required for DSPy optimization and evals:**
```bash
# Cerebras API (for LLM judge)
CEREBRAS_API_KEY=<your_api_key>

# Already configured in .env - do not modify unless rotating keys
```

**If you need to add new secrets:**
1. Add to `.env` file only
2. Add template to `.env.example` (with placeholder values)
3. Verify `.gitignore` includes `.env`

---

## Skill Auto-Load Rules

Skills in `.claude/skills/` provide domain expertise. Load the right skill based on the files you're touching:

| File Pattern                                                        | Skill              | When                                       |
| ------------------------------------------------------------------- | ------------------ | ------------------------------------------ |
| `jarvis/socket_server.py`, `jarvis/prefetch/`, `jarvis/watcher.py`  | backend-expert     | Server, IPC, background tasks              |
| `jarvis/reply_service.py`, `models/`, `jarvis/prompts/`             | ai-llm-expert      | LLM generation, prompts, inference         |
| `jarvis/search/`, `jarvis/contacts/`, `jarvis/graph/`, `jarvis/db/` | data-expert        | Embeddings, RAG, knowledge graph, DB       |
| `jarvis/classifiers/`, `jarvis/features/`, `scripts/train_*`        | ml-expert          | Classifiers, training, feature engineering |
| `desktop/src/**/*.svelte`, `desktop/src/**/*.ts`                    | frontend-expert    | Svelte components, TypeScript, stores      |
| `desktop/src-tauri/**/*.rs`                                         | tauri-expert       | Rust backend, Tauri IPC, permissions       |
| `tests/`, `test_*.py`                                               | testing-expert     | Test patterns, fixtures, mocking           |
| `scripts/`, `jarvis/prefetch/`, `models/` (I/O, encoding)           | performance-expert | Memory, batching, caching, profiling       |
| `jarvis/socket_server.py`, `jarvis/db/`, `api/routers/`             | security-expert    | Injection, permissions, secrets            |
| `docs/` or architectural changes                                    | docs-expert        | Documentation sync                         |
| Any `.py`/`.ts` file with functions >50 lines                       | refactor-expert    | Code quality, dead code, complexity        |

**Cross-cutting rules:**

- Performance reviews: always load `performance-expert` alongside the domain skill
- Security-sensitive changes: always load `security-expert` alongside the domain skill
- When writing tests for new code: load `testing-expert` alongside the domain skill
