# Performance Guide

> **Last Updated:** 2026-02-15

This document consolidates all performance-related guidelines, optimizations, and rules for the JARVIS codebase.

---

## Table of Contents

1. [Performance Rules & N+1 Prevention](#performance-rules--n1-prevention)
2. [Optimization Summary](#optimization-summary)
3. [Model Reuse & Memory Management](#model-reuse--memory-management)

---

## Performance Rules & N+1 Prevention

**Status**: Established after fixing critical N+1 queries that caused 1400ms startup delay.

### What Happened

Development built code without performance testing against realistic data (400k messages). Result: 5 critical performance bugs:

| Issue               | Impact          | Root Cause                   |
| ------------------- | --------------- | ---------------------------- |
| Conversations query | 1400ms startup  | 5 correlated subqueries      |
| Message loading     | 500ms per page  | N+1 on attachments/reactions |
| Fact extraction     | 150ms per batch | N individual INSERTs         |
| Search filtering    | 5x data wastage | Post-query filtering in code |
| Graph building      | 200ms startup   | Sequential add_node() calls  |

**Common pattern**: Doing per-item operations (query/insert/call) instead of batch operations.

### Prevention System

#### 1. Code Review Checklist (Required for All PRs)

```markdown
## Performance Checklist

- [ ] **Database queries**: No loops containing `db.query()` or `db.execute()`
- [ ] **Subqueries**: Uses CTEs/JOINs, not correlated subqueries
- [ ] **Batch operations**: Uses `executemany()` not loop with `execute()`
- [ ] **Filtering**: Filters in SQL WHERE clause, not Python loops
- [ ] **Benchmarks**: Performance-critical code has `<Xms` assertion
- [ ] **Monitoring**: Wrapped with `track_latency()` context manager
- [ ] **Tests**: `make test` passes, no new slow operations logged
```

**Anti-patterns to reject:**

```python
# ❌ REJECT: N+1 loop pattern
for item in items:
    result = db.query(f"SELECT ... WHERE id = {item.id}")  # N queries

# ✅ ACCEPT: Batch pattern
ids = [item.id for item in items]
results = db.query(f"SELECT ... WHERE id IN ({ids})")  # 1 query

# ❌ REJECT: Individual INSERTs
for fact in facts:
    db.execute("INSERT INTO fact VALUES (...)", ...)  # N INSERTs

# ✅ ACCEPT: Batch INSERT
db.executemany("INSERT INTO fact VALUES (?, ?, ...)", batch_data)  # 1 INSERT
```

#### 2. Performance Thresholds

Operations must complete in these times with realistic data (400k messages):

```python
LATENCY_THRESHOLDS = {
    "conversations_fetch": 100,      # 50 conversations
    "message_load": 100,             # 20 messages with attachments/reactions
    "fact_save": 50,                 # Batch insert of facts
    "search_filter": 100,            # Index search results
    "graph_build": 100,              # Build knowledge graph
    "socket_startup": 500,           # Socket ready for requests
    "db_query": 200,                 # Single SQL query
}
```

**Usage:**

```python
from jarvis.utils.latency_tracker import track_latency

with track_latency("conversations_fetch", limit=50):
    conversations = get_conversations(limit=50)
    # If >100ms, logs WARNING with "possible N+1 pattern"
```

#### 3. Key Takeaways

1. **N+1 queries scale exponentially with data**: 10 items → slow. 400k items → blocking UI.
2. **Batch operations are not optional**: They're correctness requirement at scale.
3. **Measure before optimizing**: Add timing before you have a slow operation.
4. **Catch at code review**: Prevent N+1 from merging, don't fix after deployment.
5. **Performance is a feature**: Slow startup/search/load is a bug, not "optimization later".

---

## Optimization Summary

### Recent Optimizations (Feb 2026)

#### 1. SQLite Performance Tuning ✅

**Files**: `jarvis/db/core.py`, `desktop/src/lib/db/direct.ts`

| Database | Cache | mmap | Notes |
|----------|-------|------|-------|
| chat.db | 64MB | 128MB | Messages (~200MB) |
| jarvis.db | 64MB | 512MB | Embeddings (~827MB) |

- WAL mode for concurrent reads
- Memory-mapped I/O for faster reads
- Temp tables in memory
- App-level message cache: 500 chats (~30k messages)

#### 2. Health Check Fast ✅

**File**: `jarvis/handlers/health.py`, `api/routers/health_readiness.py`

- Replaced slow `psutil` (~200-500ms) with native `vm_stat` command (~50ms)
- Added 5-second cache for health status
- Falls back to psutil if vm_stat fails

#### 3. Streaming Timeouts ✅

**Files**: `desktop/src/lib/components/SuggestionBar.svelte`, `desktop/src/lib/socket/stream-manager.ts`

- Increased frontend timeouts: 15s→45s stream, 12s→30s fallback
- Increased StreamManager idle: 60s→90s
- Added stale generation cancellation when switching chats

#### 4. Lazy Loading ✅

**File**: `desktop/src/lib/db/direct.ts`

- Skip attachment/reaction queries for "load more" (scroll-up)
- Only fetch full context for initial message load
- Reduces scroll-up time by ~50%

#### 5. Reduced Page Sizes ✅

**File**: `desktop/src/lib/stores/conversations.svelte.ts`

- Conversation page: 150→50
- Message page: 40→60
- Fewer rows per query = faster

#### 6. Generation Pipeline Optimization ✅

**Files**: `jarvis/config.py`, `jarvis/prompts/constants.py`, `evals/sweep_pipeline.py`

| Parameter | Previous | Optimized | Impact |
|-----------|----------|-----------|--------|
| Context Depth | 7-10 turns | 3 turns | **~25% better quality** on small models |
| Repetition Penalty | 1.05 | 1.1 | Reduced echoing/looping |
| Prompt Format | Plain Text | ChatML | Better instruction following for LFM |
| Logit Bias | None | AI-filtering | Reduced "As an AI..." filler |

- Discovered that small models (0.7B/1.2B) perform significantly better with **less context** (3 turns vs 10).
- Standardized on Liquid AI **ChatML** format for all generation.
- Implemented **MIPROv2** sweep to find winning sampling parameters (Top-P 0.9, Top-K 40, RP 1.1).

---

### Older Optimizations

#### 1. Batched Fact Extraction ✅

**File**: `jarvis/contacts/batched_extractor.py`

- Processes 5 segments per LLM call instead of 1
- 5x speedup on backfill operations
- Model kept warm during batch processing

#### 2. Optimized vec_chunks INSERT ✅

**File**: `jarvis/search/vec_search.py`

- Changed from individual INSERTs to single transaction with RETURNING
- Proper rollback on failure

#### 3. SQL Query Builder ✅

**File**: `jarvis/db/query_builder.py`

- Centralized safe SQL generation
- Automatic IN clause parameter limits (900 max)
- Eliminates scattered f-string SQL

#### 4. Single Transaction Pipeline ✅

**File**: `jarvis/topics/segment_pipeline.py`

- All DB operations in single transaction
- Atomic persist → index → link operations

### Performance Impact

| Optimization      | Before               | After                   | Improvement           |
| ----------------- | -------------------- | ----------------------- | --------------------- |
| Fact Extraction   | 1 segment/call       | 5 segments/call         | **5x faster**         |
| vec_chunks INSERT | Individual INSERTs   | Transaction + RETURNING | **~3x faster**        |
| DB Operations     | Multiple connections | Single transaction      | **Atomic + less I/O** |
| Model Memory      | 350M default         | 700M default            | **Better quality**    |

---

## Model Reuse & Memory Management

### Memory Budget

JARVIS operates within strict memory constraints on consumer hardware:

| Component            | Memory  | Notes                         |
| -------------------- | ------- | ----------------------------- |
| **Embedding Model**  | ~200MB  | BGE-small-en-v1.5 (int8)      |
| **Generation Model** | ~1.2GB  | LFM-2.5-1.2B-Instruct (4-bit) |
| **Extraction Model** | ~0.35GB | LFM-0.7B (4-bit)              |
| **System Overhead**  | ~500MB  | Python, SQLite, caches        |
| **Total Peak**       | ~2.2GB  | When both models loaded       |

**Target**: Stay under 3GB total to run on 8GB systems.

### Load/Unload Strategy

**Problem**: Loading models is expensive (2-15s cold start). Unloading too aggressively causes repeated load overhead.

**Solution**: Lazy loading with smart unloading based on memory pressure.

#### Embedding Model

- **Load**: On first search/indexing request
- **Keep**: Until generation model needed
- **Unload**: Before loading generation model (memory constraint)
- **Reload**: After generation completes (for next search)

#### Generation Model

- **Load**: On first draft/chat request
- **Keep**: For 5 minutes after last use (configurable)
- **Unload**: After idle timeout or memory pressure
- **Reload**: On next generation request

#### Extraction Model

- **Load**: During backfill operations
- **Keep**: For entire batch processing session
- **Unload**: After batch completes
- **Reload**: On next extraction request

### Implementation

**Model Manager** (`jarvis/model_manager.py`):

```python
class ModelManager:
    def __init__(self):
        self._embedding_model = None
        self._generation_model = None
        self._extraction_model = None
        self._last_gen_use = None
        self._idle_timeout = 300  # 5 minutes

    async def get_embedder(self):
        if self._embedding_model is None:
            # Unload generation if loaded
            if self._generation_model:
                await self._unload_generation()
            self._embedding_model = load_embedding_model()
        return self._embedding_model

    async def get_generator(self):
        if self._generation_model is None:
            # Unload embedding if loaded
            if self._embedding_model:
                await self._unload_embedding()
            self._generation_model = load_generation_model()
        self._last_gen_use = time.time()
        return self._generation_model
```

### Best Practices

1. **Batch operations**: Load model once, process all items, unload once
2. **Predictive loading**: Preload models when user hovers over conversation
3. **Memory monitoring**: Track RSS, unload proactively if approaching limit
4. **Graceful degradation**: Fall back to simpler models if memory constrained

### Metrics

Track model lifecycle events:

```python
from jarvis.observability.logging import log_event

log_event("model_load", model="lfm-1.2b", duration_ms=2340)
log_event("model_unload", model="bge-small", reason="memory_pressure")
```

---

## References

- `CLAUDE.md` - Core behavioral rules (includes N+1 section)
- `tests/performance_baseline.py` - Performance tests
- `jarvis/utils/latency_tracker.py` - Latency monitoring
- `jarvis/socket_server.py` - Integrated performance tracking
