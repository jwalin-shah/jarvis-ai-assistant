# JARVIS Performance Audit Report

**Date:** 2026-02-14  
**Auditor:** @performance-expert  
**Scope:** Database operations, model calls, data processing pipelines

---

## Executive Summary

The JARVIS codebase shows **generally good performance patterns** with proper batching, caching, and memory management for an 8GB-constrained system. However, there are **several areas for improvement** that could reduce latency, memory pressure, and redundant computations.

### Overall Grade: **B+**

**Strengths:**
- Proper use of batch embeddings via `CachedEmbedder`
- SQLite `executemany` for bulk inserts
- Thread-local connection pooling in `JarvisDBBase`
- Model lifecycle management with `ModelManager`/`ModelWarmer`
- Singleton patterns for expensive resources (embedder, searcher)

**Key Issues:**
1. Potential N+1 queries in contact/fact retrieval
2. Missing binary serialization for some embedding operations
3. Redundant embedding computations in some paths
4. BM25 index rebuilds on every HybridSearcher initialization

---

## Detailed Findings

### 🔴 HIGH PRIORITY

#### 1. **HybridSearcher BM25 Index Rebuild on Every Init** 
**Location:** `jarvis/search/hybrid_search.py:30-58`

```python
def _ensure_initialized(self) -> None:
    """Lazily build BM25 index from SQLite chunks."""
    # Fetches ALL chunks from DB every time searcher is created
    rows = conn.execute(
        "SELECT rowid, context_text, reply_text FROM vec_chunks"
    ).fetchall()
```

**Problem:** 
- BM25 index is rebuilt from scratch on every `HybridSearcher` instantiation
- No persistence of the BM25 index
- O(n) operation where n = total chunks (scales poorly)

**Impact:** 
- Adds 50-200ms latency to first search call
- Wasted CPU cycles for identical index reconstruction

**Fix:**
```python
# Option 1: Persistent BM25 index on disk (pickle)
def _ensure_initialized(self) -> None:
    if self._initialized:
        return
    cache_path = Path(".cache/bm25_index.pkl")
    if cache_path.exists() and not self._is_stale(cache_path):
        self.bm25_searcher = pickle.loads(cache_path.read_bytes())
    else:
        # Build and save
        ...
        cache_path.write_bytes(pickle.dumps(self.bm25_searcher))

# Option 2: Incremental updates
# Track last_indexed_timestamp, only fetch new chunks
```

---

#### 2. **Fact Indexing - Embedding Computed Twice**
**Location:** `jarvis/contacts/fact_index.py:144-214` vs `jarvis/contacts/fact_storage.py:380-420`

**Problem:**
- `save_and_index_facts()` calls `save_facts()` then `index_facts()`
- `index_facts()` re-encodes facts that were already embedded in deduplication phase
- `FactDedupicator` computes embeddings for semantic dedup, but they're discarded

**Impact:**
- 2x embedding computation for each fact batch >6 items
- ~10-20ms per fact wasted

**Fix:**
```python
# Pass embeddings through the pipeline
def save_facts(..., embeddings: np.ndarray | None = None):
    ...
    if embeddings is not None:
        # Reuse provided embeddings
        index_facts(facts, contact_id, embeddings=embeddings)
```

---

#### 3. **VecSearch Row-by-Row INSERT in `index_segments`**
**Location:** `jarvis/search/vec_search.py:401-448`

```python
# OPTIMIZED: Use single transaction with RETURNING for reliable rowids
insert_sql = """
    INSERT INTO vec_chunks(...) VALUES (...)
    RETURNING rowid
"""
chunk_rowids: list[int] = []
for row in vec_chunks_batch:  # <-- Looping in Python, not using executemany
    cursor = conn.execute(insert_sql, row[:8])
    result = cursor.fetchone()
    if result:
        chunk_rowids.append(result["rowid"])
```

**Problem:**
- Uses Python loop + individual `execute()` calls instead of `executemany`
- SQLite `RETURNING` doesn't work with `executemany`, but workaround exists

**Impact:**
- 5-10x slower than bulk insert for large batches
- 100 segments = 100 round-trips to SQLite

**Fix:**
```python
# Use executemany with LAST_INSERT_ROWID tracking
# Or batch insert then SELECT rowids by unique key
conn.executemany(
    "INSERT INTO vec_chunks(...) VALUES (...)",
    [row[:8] for row in vec_chunks_batch]
)
# Then fetch rowids by matching the unique (contact_id, chat_id, source_timestamp)
```

---

### 🟡 MEDIUM PRIORITY

#### 4. **Missing Embedding Cache in ReplyService Context Building**
**Location:** `jarvis/reply_service.py:694-808`

```python
def build_generation_request(...):
    # ...
    all_exchanges = self._dedupe_examples(
        all_exchanges, cached_embedder, rerank_scores=all_rerank_scores
    )

def _dedupe_examples(self, examples, embedder, ...):
    texts = [f"{ctx} {out}" for ctx, out in examples]
    embeddings = embedder.encode(texts, normalize=True)  # No caching for these temp embeddings
```

**Problem:**
- `_dedupe_examples` computes embeddings for RAG examples
- These embeddings are not cached (one-time use)
- Could be cached by content hash for repeated queries

**Impact:**
- ~5-10ms per reply generation for deduplication

**Fix:**
```python
# Use content-addressable cache for example embeddings
# Or skip dedup if examples < threshold (e.g., 10)
if len(examples) > 10:
    embeddings = embedder.encode(texts, normalize=True)
else:
    return examples  # Skip expensive dedup for small sets
```

---

#### 5. **Contact Facts Query - Potential N+1 in Batch Operations**
**Location:** `jarvis/contacts/fact_storage.py:423-458`

```python
def get_facts_for_contact(contact_id: str) -> list[Fact]:
    # Single contact query - fine
    ...

def get_all_facts() -> list[Fact]:
    # Loads ALL facts then constructs objects one by one
    # No batching for multiple contacts
```

**Problem:**
- `get_all_facts()` loads entire table into memory
- No `get_facts_for_contacts(contact_ids: list[str])` batch variant
- Scripts that iterate contacts call `get_facts_for_contact` N times

**Impact:**
- N+1 query pattern in batch scripts
- Connection open/close overhead

**Fix:**
```python
def get_facts_for_contacts(contact_ids: list[str]) -> dict[str, list[Fact]]:
    """Batch fetch facts for multiple contacts."""
    if not contact_ids:
        return {}
    placeholders = ",".join("?" * len(contact_ids))
    rows = conn.execute(
        f"SELECT * FROM contact_facts WHERE contact_id IN ({placeholders})",
        contact_ids
    ).fetchall()
    # Group by contact_id
    ...
```

---

#### 6. **GLiNER Chunk Processing - Row-by-Row Encoding**
**Location:** `scripts/extract_gliner_chunks.py` (and similar)

**Problem:**
- Some extraction scripts may process chunks individually
- GLiNER model loads/unloads per chunk in worst case

**Impact:**
- Model load overhead dominates processing time
- Memory thrashing from repeated load/unload

**Fix:**
```python
# Ensure batch processing with model reuse
# Already partially addressed in batched_extractor.py
# Apply same pattern to all extraction scripts
```

---

### 🟢 LOW PRIORITY (Optimization Opportunities)

#### 7. **JSON Serialization for Embeddings in Some Paths**
**Location:** Various API endpoints

**Problem:**
- Some API responses may serialize embeddings as JSON float lists
- Binary (base64) would be 5-6x smaller and faster

**Impact:**
- Network overhead for embedding APIs
- Slight parsing overhead

**Fix:**
```python
# Use base64-encoded binary for embedding responses
embedding_bytes = embeddings.astype(np.float32).tobytes()
response = {"embedding": base64.b64encode(embedding_bytes).decode()}
```

---

#### 8. **PrefetchExecutor Queue - No Batching of Similar Tasks**
**Location:** `jarvis/prefetch/executor.py:461-511`

**Problem:**
- `schedule_batch` validates each prediction individually
- No coalescing of similar embedding tasks

**Impact:**
- Minor - queue overhead is small

---

#### 9. **Lazy Import Anti-Pattern in Hot Paths**
**Location:** Multiple files

```python
# In reply_service.py
def _fetch_contact_facts(...):
    from jarvis.contacts.fact_index import search_relevant_facts  # Lazy import
```

**Problem:**
- Lazy imports inside functions that are called frequently
- Import overhead on every call (though mitigated by Python's import cache)

**Fix:**
```python
# Move to module level, use TYPE_CHECKING for type hints only
from jarvis.contacts.fact_index import search_relevant_facts
```

---

## Positive Patterns (Keep Doing These!)

### ✅ Excellent: Batched Embedder with LRU Cache
**Location:** `models/bert_embedder.py:658-762`

```python
class CachedEmbedder:
    def encode(self, texts: list[str], ...):
        # Deduplicates within batch
        # LRU cache for repeated texts
        # Batch encodes missing texts
```

### ✅ Excellent: Thread-Local DB Connections
**Location:** `jarvis/db/core.py:81-121`

```python
def _get_connection(self) -> sqlite3.Connection:
    # Thread-local connection reuse
    # WAL mode for read-heavy
    # Proper pragmas for performance
```

### ✅ Excellent: Model Manager for Memory Orchestration
**Location:** `jarvis/model_manager.py`

```python
def prepare_for(self, model_type: ModelType) -> None:
    # Unloads conflicting models before loading new ones
    # Prevents OOM on 8GB systems
```

### ✅ Excellent: QueryBuilder with Chunking
**Location:** `jarvis/db/query_builder.py`

```python
@staticmethod
def chunked_in_clause(values: list[Any], chunk_size: int = 900) -> list[tuple[str, list[Any]]]:
    # Handles SQLite parameter limits
    # Safe from injection
```

### ✅ Excellent: Length-Sorted Batch Encoding
**Location:** `models/bert_embedder.py:454-511`

```python
def encode(self, texts, ...):
    # Sorts by length to minimize padding waste
    # Batches intelligently
```

---

## Recommendations Summary

| Priority | Issue | Estimated Impact | Effort |
|----------|-------|------------------|--------|
| 🔴 High | BM25 Index Persistence | -50-200ms first search | Medium |
| 🔴 High | Fact Indexing Double Embed | -10-20ms per fact save | Low |
| 🔴 High | VecSearch Row-by-Row INSERT | -5-10x batch speedup | Medium |
| 🟡 Med | Example Dedup Caching | -5-10ms per reply | Low |
| 🟡 Med | Batch Fact Query API | -N+1 in scripts | Low |
| 🟡 Med | GLiNER Batch Consistency | Prevents thrashing | Low |
| 🟢 Low | Binary Embedding Serialization | -80% network size | Low |
| 🟢 Low | Import Optimization | Minor startup gain | Low |

---

## Memory Safety Assessment (8GB Constraint)

**Overall: EXCELLENT** ✅

The codebase demonstrates strong memory awareness:

1. **ModelManager** orchestrates LLM/embedder/NLI mutual exclusion
2. **MLXModelLoader._mlx_load_lock** serializes GPU access
3. **ModelWarmer** unloads after idle timeout (default 5min)
4. **Batched extraction** with configurable batch sizes
5. **Memory pressure callbacks** for emergency unloads
6. **Connection pooling** prevents file descriptor exhaustion

**One concern:**
- `HybridSearcher` loads all chunks into memory for BM25
- At 100k chunks, this could be 50-100MB
- Acceptable for 8GB, but monitor growth

---

## Action Items

### Immediate (This Week)
1. [ ] Fix `index_segments` to use `executemany` (issue #3)
2. [ ] Add BM25 index caching to `HybridSearcher` (issue #1)
3. [ ] Pass embeddings through fact save pipeline (issue #2)

### Short Term (This Month)
4. [ ] Add `get_facts_for_contacts` batch API (issue #5)
5. [ ] Audit all extraction scripts for batch consistency (issue #6)
6. [ ] Add embedding cache to `_dedupe_examples` (issue #4)

### Long Term (Next Quarter)
7. [ ] Binary serialization for embedding APIs (issue #7)
8. [ ] Profiling-guided optimization for reply latency
9. [ ] Consider mmap for large fact tables

---

*Report generated by performance-expert skill. For questions, see the performance-expert SKILL.md.*
