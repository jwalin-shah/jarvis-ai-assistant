# JARVIS Codebase Logic Review Report

## Executive Summary

This review identifies **15 logic issues** across the codebase ranging from critical bugs that break core functionality to subtle edge cases and code quality issues. The most severe issue is in the orchestrator circuit breaker logic which prevents the system from functioning.

---

## 🔴 Critical Issues (System-Breaking)

### 1. **Circuit Breaker Logic Inverted** (CRITICAL)
**File:** `jarvis/agents/orchestrator.py` (line 57)

```python
# Step 1: Check circuit breakers
if self.circuit_breakers["rag"] or self.circuit_breakers["classifier"]:
    return AgentResponse(
        intent="error",
        confidence=0.0,
        response="Circuit breaker triggered: system temporarily unavailable",
        context=None
    )
```

**Problem:** Circuit breakers are initialized as `True` (enabled) in line 48-52:
```python
self.circuit_breakers = {
    "rag": True,
    "classifier": True,
    "mlx": True
}
```

The condition checks if they are `True` and immediately returns an error. This means:
- The system **never processes any queries**
- The orchestrator is completely non-functional

**Fix:** Circuit breakers should be `False` when enabled/closed (allowing traffic), and set to `True` when open (blocking traffic). Invert the logic:
```python
# Circuit breakers: False = closed (working), True = open (tripped)
self.circuit_breakers = {
    "rag": False,
    "classifier": False,
    "mlx": False
}

if self.circuit_breakers["rag"] or self.circuit_breakers["classifier"]:
    # Return error when tripped
```

---

### 2. **Unreachable Code After Return**
**File:** `jarvis/agents/confidence.py` (lines 218-228)

```python
def _self_consistency_confidence(self, query: str, response: str) -> float:
    try:
        response_length = len(response.split())

        # Check response length appropriateness
        if response_length < 3:
            return 0.0  # ← Exits here
        elif response_length > 100:
            return 0.6  # ← Or exits here
        else:
            return 0.8  # ← Or exits here

        # THIS CODE IS NEVER REACHED:
        if "but" in response.lower() and "however" in response.lower():
            return 0.6  # Potential contradiction

        return 0.7  # Never reached
```

**Problem:** The `if/elif/else` block covers all cases and always returns. Lines 225-228 are dead code.

**Fix:** Reorder the logic:
```python
def _self_consistency_confidence(self, query: str, response: str) -> float:
    response_length = len(response.split())
    
    # Check for contradictions first
    if "but" in response.lower() and "however" in response.lower():
        return 0.6
    
    if response_length < 3:
        return 0.0
    elif response_length > 100:
        return 0.6
    else:
        return 0.8
```

---

## 🟠 High Severity Issues

### 3. **Variable May Be Unreferenced**
**File:** `jarvis/reply_service.py` (lines 393-416)

```python
if classification is None:
    # ... create default classification ...
    category_name = "statement"  # ← Set here

category_name_val: str = classification.metadata.get("category_name", "")
if not category_name_val:
    category_name_val = getattr(
        classification.category, "value", classification.category
    )
category_name = str(category_name_val)  # ← Always overwrites
```

**Problem:** If `classification is None`, `category_name` is set to `"statement"` but then immediately overwritten on line 416. Also, `classification` could be used before assignment in line 411 if the code flow changes.

**Fix:** Use a clearer control flow:
```python
if classification is None:
    classification = ClassificationResult(...)

category_name_val = classification.metadata.get("category_name", "")
if not category_name_val:
    category_name_val = getattr(classification.category, "value", classification.category)
category_name = str(category_name_val)
```

---

### 4. **Migration Logic Runs Multiple Times**
**File:** `jarvis/db/core.py` (lines 370-376)

```python
for max_version, exact_only, migrate_fn in _MIGRATIONS:
    if exact_only:
        should_run = current_version == max_version
    else:
        should_run = current_version <= max_version  # ← PROBLEM
    if should_run:
        migrate_fn(conn)
```

**Problem:** With `exact_only=False`, if `current_version=10` and we have migrations for versions 15, 16, 17, ALL would run because `10 <= 15`, `10 <= 16`, `10 <= 17`. This could apply migrations multiple times or out of order.

**Fix:** The condition should be:
```python
should_run = current_version < max_version  # Run only if DB is older
```

---

### 5. **Empty IN Clause Returns Invalid SQL**
**File:** `jarvis/db/query_builder.py` (lines 33-34)

```python
@staticmethod
def in_clause(values: list[Any]) -> tuple[str, list[Any]]:
    if not values:
        return "NULL", []  # ← "NULL" without quotes is invalid
```

**Problem:** When values is empty, it returns `"NULL"` as the placeholder. Using this in a query:
```sql
SELECT * FROM t WHERE id IN (NULL)
```

This is syntactically valid but semantically wrong - `IN (NULL)` never matches anything (NULL comparisons are NULL, not True).

**Fix:** Either raise an exception or return a condition that matches nothing properly:
```python
if not values:
    return "NULL", []  # Document: caller must handle empty case
    # OR raise ValueError("Empty values for IN clause")
    # OR return "1=0", []  # False condition
```

---

## 🟡 Medium Severity Issues

### 6. **Duplicate Strip Check**
**File:** `jarvis/reply_service.py` (line 387)

```python
incoming = context.message_text.strip()
if not incoming or not incoming.strip():  # ← Second strip is redundant
    return self._empty_message_response()
```

**Problem:** `incoming` is already stripped on line 386. The second `.strip()` is redundant.

**Fix:**
```python
incoming = context.message_text.strip()
if not incoming:
    return self._empty_message_response()
```

---

### 7. **Unused Variable in Contact Scores**
**File:** `jarvis/prefetch/predictor.py` (lines 249-264)

```python
max_count = 1  # Avoid division by zero

for row in cursor.fetchall():
    chat_id = row["chat_id"]
    msg_count = row["msg_count"]
    max_count = max(max_count, msg_count)  # ← Updates but...
    
    # Score calculation that doesn't use max_count properly:
    self._contact_scores[chat_id] = (msg_count / max_count) * 50 + recency_boost * 50
```

**Problem:** `max_count` is updated inside the loop, but used immediately in the same iteration. This means the first iteration always divides by 1, and subsequent iterations use the current row's count as max, not the true maximum.

**Fix:** Two-pass approach or track max first:
```python
rows = cursor.fetchall()
max_count = max((row["msg_count"] for row in rows), default=1)

for row in rows:
    chat_id = row["chat_id"]
    msg_count = row["msg_count"]
    # Now max_count is stable
    self._contact_scores[chat_id] = (msg_count / max_count) * 50 + recency_boost * 50
```

---

### 8. **Query Result Order Lost After Enrichment**
**File:** `jarvis/search/hybrid_search.py` (lines 303-367)

```python
# 4. Sort by fused score
sorted_rowids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:limit]

# 5. Fetch full metadata...
enriched = self._enrich_results([rid for rid, _ in sorted_rowids])

def _enrich_results(self, rowids: list[int]) -> list[dict[str, Any]]:
    # ... builds a dict then returns in original rowid order
    row_map = {r["rowid"]: dict(r) for r in all_rows}
    results = []
    for rid in rowids:
        if rid in row_map:
            results.append(row_map[rid])
    return results
```

**Problem:** The enrichment maintains the order of `rowids` passed in, which is correct. However, the actual sorting by `fused_score` happens after enrichment on lines 319-321:

```python
for item in enriched:
    item["fused_score"] = rrf_scores.get(item["rowid"], 0.0)
```

The scores are added but the list is NOT re-sorted. The results may not be in the correct RRF order.

**Fix:** Sort after enrichment:
```python
for item in enriched:
    item["fused_score"] = rrf_scores.get(item["rowid"], 0.0)

# Re-sort by fused_score
enriched.sort(key=lambda x: x["fused_score"], reverse=True)
```

---

### 9. **ModelManager Singleton Pattern Issue**
**File:** `jarvis/model_manager.py` (lines 27-38)

```python
def __new__(cls) -> ModelManager:
    with cls._lock:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False  # ← Set in __new__
        return cls._instance

def __init__(self) -> None:
    if getattr(self, "_initialized", False):  # ← Checks in __init__
        return
    self._active_type: ModelType | None = None
    self._initialized = True
```

**Problem:** `_initialized` is set in `__new__` but `__init__` may run multiple times. The check uses `getattr` with default `False`, but `_initialized` was already set to `False` in `__new__`. This is technically correct but fragile.

**Risk:** If someone subclasses or modifies, the double-initialization protection may fail.

**Fix:** Use a more robust singleton pattern or clearer initialization:
```python
def __new__(cls) -> ModelManager:
    if cls._instance is None:
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._active_type = None
                cls._instance = instance
    return cls._instance

def __init__(self) -> None:
    pass  # Already initialized in __new__
```

---

### 10. **Division by Zero Risk in Frequency Calculation**
**File:** `jarvis/prefetch/predictor.py` (lines 97-105)

```python
@property
def frequency(self) -> float:
    if len(self.access_times) < 2:
        return 0.0
    duration = self.access_times[-1] - self.access_times[0]
    if duration < 1:
        return 0.0  # ← Returns 0 for any duration < 1 second
    return (len(self.access_times) - 1) / (duration / 3600)
```

**Problem:** If `duration` is between 0 and 1, it returns 0.0 even though there were valid accesses. A rapid double-click would show frequency=0.

**Fix:** Use a minimum threshold or handle sub-second durations:
```python
if duration < 0.001:  # 1ms minimum
    return 0.0
# Or calculate frequency properly for short durations
```

---

## 🟢 Low Severity / Code Quality Issues

### 11. **Semantic Confidence Calculation Logic Issue**
**File:** `jarvis/agents/confidence.py` (lines 115-144)

```python
def _semantic_confidence(self, query: str, response: str, context: Optional[str]) -> float:
    query_lower = query.lower()
    response_lower = response.lower()

    query_words = set(query_lower.split())
    response_words = set(response_lower.split())

    if len(query_words) == 0:
        return 0.0

    overlap = len(query_words.intersection(response_words))
    raw_similarity = overlap / len(query_words)  # ← Asymmetric similarity
```

**Problem:** The similarity is calculated as `|query ∩ response| / |query|`. This means a response containing all query words PLUS many others would have similarity=1.0, which is wrong for response quality.

**Fix:** Use Jaccard similarity or proper containment measure:
```python
# Jaccard similarity
union = len(query_words | response_words)
jaccard = overlap / union if union > 0 else 0.0
```

---

### 12. **Response Contains Both "but" and "however" Check Is Weak**
**File:** `jarvis/agents/confidence.py` (line 226)

```python
if "but" in response.lower() and "however" in response.lower():
    return 0.6  # Potential contradiction
```

**Problem:** This would flag valid sentences like:
- "I like it but however you feel is fine"
- "Nothing but however hard I try..."

**Fix:** Use more sophisticated contradiction detection or NLI model.

---

### 13. **Cache Directory Not Thread-Safe**
**File:** `jarvis/search/hybrid_search.py` (lines 176-189)

```python
def _save_index_to_cache(self, metadata: dict[str, Any]) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_BM25_CACHE_FILE, "wb") as f:
            pickle.dump({...}, f)
    except Exception as e:
        logger.debug("Failed to save BM25 cache: %s", e)
```

**Problem:** No file locking. Concurrent processes could corrupt the cache file.

**Fix:** Use atomic writes or file locking:
```python
def _save_index_to_cache(self, metadata: dict[str, Any]) -> None:
    import tempfile
    import os
    
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temp_file = _BM25_CACHE_FILE.with_suffix('.tmp')
    
    with open(temp_file, "wb") as f:
        pickle.dump({...}, f)
    
    os.replace(temp_file, _BM25_CACHE_FILE)  # Atomic on POSIX
```

---

### 14. **get_mlx_memory_info Returns Inconsistent Types**
**File:** `jarvis/utils/memory.py` (lines 48-98)

```python
@lru_cache(maxsize=1)
def get_mlx_memory_info() -> dict[str, Any] | None:
    if not _is_mlx_available():
        return None  # ← Returns None
    # ... returns dict with specific keys
```

**Problem:** The function returns `None` when MLX unavailable, but a dict otherwise. Callers must always check for None.

**Fix:** Return an empty dict or sentinel value instead of None for consistency.

---

### 15. **Thermal State Default May Hide Issues**
**File:** `jarvis/utils/memory.py` (lines 322-323)

```python
except Exception:
    return "nominal"  # Assume nominal if we can't read it
```

**Problem:** Silently returning "nominal" when thermal state can't be read may hide system issues. The caller thinks everything is fine when it might not be.

**Fix:** Return `None` or `"unknown"` to indicate the reading failed:
```python
except Exception:
    logger.debug("Could not read thermal state")
    return None
```

---

## Summary Table

| Issue | File | Line | Severity | Category |
|-------|------|------|----------|----------|
| Circuit breaker inverted | `orchestrator.py` | 57 | 🔴 Critical | Logic Bug |
| Unreachable code | `confidence.py` | 225-228 | 🔴 Critical | Dead Code |
| Variable unreferenced | `reply_service.py` | 393-416 | 🟠 High | Control Flow |
| Migration runs multiple times | `db/core.py` | 370-376 | 🟠 High | Data Integrity |
| Empty IN clause SQL | `query_builder.py` | 33-34 | 🟠 High | SQL Generation |
| Duplicate strip | `reply_service.py` | 387 | 🟡 Medium | Redundancy |
| Contact scores calculation | `predictor.py` | 249-264 | 🟡 Medium | Algorithm |
| Result order not preserved | `hybrid_search.py` | 303-367 | 🟡 Medium | Search Ranking |
| Singleton pattern fragile | `model_manager.py` | 27-38 | 🟡 Medium | Concurrency |
| Division by zero risk | `predictor.py` | 97-105 | 🟡 Medium | Edge Case |
| Asymmetric similarity | `confidence.py` | 130 | 🟢 Low | Algorithm |
| Weak contradiction check | `confidence.py` | 226 | 🟢 Low | NLP |
| Cache not thread-safe | `hybrid_search.py` | 176-189 | 🟢 Low | Concurrency |
| Inconsistent return types | `memory.py` | 48-98 | 🟢 Low | API Design |
| Silent failure on thermal | `memory.py` | 322-323 | 🟢 Low | Error Handling |

---

## Recommendations

### Immediate Actions
1. **Fix the circuit breaker logic** - This breaks the entire orchestrator
2. **Fix unreachable code** in confidence calculation
3. **Review migration logic** to ensure data integrity

### Short-term
4. Add proper ordering to hybrid search results
5. Fix contact score calculation to use true max
6. Add thread safety to cache operations

### Long-term
7. Refactor semantic confidence to use proper metrics
8. Improve error handling (don't silently return defaults)
9. Add comprehensive unit tests for edge cases

---

*Review Date: 2026-02-20*
*Files Reviewed: 15+ core modules*
*Total Lines Analyzed: ~4,500*
