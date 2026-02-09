# Performance Rules & N+1 Prevention Strategy

**Status**: Established after fixing critical N+1 queries that caused 1400ms startup delay.

## What Happened

Development built code without performance testing against realistic data (400k messages). Result: 5 critical performance bugs:

| Issue | Impact | Root Cause |
|-------|--------|-----------|
| Conversations query | 1400ms startup | 5 correlated subqueries |
| Message loading | 500ms per page | N+1 on attachments/reactions |
| Fact extraction | 150ms per batch | N individual INSERTs |
| Search filtering | 5x data wastage | Post-query filtering in code |
| Graph building | 200ms startup | Sequential add_node() calls |

**Common pattern**: Doing per-item operations (query/insert/call) instead of batch operations.

---

## Prevention System

### 1. Code Review (Required for All PRs)

**Mandatory checks before merge:**

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

**Specific anti-patterns to reject:**

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

# ❌ REJECT: Correlated subquery
SELECT chat.id,
  (SELECT COUNT(*) FROM message WHERE chat_id = chat.id),
  (SELECT MAX(date) FROM message WHERE chat_id = chat.id)
FROM chat

# ✅ ACCEPT: CTE join
WITH stats AS (
  SELECT chat_id, COUNT(*) as cnt, MAX(date) as last_date FROM message GROUP BY chat_id
)
SELECT chat.*, stats.cnt, stats.last_date FROM chat JOIN stats ...
```

---

### 2. Performance Thresholds (Enforced by Tests)

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

**Usage in code:**
```python
from jarvis.utils.latency_tracker import track_latency

with track_latency("conversations_fetch", limit=50):
    conversations = get_conversations(limit=50)
    # If >100ms, logs WARNING with "possible N+1 pattern"
```

**Automatic test for every PR:**
```bash
make test -k performance    # Runs performance_baseline.py tests
```

If a test fails:
```
AssertionError: getConversations too slow: 1200.5ms
(indicates N+1 query pattern)
```

---

### 3. Continuous Monitoring

**Socket server tracks all operations:**
```python
async def _list_conversations(self, limit: int = 50):
    with track_latency("conversations_fetch", limit=limit):
        # Automatically tracks timing, logs if slow
        conversations = reader.get_conversations(limit=limit)
    return conversations
```

**Dashboard metrics** (can be wired to monitoring system):
```python
tracker = get_tracker()
print(tracker.summary())
# {
#   "total_operations": 1247,
#   "slow_operations": 3,
#   "slow_ops_pct": 0.24,
#   "average_ms": 45.2,
#   "slow_operations_detail": [...]
# }
```

**Alerts on slow operations:**
```
[WARNING] [LATENCY] conversations_fetch took 1203.5ms (threshold: 100ms)
- possible N+1 pattern. Metadata: {'limit': 50}
```

---

### 4. Automated Detection

**Git pre-commit hook** (`.git/hooks/pre-commit`):
```bash
# Catches obvious N+1 patterns before commit
if git diff --cached | grep -E "for .* in .*:\n.*db\\.query\\(|for .* in .*:\n.*await db"; then
    echo "❌ Commit rejected: Found N+1 query pattern (loop with db.query)"
    exit 1
fi
```

**Run before submitting PR:**
```bash
make verify    # Runs lint + typecheck + test + performance checks
```

---

### 5. Performance Testing Standards

**Every performance-critical function must have a test:**

```python
def test_get_conversations_performance():
    """Performance baseline with realistic data (400k messages)."""
    start = time.perf_counter()
    convos = get_conversations(limit=50)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Hard assertion: MUST be fast
    assert elapsed_ms < 100, f"Too slow: {elapsed_ms:.1f}ms (threshold: 100ms)"
    assert len(convos) <= 50
    assert all(isinstance(c, Conversation) for c in convos)
```

**If test fails, investigate immediately:**
1. Profile the operation: Which query is slow?
2. Check for N+1 pattern: Loop with database calls?
3. Fix the root cause, not symptom
4. Re-test until <threshold

---

### 6. Code Quality Checklist

**Before marking code "ready for review":**

- [ ] All functions that query DB have performance tests
- [ ] All batch operations use proper API (executemany, add_nodes_from, etc.)
- [ ] No loops containing database operations
- [ ] Filters applied in SQL, not Python
- [ ] Correlated subqueries replaced with CTEs/JOINs
- [ ] `make test` passes without slow operation warnings
- [ ] `make lint` and `make format` pass
- [ ] Commit message explains performance impact

**Example commit message:**
```
Fix N+1 query on message attachments (30x speedup)

Problem: getMessages() looped over 100 messages, fetching attachments
individually = 201 queries, ~500ms delay.

Solution: Batch fetch all attachments with WHERE ... IN (...), build
Map for O(1) lookup. 201 queries → 3 queries, 500ms → 25ms.

Verified with latency_tracker showing <100ms for message loads.
All performance tests pass.
```

---

## Integration Checklist

To fully wire performance monitoring across the codebase:

- [ ] Import `track_latency` in all DB query functions
- [ ] Wrap queries in `with track_latency("operation_name")`
- [ ] Add performance tests for critical paths
- [ ] Update CI/CD to fail on `make test -k performance` failures
- [ ] Add performance metrics to socket server responses
- [ ] Wire latency tracker to metrics dashboard (if exists)
- [ ] Review slow operations weekly during stand-ups

---

## Key Takeaways

1. **N+1 queries scale exponentially with data**: 10 items → slow. 400k items → blocking UI.
2. **Batch operations are not optional**: They're correctness requirement at scale.
3. **Measure before optimizing**: Add timing before you have a slow operation.
4. **Catch at code review**: Prevent N+1 from merging, don't fix after deployment.
5. **Performance is a feature**: Slow startup/search/load is a bug, not "optimization later".

---

## References

- `CLAUDE.md` - Core behavioral rules (includes N+1 section)
- `tests/performance_baseline.py` - Performance tests catching 5 anti-patterns
- `jarvis/utils/latency_tracker.py` - Latency monitoring infrastructure
- `jarvis/socket_server.py` - Example of integrated performance tracking
- `desktop/src/lib/db/queries.ts` - Optimized CTE query pattern
