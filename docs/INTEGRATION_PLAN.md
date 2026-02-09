# Integration Plan: Wiring Latency & Memory Tracking

**Goal**: Every performance-critical operation reports latency automatically.

## Quick Integration (Add 2 Lines to Each Function)

### Backend (Python)

**Before:**
```python
def get_conversations(limit: int = 50) -> list[Conversation]:
    with ChatDBReader() as reader:
        return reader.get_conversations(limit=limit)
```

**After:**
```python
from jarvis.utils.latency_tracker import track_latency

def get_conversations(limit: int = 50) -> list[Conversation]:
    with track_latency("conversations_fetch", limit=limit):
        with ChatDBReader() as reader:
            return reader.get_conversations(limit=limit)
```

That's it. Now:
- Automatically measures execution time
- Logs warning if >100ms
- Tracks in global metrics
- Reports in JSON for dashboards

### Frontend (TypeScript)

**Before:**
```typescript
const messages = await getMessagesDirect(chatId, 20);
```

**After:**
```typescript
const start = performance.now();
const messages = await getMessagesDirect(chatId, 20);
const elapsed = performance.now() - start;
console.log(`[LATENCY] message_load took ${elapsed.toFixed(1)}ms`);
if (elapsed > 100) console.warn("Possible N+1 pattern detected");
```

Or use a helper:
```typescript
async function trackLatency(name: string, fn: () => Promise<any>) {
  const start = performance.now();
  const result = await fn();
  const elapsed = performance.now() - start;
  console.log(`[LATENCY] ${name} took ${elapsed.toFixed(1)}ms`);
  if (elapsed > THRESHOLDS[name]) console.warn(`${name} exceeded threshold`);
  return result;
}

// Usage:
const messages = await trackLatency("message_load", () =>
  getMessagesDirect(chatId, 20)
);
```

---

## Integration Checklist by Module

### 🔴 Critical Path (Do First)

**Backend:**
- [ ] `integrations/imessage/reader.py` - Wrap `get_conversations()`, `get_messages()`
- [ ] `jarvis/search/semantic_search.py` - Wrap search operations
- [ ] `jarvis/contacts/fact_storage.py` - Wrap `save_facts()`
- [ ] `jarvis/socket_server.py` - Wrap RPC handlers

**Frontend:**
- [ ] `desktop/src/lib/stores/conversations.ts` - Wrap fetch operations
- [ ] `desktop/src/lib/db/direct.ts` - Wrap `getMessages()`, `getConversations()`

### 🟡 Important (Phase 2)

**Backend:**
- [ ] `jarvis/reply_service.py` - Track generation latency
- [ ] `jarvis/prefetch/executor.py` - Track prefetch operations
- [ ] `jarvis/graph/knowledge_graph.py` - Track graph building

**Frontend:**
- [ ] `desktop/src/lib/components/MessageView.svelte` - Track render performance
- [ ] `desktop/src/lib/socket.ts` - Track socket operations

### 🟢 Nice-to-Have (Phase 3)

- [ ] Model loading (`models/loader.py`)
- [ ] Embedding generation (`models/bert_embedder.py`)
- [ ] Batch classification (`jarvis/classifiers/`)

---

## Specific Integration Examples

### Example 1: Fact Storage (Python)

**File**: `jarvis/contacts/fact_storage.py`

```python
from jarvis.utils.latency_tracker import track_latency

class FactStorage:
    def save_facts(self, facts: list[Fact]) -> int:
        """Save facts for a contact."""
        with track_latency("fact_save", contact_id=contact_id, count=len(facts)):
            # existing code here
            batch_data = [...]
            inserted = db.executemany("INSERT OR IGNORE INTO contact_facts ...", batch_data)
        return inserted
```

**What happens:**
- Operation completes: logs `[LATENCY] fact_save took 3.2ms (ok)`
- Operation slow (>50ms): logs `[WARNING] [LATENCY] fact_save took 150ms (threshold: 50ms) - possible N+1 pattern`

### Example 2: Message Loading (TypeScript)

**File**: `desktop/src/lib/db/direct.ts`

```typescript
export async function getMessages(chatId: string, limit: number = 100): Promise<Message[]> {
  const start = performance.now();

  try {
    const query = getMessagesQuery();
    const rows = await chatDb.select<MessageRow[]>(query, [chatId, limit]);

    // Batch fetch attachments and reactions
    const messageIds = rows.map(r => r.id);
    const guidIds = rows.map(r => r.guid);

    const attachmentsMap = await getAttachmentsForMessages(messageIds);
    const reactionsMap = await getReactionsForMessages(guidIds);

    const messages: Message[] = rows.map(row => ({
      ...row,
      attachments: attachmentsMap.get(row.id) || [],
      reactions: reactionsMap.get(row.guid) || [],
    }));

    const elapsed = performance.now() - start;
    console.log(`[LATENCY] message_load took ${elapsed.toFixed(1)}ms (${messages.length} messages)`);
    if (elapsed > 100) console.warn("Possible N+1 pattern in message loading");

    return messages;
  } catch (error) {
    const elapsed = performance.now() - start;
    console.error(`[LATENCY] message_load failed after ${elapsed.toFixed(1)}ms`, error);
    throw error;
  }
}
```

### Example 3: Socket Server (Python)

**File**: `jarvis/socket_server.py` (already has this pattern)

```python
async def _list_conversations(self, limit: int = 50) -> dict[str, Any]:
    """List conversations via socket."""
    start_time = time.time()

    try:
        with track_latency("socket_list_conversations", limit=limit):
            with ChatDBReader() as reader:
                conversations = reader.get_conversations(limit=limit)

        return {
            "conversations": [...],
            "total": len(conversations),
        }
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.exception(f"Error listing conversations (after {elapsed_ms:.1f}ms)")
        raise JsonRpcError(INTERNAL_ERROR, "Failed") from e
```

---

## Monitoring Dashboard Integration

### Expose Metrics Endpoint

**In socket_server.py:**
```python
async def _get_latency_metrics(self) -> dict[str, Any]:
    """Return latency metrics for monitoring dashboard."""
    from jarvis.utils.latency_tracker import get_tracker

    tracker = get_tracker()
    return tracker.summary()
```

**Then register:**
```python
self.register("get_latency_metrics", self._get_latency_metrics)
```

### Frontend Dashboard

```typescript
// In a monitoring component
const metrics = await jarvis.call("get_latency_metrics", {});
console.log("Performance Summary:", metrics);
// {
//   total_operations: 1247,
//   slow_operations: 3,
//   slow_ops_pct: 0.24,
//   average_ms: 45.2,
//   slow_operations_detail: [...]
// }
```

### CI/CD Integration

Add to `Makefile`:
```makefile
perf-check:
	@echo "Running performance tests..."
	@make test -k performance
	@echo "✓ All operations within thresholds"

verify: lint typecheck test perf-check
	@echo "✓ Full verification passed (lint, types, tests, perf)"
```

---

## Memory Tracking Integration

The system already has `jarvis/utils/memory.py` for memory pressure tracking.

**Wire it to critical operations:**

```python
from jarvis.utils.memory import get_memory_pressure

async def _list_conversations(self, limit: int = 50) -> dict:
    mem_before = get_memory_pressure()

    with track_latency("conversations_fetch"):
        conversations = reader.get_conversations(limit=limit)

    mem_after = get_memory_pressure()
    if mem_after > 75:  # >75% is concerning
        logger.warning(
            f"[MEMORY] conversations_fetch increased pressure from "
            f"{mem_before}% to {mem_after}%"
        )

    return {"conversations": [...]}
```

---

## Verification Steps

**After integrating latency tracking:**

1. **Run app normally:**
   ```bash
   make launch
   ```
   Watch logs for `[LATENCY]` messages

2. **Check for slow operations:**
   ```bash
   grep "\[LATENCY\]" <log_file> | grep -v "ok)"
   ```
   Should be empty (no slow operations)

3. **Run performance tests:**
   ```bash
   make test -k performance
   ```
   All should pass (all operations <threshold)

4. **Query metrics:**
   ```python
   from jarvis.utils.latency_tracker import get_tracker
   tracker = get_tracker()
   print(tracker.summary())
   ```
   Should show `slow_operations: 0`

---

## Rollout Timeline

**Week 1**: Critical path (messaging, conversations, searches)
**Week 2**: Important modules (reply service, prefetch)
**Week 3**: Nice-to-have (model loading, embeddings)
**Ongoing**: Monitor logs, alert on regressions

Each operation can be integrated in **<5 minutes** (2-3 lines of code).

---

## FAQ

**Q: Does this add overhead?**
A: Minimal. `time.perf_counter()` is ~1-2μs, negligible vs database queries (ms).

**Q: What if operation is intentionally slow?**
A: Update threshold in `LATENCY_THRESHOLDS` for that operation. Always document why.

**Q: Can I disable tracking in production?**
A: Yes, set `ENABLE_LATENCY_TRACKING=false` env var (not implemented yet, but easy to add).

**Q: How do I know if tracking is working?**
A: Look for `[LATENCY]` lines in logs. If none, tracking not wired up yet.

