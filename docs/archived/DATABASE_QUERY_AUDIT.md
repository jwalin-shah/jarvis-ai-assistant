# Database & Query Performance Audit (File-by-File)

Walkthrough of DB-touching code with bottlenecks identified and fixes applied.

---

## 1. `jarvis/db/core.py`

**Summary:** Connection management and schema init.

- **Connections:** Thread-local reuse, semaphore cap (20), WAL + 8MB cache. Good.
- **Caches:** TTL caches for contacts, stats, trigger patterns. Good.
- **Queries:** Migrations and schema only. No N+1.

**Verdict:** No changes needed.

---

## 2. `jarvis/db/contacts.py`

**Summary:** Contact CRUD.

- **UPSERT:** Single `INSERT ... ON CONFLICT DO UPDATE RETURNING id`. Good.
- **Batch:** `get_contact_by_handles(handles)` uses one `IN` query. Good.
- **Cache:** 30s TTL for get_contact / get_contact_by_chat_id. Good.
- **delete_contact:** Several DELETEs in one transaction; batching DELETEs is not a win. Fine.

**Verdict:** No changes needed.

---

## 3. `jarvis/topics/segment_storage.py`

**Summary:** Persist/retrieve conversation segments.

- **Issue (fixed):** `persist_segments` did one INSERT per segment to get `RETURNING id` / `lastrowid`, i.e. N round-trips.
- **Fix:** Single `executemany(INSERT, segment_rows)` then one `SELECT id FROM conversation_segments WHERE chat_id = ? ORDER BY id DESC LIMIT ?`; reverse to get IDs in insert order.
- **Other:** `link_vec_chunk_rowids`, `mark_facts_extracted`, `delete_segments_for_chat`, `get_segments_for_chat` already use batch fetch (segment IDs → one query for segment_messages). Good.

**Verdict:** Optimized.

---

## 4. `jarvis/topics/segment_pipeline.py`

**Summary:** Persist → index → extract facts.

- **Issue (fixed earlier):** Per-segment `INSERT OR IGNORE` into `segment_fact_fingerprints` + `SELECT changes()` in a loop (N+1).
- **Fix:** One query for existing fingerprints, build batch, single `executemany(INSERT OR IGNORE)`, compute “new” in memory.
- **Other:** Single transaction for persist + link; fact extraction uses batch extract + `save_facts` (batch). Good.

**Verdict:** Optimized.

---

## 5. `jarvis/contacts/fact_storage.py`

**Summary:** Persist facts and pipeline metrics.

- **save_facts:** Uses `executemany` for contact_facts; uses `total_changes()` once to count inserts. Good.
- **get_facts_for_contacts:** Chunked `IN` (900) for multiple contacts. Good.
- **log_pass1_claims:** `executemany` for pass1 log. Good.
- **Semantic deduper:** Singleton; embeddings reused for indexing. Good.

**Verdict:** No changes needed.

---

## 6. `jarvis/contacts/fact_index.py`

**Summary:** vec_facts index and semantic fact search.

- **index_facts:** Batch encode, one query for existing fact IDs, `executemany` for vec_facts. Good.
- **reindex_all_facts:** One query for all facts, batch encode, one `executemany`. Good.
- **search_relevant_facts:** One connection, count check → encode query → vector search → one `IN` for full fact rows. Good.

**Verdict:** No changes needed.

---

## 7. `jarvis/search/segment_ingest.py`

**Summary:** Ingest segments and extract facts from iMessage.

- **Issue (fixed):** In the conversation loop, each iteration did `SELECT MAX(end_time) FROM conversation_segments WHERE chat_id = ?` (N+1).
- **Fix:** Before the loop, one query: `SELECT chat_id, MAX(end_time) AS last_time FROM conversation_segments WHERE chat_id IN (...) GROUP BY chat_id`; in the loop use `last_processed_by_chat.get(conv.chat_id)`.
- **Other:** Contacts batch-loaded by `chat_id IN (...)`. Good.

**Verdict:** Optimized.

---

## 8. `jarvis/search/vec_search.py`

**Summary:** Vector chunks and search.

- **index_segments:** `executemany` for vec_chunks; then one SELECT for rowids (by key); `executemany` for vec_binary. Good.
- **search:** Single encode + one query. Good.
- **delete_chunks_for_chat:** One SELECT for rowids, batched DELETE for vec_binary, one DELETE for vec_chunks. Good.

**Verdict:** No changes needed.

---

## 9. `jarvis/search/hybrid_search.py`

**Summary:** BM25 + vector hybrid.

- **Metadata / cache:** Single query for count and max timestamp. Good.
- **\_enrich_results:** Chunks rowids (900), one `IN` query per chunk. Good.

**Verdict:** No changes needed.

---

## 10. `jarvis/watcher.py`

**Summary:** chat.db watcher and new-message handling.

- **\_get_new_messages:** Persistent read-only connection; one query for new messages. Good.
- **\_resegment_chats:** One chat at a time by design (per-chat lock); each does one `MAX(end_time)` and one `process_segments`. Acceptable.
- **\_validate_schema:** One-off at startup. Fine.

**Verdict:** No changes needed.

---

## 11. `jarvis/graph/knowledge_graph.py`

**Summary:** Build graph from contacts and facts.

- **build:** One query for contacts, one for facts; batch `add_nodes_from` / `add_edges_from`. Already optimized. Good.

**Verdict:** No changes needed.

---

## 12. `jarvis/graph/context.py`

**Summary:** Graph-based context for replies.

- **get_graph_context:** Calls \_get_fact_summary, \_get_interaction_recency, \_get_shared_connections; each does one or two focused queries. Fine.

**Verdict:** No changes needed.

---

## 13. `jarvis/graph/builder.py`

**Summary:** Graph builder and message stats.

- **\_get_contacts:** Single SELECT. Good.
- **\_get_message_stats:** Uses `_batch_get_messages(reader, chat_ids, ...)` (one batched query). Good.

**Verdict:** No changes needed.

---

## 14. `jarvis/contacts/contact_profile.py`

**Summary:** Profile build and DB facts.

- **\_fetch_db_facts:** One query per contact. Used during profile build (one contact at a time). Fine.
- **\_discover_topics:** One query for segments, then vec searcher. Fine.

**Verdict:** No changes needed.

---

## 15. `jarvis/tasks/worker.py`

**Summary:** Fact extraction task.

- **FACT_EXTRACTION task:** One query for `last_extracted_rowid`, one for `display_name`; then message fetch and windowed extraction; `save_facts` is batched; one UPDATE at end. No N+1 in DB. Good.

**Verdict:** No changes needed.

---

## 16. `jarvis/prefetch/predictor.py`

**Summary:** Prediction strategies (recency, time-of-day).

- **RecencyStrategy / TimeOfDayStrategy:** One query per strategy update (GROUP BY chat or similar). Fine.

**Verdict:** No changes needed.

---

## 17. Other files (brief)

- **jarvis/db/reply_logs.py:** Single-query patterns. Fine.
- **jarvis/db/stats.py:** Few queries, simple. Fine.
- **jarvis/db/migration.py, backup.py, reliability.py:** Admin/maintenance. Fine.
- **jarvis/infrastructure/cache/sqlite.py:** Single-key / batch delete. Fine.
- **jarvis/observability/metrics_router.py:** Batched inserts. Fine.
- **jarvis/classifiers/relationship_classifier.py:** Uses parameterized queries; no loop-per-row. Fine.

---

## Summary of code changes

| File                                | Change                                                                                                      |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `jarvis/topics/segment_storage.py`  | Replace N single-row INSERTs in `persist_segments` with one `executemany` + one SELECT for IDs.             |
| `jarvis/search/segment_ingest.py`   | Replace per-conversation `MAX(end_time)` in loop with one batched `GROUP BY chat_id` query before the loop. |
| `jarvis/topics/segment_pipeline.py` | (Earlier) Replace per-segment INSERT + changes() with batch fingerprint insert and in-memory “new” set.     |

---

## Checklist for future DB code

- [ ] No `db.query()` / `conn.execute()` inside a loop over items.
- [ ] Use CTEs/JOINs instead of correlated subqueries.
- [ ] Use `executemany` (or one multi-row INSERT) instead of looped INSERT/UPDATE.
- [ ] Filter in SQL (WHERE) instead of in Python.
- [ ] Batch “load by IDs” (e.g. attachments, facts) then map in memory.
