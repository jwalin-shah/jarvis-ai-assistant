# Code Audit Report

Date: 2026-02-10  
Scope: `jarvis/`, `scripts/`, `models/` (with targeted DB query audit in `jarvis/db/` and `integrations/imessage/`)

## 1) Dead Code

| File Path | Line | Severity | Description | Suggested Fix |
|---|---:|---|---|---|
| `models/generator.py` | 386 | medium | `ThreadAwareGenerator` is defined but has no in-repo call sites (only declaration/docstring hits). | Remove it if deprecated, or wire it into runtime/API and add tests that instantiate/use it. |
| `jarvis/metrics.py` | 61 | low | `LatencyStats` dataclass appears unused in this module and across runtime code. | Delete it or refactor histogram code to use it explicitly; add tests if kept. |
| `jarvis/pdf_generator.py` | 553 | low | `generate_pdf_base64()` has no call sites. | Remove or expose through API/CLI route and add a unit test that exercises it. |
| `jarvis/prompts.py` | 1862 | low | `reset_optimized_programs()` has no references (including tests). | Remove dead reset hook or add tests that require this reset behavior. |
| `jarvis/retry.py` | 91 | medium | `retry_async_with_backoff()` has no references; sync variant is tested/used but async variant is orphaned. | Either adopt it in async call paths and add tests, or remove to reduce maintenance surface. |
| `models/registry.py` | 374 | low | `clear_availability_cache()` is defined but not used outside its own module exports. | Remove it if not part of intended public API, or call it from model-management flows/tests. |
| `models/registry.py` | 382 | low | `ensure_model_available()` has no in-repo callers besides re-export. | Integrate into settings/model-download flows or remove if out of scope. |
| `models/loader.py` | 31 | low | Unused imports: `TYPE_CHECKING`, `Protocol`. | Drop unused imports to reduce lint noise. |
| `scripts/eval_and_retrain_gemini.py` | 30 | low | Unused import: `precision_recall_fscore_support`. | Remove import or use metric in output/reporting path. |
| `scripts/train_mobilization_logistic.py` | 11 | low | Unused import: `json`. | Remove import. |
| `scripts/train_mobilization_lightgbm.py` | 11 | low | Unused import: `json`. | Remove import. |

## 2) N+1 Query Patterns

| File Path | Line | Severity | Description | Suggested Fix |
|---|---:|---|---|---|
| `integrations/imessage/reader.py` | 1341 | high | `search()` loops over rows and calls `_row_to_message()` per row. In this path, `_row_to_message()` falls back to per-message DB lookups for attachments/reactions/reply links, producing query-per-row behavior. | Replace loop with batched conversion (`_rows_to_messages`) or add a batched search-specific prefetch for attachments, reactions, and reply GUID mapping. |
| `integrations/imessage/reader.py` | 1660 | medium | `reply_to_guid` resolution does per-message `_get_message_rowid_by_guid()` calls. Cache helps only for repeated GUIDs, but typical GUIDs are unique, so this still scales poorly on large result sets. | Batch resolve reply GUIDs in one `IN (...)` query and join map in memory. |
| `jarvis/db/stats.py` | 87 | low | `get_gate_stats()` executes a separate `COUNT(*)` query per status in a loop. Small fixed loop, but still repetitive query pattern. | Use one grouped query (`SELECT validity_status, COUNT(*) ... GROUP BY validity_status`) and map defaults in Python. |

## 3) Duplicate Logic

| File Path | Line | Severity | Description | Suggested Fix |
|---|---:|---|---|---|
| `jarvis/_cli_main.py` | 58 | medium | `_format_jarvis_error()` logic is nearly duplicated in `jarvis/cli/utils.py:71`, risking drift in user-facing guidance. | Extract one shared formatter utility and call it from both entrypoints. |
| `scripts/prepare_gemini_training_data.py` | 89 | medium | `create_splits()` is duplicated with near-identical bodies in `scripts/prepare_gemini_training_with_embeddings.py:130` and `scripts/prepare_gemini_with_full_features.py:155`. | Move split/report helper to a shared script utility module and reuse. |
| `jarvis/prefetch/cache.py` | 472 | medium | L2 `_serialize/_deserialize` is duplicated in L3 implementation at `jarvis/prefetch/cache.py:742` and `:763`. | Extract serialization codec helpers (module-level or mixin) to one implementation. |
| `jarvis/prompts.py` | 470 | low | `CASUAL_INDICATORS` token set overlaps heavily with `jarvis/relationships.py:521` and `jarvis/contacts/contact_profile.py:177`. | Centralize tone/formality lexicons in one constants module. |
| `jarvis/contacts/contact_profile.py` | 323 | low | Large stopword-like token sets are replicated in `jarvis/search/embeddings.py:192` and `jarvis/topics/topic_segmenter.py:983`. | Create shared stopword/token-filter constants to avoid divergence and repeated maintenance. |

## 4) Memory Leaks / Retention Risks

| File Path | Line | Severity | Description | Suggested Fix |
|---|---:|---|---|---|
| `jarvis/search/embeddings.py` | 387 | medium | `_profile_cache` is an unbounded dict on a long-lived store singleton. TTL checks happen on read but stale keys are not actively evicted; cardinality can grow with contact churn. | Add max-size LRU behavior and periodic stale-key sweep; evict on insert when over cap. |
| `jarvis/scheduler/timing.py` | 59 | medium | `_interaction_cache` stores full interaction lists per contact in singleton analyzer with no TTL/size cap. Memory can grow with number of contacts/history depth. | Store aggregates instead of raw histories, or enforce per-contact/history caps plus TTL eviction. |
| `jarvis/prefetch/cache.py` | 248 | medium | `L2Cache` uses thread-local SQLite connections but provides no close lifecycle; long-lived thread churn can retain open descriptors and page cache memory. | Add `close()` for per-thread connection cleanup and invoke during shutdown/reset paths. |
| `scripts/prepare_gemini_with_full_features.py` | 60 | medium | Builds large `features_list` in memory, then materializes a second full copy via `np.array(features_list)` (`line 105`), causing peak-memory spikes. | Stream to chunked `.npy/.npz` or memmap output; avoid keeping list + full array simultaneously. |
| `scripts/prepare_gemini_training_with_embeddings.py` | 56 | medium | Same retention pattern as above (`features_list` + `np.array` at `line 84`) with embedding vectors per row. | Write features in chunks/memmap and free chunk buffers aggressively. |

## 5) Performance Anti-Patterns

| File Path | Line | Severity | Description | Suggested Fix |
|---|---:|---|---|---|
| `scripts/prepare_gemini_with_full_features.py` | 112 | high | Unbatched embedding calls: `embedder.encode(text)` per example (and context at `line 119`). This is N-round-trip embedding work and violates batch-first guidance. | Batch encode texts/contexts with `encode(list[str])`, then join results by index. |
| `scripts/prepare_gemini_training_with_embeddings.py` | 102 | high | Unbatched embedding extraction in a row loop. | Encode in batches and vectorize concatenation to reduce model invocations and Python overhead. |
| `scripts/prepare_gemini_training_data.py` | 67 | medium | Per-row spaCy-heavy feature extraction in loop with no batching (`nlp.pipe`) path. | Add batched extractor API using `nlp.pipe` and process rows in chunks. |
| `jarvis/graph/builder.py` | 276 | high | For each conversation, code calls `reader.get_messages(...)` (`line 282`), creating conversation-level N+1 I/O and repeated parsing overhead. | Fetch message stats in a single aggregated query or bulk reader API keyed by conversation IDs. |
| `jarvis/graph/builder.py` | 222 | medium | `_contact_cache` and `_message_stats_cache` are declared but never used, so expensive fetches are repeated without reuse. | Implement cache usage with invalidation policy or remove fields. |
| `integrations/imessage/reader.py` | 1341 | high | Search path bypasses existing batch prefetch pipeline (`_rows_to_messages`), causing avoidable per-row DB work. | Reuse `_rows_to_messages` or build equivalent batched prefetch in search path. |
| `jarvis/observability/metrics_router.py` | 287 | low | Per-record `json.dumps(m.latency_ms)` in hot flush path adds CPU/alloc pressure and larger storage footprint. | Store summary stats in typed columns; if raw latency vectors are required, store compact binary format. |

## Notes

- Embedding storage paths generally use binary serialization (`.tobytes()` / `np.frombuffer()`), which is good and avoids the worst JSON-float payload anti-pattern for vectors.
- Several looped DB calls in `jarvis/db/` are chunked `IN (...)` queries (good pattern), not true N+1 by row.
