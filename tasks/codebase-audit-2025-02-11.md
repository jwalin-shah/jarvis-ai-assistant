# Codebase Audit - February 11, 2025

Full codebase review across 9 domains: backend/server, AI/LLM models, data/search/contacts, ML classifiers, scripts, integration tests, unit tests, frontend/Tauri, API routers. ~200 issues identified, deduplicated and prioritized below.

**Suitable for Claude Code Web sessions**: Each task is scoped to 1-3 files, no MLX/GPU or `make test` required. Branch per task.

---

## P0 - Critical (Fix First)

### Security

- [ ] **SEC-01**: Path traversal in send-attachment endpoint. `api/routers/conversations.py:690-704` uses string `startswith()` for path validation — vulnerable to symlinks and race conditions. Fix: use `Path.resolve().relative_to(home)`.
- [ ] **SEC-02**: WebSocket auth token never rotated. `jarvis/socket_server.py:274-277` generates token once at startup, stores in plaintext `~/.jarvis/ws_token`. Fix: rotate every 24h, add token expiry.
- [ ] **SEC-03**: Untrusted cache deserialization. `jarvis/prefetch/cache_utils.py:43-70` deserializes JSON/NumPy without size or depth limits. Fix: add JSON depth limit, whitelist numpy dtypes, add size cap before deserializing.

### Thread Safety

- [ ] **SAFE-01**: GPU lock held for entire generation. `models/loader.py:757-805` holds `_mlx_load_lock` during full token generation loop, blocking all other GPU ops. Fix: lock only wraps forward pass setup, not token iteration.
- [ ] **SAFE-02**: GPU lock held for entire stream generation. `models/loader.py:935-984` same issue for streaming — holds lock while yielding tokens. Fix: release lock after stream setup.
- [ ] **SAFE-03**: Tokenizer/GPU lock race in bert_embedder. `models/bert_embedder.py:407-425` — tokenization happens outside GPU lock, allowing another thread to corrupt tokenizer state (padding on/off) between tokenize and forward pass. Fix: single critical section per batch.
- [ ] **SAFE-04**: Same tokenizer/GPU lock race in cross_encoder. `models/cross_encoder.py:288-342`. Fix: same as SAFE-03.

### Performance

- [ ] **PERF-01**: Redundant tokenization in bert_embedder. `models/bert_embedder.py:407-415` tokenizes twice — once without padding for lengths, again with padding for batch. Fix: single tokenize pass (~10ms saved per encode).
- [ ] **PERF-02**: Same redundant tokenization in cross_encoder. `models/cross_encoder.py:288-294`. Fix: single pass.
- [ ] **PERF-03**: N+1 query in stats endpoint. `api/routers/stats.py:537-540` fetches 100 conversations then loops to find one. Fix: add `get_conversation(chat_id)` method.
- [ ] **PERF-04**: N+1 in get_relationship_profile. `jarvis/search/embeddings.py:904-924` fetches stats then loads all messages separately. Fix: combine into single query.

### ML Quality

- [ ] **ML-01**: Train-serve skew with context BERT features. `jarvis/classifiers/category_classifier.py:12-260` — model trained with 384-dim context BERT embeddings, but inference always zeros them out. This is NOT valid auxiliary supervision — it's distribution mismatch. Fix: either (a) train without context BERT, or (b) compute context BERT at inference too. Validate F1 impact.

---

## P1 - Important (High Impact)

### Performance

- [ ] **PERF-05**: Redundant spaCy parsing in feature extraction. `jarvis/features/category_features.py:1029-1077` — `extract_all_batch()` calls `nlp.pipe()` but `extract_hand_crafted()` re-tokenizes via `text.split()`. Fix: pass parsed doc to all extract methods. 3-5x speedup for batch.
- [ ] **PERF-06**: Cache key computed 3x per message in batch classification. `jarvis/classifiers/category_classifier.py:301-452` — MD5 hash computed at cache check, pipeline, and post-prediction. Fix: pre-compute cache keys once at batch start.
- [ ] **PERF-07**: HDBSCAN memory leak on repeated calls. `jarvis/topics/topic_discovery.py:165-250` — clusterer never cleaned up, accumulates C++ memory. Fix: `del clusterer; gc.collect()` in finally block.
- [ ] **PERF-08**: Topic segmenter re-encodes per contact. `jarvis/topics/topic_segmenter.py:568-613` — no cross-contact batching of embedding lookups. Fix: batch fetch all message embeddings once.
- [ ] **PERF-09**: Redundant tokenization in nli_cross_encoder. `models/nli_cross_encoder.py:215-219`. Fix: single pass with padding.
- [ ] **PERF-10**: Redundant tokenization in prefill_prompt_cache. `models/loader.py:531` encodes prefix, then `generate_step` re-encodes. Fix: reuse tokens.
- [ ] **PERF-11**: `analyze_user_style()` called every time without caching. `jarvis/prompts/builders.py:814-841`. Fix: accept pre-computed analysis parameter.
- [ ] **PERF-12**: GLiNER model loaded per-instance, not cached. `jarvis/contacts/candidate_extractor.py:532-551`. Fix: singleton pattern like fact_extractor's `_get_shared_nlp()`.
- [ ] **PERF-13**: Contact cache per-instance in fact_extractor. `jarvis/contacts/fact_extractor.py:232`. Fix: module-level cache with TTL.
- [ ] **PERF-14**: Blocking I/O without timeout in avatar endpoint. `api/routers/contacts.py:408-410`. Fix: wrap in `asyncio.timeout(5.0)`.
- [ ] **PERF-15**: Missing connection pool limits in search. `api/routers/search.py:31-58` creates dedicated reader with own pool. Fix: share reader or limit instances.
- [ ] **PERF-16**: Double data load in gemini prep scripts. `scripts/prepare_gemini_training_with_embeddings.py:111-140`. Fix: pass pre-loaded examples directly.
- [ ] **PERF-17**: Streaming token memory leak in frontend. `desktop/src/lib/socket/client.ts:305-325` — `streamingRequests.tokens[]` never cleared after completion. Fix: clear entries after `onComplete()`.
- [ ] **PERF-18**: Prefetch executor reader lazily initialized. `jarvis/prefetch/executor.py:289-303`. Fix: init in `start()`.
- [ ] **PERF-19**: `find_similar()` loads up to 1000 embeddings when only 10 needed. `jarvis/search/embeddings.py:647-717`. Fix: early termination or sqlite-vec.
- [ ] **PERF-20**: Missing pagination on export backup. `api/routers/export.py:255-277`. Fix: add offset param, timeout, partial results.

### Security

- [ ] **SEC-04**: Insufficient search query validation — no max_length. `api/routers/conversations.py:359-365`. Fix: add `max_length=1000`.
- [ ] **SEC-05**: Incomplete prompt injection defense. `api/routers/drafts.py:77-111`. Fix: allowlist approach instead of regex deny-list.
- [ ] **SEC-06**: SQLite parameter limit not checked. `jarvis/search/vec_search.py:42-58` — placeholder validation doesn't check against SQLite's 999 param limit. Fix: `if len(rowids) > 900: raise`.
- [ ] **SEC-07**: XSS risk in avatar blob URLs. `desktop/src/lib/components/ConversationList.svelte:232`. Fix: validate blob MIME type is image/jpeg or image/png.
- [ ] **SEC-08**: Method name injection in Tauri event emission. `desktop/src-tauri/src/socket.rs:230-237`. Fix: whitelist allowed method names.
- [ ] **SEC-09**: Watcher schema validation incomplete. `jarvis/watcher.py:669-713` — only checks table/column existence, not types. Fix: validate column types via `PRAGMA table_info()`.
- [ ] **SEC-10**: Router prefetch cache missing validation. `jarvis/router.py:402-434` — cached dict returned without structure validation. Fix: validate required keys/types.

### Refactoring

- [ ] **REF-01**: Reply service `generate_reply()` is 133 lines. `jarvis/reply_service.py:294-426`. Fix: extract search phase, generation phase, metrics recording.
- [ ] **REF-02**: Router `route()` is 131 lines. `jarvis/router.py:458-588`. Fix: extract message context builder, prefetch check, legacy mapping.
- [ ] **REF-03**: BERT weight mapping duplicated in 3 files. `models/bert_embedder.py:212-246`, `models/cross_encoder.py:79-136`, `models/nli_cross_encoder.py`. Fix: extract to shared `models/weight_mapping.py`.
- [ ] **REF-04**: Snapshot finding duplicated in 3 files. bert_embedder, cross_encoder, nli_cross_encoder. Fix: extract to `models/utils.py:find_model_snapshot()`.
- [ ] **REF-05**: `build_reply_prompt()` is 98 lines with 5+ nested branches. `jarvis/prompts/builders.py:338-436`. Fix: extract `_determine_effective_tone()`, `_select_examples_for_tone()`, `_build_custom_instructions()`.
- [ ] **REF-06**: `build_threaded_reply_prompt()` is 75 lines. `jarvis/prompts/builders.py:627-702`. Fix: introduce `ThreadedPromptComponents` dataclass.
- [ ] **REF-07**: Regex patterns duplicated between response_mobilization.py and category_features.py. Fix: extract to shared `jarvis/nlp/patterns.py`.
- [ ] **REF-08**: 70+ magic threshold numbers in response_mobilization.py. Fix: extract to `MobilizationConfig` dataclass for tunability.
- [ ] **REF-09**: `_apply_negative_signals()` 60+ lines, untestable. `jarvis/topics/topic_segmenter.py:720-781`. Fix: extract individual signal methods.
- [ ] **REF-10**: `_extract_rule_based()` 120 lines. `jarvis/contacts/fact_extractor.py:568-688`. Fix: split into `_extract_relationships()`, `_extract_locations()`, `_extract_preferences()`.
- [ ] **REF-11**: `_is_coherent_subject()` 67 lines. `jarvis/contacts/fact_extractor.py:347-414`. Fix: split into `_is_vague_pronoun()`, `_is_incomplete_phrase()`, `_is_malformed()`.
- [ ] **REF-12**: Duplicated evaluation logic across train scripts. `scripts/train_fact_filter.py`, `train_message_gate.py`, `train_category_svm.py`. Fix: create `scripts/training_utils.py`.
- [ ] **REF-13**: Duplicated data prep pipeline. `scripts/prepare_gemini_training_with_embeddings.py` vs `prepare_gemini_with_full_features.py` — 95% identical. Fix: extract shared `gemini_prepare_shared.py`.
- [ ] **REF-14**: JarvisSocket class 1230 lines. `desktop/src/lib/socket/client.ts`. Fix: split into TauriSocketClient, WebSocketClient, SocketBatchManager, SocketStreamManager.
- [ ] **REF-15**: Avatar endpoint 152 lines. `api/routers/contacts.py:335-487`. Fix: extract normalize, cache lookup, image processing, default generation.
- [ ] **REF-16**: Duplicated fetch_context SQL. `scripts/build_fact_goldset.py:200-275`. Fix: parameterize direction.
- [ ] **REF-17**: Cache tier invalidation iterates all 3 tiers separately. `jarvis/prefetch/cache.py:910-955`. Fix: single `_invalidate_all_tiers()` helper.

### Code Quality

- [ ] **QUAL-01**: Deprecated MLX API in nli_cross_encoder. `models/nli_cross_encoder.py:172-173` uses `mx.metal.clear_cache()`. Fix: use `mx.clear_cache()`.
- [ ] **QUAL-02**: Broad exception handling hides errors. `models/cross_encoder.py:235-249` catches all Exception during download. Fix: catch ImportError separately, let others propagate.
- [ ] **QUAL-03**: Header-only parsing could fail silently. `models/bert_embedder.py:254-259` — `_check_has_pooler()` doesn't validate read bytes match header_size. Fix: verify `len(data) == header_size`.
- [ ] **QUAL-04**: `speculative_generate_step` variable naming misleading. `models/loader.py:787-805` — `accepted` should be `draft_tokens_verified`.
- [ ] **QUAL-05**: Type hints use `Any` in socket server. `jarvis/socket_server.py:177-198` — `_watcher: Any`, `_prefetch_manager: Any`. Fix: use proper types with TYPE_CHECKING.
- [ ] **QUAL-06**: Error handling inconsistent in watcher. `jarvis/watcher.py:290-360` — no distinction between transient and permanent errors. Fix: create exception hierarchy.
- [ ] **QUAL-07**: Executor handlers catch generic Exception, swallow KeyboardInterrupt. `jarvis/prefetch/executor.py:677-909`. Fix: re-raise BaseException subclasses, use `logger.exception()`.
- [ ] **QUAL-08**: Generator stream state management complex. `models/generator.py:309-449` — queue + thread bridge hard to follow. Fix: use sentinel-based queue protocol.

### Test Gaps

- [ ] **TEST-01**: No thread safety tests for bert_embedder concurrent encode(). Fix: add ThreadPoolExecutor test with 10 concurrent threads.
- [ ] **TEST-02**: No thread safety tests for cross_encoder concurrent predict(). Fix: same pattern.
- [ ] **TEST-03**: Generation timeout never tested. `models/loader.py:807-830`. Fix: verify exception raised, next generation works, thread cleanup.
- [ ] **TEST-04**: 11 API routers have ZERO tests: search, drafts, export, batch, graph, threads, relationships, tags, settings, feedback, experiments. Fix: create test files with happy path + error cases.
- [ ] **TEST-05**: No security test cases for injection attacks. Fix: add `tests/security/test_injection_attacks.py`.
- [ ] **TEST-06**: Socket server streaming response untested. `jarvis/socket_server.py:1049-1113`. Fix: test token ordering, error during streaming.
- [ ] **TEST-07**: Watcher concurrent resegmentation untested. `jarvis/watcher.py:618-668`. Fix: test lock behavior with concurrent calls.
- [ ] **TEST-08**: No test for fact index consistency after DELETE. Fix: verify vec_facts cleaned up.
- [ ] **TEST-09**: No test for relationship profiles with <20 messages. Fix: test minimal profile.
- [ ] **TEST-10**: Mixed batch classification paths untested (cache/fast-path/pipeline). Fix: parameterized test with mixed inputs.

### UX (Frontend)

- [ ] **UX-01**: Silent WebSocket fallback — user not notified when local socket fails. `desktop/src/lib/socket/client.ts:336-372`. Fix: show indicator badge.
- [ ] **UX-02**: Missing delivery/read status after sending. `desktop/src/lib/components/MessageView.svelte:226-292`. Fix: 3-state status (sending/sent/error).
- [ ] **UX-03**: Scroll-to-bottom FAB not keyboard accessible. `desktop/src/lib/components/MessageView.svelte:729-737`. Fix: add ESC handler, tabindex.
- [ ] **UX-04**: No loading state distinction between initial load and polling. `desktop/src/lib/stores/conversations.svelte.ts:212-255`. Fix: split into `isInitialLoad` vs `isPolling`.
- [ ] **UX-05**: Search input in conversation list is non-functional. `desktop/src/lib/components/ConversationList.svelte:438-440`. Fix: implement filter handler.

### API Design

- [ ] **API-01**: Inconsistent error response formats. `api/dependencies.py:44-56` returns nested JSON vs `ErrorResponse` elsewhere. Fix: standardize.
- [ ] **API-02**: Missing pagination cursors. `api/routers/conversations.py:80-100` — no `has_more` or `next_cursor`. Fix: add to response schema.
- [ ] **API-03**: Cache key includes microsecond timestamps, defeating caching. `api/routers/conversations.py:164-182`. Fix: truncate to minute/hour.

### Frontend Quality

- [ ] **FE-01**: Missing null checks on dynamic Tauri imports. `desktop/src/lib/socket/client.ts:43-45, 207-213`. Fix: add explicit guards.
- [ ] **FE-02**: @ts-expect-error abuse. `desktop/src/lib/components/ConversationList.svelte:27-30`. Fix: declare proper type.
- [ ] **FE-03**: Unsafe type coercion in stream token handling. `desktop/src/lib/socket/client.ts:451-452`. Fix: validate params structure.
- [ ] **FE-04**: Race condition in selectConversation(). `desktop/src/lib/stores/conversations.svelte.ts:344-393` — prefetch_focus not awaited before setting messages. Fix: await or skip cache.
- [ ] **FE-05**: Unchecked .unwrap() in Rust error handler. `desktop/src-tauri/src/socket.rs:319-327`. Fix: use try_write().
- [ ] **FE-06**: Two concurrent message processors with backpressure risk. `desktop/src-tauri/src/socket.rs:656-717`. Fix: increase channel capacity with monitoring.

---

## P2 - Nice to Have

### Robustness (Scripts)

- [ ] No crash recovery in eval_gliner_candidates — reruns from scratch on failure. Fix: incremental JSONL saves.
- [ ] No incremental save in generate_preference_pairs. Fix: append pairs as generated.
- [ ] No retry on database locks in build_fact_goldset. Fix: exponential backoff.
- [ ] No input schema validation in train_fact_filter. Fix: validate required fields.
- [ ] cleanup_legacy_facts.py uses print() instead of logging. Fix: use setup_script_logging().

### Test Quality

- [ ] Tests mirror implementation instead of testing behavior (3-5 files). Fix: test from spec.
- [ ] Over-mocking hides real bugs (8-10 files). Fix: use real implementations or realistic data.
- [ ] Flaky timing-dependent performance tests. Fix: relative measurements or skip on slow CI.
- [ ] Fixture overuse and test interdependence. Fix: move to smallest scope.
- [ ] Missing negative test cases for boundary values. Fix: parameterized edge cases.
- [ ] Performance baselines measure mock overhead, not real code. Fix: use real models (mark slow) or remove.
- [ ] Duplicated router fixtures across 3+ integration test files. Fix: shared conftest.py.

### Frontend Performance

- [ ] N+1 polling pattern in conversations store — filters in-memory. Fix: push to SQL WHERE.
- [ ] Topic fetching burst on load (10+ requests). Fix: lazy-load for visible only.
- [ ] ResizeObserver on every message. Fix: IntersectionObserver hybrid.
- [ ] Two separate IntersectionObservers for avatars/topics. Fix: merge into one.

### Code Quality (Minor)

- [ ] Missing type annotations throughout (scattered).
- [ ] Magic numbers in watcher segment thresholds (15, 50). Fix: named constants with docs.
- [ ] Dead fallback path in topic_discovery `_extract_keywords()`. Fix: make precomputed_idf required.
- [ ] Cache eviction uses O(N log N) sort. Fix: use TTLCache from cachetools.
- [ ] Inconsistent `.match()` vs `.search()` in response_mobilization patterns. Fix: document intent.
- [ ] FOOD_WORDS list unorganized. Fix: group by category.
- [ ] Massive type imports without documentation in `desktop/src/lib/api/client.ts`. Fix: add JSDoc clusters.

---

## Metrics

| Domain | P0 | P1 | P2 | Total |
|--------|----|----|-----|-------|
| Security | 3 | 7 | 0 | 10 |
| Thread Safety | 4 | 0 | 0 | 4 |
| Performance | 4 | 16 | 4 | 24 |
| ML Quality | 1 | 0 | 0 | 1 |
| Refactoring | 0 | 17 | 0 | 17 |
| Code Quality | 0 | 8 | 7 | 15 |
| Test Gaps | 0 | 10 | 7 | 17 |
| UX | 0 | 5 | 0 | 5 |
| API Design | 0 | 3 | 0 | 3 |
| Frontend | 0 | 6 | 4 | 10 |
| Robustness | 0 | 0 | 5 | 5 |
| **Total** | **12** | **72** | **27** | **111** |

---

## Recommended Claude Code Web Task Sizing

**Small (1 branch, ~30 min):** SEC-04, SEC-06, QUAL-01, QUAL-04, PERF-01, PERF-02, PERF-09, REF-04, API-03

**Medium (1 branch, ~1 hr):** SEC-01, SAFE-03, SAFE-04, PERF-05, PERF-06, REF-03, REF-07, REF-12, REF-13, TEST-04 (per router)

**Large (1 branch, ~2 hr):** SAFE-01+SAFE-02 (together), REF-01, REF-02, REF-14, ML-01, TEST-01+TEST-02

**Do NOT assign to Web (needs local):** Anything requiring `make test`, MLX GPU verification, iMessage DB access
