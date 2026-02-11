# Claude Code Web Session Prompts

## Batch 1: All P0 Issues (8 parallel sessions)

All sessions can run simultaneously — no file overlaps between groups.

---

### Session 1: GPU Lock Scope (SAFE-01 + SAFE-02)
**Branch:** `web/safe-01-gpu-lock-scope`
**Files:** `models/loader.py`

```
Create branch web/safe-01-gpu-lock-scope from main.

Fix GPU lock scope in models/loader.py. There are two critical issues:

1. In _do_generate() (around line 757-805): The _mlx_load_lock is held for the ENTIRE generation loop including all token sampling. The lock should only protect the initial model forward pass setup and GPU memory operations, NOT the token-by-token iteration loop. Refactor so the lock is acquired for the setup/first forward pass, then released before iterating tokens.

2. In stream_generate() (around line 935-984): Same issue — GPU lock held for entire streaming loop, blocking all other GPU operations (embeddings, other generations) while tokens are yielded. Refactor so the lock wraps the mlx_lm.stream_generate setup, not the iteration/yielding loop.

IMPORTANT CONTEXT:
- MLX/Metal GPU is NOT thread-safe for concurrent operations (loads, generate, encode forward pass, mx.eval)
- The lock MUST still protect actual GPU operations (model forward pass, mx.eval calls)
- But yielding/iterating tokens between forward passes does NOT need the lock
- The class-level lock is `_mlx_load_lock` (a threading.Lock)
- Other code (bert_embedder, cross_encoder) also acquires this same lock for their GPU ops

Read the full file first to understand the locking pattern. Make minimal, surgical changes. Commit when done.

Do NOT run make test, pytest, or any Python commands.
```

---

### Session 2: Tokenizer Lock Race + Redundant Tokenization (SAFE-03 + SAFE-04 + PERF-01 + PERF-02)
**Branch:** `web/safe-03-tokenizer-lock-race`
**Files:** `models/bert_embedder.py`, `models/cross_encoder.py`

```
Create branch web/safe-03-tokenizer-lock-race from main.

Fix two related issues in models/bert_embedder.py and models/cross_encoder.py:

ISSUE 1 — Thread safety race condition (both files):
In bert_embedder.py encode() (around line 407-425) and cross_encoder.py predict_batch() (around line 288-342):
Tokenization happens OUTSIDE the GPU lock, then the tokenized result is used INSIDE the GPU lock. Between these two sections, another thread could call the tokenizer and change its padding state (enable_padding/no_padding), corrupting the batch.

Fix: Make tokenization and GPU forward pass a single critical section. Either:
(a) Move tokenization inside the GPU lock, OR
(b) Use a single encode_lock that covers both tokenization and forward pass

ISSUE 2 — Redundant tokenization (both files):
In bert_embedder.py encode(): Tokenization happens TWICE — once without padding to get sequence lengths for sorting (line ~410), then again with padding for the actual batch (line ~431).
Same pattern in cross_encoder.py predict_batch() (line ~292 and ~305).

Fix: Tokenize ONCE with padding enabled, extract both the encodings AND the lengths from that single pass. This saves ~10ms per encode call.

Read both files fully first. The fix pattern is the same in both files. Commit when done.

Do NOT run make test, pytest, or any Python commands.
```

---

### Session 3: Path Traversal Fix (SEC-01)
**Branch:** `web/sec-01-path-traversal`
**Files:** `api/routers/conversations.py`, `api/schemas/drafts.py`

```
Create branch web/sec-01-path-traversal from main.

Fix path traversal vulnerability in the send-attachment endpoint.

In api/routers/conversations.py (around line 690-704) and api/schemas/drafts.py (around line 94-103):
The file_path validation uses string startswith() to check if path is within home directory. This is vulnerable to:
- Symlink attacks (symlink pointing outside home dir)
- Race conditions between validation and file access (TOCTOU)
- Case sensitivity issues on case-insensitive filesystems

Fix:
1. In the validation function, use Path.resolve() to resolve all symlinks FIRST
2. Then use .relative_to(home) which raises ValueError if path is outside home
3. Wrap in try/except ValueError to return proper error
4. Verify the resolved path exists and is a regular file (not a directory, device, etc.)

Example pattern:
```python
resolved = Path(file_path).resolve(strict=True)  # Resolves symlinks, raises if doesn't exist
try:
    resolved.relative_to(Path.home())
except ValueError:
    raise ValueError("File must be within home directory")
if not resolved.is_file():
    raise ValueError("Path must be a regular file")
```

Read both files first. Make minimal changes. Commit when done.

Do NOT run make test, pytest, or any Python commands.
```

---

### Session 4: WebSocket Auth Token Rotation (SEC-02)
**Branch:** `web/sec-02-ws-token-rotation`
**Files:** `jarvis/socket_server.py`

```
Create branch web/sec-02-ws-token-rotation from main.

Fix WebSocket auth token security in jarvis/socket_server.py.

Current issue (around line 274-277): The WebSocket auth token is generated ONCE at startup and never rotated. It's stored in plaintext at ~/.jarvis/ws_token with 0o600 permissions.

Fix:
1. Add a token generation timestamp alongside the token
2. Add a method _rotate_ws_token() that generates a new token and writes it to the file
3. In the WebSocket auth check, also check if the token is older than 24 hours — if so, rotate it
4. When rotating, the OLD token should remain valid for a 60-second grace period (so existing connections aren't immediately killed)
5. Add a _token_created_at: float attribute to track when current token was generated

Keep changes minimal and focused on the rotation logic. Don't refactor unrelated code. Read the full file first to understand the auth flow.

Commit when done.

Do NOT run make test, pytest, or any Python commands.
```

---

### Session 5: Cache Deserialization Safety (SEC-03)
**Branch:** `web/sec-03-cache-deserialization`
**Files:** `jarvis/prefetch/cache_utils.py`

```
Create branch web/sec-03-cache-deserialization from main.

Fix unsafe deserialization in jarvis/prefetch/cache_utils.py (around line 43-70).

Current issues in deserialize_value():
1. JSON deserialization has no size or depth limits — deeply nested JSON could cause DoS
2. NumPy deserialization trusts the dtype string without validation — malformed dtype could cause issues
3. No overall size limit before attempting deserialization

Fix:
1. Before JSON deserialization, check that the raw data size is under a reasonable limit (e.g., 10MB)
2. For NumPy deserialization, validate dtype is in a whitelist: float32, float64, int32, int64, uint8, bool
3. For JSON, add a max nesting depth check (e.g., reject if nesting > 20 levels) — you can do this by checking for excessive bracket depth in the raw string before parsing, or use a custom decoder
4. Add similar protections in serialize_value() if applicable

Read the file first. Keep changes minimal. Commit when done.

Do NOT run make test, pytest, or any Python commands.
```

---

### Session 6: Stats N+1 Query (PERF-03)
**Branch:** `web/perf-03-stats-n1`
**Files:** `api/routers/stats.py`

```
Create branch web/perf-03-stats-n1 from main.

Fix N+1 query pattern in api/routers/stats.py (around line 537-540).

Current code fetches ALL conversations (limit=100) then loops to find the matching one:
```python
conversations = reader.get_conversations(limit=100)
for conv in conversations:
    if conv.chat_id == chat_id:
        ...
```

This is an O(n) search that should be a direct lookup. Fix by:

1. Check if the reader object (ChatDBReader from integrations/imessage/reader.py) has a method to get a single conversation by chat_id. Read integrations/imessage/reader.py to check.
2. If it exists, use it directly.
3. If it doesn't exist, add a get_conversation(chat_id) method to ChatDBReader that queries for a single conversation. Model it after get_conversations() but with a WHERE clause on chat_id.
4. Update the stats endpoint to use the direct lookup.

Read both files first. Commit when done.

Do NOT run make test, pytest, or any Python commands.
```

---

### Session 7: Embeddings N+1 (PERF-04)
**Branch:** `web/perf-04-embeddings-n1`
**Files:** `jarvis/search/embeddings.py`

```
Create branch web/perf-04-embeddings-n1 from main.

Fix N+1 query pattern in jarvis/search/embeddings.py (around line 904-924).

In get_relationship_profile(): The function fetches basic stats in one query, then loads ALL messages for tone detection in a separate query. When called in a batch (e.g., for 100+ contacts), this multiplies queries.

Fix:
1. Read the function and understand what data it needs from messages (likely just text content for tone analysis)
2. Combine the stats query and message fetch into a single query using CTEs or JOINs
3. If full combination isn't possible, at least ensure the message fetch uses LIMIT and only retrieves the columns needed for tone detection (text, is_from_me) — not full Message objects

Also look for any callers that loop over contacts calling this function — if found, note it as a future batch optimization opportunity in a comment.

Read the full file first. Make minimal changes. Commit when done.

Do NOT run make test, pytest, or any Python commands.
```

---

### Session 8: Train-Serve Skew Investigation (ML-01)
**Branch:** `web/ml-01-context-bert-skew`
**Files:** `jarvis/classifiers/category_classifier.py`, `jarvis/features/category_features.py`, `scripts/train_category_svm.py`

```
Create branch web/ml-01-context-bert-skew from main.

Investigate and fix train-serve skew in the category classifier.

ISSUE: In jarvis/classifiers/category_classifier.py (around line 258-260), context BERT embeddings (feature indices 384:768) are ALWAYS zeroed at inference:
```python
features = np.concatenate([embedding, _ZERO_CONTEXT, non_bert_features])
```

But the model was TRAINED with real context BERT embeddings present. This creates a distribution mismatch between training and inference — the model learned feature interactions with context BERT that don't exist at inference time.

Tasks:
1. Read category_classifier.py to understand the full feature pipeline (915 features = 384 BERT + 384 context BERT + 147 hand-crafted)
2. Read the training script (scripts/train_category_svm.py or any LightGBM training script) to confirm context BERT IS present during training
3. Read jarvis/features/category_features.py to understand how context features are extracted

Then fix by choosing ONE approach:
(a) PREFERRED: Remove context BERT from training entirely. Update the training script to use 531 features (384 BERT + 147 hand-crafted) instead of 915. Update inference to match. This is cleaner — if you're not using it at inference, don't train with it.
(b) ALTERNATIVE: Actually compute context BERT at inference time (remove the zeroing). This requires the context messages to be embedded, which adds latency.

Choose approach (a) unless approach (b) is clearly simpler. Update both the classifier AND the training script to be consistent. Add a comment explaining the feature layout.

Do NOT retrain the model — just update the code so the NEXT training run will be correct.

Commit when done.

Do NOT run make test, pytest, or any Python commands.
```

---

## Batch 2: High-Impact P1s (run after Batch 1 merges)

These should run after you've merged Batch 1 to avoid conflicts.

---

### Session 9: Weight Mapping + Snapshot Dedup (REF-03 + REF-04)
**Branch:** `web/ref-03-weight-mapping-dedup`
**Files:** `models/bert_embedder.py`, `models/cross_encoder.py`, `models/nli_cross_encoder.py`, new `models/utils.py`

```
Create branch web/ref-03-weight-mapping-dedup from main.

Deduplicate two patterns copied across 3 model files.

ISSUE 1 — BERT weight name mapping (REF-03):
The HuggingFace-to-MLX weight name mapping logic is duplicated in:
- models/bert_embedder.py (around line 212-246, in load_bert_weights)
- models/cross_encoder.py (around line 79-136, in load_cross_encoder_weights)
- models/nli_cross_encoder.py (similar convert_hf_weights)

Fix: Create models/weight_mapping.py (or add to models/utils.py) with a shared function:
```python
def map_hf_bert_weights(weights: dict) -> dict:
    """Map HuggingFace BERT weight names to MLX model format."""
    ...
```
Then have all three files import and use it.

ISSUE 2 — Snapshot finding (REF-04):
The logic to find the latest model snapshot directory is duplicated in:
- models/bert_embedder.py (around line 322-327)
- models/cross_encoder.py (around line 200-205)
- models/nli_cross_encoder.py (around line 132-140)

Fix: Add to models/utils.py:
```python
def find_model_snapshot(model_dir: Path) -> Path:
    """Find the latest HuggingFace snapshot directory."""
    ...
```

Read all three files first. Extract the COMMON logic — don't force differences to be the same. The cross_encoder has an extra classifier weight step that should stay in its own file.

Commit when done.

Do NOT run make test, pytest, or any Python commands.
```

---

### Session 10: Shared NLP Patterns (REF-07)
**Branch:** `web/ref-07-shared-patterns`
**Files:** `jarvis/classifiers/response_mobilization.py`, `jarvis/features/category_features.py`, new `jarvis/nlp/patterns.py`

```
Create branch web/ref-07-shared-patterns from main.

Extract duplicated regex patterns into a shared module.

ISSUE: Many text patterns (greetings, emotional markers, request indicators, proposal patterns) are defined independently in both:
- jarvis/classifiers/response_mobilization.py
- jarvis/features/category_features.py

They've diverged over time and are a maintenance headache.

Fix:
1. Read both files and identify ALL overlapping pattern lists (GREETING_PATTERNS, EMOTIONAL_MARKERS, REQUEST_PATTERNS, PROPOSAL_PATTERNS, etc.)
2. Create jarvis/nlp/__init__.py (empty) and jarvis/nlp/patterns.py
3. Move the canonical version of each pattern list to patterns.py with clear docstrings
4. Update both source files to import from jarvis.nlp.patterns
5. Where patterns differ between the two files, prefer the MORE COMPREHENSIVE version
6. Pre-compile all regex patterns in patterns.py (use re.compile)

Keep imports clean. Don't change any logic — only move pattern definitions.

Commit when done.

Do NOT run make test, pytest, or any Python commands.
```

---

### Session 11: Training Utils Dedup (REF-12)
**Branch:** `web/ref-12-training-utils`
**Files:** `scripts/train_fact_filter.py`, `scripts/train_message_gate.py`, `scripts/train_category_svm.py`, new `scripts/training_utils.py`

```
Create branch web/ref-12-training-utils from main.

Extract duplicated training evaluation logic into a shared module.

ISSUE: Three training scripts reimplement the same compute_metrics(), evaluate(), and classification report printing — about 100 lines of duplication each.

Files:
- scripts/train_fact_filter.py
- scripts/train_message_gate.py
- scripts/train_category_svm.py

Fix:
1. Read all three files and identify the common evaluation/metrics code
2. Create scripts/training_utils.py with shared functions:
   - compute_metrics(y_true, y_pred) -> dict
   - print_classification_report(y_true, y_pred, labels)
   - evaluate_model(model, X_test, y_test) -> dict
   - Any other obviously duplicated utilities
3. Update all three scripts to import from training_utils
4. Keep script-specific logic in each script

Don't change any behavior — just move common code. Commit when done.

Do NOT run make test, pytest, or any Python commands.
```

---

### Session 12: Missing API Router Tests (TEST-04, partial)
**Branch:** `web/test-04-search-drafts-tests`
**Files:** new `tests/api/test_search_router.py`, new `tests/api/test_drafts_router.py`

```
Create branch web/test-04-search-drafts-tests from main.

Add test coverage for two untested API routers: search and drafts.

Currently these routers have ZERO tests:
- api/routers/search.py (semantic search, cache management)
- api/routers/drafts.py (draft generation, prompt injection defense)

Steps:
1. Read api/routers/search.py and api/routers/drafts.py fully
2. Read existing test files (tests/api/test_contacts_router.py, tests/api/test_attachments_router.py) to understand the project's test patterns, fixtures, and FastAPI TestClient setup
3. Create tests/api/test_search_router.py with tests for:
   - Semantic search happy path (mock embedder, verify response format)
   - Search with empty query (should return 400 or empty)
   - Search with very long query (should be rejected or truncated)
   - Cache stats endpoint
   - Cache clear endpoint
4. Create tests/api/test_drafts_router.py with tests for:
   - Draft generation happy path (mock generator)
   - Prompt injection defense (_sanitize_instruction should strip dangerous patterns)
   - Missing required fields (400 response)
   - Rate limiting behavior

IMPORTANT test quality rules:
- Assert on RESPONSE DATA (response.json()), NOT on mock calls
- Test BEHAVIOR from the spec, not implementation details
- Use realistic test data, not trivial mocks
- Each test should fail if the behavior breaks, regardless of implementation

Commit when done.

Do NOT run make test, pytest, or any Python commands.
```

---

### Session 13: Feature Extraction Performance (PERF-05 + PERF-06)
**Branch:** `web/perf-05-feature-extraction`
**Files:** `jarvis/features/category_features.py`, `jarvis/classifiers/category_classifier.py`

```
Create branch web/perf-05-feature-extraction from main.

Fix two performance issues in the classification pipeline.

ISSUE 1 (PERF-05) — Redundant text parsing in jarvis/features/category_features.py:
In extract_all_batch() (around line 1029-1077): nlp.pipe() is called to parse docs, but extract_hand_crafted() does NOT accept a doc parameter — it re-tokenizes via text.split(). Other extract methods (extract_spacy_features, etc.) DO accept doc.

Fix: Add doc parameter to extract_hand_crafted() and any other extract methods that don't have it. Pass the parsed doc from extract_all_batch() through to ALL extraction methods. Use doc tokens instead of text.split() where applicable. This avoids 4 separate text normalizations per message.

ISSUE 2 (PERF-06) — Cache key computed 3x per message in jarvis/classifiers/category_classifier.py:
In classify_batch() (around line 327-454): The MD5 cache key (text + "|" + "|".join(context)) is computed at:
- Cache lookup (pass 1)
- After pipeline prediction (pass 3)
- After each result is computed

Fix: Pre-compute ALL cache keys once at the start of classify_batch():
```python
cache_keys = [
    hashlib.md5((texts[i] + "|" + "|".join(contexts[i] or [])).encode()).hexdigest()
    for i in range(n)
]
```
Then reference cache_keys[i] throughout instead of recomputing.

Read both files fully first. Commit when done.

Do NOT run make test, pytest, or any Python commands.
```

---

### Session 14: Reply Service + Router Refactor (REF-01 + REF-02)
**Branch:** `web/ref-01-reply-service-refactor`
**Files:** `jarvis/reply_service.py`, `jarvis/router.py`

```
Create branch web/ref-01-reply-service-refactor from main.

Refactor two oversized methods into smaller, testable pieces.

ISSUE 1 (REF-01) — jarvis/reply_service.py generate_reply() is 133 lines (around line 294-426):
Has 7 nested sections: health check, search, classification, generation, metrics.

Fix: Extract into focused private methods:
- _search_context(chat_id, message) -> SearchResult
- _generate_response(prompt, request) -> GenerationResult
- _record_metrics(result, timings) -> None

The main generate_reply() becomes orchestration only (~30 lines).

ISSUE 2 (REF-02) — jarvis/router.py route() is 131 lines (around line 458-588):
Has prefetch cache check, message context building, and legacy response mapping all interleaved.

Fix: Extract into:
- _build_message_context(messages, chat_id) -> MessageContext
- _check_prefetch_cache(chat_id, message) -> CachedResult | None
- _map_legacy_response(result) -> dict

Read both files fully first. Keep all existing behavior identical — pure refactor, no logic changes. Commit when done.

Do NOT run make test, pytest, or any Python commands.
```

---

### Session 15: Frontend UX Fixes (UX-01 + UX-03 + UX-05)
**Branch:** `web/ux-fixes-batch1`
**Files:** `desktop/src/lib/socket/client.ts`, `desktop/src/lib/components/MessageView.svelte`, `desktop/src/lib/components/ConversationList.svelte`

```
Create branch web/ux-fixes-batch1 from main.

Fix three UX issues in the desktop frontend.

ISSUE 1 (UX-01) — Silent WebSocket fallback:
In desktop/src/lib/socket/client.ts (around line 336-372): When Unix socket fails 5 times and falls back to WebSocket, the user gets NO notification. They may not realize they're using network instead of local socket.

Fix: After fallback, emit a custom event (e.g., window.dispatchEvent or use the existing event system) that the UI can listen for. Add a comment noting where a UI indicator should be added.

ISSUE 2 (UX-03) — Scroll-to-bottom FAB not keyboard accessible:
In desktop/src/lib/components/MessageView.svelte (around line 729-737): The scroll-to-bottom floating action button can't be dismissed by keyboard users.

Fix: Add tabindex="0" to the FAB button, add an onkeydown handler that scrolls to bottom on Enter/Space and dismisses on Escape.

ISSUE 3 (UX-05) — Search input in conversation list is non-functional:
In desktop/src/lib/components/ConversationList.svelte (around line 438-440): Search input has no change handler.

Fix: Add a reactive filter that filters the sortedConversations list based on search input text. Match against conversation display_name and last message text. Debounce the filter by 200ms.

Read each file first. Commit when done.

Do NOT run any build commands.
```

---

## Batch 3: More P1s (run after Batch 2 merges)

### Session 16: HDBSCAN Memory + Topic Segmenter (PERF-07 + PERF-08)
**Branch:** `web/perf-07-topic-memory`
**Files:** `jarvis/topics/topic_discovery.py`, `jarvis/topics/topic_segmenter.py`

```
Create branch web/perf-07-topic-memory from main.

Fix two memory/performance issues in the topic system.

ISSUE 1 (PERF-07) — HDBSCAN memory leak in jarvis/topics/topic_discovery.py:
In discover_topics() (around line 165-250): Each call creates a new HDBSCAN clusterer that stores minimum spanning trees, distance matrices, and condensation trees. These are never explicitly cleaned up. On 8GB RAM with 50+ contacts processed sequentially, this accumulates 500MB-1GB.

Fix: Add explicit cleanup in a try/finally block:
```python
try:
    clusterer = HDBSCAN(...)
    labels = clusterer.fit_predict(embeddings)
    # ... use labels ...
finally:
    del clusterer
    gc.collect()
```

ISSUE 2 (PERF-08) — Topic segmenter re-encodes per contact:
In jarvis/topics/topic_segmenter.py _compute_embeddings() (around line 568-613): Looks up cached embeddings via vec.get_embeddings_by_ids() per contact. When segmenting 50 contacts, that's 50 separate DB queries.

Fix: If there's a batch entry point (like segment_conversations_batch or similar), collect all message IDs across contacts and do ONE get_embeddings_by_ids() call. If no batch entry point exists, add a comment suggesting one for future optimization.

Read both files first. Commit when done.

Do NOT run make test, pytest, or any Python commands.
```

---

### Session 17: Mobilization Config (REF-08)
**Branch:** `web/ref-08-mobilization-config`
**Files:** `jarvis/classifiers/response_mobilization.py`

```
Create branch web/ref-08-mobilization-config from main.

Extract 70+ hardcoded confidence thresholds into a configuration dataclass.

In jarvis/classifiers/response_mobilization.py: Confidence thresholds (0.90, 0.85, 0.80, 0.75, etc.) are scattered throughout every classification function. This makes threshold tuning impossible without editing code.

Fix:
1. Read the full file to catalog ALL threshold values and their purposes
2. Create a MobilizationConfig dataclass at the top of the file with ALL thresholds as fields with default values matching current behavior
3. Update classify_response_pressure() to accept an optional config parameter (default to MobilizationConfig())
4. Replace all hardcoded threshold values with config.field_name references
5. Group thresholds logically in the config (question_mark_thresholds, request_thresholds, greeting_thresholds, etc.)

This is a pure refactor — behavior must remain identical with default config. The value is enabling future grid search tuning.

Commit when done.

Do NOT run make test, pytest, or any Python commands.
```

---

### Session 18: Fact Extractor Refactor (REF-10 + REF-11)
**Branch:** `web/ref-10-fact-extractor-refactor`
**Files:** `jarvis/contacts/fact_extractor.py`

```
Create branch web/ref-10-fact-extractor-refactor from main.

Refactor two oversized functions in jarvis/contacts/fact_extractor.py.

ISSUE 1 (REF-10) — _extract_rule_based() is 120 lines (around line 568-688):
Does relationship extraction, location/work extraction, and preference extraction all in one function.

Fix: Split into private methods:
- _extract_relationships(text, ...) -> list[Fact]
- _extract_locations_and_work(text, ...) -> list[Fact]
- _extract_preferences(text, ...) -> list[Fact]

Then _extract_rule_based() orchestrates by calling each and combining results.

ISSUE 2 (REF-11) — _is_coherent_subject() is 67 lines (around line 347-414):
Has 6 rejection rules with complex regex patterns. Hard to test individual rules.

Fix: Split into focused checkers:
- _is_vague_pronoun(subject) -> bool
- _is_incomplete_phrase(subject) -> bool
- _is_malformed(subject) -> bool

Then _is_coherent_subject() calls each and returns False if any returns True.

Read the full file first. Pure refactor — no logic changes. Commit when done.

Do NOT run make test, pytest, or any Python commands.
```

---

### Session 19: Gemini Prep Dedup (REF-13)
**Branch:** `web/ref-13-gemini-prep-dedup`
**Files:** `scripts/prepare_gemini_training_with_embeddings.py`, `scripts/prepare_gemini_with_full_features.py`, new `scripts/gemini_prepare_shared.py`

```
Create branch web/ref-13-gemini-prep-dedup from main.

Deduplicate two nearly identical data preparation scripts.

ISSUE: scripts/prepare_gemini_training_with_embeddings.py and scripts/prepare_gemini_with_full_features.py are 95% identical. Both have: setup_logging, parse_args, load_labeled_examples, create_splits, save_training_data. Only the feature extraction differs.

Fix:
1. Read both files fully to identify ALL shared code
2. Create scripts/gemini_prepare_shared.py with:
   - setup_logging() (or use jarvis.utils.logging.setup_script_logging)
   - parse_args() with common arguments
   - load_labeled_examples(path) -> list[dict]
   - create_train_dev_split(examples, dev_ratio) -> tuple
   - save_training_data(features, labels, output_dir)
   - run_pipeline(feature_extractor_fn, args) — main orchestrator that takes a feature extraction callback
3. Simplify both scripts to just define their unique feature_extractor function and call run_pipeline()

Keep behavior identical. Commit when done.

Do NOT run make test, pytest, or any Python commands.
```

---

### Session 20: API Design Fixes (API-01 + API-02 + API-03)
**Branch:** `web/api-design-fixes`
**Files:** `api/dependencies.py`, `api/routers/conversations.py`

```
Create branch web/api-design-fixes from main.

Fix three API design inconsistencies.

ISSUE 1 (API-01) — Inconsistent error response formats:
In api/dependencies.py (around line 44-56): get_imessage_reader() returns a nested JSON error with {error, message, instructions} structure. But other endpoints use a flat ErrorResponse schema.

Fix: Standardize to use the same error format everywhere. Check what ErrorResponse looks like in api/schemas/ and use that pattern in the dependency too.

ISSUE 2 (API-02) — Missing pagination cursors:
In api/routers/conversations.py (around line 80-100): The conversations list endpoint returns a flat list without has_more or next_cursor fields. Clients must guess when to stop paginating.

Fix: Add has_more and next_cursor to the response. has_more is True if len(results) == limit. next_cursor is the last conversation's last_message_date (which the client passes as 'before' on next request).

ISSUE 3 (API-03) — Cache key includes microseconds:
In api/routers/conversations.py (around line 164-182): Cache key uses full isoformat() timestamps including microseconds. Two identical queries a millisecond apart create separate cache entries.

Fix: Truncate timestamps to minute resolution in cache key: since.replace(second=0, microsecond=0).isoformat()

Read the files first. Commit when done.

Do NOT run make test, pytest, or any Python commands.
```
