# JARVIS Codebase Fix Plan - Implementation Status

## P0 - Critical Issues (Completed)

### 1. Fix FAISS IVF Segfault ✅
- Added validation that num_vectors >= nlist * 39 before training IVF index
- Added is_trained check after training with fallback to flat index
- Applied to both `jarvis/index_v2.py` and `jarvis/index.py`

### 2. Fix Thread-Unsafe EmbeddingCache ✅
- Reviewed code - lock scope already covers all operations atomically
- No changes needed - implementation was correct

### 3. Add Validity Filtering Before Index ✅
- Added filtering in `add_pairs()` to skip pairs with `validity_status == "invalid"`
- Applied to both `jarvis/index_v2.py` and `jarvis/index.py`
- Added logging for skipped invalid pairs

## P1 - High Priority (Completed)

### 4. Commit ContactProfiler Tests ✅
- Tests were already committed (verified in git history: commit 3e760d5)

### 5. Consolidate V1/V2 Response Classifiers ✅
- Added deprecation warnings to V1's `get_response_classifier()` and `reset_response_classifier()`
- Copied `COMMITMENT_RESPONSE_TYPES` and `TRIGGER_TO_VALID_RESPONSES` constants to V2
- Updated internal modules (`jarvis/retrieval.py`, `jarvis/multi_option.py`) to import from V2
- Updated test files to import from V2 or suppress deprecation warnings

### 6. Add Query Embedding Cache to IntentClassifier ✅
- Added `QueryEmbeddingCache` class with thread-safe LRU caching
- Added module-level cache (1000 entries max)
- Modified `classify()` method to check cache before encoding
- Added `get_query_cache_stats()` function for monitoring

### 7. Fix Desktop State Race Conditions ✅
- Added `pendingMessageOperations` Map for per-chat request deduplication
- Updated `handleNewMessagePush` with AbortController to prevent race conditions
- Added toast error notifications for message fetch failures
- Imported toast from `./toast` store

### 8. Enable Real-Time Socket Updates ✅
- Increased `MESSAGE_POLL_INTERVAL_CONNECTED` from 30s to 300s (5 min fallback)
- Increased `CONVERSATION_POLL_INTERVAL_CONNECTED` from 60s to 300s
- Socket push now handles real-time updates; polling is just fallback

## P2 - Medium Priority (Completed)

### 9. Bound Desktop Message Cache with LRU ✅
- Added `MAX_CACHED_CONVERSATIONS = 20`
- Added `cacheAccessOrder` array for LRU tracking
- Created `setMessageCache()` helper with LRU eviction
- Created `getMessageCache()` helper with access order update
- Updated `invalidateMessageCache()` to also clean access order

### 9b. Desktop Hardening: TypeScript Strict Mode ✅
- Already enabled in tsconfig.json - SKIPPED

### 9c. Desktop Hardening: Error Boundaries ✅
- Created `ErrorBoundary.svelte` component
- Catches global `error` and `unhandledrejection` events
- Shows overlay with error message, stack trace, and retry button
- Integrates with toast notifications
- Added to `App.svelte`

### 10. Complete V3 Module Integrations ✅
- Added V3 quality imports to `api/routers/quality.py`
- New endpoints added:
  - `POST /quality/v3/check` - Hallucination detection via ensemble
  - `GET /quality/v3/dashboard` - V3 dashboard summary
  - `GET /quality/v3/trends` - Quality trends over time
  - `GET /quality/v3/alerts` - Quality alerts list
  - `POST /quality/v3/reset` - Reset V3 dashboard
- New models: `V3QualityCheckRequest`, `V3HallucinationResult`, `V3DashboardSummary`, `V3QualityTrend`

### 11. Add Documentation ✅
- Created `CONTRIBUTING.md` - Developer onboarding guide
- Created `docs/DEPLOYMENT.md` - Production deployment guide
- Created `docs/SCHEMA.md` - Database schema documentation

### 12. Expand Slang Map ✅
- Added 50+ Gen-Z slang terms to `jarvis/slang.py`:
  - slay, bussin, mid, sus, cap, no cap, periodt, snatched
  - stan, simp, hits different, main character, ate, bestie, bffr
  - ong, npc, gyat, rizz, valid, based, cringe, vibe/vibes/vibing
  - yeet, ghosted/ghosting, deadass, slaps, fire, lit, goat/goated
  - tea, spill the tea, salty, extra, flex/flexing, thicc, clout
  - fomo, jomo, oof, yikes
- Added tests for new slang terms

## Verification Results

- [x] `pnpm run check` - No errors in changed files
- [x] `ruff check` - All changed Python files pass
- [x] `pytest tests/unit/test_slang.py` - All 15 tests pass

## Files Modified

| File | Changes |
|------|---------|
| `jarvis/index.py` | FAISS safety checks, validity filtering |
| `jarvis/index_v2.py` | FAISS safety checks, validity filtering |
| `jarvis/intent.py` | Query embedding cache |
| `jarvis/response_classifier.py` | Deprecation warnings |
| `jarvis/response_classifier_v2.py` | Added V1 constants for backward compat |
| `jarvis/retrieval.py` | Import from V2 |
| `jarvis/multi_option.py` | Import from V2 |
| `jarvis/slang.py` | 50+ Gen-Z slang terms |
| `api/routers/quality.py` | V3 quality endpoints |
| `desktop/src/App.svelte` | ErrorBoundary import |
| `desktop/src/lib/stores/conversations.ts` | Race condition fixes, LRU cache |
| `desktop/src/lib/components/ErrorBoundary.svelte` | New component |
| `tests/unit/test_response_classifier.py` | Suppress deprecation warnings |
| `tests/unit/test_multi_option.py` | Import from V2 |
| `tests/unit/test_retrieval.py` | Import from V2 |
| `tests/unit/test_slang.py` | Gen-Z slang tests |
| `CONTRIBUTING.md` | New file - developer guide |
| `docs/DEPLOYMENT.md` | New file - deployment guide |
| `docs/SCHEMA.md` | New file - schema documentation |

---

## Classifier Cleanup: V3 Preprocessing-Based Classifier (COMPLETED)

### Goal
Remove old SVM-trained trigger classifier (V1) and wire up preprocessing-based V3 as the new default.

### Archived
- [x] `jarvis/trigger_classifier.py` (V1) → `archive/classifiers/jarvis/trigger_classifier_svm.py`
- [x] `jarvis/index_v2.py` → `archive/`
- [x] `tests/unit/test_index_v2.py` → `archive/`
- [x] Training data files → `archive/data/`
- [x] `scripts/compare_trigger_classifiers.py` → `archive/classifiers/scripts/`

### Renamed V3 to Main
- [x] `trigger_classifier_v3.py` → `trigger_classifier.py`
- [x] `HybridTriggerClassifierV3` → `HybridTriggerClassifier`
- [x] `get_trigger_classifier_v3` → `get_trigger_classifier`
- [x] `reset_trigger_classifier_v3` → `reset_trigger_classifier`
- [x] `classify_trigger_v3` → `classify_trigger`

### Updated Imports
- [x] `scripts/benchmark_small_nli.py`
- [x] `scripts/overnight_batch.py`
- [x] Tests: removed `use_svm` args, skipped SVM threshold tests

### Added Missing Constants
- [x] Added `COMMITMENT_RESPONSE_TYPES` to `response_classifier_v2.py`
- [x] Added `TRIGGER_TO_VALID_RESPONSES` to `response_classifier_v2.py`

### Verified
- [x] Trigger classifier tests pass (53 passed, 4 skipped)
- [x] Manual tests confirm correct classification
