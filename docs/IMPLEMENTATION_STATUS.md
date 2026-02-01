# Implementation Status Tracker

**Last Updated**: 2026-01-31
**Reference Document**: `docs/FROM_SCRATCH_PLAN.md`

This document tracks what has been implemented vs. what the FROM_SCRATCH_PLAN.md outlined, to prevent redoing work and clarify open questions.

---

## Executive Summary

| Phase | Plan Status | Implementation Status | Notes |
|-------|-------------|----------------------|-------|
| Phase 0: Day 1 Validation | COMPLETE | COMPLETE | Gold pair quality validated |
| Phase 1: Data Extraction | COMPLETE | COMPLETE | 140k+ pairs extracted |
| Phase 2: FAISS Indexing | COMPLETE | COMPLETE | 100k vectors indexed |
| Phase 3: Router + API | COMPLETE (in plan) | PARTIAL | Different architecture than plan |
| Phase 4: Evaluation | COMPLETE (in plan) | COMPLETE | Extensive analysis done |

**Key Divergence**: The plan's "Updated Combined Architecture" (multi-option generation with DA-filtered retrieval) was documented but **NOT implemented**. The actual `router.py` uses a simpler approach.

---

## Data Artifacts (DO NOT REGENERATE)

### Extracted Data Files

| File | Location | Size | Records | Status |
|------|----------|------|---------|--------|
| `threaded_conversations.jsonl` | Project root | 23 MB | 17,918 pairs | COMPLETE |
| `semantic_conversations.jsonl` | Project root | 344 MB | 122,567 pairs | COMPLETE |
| `gold_pairs_sample.jsonl` | Project root | 32 MB | Sample set | COMPLETE |

### Database

| Item | Location | Details |
|------|----------|---------|
| Main DB | `~/.jarvis/jarvis.db` | 286 MB, Schema v7 |
| Pairs table | In DB | 132,073 unique pairs (deduplicated) |
| Train/Test split | In DB | 105,659 training / 26,414 holdout (80/20) |
| DA columns | In DB | `trigger_da_type`, `response_da_type`, `cluster_id` |

### FAISS Indexes

| Index | Location | Size | Status |
|-------|----------|------|--------|
| Trigger index | `~/.jarvis/indexes/triggers/bge-small-en-v1.5/` | 146.5 MB each | 14 versions |
| Active version | `20260131-101355` | 100k vectors | ACTIVE |
| DA classifier (trigger) | `~/.jarvis/da_classifiers/trigger/index.faiss` | — | COMPLETE |
| DA classifier (response) | `~/.jarvis/da_classifiers/response/index.faiss` | — | COMPLETE |

### Clustering Data

| File | Location | Contents |
|------|----------|----------|
| Response embeddings | `~/.jarvis/response_clusters/response_embeddings.npy` | 155 MB, 105k × 384 |
| UMAP reduced | `~/.jarvis/response_clusters/reduced_embeddings.npy` | 5D projections |
| HDBSCAN labels | `~/.jarvis/response_clusters/hdbscan_labels.npy` | 240 clusters |

---

## Scripts Status

### All Plan Scripts - IMPLEMENTED

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/extract_threaded_conversations.py` | Extract pairs from Apple threading | COMPLETE |
| `scripts/extract_semantic_conversations.py` | Extract pairs via semantic chunking | COMPLETE |
| `scripts/build_training_index.py` | Merge, dedupe, import, build FAISS | COMPLETE |
| `scripts/evaluate_retrieval.py` | Holdout evaluation | COMPLETE |
| `scripts/cluster_response_types.py` | UMAP + HDBSCAN clustering | COMPLETE |
| `scripts/build_da_classifier.py` | DA classification (SWDA-based) | COMPLETE |
| `scripts/populate_da_and_clusters.py` | Populate DA + cluster columns | COMPLETE |

---

## Classification Systems Analysis

### What the Plan Proposed (Section: "Updated Combined Architecture")

```
STAGE 1: Multi-Signal Classification (parallel)
├── DA Classifier (trigger type: INVITATION, YN_QUESTION, etc.)
├── HDBSCAN Topic Cluster (food, games, scheduling)
└── Structural Features (starts_with_yes → AGREE)

STAGE 2: Valid Response Types
├── TRIGGER_TO_VALID_RESPONSES mapping
└── Filter retrieval by response_da_type

STAGE 3: Type-Filtered FAISS Retrieval
└── Get examples filtered by response DA type

STAGE 4: Multi-Option Generation (Gmail Smart Reply style)
├── Option 1: AGREE
├── Option 2: DECLINE
├── Option 3: DEFER
└── Diversity enforcement (positive/negative balance)
```

### What's Actually Implemented (`jarvis/router.py`)

```
STEP 1: Message Classification (MessageClassifier)
├── Types: QUESTION, STATEMENT, ACKNOWLEDGMENT, REACTION, GREETING, FAREWELL
└── Context requirement: STANDALONE, NEEDS_CONTEXT, VAGUE

STEP 2: Intent Classification (IntentClassifier)
└── Types: REPLY, SUMMARIZE, SEARCH, QUICK_REPLY, GENERAL

STEP 3: Quick-exit paths
├── Acknowledgments → Generic response
├── Reactions → Generic response
├── Greetings → Generic response
└── Vague context → Ask for clarification

STEP 4: FAISS Search
└── Get top-5 similar triggers (no DA filtering)

STEP 5: Threshold-based routing
├── >= 0.95: Template response (pick from matches)
├── 0.65-0.95: LLM generation with examples
├── 0.45-0.65: Cautious LLM generation
└── < 0.45: Clarification request
```

### Gap Analysis

| Plan Component | Planned | Implemented | Gap |
|----------------|---------|-------------|-----|
| DA Classifier integration | In router | Built but NOT used in router | NOT INTEGRATED |
| HDBSCAN topic filtering | Filter FAISS by cluster | Built but NOT used in router | NOT INTEGRATED |
| Type-filtered retrieval | Filter by response_da_type | NOT implemented | MISSING |
| Multi-option generation | 3 options (agree/decline/defer) | NOT implemented | MISSING |
| Diversity enforcement | Force positive/negative balance | NOT implemented | MISSING |
| Structural features | Rule-based backup | PARTIAL (in message_classifier) | PARTIAL |
| Cold start handling | Explicit COLD_START route | NOT explicit | MISSING |
| Skip classification | Explicit SKIP route | Via clarify/acknowledgment paths | PARTIAL |

---

## Files Status

### Files That Should Exist (Per Plan) - NOT CREATED

| File | Purpose | Status |
|------|---------|--------|
| `jarvis/response_classifier.py` | Unified DA + structural classifier | NOT FOUND |
| `jarvis/multi_option.py` | Multi-option generation with diversity | NOT FOUND |
| `jarvis/retrieval.py` | Type-filtered FAISS retrieval | NOT FOUND |

### Files That DO Exist

| File | Purpose | Lines | Notes |
|------|---------|-------|-------|
| `jarvis/router.py` | Reply routing | 1,426 | Different architecture than plan |
| `jarvis/message_classifier.py` | Message type classification | — | Partial coverage of plan |
| `jarvis/intent.py` | Intent classification | — | Not in original plan |
| `jarvis/index.py` | FAISS index operations | — | No DA filtering |
| `jarvis/db.py` | Database with DA columns | ~2,000 | Schema v7 with DA columns |

---

## Open Questions from Plan (Status Update)

### Question 1: Memory Budget
> "Does FAISS + embedder + LLM fit in 8GB?"

**Status**: PARTIALLY ANSWERED
- Memory profiling infrastructure exists (`benchmarks/memory/`)
- Overnight evaluation runs exist
- **Specific measurement needed**: Run `uv run python -m benchmarks.memory.run` to get current numbers

### Question 2: Incremental Updates
> "Best strategy for adding new pairs?"

**Status**: NOT ADDRESSED
- No incremental update mechanism in place
- Index rebuild is full rebuild currently
- **Recommendation**: Design incremental update in Phase 3 if needed

### Question 3: Desktop App Needs
> "What endpoints does Tauri app actually require?"

**Status**: OVER-ADDRESSED
- 29 API routers implemented
- Likely more than needed for V1
- **Recommendation**: Audit which endpoints are actually used by desktop app

### Question 4: STATEMENT Problem
> "78% of responses are STATEMENT - need better ANSWER/AGREE/DECLINE exemplars"

**Status**: DOCUMENTED BUT NOT RESOLVED
- Problem well-documented in FROM_SCRATCH_PLAN.md
- DA classifier still has this issue
- **Recommendation**: Mine non-STATEMENT clusters (7, 9, 32, 12, 45, 80) for better exemplars

### Question 5: Cross-Encoder for V2
> "Research suggests +20-40% accuracy, but latency concerns for short text"

**Status**: DEFERRED (correct per plan)
- V1 uses FAISS only (no reranker)
- Decision to be made based on V1 performance

### Question 6: User Preference Learning
> "How to learn from which option user picks over time?"

**Status**: NOT ADDRESSED
- No multi-option generation → no preference learning
- **Blocker**: Requires multi-option generation first

### Question 7: Topic Cluster Granularity
> "240 clusters vs fewer broader topics - what's optimal?"

**Status**: DATA EXISTS, NOT ANALYZED
- 240 HDBSCAN clusters created
- Purity analysis done
- **Recommendation**: Test with ~50 broader clusters if needed

---

## Recommended Next Steps

### Option A: Implement Plan's Multi-Option Architecture

If the goal is to match the FROM_SCRATCH_PLAN.md:

1. **Create `jarvis/response_classifier.py`**
   - Combine DA classifier + structural features
   - Use existing DA classifier indexes in `~/.jarvis/da_classifiers/`

2. **Create `jarvis/retrieval.py`**
   - Add `get_typed_examples()` function
   - Filter FAISS results by `response_da_type`

3. **Create `jarvis/multi_option.py`**
   - Implement `generate_response_options()`
   - Add diversity enforcement

4. **Modify `jarvis/router.py`**
   - Integrate DA classification into routing
   - Return multiple options instead of single response

5. **Modify `api/routers/drafts.py`**
   - Return array of options
   - Update response schema

### Option B: Accept Current Architecture

If the current implementation is sufficient:

1. **Update FROM_SCRATCH_PLAN.md**
   - Mark "Updated Combined Architecture" as FUTURE/DEFERRED
   - Document current simpler architecture

2. **Focus on Quality**
   - Tune thresholds (currently 0.95/0.65/0.45)
   - Improve generation prompts
   - Better acknowledgment handling

3. **Clean Up**
   - Remove unused DA classifier infrastructure if not needed
   - Simplify to what's actually used

---

## Decision Log Update

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-31 | Document plan vs implementation gap | Prevent confusion and rework |
| — | PENDING: Choose Option A or B | User decision needed |

---

## Files to NOT Regenerate

These artifacts exist and should not be recreated unless data has changed:

1. `threaded_conversations.jsonl` - 17,918 pairs
2. `semantic_conversations.jsonl` - 122,567 pairs
3. `gold_pairs_sample.jsonl` - Sample set
4. `~/.jarvis/jarvis.db` - 132,073 pairs, schema v7
5. `~/.jarvis/indexes/triggers/*/` - FAISS indexes (14 versions)
6. `~/.jarvis/da_classifiers/` - DA classifier indexes
7. `~/.jarvis/response_clusters/` - Clustering artifacts
8. Train/test split - Already done in DB (`is_holdout` column)
