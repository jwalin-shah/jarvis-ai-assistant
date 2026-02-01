# Refactor Map: Current → New Architecture

## Current jarvis/ Files (51 files)

### DELETE (Obsolete or Replaced)

| File | Lines | Reason |
|------|-------|--------|
| `_cli_main.py` | 3,151 | Replace with minimal CLI |
| `cli/` | ~600 | Half-baked refactor |
| `cli_examples.py` | ~200 | Not needed |
| `prompts.py` | 1,940 | Replace with retrieval-based |
| `fallbacks.py` | ~200 | Hardcoded fallbacks, not needed |
| `simple_reply.py` | ~150 | Superseded by new router |
| `experiments.py` | ~300 | Dev experiments |
| `pdf_generator.py` | ~200 | Not core functionality |
| `digest.py` | ~300 | Not core functionality |
| `insights.py` | ~400 | Not core functionality |
| `metrics_router.py` | ~200 | Consolidate into metrics |
| `metrics_validation.py` | ~200 | Consolidate into metrics |
| `pair_balancer.py` | ~200 | Unused |
| `tasks/` | ~400 | Background tasks, not needed initially |

**Total to delete: ~8,500 lines**

### KEEP (Works Well)

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 50 | Package exports |
| `__main__.py` | 20 | Entry point |
| `config.py` | 400 | Pydantic config (keep) |
| `errors.py` | 300 | Exception hierarchy (keep) |
| `db.py` | 1,800 | Database (keep, will extend) |

### KEEP & REFACTOR (Good Base, Needs Updates)

| File | Lines | What to Change |
|------|-------|----------------|
| `extract.py` | 1,447 | Add threaded extraction, semantic chunking |
| `index.py` | 400 | Add reranker support |
| `router.py` | 1,415 | Simplify, use new retrieval pipeline |
| `intent.py` | 500 | Refactor to learn from YOUR data |
| `embeddings.py` | 1,000 | Keep MLX embeddings |
| `embedding_adapter.py` | 300 | Keep adapter pattern |
| `context.py` | 400 | Keep context fetching |
| `generation.py` | 300 | Simplify |
| `system.py` | 200 | Keep system init |
| `setup.py` | 400 | Keep setup wizard |
| `export.py` | 300 | Keep export functionality |
| `metrics.py` | 500 | Keep metrics |
| `semantic_search.py` | 400 | Keep, use for retrieval |

### EVALUATE (May Keep or Merge)

| File | Lines | Decision Needed |
|------|-------|-----------------|
| `relationship_classifier.py` | ? | Keep if good, else rewrite |
| `relationships.py` | 1,314 | Useful for profiling |
| `message_classifier.py` | 300 | Merge into intent? |
| `quality_metrics.py` | 500 | Keep for validation |
| `evaluation.py` | 500 | Keep for validation |
| `validity_gate.py` | 300 | Merge into extraction filters |
| `cluster.py` | 400 | Keep for intent clustering |
| `text_normalizer.py` | 200 | Keep utility |
| `threading.py` | 300 | Keep for conversation threads |
| `exchange.py` | 300 | Merge into extraction |
| `embedding_profile.py` | 400 | Keep for profiling |
| `model_warmer.py` | 200 | Keep for performance |
| `api_models.py` | 200 | Keep pydantic models |
| `api.py` | 200 | Keep async utilities |
| `retry.py` | 100 | Keep retry logic |
| `priority.py` | 200 | Keep for message priority |

---

## New Directory Structure

```
jarvis/
├── __init__.py                  # Keep
├── __main__.py                  # Keep
├── cli.py                       # NEW: Minimal CLI (300 lines)
├── config.py                    # Keep
├── errors.py                    # Keep
├── db.py                        # Keep & extend
│
├── extraction/                  # NEW PACKAGE
│   ├── __init__.py
│   ├── threaded.py              # NEW: Gold pair extraction
│   ├── semantic.py              # NEW: Semantic topic segmentation
│   ├── chunker.py               # NEW: Topic boundary detection
│   └── filters.py               # MERGE: From validity_gate.py + extract.py
│
├── profiling/                   # NEW PACKAGE
│   ├── __init__.py
│   ├── relationship.py          # REFACTOR: From relationship_classifier.py
│   ├── intent.py                # REFACTOR: From intent.py
│   ├── style.py                 # NEW: Your texting style analysis
│   ├── response_time.py         # NEW: Response time modeling
│   └── tone.py                  # NEW: Tone/energy detection
│
├── retrieval/                   # NEW PACKAGE
│   ├── __init__.py
│   ├── embedder.py              # REFACTOR: From embeddings.py
│   ├── index.py                 # REFACTOR: From index.py
│   ├── reranker.py              # NEW: Cross-encoder reranking
│   └── pipeline.py              # NEW: Two-stage orchestration
│
├── generation/                  # NEW PACKAGE
│   ├── __init__.py
│   ├── prompts.py               # NEW: Minimal prompts (50 lines)
│   └── generator.py             # REFACTOR: From generation.py
│
├── clarification/               # NEW PACKAGE
│   ├── __init__.py
│   ├── detector.py              # NEW: Detect vague messages
│   └── responses.py             # NEW: Clarifying questions (from YOUR data)
│
├── tracking/                    # NEW PACKAGE
│   ├── __init__.py
│   ├── commitments.py           # NEW: Promise/commitment tracking
│   ├── conversation_memory.py   # NEW: Recent topic memory
│   ├── relationship_health.py   # NEW: Relationship health monitoring
│   └── rejections.py            # NEW: Learn from rejected suggestions
│
├── validation/                  # NEW PACKAGE
│   ├── __init__.py
│   ├── metrics.py               # REFACTOR: From quality_metrics.py
│   ├── evaluation.py            # REFACTOR: From evaluation.py
│   └── dashboard.py             # NEW: Verification dashboard
│
└── router.py                    # REFACTOR: Simplified routing
```

---

## Refactoring Steps

### Step 1: Clean Up (Delete Cruft)

```bash
# Delete obsolete files
rm jarvis/_cli_main.py
rm jarvis/cli_examples.py
rm jarvis/fallbacks.py
rm jarvis/simple_reply.py
rm jarvis/experiments.py
rm jarvis/pdf_generator.py
rm jarvis/digest.py
rm jarvis/insights.py
rm jarvis/metrics_router.py
rm jarvis/metrics_validation.py
rm jarvis/pair_balancer.py
rm -rf jarvis/cli/
rm -rf jarvis/tasks/

# Keep prompts.py temporarily until new system is ready
```

### Step 2: Create New Package Structure

```bash
mkdir -p jarvis/extraction
mkdir -p jarvis/profiling
mkdir -p jarvis/retrieval
mkdir -p jarvis/generation
mkdir -p jarvis/clarification
mkdir -p jarvis/tracking
mkdir -p jarvis/validation

# Create __init__.py files
touch jarvis/extraction/__init__.py
touch jarvis/profiling/__init__.py
touch jarvis/retrieval/__init__.py
touch jarvis/generation/__init__.py
touch jarvis/clarification/__init__.py
touch jarvis/tracking/__init__.py
touch jarvis/validation/__init__.py
```

### Step 3: Create Minimal CLI

Replace 3,151 line CLI with ~300 line version:
- `jarvis serve`
- `jarvis db init|extract-gold|extract-semantic|build-index|stats`
- `jarvis profile <contact>` (NEW)
- `jarvis validate` (NEW)
- `jarvis health`
- `jarvis benchmark`

### Step 4: Implement Extraction Package

Start with gold pair extraction (thread_originator_guid).

### Step 5: Implement Profiling Package

Build relationship classifier, intent classifier from YOUR data.

### Step 6: Implement Retrieval Package

FAISS + cross-encoder reranker.

### Step 7: Wire Up Router

Connect everything into the routing logic.

---

## Migration Strategy

**Approach: Parallel Development**

1. Keep old code working while building new
2. Feature flag to switch between old/new
3. Validate new system matches or beats old
4. Delete old code only after new is validated

```python
# In router.py during migration
USE_NEW_PIPELINE = os.getenv("JARVIS_NEW_PIPELINE", "false") == "true"

if USE_NEW_PIPELINE:
    from jarvis.retrieval.pipeline import retrieve_and_rank
    from jarvis.profiling.intent import classify_intent
else:
    # Old code path
    from jarvis.index import search_similar
```
