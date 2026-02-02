# JARVIS Architecture

Technical architecture and implementation status for the JARVIS AI assistant.

**Related Documentation:**
- [DESIGN.md](./DESIGN.md) - **Comprehensive design document** with rationale, decisions, and lessons learned (recommended for understanding the "why")
- [ARCHITECTURE_V2.md](./ARCHITECTURE_V2.md) - Direct SQLite + Unix Socket architecture for faster desktop performance
- [CLASSIFIER_SYSTEM.md](./CLASSIFIER_SYSTEM.md) - Deep dive into the hybrid classifier system
- [design/EMBEDDINGS.md](./design/EMBEDDINGS.md) - Embedding models, multi-model support, FAISS
- [design/TEXT_NORMALIZATION.md](./design/TEXT_NORMALIZATION.md) - Text normalization for consistent embeddings

## Quick Overview

JARVIS is a **privacy-first AI assistant** for iMessage on Apple Silicon. Key innovations:

| Feature | Approach | Result |
|---------|----------|--------|
| Classification | 3-layer hybrid (structural → centroid → SVM) | 82% F1 |
| Response Generation | Retrieval-augmented generation (RAG) | Personalized |
| Performance | Unix sockets + direct SQLite (V2) | 30-50x faster |
| Privacy | All local, MLX on Apple Silicon | No cloud |

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Contracts/Interfaces | COMPLETE | 9 protocol definitions in `contracts/` |
| Model Generator (WS8) | COMPLETE | MLX loader, template fallback, RAG support |
| iMessage Reader (WS10) | COMPLETE | Schema detection, attachments, reactions |
| Memory Profiler (WS1) | COMPLETE | MLX memory profiling with model unload |
| HHEM Benchmark (WS2) | COMPLETE | Vectara HHEM model evaluation |
| Latency Benchmark (WS4) | COMPLETE | Cold/warm/hot start scenarios |
| Memory Controller (WS5) | COMPLETE | Three-tier modes (FULL/LITE/MINIMAL) |
| Degradation Controller (WS6) | COMPLETE | Circuit breaker pattern |
| Setup Wizard | COMPLETE | Environment validation, config init, health report |
| CLI Entry Point | COMPLETE | `jarvis/_cli_main.py` with chat, search, reply, summarize, export, serve |
| FastAPI Layer | COMPLETE | `api/` module for Tauri frontend integration |
| Config System | COMPLETE | `jarvis/config.py` with nested sections and migration |
| Model Registry | COMPLETE | `models/registry.py` with multi-model support |
| Intent Classification | COMPLETE | `jarvis/intent.py` with semantic similarity routing |
| Metrics System | COMPLETE | `jarvis/metrics.py` for performance monitoring |
| Export System | COMPLETE | `jarvis/export.py` for JSON/CSV/TXT export |
| Error Handling | COMPLETE | `jarvis/errors.py` unified exception hierarchy |
| Prompts Registry | COMPLETE | `jarvis/prompts.py` centralized prompt templates |
| MLX Embeddings | COMPLETE | `models/embeddings.py` + `jarvis/embedding_adapter.py` multi-model support |
| Reply Router | COMPLETE | `jarvis/router.py` with template/generate/clarify routing |
| FAISS Index | COMPLETE | `jarvis/index.py` for trigger similarity search |
| JARVIS Database | COMPLETE | `jarvis/db.py` with contacts, pairs, clusters |
| Response Classifier | COMPLETE | `jarvis/response_classifier.py` hybrid 3-layer (81.9% F1) |
| Trigger Classifier | COMPLETE | `jarvis/trigger_classifier.py` hybrid structural+SVM (82.0% F1) |
| Multi-Option Generation | COMPLETE | `jarvis/multi_option.py` for AGREE/DECLINE/DEFER |
| Typed Retrieval | COMPLETE | `jarvis/retrieval.py` for DA-filtered FAISS |
| Unix Socket Server | COMPLETE | `jarvis/socket_server.py` for desktop IPC (V2) |
| File Watcher | COMPLETE | `jarvis/watcher.py` for real-time notifications (V2) |

## Contract-Based Design

Python Protocols in `contracts/` enable parallel development:

| Contract | Protocol(s) | Implementation |
|----------|-------------|----------------|
| `contracts/memory.py` | MemoryProfiler, MemoryController | `benchmarks/memory/`, `core/memory/` |
| `contracts/hallucination.py` | HallucinationEvaluator | `benchmarks/hallucination/` |
| `contracts/latency.py` | LatencyBenchmarker | `benchmarks/latency/` |
| `contracts/health.py` | DegradationController, PermissionMonitor, SchemaDetector | `core/health/`, `jarvis/setup.py` |
| `contracts/models.py` | Generator | `models/` |
| `contracts/imessage.py` | iMessageReader | `integrations/imessage/` |

## Module Structure

| Directory | Purpose |
|-----------|---------|
| `jarvis/` | CLI, config, errors, metrics, export, prompts, intent classification, socket server, watcher |
| `api/` | FastAPI REST layer for CLI and web clients |
| `benchmarks/` | Memory, hallucination, latency benchmarks |
| `core/` | Memory controller, health monitoring |
| `models/` | MLX model inference, registry, templates |
| `integrations/imessage/` | iMessage reader with filters |
| `desktop/` | Tauri desktop app (Svelte frontend) with direct SQLite + socket |
| `desktop/src/lib/db/` | Direct SQLite access layer (V2) |
| `desktop/src/lib/socket/` | Unix socket client (V2) |
| `tests/` | Unit and integration tests |

## Key Patterns

### Two Template Systems

1. **Static TemplateMatcher** (`models/templates.py`): ~25 canned response templates using semantic similarity (threshold: 0.70). Supports group chat context.

2. **FAISS ReplyRouter** (`jarvis/router.py`): Matches against historical (trigger, response) pairs from iMessage history. Primary routing system.

### Reply Router Thresholds

Configurable via `~/.jarvis/config.json`:
```json
{
  "routing": {
    "template_threshold": 0.90,
    "context_threshold": 0.70,
    "generate_threshold": 0.50
  }
}
```

- Score >= 0.90: Template response from FAISS
- Score 0.50-0.90: LLM generation with few-shot examples
- Score < 0.50: Clarification request

### Classifiers

**Response Classifier** (`jarvis/response_classifier.py`):
- 3-layer hybrid: structural patterns → centroid verification → SVM
- **81.9% macro F1** [95% CI: 78.4% - 84.9%] on held-out test set
- 6 labels: AGREE, DECLINE, DEFER, OTHER, QUESTION, REACTION
- Model: `~/.jarvis/embeddings/{model_name}/response_classifier_model/`
- Training: `scripts/train_response_classifier.py`

**Trigger Classifier** (`jarvis/trigger_classifier.py`):
- Hybrid: structural patterns → SVM with per-class thresholds
- **82.0% macro F1** [95% CI: 79.3% - 84.4%] on held-out test set
- 5 labels: COMMITMENT, QUESTION, REACTION, SOCIAL, STATEMENT
- Model: `~/.jarvis/embeddings/{model_name}/trigger_classifier_model/`
- Training: `scripts/train_trigger_classifier.py`

**Note:** Classifier models are stored per embedding model. Switching embedding models requires retraining classifiers.

### Singleton Pattern

All expensive resources use lazy-loaded singletons:
- `get_generator()` - MLX model
- `get_embedder()` - Embedding model
- `get_response_classifier()` - Response classifier
- `get_trigger_classifier()` - Trigger classifier
- `get_reply_router()` - FAISS router
- `get_db()` - Database connection
- `get_memory_controller()` - Memory controller
- `get_degradation_controller()` - Circuit breaker

### Data Flow for Text Generation

1. Intent classification → route to handler
2. Message classification → detect type
3. FAISS similarity search → get score
4. Route based on thresholds (template/generate/clarify)
5. Memory check → operating mode
6. Context fetching for iMessage intents
7. Prompt building with tone detection
8. MLX model generation
9. (Optional) HHEM quality validation

## Validation Gates

| Gate | Metric | Pass | Conditional | Fail |
|------|--------|------|-------------|------|
| G1 | Model stack memory | <5.5GB | 5.5-6.5GB | >6.5GB |
| G2 | Mean HHEM score | >=0.5 | 0.4-0.5 | <0.4 |
| G3 | Warm-start latency | <3s | 3-5s | >5s |
| G4 | Cold-start latency | <15s | 15-20s | >20s |

Run: `uv run python -m benchmarks.{memory,hallucination,latency}.run`

## Evaluation Scripts

```bash
# Train/test split and evaluation
uv run python -m scripts.eval_pipeline --setup
uv run python -m scripts.eval_pipeline --limit 100

# Classifier training
uv run python -m scripts.train_response_classifier --save-best
uv run python -m scripts.train_trigger_classifier --save-best

# Quality analysis
uv run python -m scripts.score_pair_quality --analyze
```
