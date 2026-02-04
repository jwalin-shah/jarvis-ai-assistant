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

**Legend:**
- ✅ **OPERATIONAL** - Code implemented, tested, and working in production
- 🟡 **IMPLEMENTED** - Code exists but has known issues or limitations
- 📝 **PLANNED** - Design complete, implementation pending

### Core Components

| Component | Status | Notes |
|-----------|--------|-------|
| Contracts/Interfaces | ✅ OPERATIONAL | 9 protocol definitions in `contracts/` |
| Model Generator (WS8) | ✅ OPERATIONAL | MLX loader, template fallback, RAG support |
| iMessage Reader (WS10) | ✅ OPERATIONAL | Schema detection, attachments, reactions |
| Memory Profiler (WS1) | ✅ OPERATIONAL | MLX memory profiling with model unload |
| HHEM Benchmark (WS2) | ✅ OPERATIONAL | Vectara HHEM model evaluation |
| Latency Benchmark (WS4) | ✅ OPERATIONAL | Cold/warm/hot start scenarios |
| Memory Controller (WS5) | ✅ OPERATIONAL | Three-tier modes (FULL/LITE/MINIMAL) |
| Degradation Controller (WS6) | ✅ OPERATIONAL | Circuit breaker pattern |
| Setup Wizard | ✅ OPERATIONAL | Environment validation, config init, health report |
| CLI Entry Point | ✅ OPERATIONAL | `jarvis/_cli_main.py` with db, health, benchmark, serve |
| FastAPI Layer | ✅ OPERATIONAL | `api/` module for Tauri frontend integration |
| Config System | ✅ OPERATIONAL | `jarvis/config.py` with nested sections and migration |
| Model Registry | ✅ OPERATIONAL | `models/registry.py` with multi-model support |
| Intent Classification | 🟡 IMPLEMENTED | `jarvis/intent.py` - Under active development |
| Metrics System | ✅ OPERATIONAL | `jarvis/metrics.py` for performance monitoring |
| Export System | ✅ OPERATIONAL | `jarvis/export.py` for JSON/CSV/TXT export |
| Error Handling | ✅ OPERATIONAL | `jarvis/errors.py` unified exception hierarchy |
| Prompts Registry | ✅ OPERATIONAL | `jarvis/prompts.py` centralized prompt templates |
| MLX Embeddings | ✅ OPERATIONAL | `models/embeddings.py` + `jarvis/embedding_adapter.py` multi-model support |
| Reply Router | ✅ OPERATIONAL | `jarvis/router.py` with template/generate/clarify routing |
| FAISS Index | ✅ OPERATIONAL | `jarvis/index.py` for trigger similarity search |
| JARVIS Database | ✅ OPERATIONAL | `jarvis/db.py` with contacts, pairs, clusters |
| Cluster Analysis | 🟡 IMPLEMENTED | `jarvis/clustering.py` - code exists, CLI command planned |
| Response Classifier | 🟡 IMPLEMENTED | `jarvis/response_classifier.py` - Under active refinement |
| Trigger Classifier | ✅ OPERATIONAL | `jarvis/trigger_classifier.py` hybrid structural+SVM (82.0% F1) |
| Multi-Option Generation | ✅ OPERATIONAL | `jarvis/multi_option.py` for AGREE/DECLINE/DEFER |
| Typed Retrieval | ✅ OPERATIONAL | `jarvis/retrieval.py` for DA-filtered FAISS |
| Unix Socket Server | ✅ OPERATIONAL | `jarvis/socket_server.py` for desktop IPC (V2) |
| File Watcher | ✅ OPERATIONAL | `jarvis/watcher.py` for real-time notifications (V2) |

### New Modules (V3)

| Component | Status | Notes |
|-----------|--------|-------|
| Analytics Module | ✅ OPERATIONAL | `jarvis/analytics/` - dashboard, trends, reports |
| Graph Visualization | 🟡 IMPLEMENTED | `jarvis/graph/` - API ready, frontend integration planned |
| Scheduler System | ✅ OPERATIONAL | `jarvis/scheduler/` - smart timing, draft scheduling |
| Tags & Smart Folders | ✅ OPERATIONAL | `jarvis/tags/` - auto-tagging, rule-based folders |
| Prefetch System | ✅ OPERATIONAL | `jarvis/prefetch/` - speculative caching, prediction |
| Quality Assurance | ✅ OPERATIONAL | `jarvis/quality/` - hallucination detection, grounding |
| Response Classifier V2 | 🟡 IMPLEMENTED | `jarvis/response_classifier_v2.py` - Batch processing, under testing |
| Sharded FAISS Index V2 | 🟡 IMPLEMENTED | `jarvis/index_v2.py` - Tiered storage, time sharding |
| Adaptive Thresholds | 🟡 IMPLEMENTED | `jarvis/adaptive_thresholds.py` - Learns from user feedback |

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
| `jarvis/` | Core logic: classifiers, router, embeddings, config, prompts |
| `jarvis/analytics/` | Dashboard metrics, trends, reports, time-series aggregation |
| `jarvis/graph/` | Relationship networks, clustering, layout, export (JSON/SVG/HTML) |
| `jarvis/scheduler/` | Draft scheduling, smart timing, priority queue, quiet hours |
| `jarvis/tags/` | Tags, smart folders, auto-tagging, rule-based filtering |
| `jarvis/prefetch/` | Multi-tier cache (L1/L2/L3), prediction, invalidation |
| `jarvis/quality/` | Hallucination detection, factuality, consistency, grounding |
| `api/` | FastAPI REST layer for CLI and web clients |
| `api/routers/` | API endpoints (analytics, graph, scheduler, tags, etc.) |
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

**Response Classifier V2** (`jarvis/response_classifier_v2.py`):
- Batch processing: 32-128 messages at once with vectorized operations
- Parallel SVM: joblib for multi-core predictions
- Streaming: Micro-batching with 50ms windows
- Extended labels: QUESTION subtypes, EMOTIONAL_SUPPORT, SCHEDULING
- Target: 10x throughput, <5ms p95 latency

### Sharded FAISS Index V2

`jarvis/index_v2.py` provides tiered storage architecture:

| Tier | Description | Access Pattern |
|------|-------------|----------------|
| HOT | Frequently accessed | Memory-mapped, always loaded |
| WARM | Recently accessed | Loaded on demand |
| COLD | Archived data | Compressed, rarely accessed |

**Features:**
- Time-based sharding (monthly shards)
- Cross-shard search with result merging
- Atomic updates with journaling
- Corruption detection and auto-repair
- Background index warming

**Storage:** `~/.jarvis/indexes_v2/<model_name>/shards/`

### Adaptive Thresholds

`jarvis/adaptive_thresholds.py` learns optimal routing thresholds from user feedback:

1. Groups feedback by similarity score buckets (e.g., 0.90-0.95)
2. Computes acceptance rate per bucket
3. Adjusts thresholds where acceptance drops below target
4. Applies learning rate to prevent volatility

```python
from jarvis.adaptive_thresholds import get_adaptive_threshold_manager
manager = get_adaptive_threshold_manager()
thresholds = manager.get_adapted_thresholds()
# {"quick_reply": 0.92, "context": 0.68, "generate": 0.48}
```

## V3 Modules

### Analytics (`jarvis/analytics/`)

Conversation analytics and insights:
- **TimeSeriesAggregator**: Pre-computed daily/hourly aggregates
- **TrendAnalyzer**: Trend detection, anomalies, seasonality
- **ReportGenerator**: Export to JSON/CSV
- **AnalyticsEngine**: Dashboard overview metrics (response time, sentiment)

### Graph (`jarvis/graph/`)

Relationship network visualization:
- **GraphBuilder**: Build network from message history
- **LayoutEngine**: Force-directed, hierarchical, radial layouts
- **detect_communities()**: Louvain clustering with auto-labels
- **Export**: JSON, GraphML, SVG, interactive HTML (D3.js)

### Scheduler (`jarvis/scheduler/`)

Smart draft scheduling:
- **DraftScheduler**: Priority queue with exponential backoff
- **TimingAnalyzer**: Contact history analysis for optimal send times
- **Quiet hours**: Configurable per-contact time zones
- Background executor with retry logic

### Tags (`jarvis/tags/`)

Organization system:
- **TagManager**: Hierarchical tags with colors/icons
- **SmartFolder**: Rule-based dynamic folders (like mail filters)
- **AutoTagger**: ML-based tag suggestions from content
- **RulesEngine**: Field/operator/value conditions

### Prefetch (`jarvis/prefetch/`)

Speculative caching for low latency:
- **MultiTierCache**: L1 (memory) / L2 (disk) / L3 (compressed)
- **PrefetchPredictor**: Strategies (contact frequency, time of day, UI focus)
- **PrefetchExecutor**: Background execution with resource management
- **CacheInvalidator**: Smart invalidation on new messages
- Target: 80% cache hit rate, 10x latency improvement

### Quality (`jarvis/quality/`)

Response quality assurance:
- **EnsembleHallucinationDetector**: Multi-model ensemble
- **FactChecker**: Claim extraction and verification
- **ConsistencyChecker**: Self-consistency across generations
- **GroundingChecker**: Source attribution tracking
- **QualityGate**: Real-time pass/warn/fail thresholds
- **QualityDashboard**: Trend tracking, regression alerts

### Frontend Redesign

See [FRONTEND_REDESIGN_PLAN.md](./FRONTEND_REDESIGN_PLAN.md):
- **Themes**: Light/Dark/System with CSS variables
- **Skeleton loading**: Shimmer placeholders for perceived speed
- **Optimistic sending**: Instant message feedback
- **Glassmorphism**: Frosted glass effects
- **Command palette**: Keyboard-first navigation (planned)

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
2. Simple acknowledgment check → fast-path for "ok", "sounds good", etc.
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

## API Endpoints (V3)

### Analytics API (`/api/analytics/*`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/overview` | GET | Dashboard metrics (messages, response time, sentiment) |
| `/timeline` | GET | Time-series data (hour/day/week/month granularity) |
| `/heatmap` | GET | Activity heatmap for calendar view |
| `/contacts/{chat_id}/stats` | GET | Per-contact detailed analytics |
| `/contacts/leaderboard` | GET | Top contacts by messages/engagement |
| `/trends` | GET | Trend detection, anomalies, seasonality |
| `/export` | GET | Export analytics (JSON/CSV) |

### Graph API (`/api/graph/*`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/network` | GET | Full relationship network graph |
| `/ego/{contact_id}` | GET | Ego-centric graph centered on contact |
| `/clusters` | GET | Community detection (Louvain algorithm) |
| `/evolution` | GET | Temporal graph snapshots |
| `/export` | POST | Export graph (JSON/GraphML/SVG/HTML) |
| `/stats` | GET | Network statistics |

### Scheduler API (`/api/scheduler/*`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/schedule` | POST | Schedule draft for future sending |
| `/smart-schedule` | POST | Schedule with smart timing analysis |
| `/{item_id}` | GET | Get scheduled item |
| `/{item_id}` | DELETE | Cancel scheduled item |
| `/{item_id}/reschedule` | PUT | Reschedule to new time |
| `/timing/suggest/{contact_id}` | GET | Get timing suggestions |
| `/stats` | GET | Scheduler statistics |
| `/start`, `/stop` | POST | Control background scheduler |

### Tags API (`/api/tags/*`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET/POST | List/create tags |
| `/{tag_id}` | GET/PATCH/DELETE | Tag CRUD |
| `/conversations/{chat_id}` | GET/POST/PUT | Conversation tagging |
| `/bulk/add`, `/bulk/remove` | POST | Bulk tag operations |
| `/folders` | GET/POST | Smart folder CRUD |
| `/folders/{folder_id}/conversations` | GET | Folder contents |
| `/folders/preview` | POST | Preview folder rules |
| `/rules` | GET/POST | Auto-tagging rules |
| `/suggestions` | POST | AI-powered tag suggestions |
| `/statistics` | GET | Tag usage statistics |

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
