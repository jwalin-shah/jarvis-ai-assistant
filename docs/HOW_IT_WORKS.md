# How JARVIS Works

> **Last Updated:** 2026-02-12

This document explains the JARVIS system architecture, services, and message flow.

---

## System Overview

JARVIS is a local-first AI assistant for iMessage on macOS. It reads messages, retrieves similar past conversations, and generates reply suggestions—all on-device.

```
┌─────────────────────────────────────────────────────────────┐
│  Tauri Desktop App                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Direct SQLite│  │ Socket Client│  │ HTTP Client  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼─────────────────┼─────────────────┼──────────────┘
          │                 │                 │
          │ ~1-5ms         │ ~1-5ms          │ ~50-150ms
          ▼                 ▼                 ▼
    ┌──────────┐    ┌──────────────┐   ┌─────────────┐
    │ chat.db  │    │ Socket Server│   │ FastAPI     │
    │ jarvis.db│    │ (Generation) │   │ (REST API)  │
    └──────────┘    └──────┬───────┘   └─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │  MLX Embedding Service   │
              │  (GPU-accelerated)       │
              └──────────────────────────┘
```

---

## Services

| Service | Port/Socket | Purpose |
|---------|-------------|---------|
| **FastAPI Backend** | HTTP `localhost:8742` | REST API for CLI and web clients |
| **Socket Server** | Unix `~/.jarvis/jarvis.sock` + WS `:8743` | Desktop IPC, LLM generation, search |
| **MLX Embedding** | `~/.jarvis/jarvis-embed.sock` | GPU-accelerated embeddings |
| **NER Server** | `~/.jarvis/jarvis-ner.sock` | Named entity recognition (optional) |
| **Desktop App** | Standalone | Tauri UI with direct SQLite access |

### Starting Services

```bash
# All services
make services-start

# Individual
uv run python -m jarvis.socket_server    # Socket server
uvicorn api.main:app --port 8742         # FastAPI
```

---

## Message Flow: End-to-End

### Phase 1: Preprocessing (Offline)

1. **Read messages** from `~/Library/Messages/chat.db`
2. **Detect topic boundaries** using linguistic features
3. **Create topic chunks** (groups of related messages)
4. **Embed chunks** → 384-dim vectors via MLX
5. **Index into sqlite-vec** for semantic search

```bash
uv run python scripts/preprocess_chunks.py --rebuild-index
```

### Phase 2: Real-Time Generation

When a new message arrives:

```
Incoming: "Want to grab lunch tomorrow?"
    │
    ├─→ 1. Encode query → embedding (384-dim)
    │
    ├─→ 2. sqlite-vec search → top 3 similar chunks
    │      └─→ "Planning Lunch" chunk with past conversations
    │
    ├─→ 3. Build prompt
    │      ├─→ System instructions
    │      ├─→ Similar chunks (style reference)
    │      └─→ Recent thread context
    │
    └─→ 4. LLM Generate (MLX, ~2-3s)
           └─→ "Sure! What time works?"
```

---

### Phase 3: Fact Extraction (Background)

When new messages arrive, the watcher extracts structured facts using a two-stage pipeline: high-recall candidate extraction followed by quality filtering.

```
Incoming: "My sister Sarah just started at Google in Austin"
    │
    ├─→ 1. Candidate extraction (two paths, merged)
    │      ├─→ GLiNER NER (zero-shot, urchade/gliner_medium-v2.1)
    │      │      ├─→ person_name: "Sarah"
    │      │      ├─→ org: "Google"
    │      │      └─→ place: "Austin"
    │      └─→ Rule-based regex patterns
    │             ├─→ Relationship: "sister Sarah" (is_family_of)
    │             ├─→ Work: "started at Google" (works_at)
    │             └─→ Location: "in Austin" (lives_in)
    │
    ├─→ 2. Quality filtering (4-filter pipeline, precision 37%→80%+)
    │      ├─→ Bot detection (CVS/Rx/LinkedIn/URL patterns)
    │      ├─→ Vague subject rejection (pronouns: me, you, that, etc.)
    │      ├─→ Short phrase filtering (min 3 words for prefs, 2 for others)
    │      └─→ Confidence recalibration (threshold ≥0.5)
    │
    ├─→ 3. NLI verification (optional, via MLX DeBERTa-v3)
    │      ├─→ "Sarah is a family member" → entailment (0.94) ✓
    │      ├─→ "Someone works at Google" → entailment (0.88) ✓
    │      └─→ "Someone lives in Austin" → entailment (0.91) ✓
    │
    ├─→ 4. NER person linking (fuzzy Jaccard match to known contacts)
    │      └─→ "Sarah" → linked_contact_id (if match ≥0.7)
    │
    └─→ 5. Persist to contact_facts table (dedup by UNIQUE constraint)
           └─→ Knowledge graph queries now return these facts
```

**GLiNER Model**: `urchade/gliner_medium-v2.1` (DeBERTa backbone, ~1.5GB). 9 span labels: person_name, family_member, place, org, date_ref, food_item, job_role, health_condition, activity. Requires `transformers<5`.

**NLI Model**: `cross-encoder/nli-deberta-v3-xsmall` (22M params, 87.77% MNLI accuracy, ~90MB in memory). Implemented as a pure MLX DeBERTa-v3 with disentangled attention. We chose this over using the LLM for entailment because:
- Dedicated NLI models are more accurate for entailment tasks than zero-shot LLM prompting
- 90MB memory footprint vs 1.2GB LLM (can run alongside other models)
- Batch inference is fast (~5ms per pair on M-series chips)
- No prompt engineering required; trained specifically on SNLI + MultiNLI

```bash
# Backfill facts from historical messages
uv run python scripts/backfill_contact_facts.py --max-contacts 50

# Without NLI verification (faster, less precise)
uv run python scripts/backfill_contact_facts.py --no-nli
```

---

## Key Components

### Topic Chunks (Not Pairs)

JARVIS uses **topic chunks** for RAG—multi-message segments about a single topic:

```
Topic: "Planning Lunch"
Messages: [
  "Want to grab lunch tomorrow?",
  "Sure! What time works?",
  "How about 12pm?",
  "Perfect, see you then!"
]
```

**Why chunks?** They capture conversational flow and style, not just Q→A patterns.

### sqlite-vec Search

- **Implementation**: Virtual tables in `jarvis.db`
- **Tables**: `vec_chunks` (int8) and `vec_binary` (sign-bit)
- **Search**: ~1-5ms for per-contact partitioned search

### Generation

- **Default Model**: LFM-2.5-1.2B-Instruct-4bit
- **Latency**: ~2-3s warm, 10-15s cold start
- **Memory**: ~1.2GB model + ~200MB embeddings
- **Model Registry**: See `models/registry.py` for all available models (Qwen, Phi-3, Gemma-3, LFM variants)

### Fact Extraction

- **Extraction Model**: LFM-2.5-350M-extract-4bit (specialized fine-tuned variant)
- **NER**: GLiNER (`urchade/gliner_medium-v2.1`) for named entity recognition
- **Verification**: MLX NLI cross-encoder for entailment checking

---

## Performance

| Operation | P50 | Target |
|-----------|-----|--------|
| Intent classification | 12ms | <50ms |
| sqlite-vec search | 3ms | <50ms |
| LLM generation | 180ms/token | <2s total |
| **Full pipeline** | **250ms** | **<3s** |

---

## Storage

| Location | Purpose |
|----------|---------|
| `~/Library/Messages/chat.db` | iMessage DB (read-only) |
| `~/.jarvis/jarvis.db` | JARVIS data (chunks, pairs, and vector index) |
| `~/.jarvis/embeddings/` | Model-specific artifacts and embeddings.db |
| `~/.jarvis/config.json` | User configuration |

---

## Model Registry

JARVIS supports multiple LLM models via `models/registry.py`. The registry automatically selects the best model based on available RAM.

| Model ID | Display Name | Size | Quality | Best For |
|----------|--------------|------|---------|----------|
| `lfm-1.2b` | LFM 2.5 1.2B (Conversational) | 1.2GB | Excellent | iMessage, quick replies |
| `lfm-1.2b-thinking` | LFM 2.5 1.2B (Thinking) | 1.2GB | Excellent | Complex reasoning |
| `lfm-1.2b-ft` | LFM 2.5 1.2B Fine-Tuned | 1.2GB | Excellent | Best for texting |
| `gemma3-4b` | Gemma 3 4B Instruct | 2.75GB | Excellent | Natural conversation |
| `qwen-3b` | Qwen 2.5 3B | 2.5GB | Excellent | Complex replies |
| `qwen-1.5b` | Qwen 2.5 1.5B | 1.5GB | Good | Balanced |
| `qwen-0.5b` | Qwen 2.5 0.5B | 0.8GB | Basic | Fast responses |
| `lfm-0.3b` | LFM 2.5 0.3B | 0.3GB | Basic | Testing |

**Extraction Models** (specialized for fact extraction):
- `lfm-350m`: `mlx-community/LFM2-350M-4bit` - 350M parameter base model for fast extraction

---

## Observability

JARVIS includes a structured observability system for debugging and monitoring:

| Module | Purpose |
|--------|---------|
| `jarvis/observability/logging.py` | Structured logging with `log_event()` and `timed_operation()` |
| `jarvis/observability/metrics_router.py` | Routing metrics collection and storage |
| `jarvis/observability/insights.py` | Sentiment analysis and conversation insights |

**Usage:**
```python
from jarvis.observability.logging import log_event, timed_operation

# Log an event
log_event("fact_extraction", contact_id="chat123", facts_found=5)

# Time an operation
with timed_operation("model_inference", model_id="lfm-350m") as ctx:
    result = generate_reply(...)
    ctx["tokens_generated"] = len(result)
```
