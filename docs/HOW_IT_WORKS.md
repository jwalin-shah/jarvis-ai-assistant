# How JARVIS Works

> **Last Updated:** 2026-02-16

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

| Service             | Port/Socket                               | Purpose                             |
| ------------------- | ----------------------------------------- | ----------------------------------- |
| **FastAPI Backend** | HTTP `localhost:8742`                     | REST API for CLI and web clients    |
| **Socket Server**   | Unix `~/.jarvis/jarvis.sock` + WS `:8743` | Desktop IPC, LLM generation, search |
| **MLX Embedding**   | `~/.jarvis/jarvis-embed.sock`             | GPU-accelerated embeddings          |
| **NER Server**      | `~/.jarvis/jarvis-ner.sock`               | Named entity recognition (optional) |
| **Desktop App**     | Standalone                                | Tauri UI with direct SQLite access  |

### Starting Services

```bash
# All services
make services-start

# Individual
uv run python -m jarvis.interfaces.desktop    # Socket server
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
    ├─→ 1. Classify mobilization level (is reply needed?)
    │      └─→ HIGH: question/request requiring response
    │
    ├─→ 2. Encode query → embedding (384-dim)
    │
    ├─→ 3. sqlite-vec search → top 3 similar chunks
    │      └─→ "Planning Lunch" chunk with past conversations
    │
    ├─→ 4. Build prompt (UNIVERSAL system prompt)
    │      ├─→ "You are NOT an AI assistant..."
    │      ├─→ Similar chunks (style reference)
    │      └─→ Recent thread context
    │
    └─→ 5. LLM Generate (MLX, ~300ms)
           └─→ "sure what time?"
```

**Note:** We use a **universal prompt** for all categories. Category-specific prompts were removed after ablation study showed they hurt quality. See [Categorization Ablation Findings](./research/CATEGORIZATION_ABLATION_FINDINGS.md).

---

### Phase 3: Fact Extraction (Background) — V4 Pipeline

When new messages arrive, the watcher extracts structured facts using the **V4 instruction-based pipeline**: turn-based grouping, LFM extraction, and two-pass self-correction.

```
Incoming messages for a chat
    │
    ├─→ 1. Group into turns (consecutive messages from same sender)
    │      └─→ "Jwalin: My sister Sarah just started at Google in Austin"
    │          "Contact: That's awesome! When did she move?"
    │
    ├─→ 2. Resolve identities (Address Book + jarvis.db contacts)
    │      ├─→ User: "Jwalin Shah" (from macOS Address Book)
    │      └─→ Contact display name from contacts table
    │
    ├─→ 3. Instruction-based extraction (LFM-0.7b, ChatML prompts)
    │      ├─→ System: "Extract 3-5 PERSONAL facts... ONLY about the PEOPLE."
    │      ├─→ User: Chat turns with resolved names
    │      └─→ Two-pass: raw extraction → self-correction / nuance check
    │
    ├─→ 4. Self-correction pass (same LFM-0.7b model)
    │      ├─→ Reviews extracted facts against source
    │      └─→ Rejects metaphors, filler, commentary; keeps verified facts
    │
    └─→ 5. Persist to contact_facts (attribution, confidence, dedup by UNIQUE)
           └─→ Contact profiles and reply context use these facts
```

**Extraction model**: LFM-0.7b (`models/lfm-0.7b-4bit`) — instruction-tuned for fact extraction with ChatML. Turn-based formatting gives the model coherent context (who said what).

**Two-Pass Architecture**:

- **Pass 1**: Raw extraction with ChatML prompts (system + user)
- **Pass 2**: Self-correction using the same model to filter noise and metaphors

> **Note**: Earlier designs used a separate NLI cross-encoder. The current V4 pipeline uses two-pass LLM self-correction for better performance.

See **docs/V4_MIGRATION_REPORT.md** for design decisions and lessons learned.

```bash
# V4 backfill (instruction-based, turn-based) — recommended
uv run python scripts/backfill_v4_final.py

# Alternative: GLiNER batch backfill (faster, different quality tradeoffs)
uv run python scripts/backfill_contact_facts.py --max-contacts 50
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

- **Default Model**: LFM-1.2b-ft (fine-tuned for iMessage)
- **Latency**: ~300ms warm, ~2s cold start
- **Memory**: ~1.2GB model + ~200MB embeddings
- **Model Registry**: See `models/registry.py` for all available models
- **Universal Prompt**: Single system prompt for all categories (categories hurt quality, see ablation study)

### Fact Extraction (V4)

- **Extraction Model**: LFM-0.7b (`lfm-0.7b`, instruction-based, ChatML)
- **Turn-based grouping**: Consecutive same-sender messages combined for context
- **Identity**: User and contact names from Address Book / contacts table
- **Verification**: Two-pass LLM self-correction (Pass 2 filters Pass 1 output)

---

## Performance

| Operation             | P50         | Target    | Status |
| --------------------- | ----------- | --------- | ------ |
| Mobilization classify | 12ms        | <50ms     | ✅     |
| sqlite-vec search     | 3ms         | <50ms     | ✅     |
| LLM generation        | 180ms/token | <2s total | ✅     |
| **Full pipeline**     | **~300ms**  | **<500ms**| ✅     |

## Quality Metrics (LLM-as-Judge)

| Metric | Baseline | Target |
|--------|----------|--------|
| Judge Score | 6.27/10 | 7.0+ |
| Anti-AI Rate | 0% | <5% ✅ |

---

## Storage

| Location                     | Purpose                                       |
| ---------------------------- | --------------------------------------------- |
| `~/Library/Messages/chat.db` | iMessage DB (read-only)                       |
| `~/.jarvis/jarvis.db`        | JARVIS data (chunks, pairs, and vector index) |
| `~/.jarvis/embeddings/`      | Model-specific artifacts and embeddings.db    |
| `~/.jarvis/config.json`      | User configuration                            |

---

## Model Registry

JARVIS supports LFM models via `models/registry.py`:

| Model ID        | Display Name            | Size       | Quality   | Best For                 |
| --------------- | ----------------------- | ---------- | --------- | ------------------------ |
| **`lfm-0.7b`**  | **LFM 0.7B (Extract)**  | **~0.5GB** | **Good**  | **Fact extraction (V4)** |
| `lfm-350m`      | LFM 2.5 350M (Base)     | 0.35GB     | Basic     | Alternative extraction   |
| `lfm-1.2b-base` | LFM 2.5 1.2B (Base)     | 1.2GB      | Good      | Few-shot, completion     |
| `lfm-1.2b-ft`   | LFM 2.5 1.2B Fine-Tuned | 1.2GB      | Excellent | **Default** - iMessage   |
| `lfm-1.2b-sft`  | LFM 2.5 1.2B SFT Only   | 1.2GB      | Excellent | iMessage                 |
| `lfm-0.3b-ft`   | LFM 2.5 0.3B Fine-Tuned | 0.3GB      | Basic     | Speculative decoding     |

---

## Observability

JARVIS includes a structured observability system for debugging and monitoring:

| Module                                   | Purpose                                                       |
| ---------------------------------------- | ------------------------------------------------------------- |
| `jarvis/observability/logging.py`        | Structured logging with `log_event()` and `timed_operation()` |
| `jarvis/observability/metrics_router.py` | Routing metrics collection and storage                        |
| `jarvis/observability/insights.py`       | Sentiment analysis and conversation insights                  |

**Usage:**

```python
from jarvis.observability.logging import log_event, timed_operation

# Log an event
log_event("fact_extraction", contact_id="chat123", facts_found=5)

# Time an operation
with timed_operation("model_inference", model_id="lfm-0.7b") as ctx:
    result = generate_reply(...)
    ctx["tokens_generated"] = len(result)
```
