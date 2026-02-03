# JARVIS Design Overview

> **TLDR**: Local-first AI assistant for iMessage on Apple Silicon. Uses 3-layer hybrid classifiers (structural → centroid → SVM) achieving ~82% F1. Retrieval + generation via MLX models. All data stays on device.

## Quick Reference

| Metric | Result |
|--------|--------|
| Response Classifier F1 | **81.9%** |
| Trigger Classifier F1 | **82.0%** |
| Message Read Latency | **1-5ms** (direct SQLite) |
| Memory Footprint | **<5.5GB** |

## Core Innovation

The system learns from **your actual messaging patterns** by extracting (trigger, response) pairs from iMessage history, then uses hybrid retrieval + generation.

## Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────┐
│  Desktop App (Tauri)  ←→  Unix Socket  ←→  JARVIS Core     │
│                                                             │
│  JARVIS Core:                                               │
│  ├─ Classifiers (Trigger/Response) - 3-layer hybrid        │
│  ├─ Router (FAISS + Thresholds)                            │
│  ├─ Generator (MLX LLM)                                    │
│  └─ Embedding Layer (MLX, 384-dim, text normalization)     │
│                                                             │
│  Data Layer:                                                │
│  ├─ chat.db (iMessage, read-only)                          │
│  ├─ jarvis.db (contacts, pairs, embeddings)                │
│  └─ FAISS Index (vector search)                            │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

| Directory | Purpose |
|-----------|---------|
| `jarvis/` | Core logic: classifiers, router, embeddings, config, prompts |
| `jarvis/analytics/` | Dashboard metrics, trends, time-series aggregation |
| `jarvis/graph/` | Relationship networks, clustering, layout algorithms |
| `jarvis/scheduler/` | Smart timing, draft scheduling, priority queue |
| `jarvis/tags/` | Tags, smart folders, auto-tagging rules |
| `jarvis/prefetch/` | Multi-tier cache, prediction, invalidation |
| `jarvis/quality/` | Hallucination detection, factuality, grounding |
| `api/` | FastAPI REST layer for HTTP clients |
| `models/` | MLX model inference, registry, templates |
| `integrations/imessage/` | iMessage database reader |
| `desktop/` | Tauri + Svelte desktop application |

## Key Files

| File | Purpose |
|------|---------|
| `jarvis/router.py` | Main routing logic, thresholds, decision flow |
| `jarvis/response_classifier.py` | 3-layer hybrid response classifier |
| `jarvis/response_classifier_v2.py` | Optimized batch classifier (10x throughput) |
| `jarvis/trigger_classifier.py` | Trigger type classifier |
| `jarvis/embedding_adapter.py` | Unified embedding interface |
| `jarvis/index_v2.py` | Sharded FAISS with tiered storage |
| `jarvis/adaptive_thresholds.py` | Learns thresholds from feedback |
| `jarvis/socket_server.py` | Unix socket JSON-RPC server |
| `jarvis/watcher.py` | File watcher for new messages |
| `jarvis/evaluation.py` | Feedback storage and analysis |

## Related Docs

- [Pipeline Details](./PIPELINE.md) - Classification and routing flow
- [V2 Architecture](./V2_ARCHITECTURE.md) - Unix sockets, direct SQLite
- [Embeddings](./EMBEDDINGS.md) - Model choices, multi-model support, FAISS
- [Text Normalization](./TEXT_NORMALIZATION.md) - Unicode/whitespace normalization for consistent embeddings
- [Feedback System](./FEEDBACK.md) - Learning from user actions
- [Design Decisions](./DECISIONS.md) - Rationale and lessons learned
- [Metrics](./METRICS.md) - Performance benchmarks
- [Frontend Redesign](../FRONTEND_REDESIGN_PLAN.md) - Themes, skeletons, command palette
