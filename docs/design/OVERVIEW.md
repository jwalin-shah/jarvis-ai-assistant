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
│  └─ Embedding Layer (bge-small-en-v1.5, 384-dim)           │
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
| `api/` | FastAPI REST layer for HTTP clients |
| `models/` | MLX model inference, registry, templates |
| `integrations/imessage/` | iMessage database reader |
| `desktop/` | Tauri + Svelte desktop application |

## Key Files

| File | Purpose |
|------|---------|
| `jarvis/router.py` | Main routing logic, thresholds, decision flow |
| `jarvis/response_classifier.py` | 3-layer hybrid response classifier |
| `jarvis/trigger_classifier.py` | Trigger type classifier |
| `jarvis/embedding_adapter.py` | Unified embedding interface |
| `jarvis/socket_server.py` | Unix socket JSON-RPC server |
| `jarvis/watcher.py` | File watcher for new messages |
| `jarvis/evaluation.py` | Feedback storage and analysis |

## Related Docs

- [Pipeline Details](./PIPELINE.md) - Classification and routing flow
- [V2 Architecture](./V2_ARCHITECTURE.md) - Unix sockets, direct SQLite
- [Embeddings](./EMBEDDINGS.md) - Model choices, caching, FAISS
- [Feedback System](./FEEDBACK.md) - Learning from user actions
- [Design Decisions](./DECISIONS.md) - Rationale and lessons learned
- [Metrics](./METRICS.md) - Performance benchmarks
