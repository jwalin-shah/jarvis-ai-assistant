# JARVIS Design Documentation

This documentation has been split into focused files for better context management.

## Quick Start

**Read first:** [Design Overview](./design/OVERVIEW.md) (~100 lines, covers everything at high level)

## Split Documentation

| Doc | Content | Lines |
|-----|---------|-------|
| [Overview](./design/OVERVIEW.md) | TLDR, architecture, key files | ~80 |
| [Pipeline](./design/PIPELINE.md) | Classification, routing flow | ~120 |
| [V2 Architecture](./design/V2_ARCHITECTURE.md) | Unix sockets, direct SQLite | ~80 |
| [Embeddings](./design/EMBEDDINGS.md) | Model choices, caching, FAISS | ~100 |
| [Text Normalization](./design/TEXT_NORMALIZATION.md) | Unicode/whitespace normalization | ~80 |
| [Feedback](./design/FEEDBACK.md) | Learning from user actions | ~120 |
| [Decisions](./design/DECISIONS.md) | Rationale, lessons learned | ~130 |
| [Metrics](./design/METRICS.md) | Benchmarks, future work | ~90 |
| [Contact Profiles](./design/CONTACT_PROFILES.md) | Per-contact style + topics | ~200 |

## V3 Features

| Feature | Module | Purpose |
|---------|--------|---------|
| Analytics | `jarvis/analytics/` | Dashboard metrics, trends, reports |
| Graph | `jarvis/graph/` | Relationship networks, community detection |
| Scheduler | `jarvis/scheduler/` | Smart timing, draft scheduling |
| Tags | `jarvis/tags/` | Auto-tagging, smart folders |
| Prefetch | `jarvis/prefetch/` | Speculative caching, prediction |
| Quality | `jarvis/quality/` | Hallucination detection, grounding |
| Classifier V2 | `jarvis/response_classifier_v2.py` | 10x batch throughput |
| Index V2 | `jarvis/index_v2.py` | Tiered sharded FAISS |
| Adaptive Thresholds | `jarvis/adaptive_thresholds.py` | Learns from feedback |
| Frontend Redesign | [Plan](./FRONTEND_REDESIGN_PLAN.md) | Themes, skeletons, optimistic UI |

## Usage

Instead of `@docs/DESIGN.md` (was ~1200 lines, ~20k tokens), use:

```
# For quick context
@docs/design/OVERVIEW.md

# For specific topics
@docs/design/PIPELINE.md      # classification details
@docs/design/EMBEDDINGS.md    # embedding model info
@docs/design/FEEDBACK.md      # feedback system
```

Or just describe what you need and let Claude search the relevant file.
