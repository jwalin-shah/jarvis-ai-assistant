# Improvements Overview

> **TLDR**: Main bottleneck is embedding latency (~100-150ms). Quick wins: faster embedding model, temporal weighting, hybrid retrieval. See priority matrix below.

## Current Architecture

| Component | Implementation | Performance |
|-----------|----------------|-------------|
| **Embeddings** | bge-small-en-v1.5 (384-dim) | **100-150ms** ← bottleneck |
| **Generation** | LFM2.5-1.2B-4bit | 600-3000ms E2E |
| **Classifiers** | 3-layer hybrid | ~82% F1, <50ms |
| **Retrieval** | FAISS flat/IVF | 5-50ms search |
| **Caching** | LRU in-memory (1000 entries) | Cache hit avoids recompute |

## Priority Matrix

### Quick Wins (1-2 weeks)

| Improvement | Impact | Effort | Risk |
|-------------|--------|--------|------|
| **Faster Embedding Models** | **High** | **Low** | Medium |
| Temporal Weighting | Medium | Low | Low |
| Multi-Option Diversity | Medium | Low | Low |
| Hybrid Retrieval | High | Low | Low |
| Longer Context | Medium | Low | Low |

### Medium-Term (2-4 weeks)

| Improvement | Impact | Effort | Risk |
|-------------|--------|--------|------|
| Feedback Loop | High | Medium | Medium |
| Adaptive Thresholds | Medium | Low | Medium |
| LoRA Fine-Tuning | High | Medium | Medium |
| Cross-Encoder Reranking | High | Low | Low |
| Domain Fine-Tuning | High | Medium | Medium |

### Long-Term (4+ weeks)

| Improvement | Impact | Effort | Risk |
|-------------|--------|--------|------|
| Contrastive Learning | High | High | High |
| Multi-Task Embeddings | High | High | High |
| Topic/Entity Tracking | High | High | Medium |

## Implementation Order

1. **Phase 1 (Foundation)**: Faster embeddings, temporal weighting, hybrid retrieval
2. **Phase 2 (Learning)**: Feedback loop, adaptive thresholds
3. **Phase 3 (Quality)**: Multi-option diversity, cross-encoder, quality scoring
4. **Phase 4 (Personalization)**: Domain fine-tuning, LoRA, summarization
5. **Phase 5 (Advanced)**: Contrastive learning, topic/entity tracking

## Critical Files

| File | Improvement Areas |
|------|-------------------|
| `jarvis/embedding_adapter.py` | Embeddings (1A-1D) |
| `jarvis/index.py` | Retrieval (4A-4C) |
| `jarvis/router.py` | Routing, context (2C, 3B, 5A) |
| `jarvis/evaluation.py` | Feedback (3A, 3C) |
| `models/generator.py` | Generation (2A, 5A) |

## Related Docs

- [Embeddings](./EMBEDDINGS.md) - Faster models, fine-tuning
- [Generation](./GENERATION.md) - LoRA, multi-option, quality scoring
- [Learning](./LEARNING.md) - Feedback loop, adaptive thresholds
- [Retrieval](./RETRIEVAL.md) - Hybrid search, reranking, temporal
- [Context](./CONTEXT.md) - Longer context, summarization, topics
- [Benchmarks](./BENCHMARKS.md) - FAISS compression data
