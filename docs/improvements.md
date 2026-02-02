# JARVIS Improvement Roadmap

This documentation has been split into focused files for better context management.

## Quick Start

**Read first:** [Improvements Overview](./improvements/OVERVIEW.md) (~80 lines, priority matrix + implementation order)

## Split Documentation

| Doc | Content | Lines |
|-----|---------|-------|
| [Overview](./improvements/OVERVIEW.md) | Priority matrix, implementation order | ~80 |
| [Embeddings](./improvements/EMBEDDINGS.md) | Faster models, fine-tuning (1A-1D) | ~100 |
| [Generation](./improvements/GENERATION.md) | LoRA, multi-option, quality (2A-2C) | ~90 |
| [Learning](./improvements/LEARNING.md) | Feedback loop, adaptive thresholds (3A-3D) | ~110 |
| [Retrieval](./improvements/RETRIEVAL.md) | Hybrid search, reranking (4A-4C) | ~100 |
| [Context](./improvements/CONTEXT.md) | Longer context, summarization (5A-5D) | ~110 |
| [Benchmarks](./improvements/BENCHMARKS.md) | FAISS compression data | ~80 |

## Usage

Instead of `@docs/improvements.md` (was ~1200 lines, ~18k tokens), use:

```
# For priorities
@docs/improvements/OVERVIEW.md

# For specific areas
@docs/improvements/EMBEDDINGS.md   # embedding improvements
@docs/improvements/RETRIEVAL.md    # search improvements
@docs/improvements/BENCHMARKS.md   # FAISS data
```

Or just describe what you need and let Claude search the relevant file.
