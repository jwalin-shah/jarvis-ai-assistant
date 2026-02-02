# Embedding Architecture

## 3-Layer Embedding Stack

```
YOUR CODE: from jarvis.embedding_adapter import get_embedder
                          │
                          ▼
LAYER 2: jarvis/embedding_adapter.py (UnifiedEmbedder)
         Model: BAAI/bge-small-en-v1.5 (384 dimensions)
         Priority: MLX service first → CPU fallback
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
LAYER 3: MLX Service              CPU Fallback
(http://127.0.0.1:8766)           SentenceTransformer
GPU-accelerated on M1/M2          Same model, slower
```

## Why bge-small-en-v1.5?

| Model | Dimensions | MTEB Score | Memory | Speed |
|-------|------------|------------|--------|-------|
| bge-small-en-v1.5 | 384 | 62.17 | ~120MB | Fast |
| bge-base-en-v1.5 | 768 | 63.55 | ~440MB | Medium |
| bge-large-en-v1.5 | 1024 | 64.23 | ~1.3GB | Slow |

**Decision:** bge-small offers best tradeoff:
- Only 1.4 points below bge-large on MTEB
- 10x smaller memory footprint
- Fast enough for real-time use

## Faster Options (TODO: Evaluate)

Current embedding latency (~100-150ms) is the main bottleneck.

| Model | Layers | MTEB | Expected Latency | Quality Loss |
|-------|--------|------|------------------|--------------|
| `bge-small` | 12 | ~62 | 100-150ms | **Baseline** |
| `gte-tiny` | 6 | ~57 | ~50-70ms | ~8% |
| `minilm-l6` | 6 | ~56 | ~50-70ms | ~10% |
| `bge-micro` | 3 | ~54 | ~30-40ms | ~13% |

**Switching models requires:**
1. Retraining SVM classifiers
2. Recomputing all centroids
3. Rebuilding FAISS index
4. Re-validating classifier accuracy

**Proposed: Embedding Model Versioning**
```
~/.jarvis/embeddings/
├── bge-small/           # Current (working)
│   ├── trigger_classifier_model/
│   ├── response_classifier_model/
│   └── embeddings.db
└── gte-tiny/            # Experimental
    ├── trigger_classifier_model/
    └── embeddings.db
```

## Embedding Caching

```python
class CachedEmbedder:
    """LRU cache for embedding results."""
    def __init__(self, maxsize: int = 1000):
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def encode(self, texts):
        # Check cache first, only compute missing
```

## FAISS Index Compression

For large histories (400K+ messages), benchmarked on 148K real messages:

| Index Type | Size | Compression | Recall@10 |
|------------|------|-------------|-----------|
| IndexFlatIP (brute force) | 217 MB | 1x | 100% |
| **IVFPQ 384x8 (4x)** | **57 MB** | **3.8x** | **92%** |
| IVFPQ 192x8 (8x) | 30 MB | 7.2x | 88% |

**Key findings:**
- 4x compression saves ~430MB on 400K messages
- Search time is NOT the bottleneck (<2ms vs ~100ms for embedding)
- **Default: IVFPQ 384x8 (4x)** - best quality/memory tradeoff

See [Benchmarks](../improvements/BENCHMARKS.md) for full data.
