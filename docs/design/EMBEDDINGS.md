# Embedding Architecture

> **Last Updated:** 2026-02-10

## Text Normalization

All text is normalized before embedding for consistency between training and inference.
See [TEXT_NORMALIZATION.md](./TEXT_NORMALIZATION.md) for details.

```python
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # Smart quotes → ASCII
    text = " ".join(text.split())                # Collapse whitespace/newlines
    return text
```

**Why?** Training data is mostly single messages, but inference sees multi-message turns
(rapid-fire texts joined with `\n`). Normalization ensures consistent embeddings.

## Multi-Model Support

JARVIS supports multiple embedding models, all with 384 dimensions for index compatibility.
Models are configured via `~/.jarvis/config.json`:

```json
{
  "embedding": {
    "model_name": "bge-small"
  }
}
```

### Available Models

| Config Name | HuggingFace ID | Layers | MTEB | Latency | Notes |
|-------------|----------------|--------|------|---------|-------|
| `bge-small` | BAAI/bge-small-en-v1.5 | 12 | ~62 | 100-150ms | **Default**, best quality |
| `gte-tiny` | TaylorAI/gte-tiny | 6 | ~57 | ~50-70ms | Good speed/quality balance |
| `minilm-l6` | sentence-transformers/all-MiniLM-L6-v2 | 6 | ~56 | ~50-70ms | Most popular fast model |
| `bge-micro` | TaylorAI/bge-micro-v2 | 3 | ~54 | ~30-40ms | Fastest, lowest quality |

## Architecture (MLX-Only)

```
YOUR CODE: from jarvis.embedding_adapter import get_embedder
                          │
                          ▼
jarvis/embedding_adapter.py (MLXEmbedder)
├─ Model Registry: maps config name → HuggingFace/MLX model
├─ Reads model from config.embedding.model_name
└─ Connects to MLX service via Unix socket
                          │
                          ▼
~/.jarvis/mlx-embed-service/ (separate process)
├─ Unix socket: ~/.jarvis/jarvis-embed.sock
├─ JSON-RPC 2.0 protocol
├─ GPU-accelerated via MLX on Apple Silicon
└─ Downloads models on first use (mlx-embedding-models)
```

**No CPU fallback** - MLX service must be running. This simplifies the codebase
and ensures consistent GPU-accelerated performance.

## Per-Model Artifact Storage

All artifacts are namespaced by model name:

```
~/.jarvis/
├── embeddings/
│   ├── bge-small/                    # Default model artifacts
│   │   ├── trigger_classifier_model/
│   │   ├── response_classifier_model/
│   │   └── embeddings.db
│   └── gte-tiny/                     # Alternative model artifacts
│       ├── trigger_classifier_model/
│       ├── response_classifier_model/
│       └── embeddings.db
└── indexes/
    ├── bge-small/                    # FAISS indexes per model
    │   └── {version}/index.faiss
    └── gte-tiny/
        └── {version}/index.faiss
```

## Switching Models

```bash
# 1. Update config
vim ~/.jarvis/config.json
# Set: "embedding": {"model_name": "gte-tiny"}

# 2. Restart MLX service (downloads model on first use)
pkill -f "mlx-embed-service"
cd ~/.jarvis/mlx-embed-service && uv run python server.py &

# 3. Retrain classifiers (required - different embedding space)
uv run python -m scripts.train_trigger_classifier --save-best
uv run python -m scripts.train_response_classifier --save-best

# 4. Rebuild FAISS index
uv run python -m jarvis.index build

# 5. Verify
uv run python -c "from jarvis.embedding_adapter import get_embedder; e = get_embedder(); print(f'Model: {e.model_name}, Backend: {e.backend}')"
```

You can switch back anytime - artifacts are preserved per model.

## Model Registry

Defined in `jarvis/embedding_adapter.py`:

```python
EMBEDDING_MODEL_REGISTRY = {
    "bge-small": ("BAAI/bge-small-en-v1.5", "bge-small"),
    "gte-tiny": ("TaylorAI/gte-tiny", "gte-tiny"),
    "minilm-l6": ("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2"),
    "bge-micro": ("TaylorAI/bge-micro-v2", "bge-micro-v2"),
}
```

## Embedding Caching

```python
class CachedEmbedder:
    """LRU cache for embedding results (1000 entries default)."""

    def encode(self, texts):
        # Check cache first (blake2b hash of text)
        # Only compute missing embeddings
        # Batch encode for efficiency
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

Configurable via `config.faiss_index.index_type`.

## Key Files

| File | Purpose |
|------|---------|
| `jarvis/embedding_adapter.py` | Unified interface, model registry |
| `jarvis/config.py` | `EmbeddingConfig`, artifact path helpers |
| `models/embeddings.py` | MLX service client (Unix socket) |
| `~/.jarvis/mlx-embed-service/server.py` | MLX embedding microservice |
