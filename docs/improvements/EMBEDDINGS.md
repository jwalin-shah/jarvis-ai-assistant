# Embedding Improvements

## 1D. Faster Embedding Models ✅ IMPLEMENTED

**Status**: Complete. Multi-model support added Feb 2026.

**Goal**: Reduce embedding latency from ~100-150ms to ~50-70ms.

### Available Models

Configure via `~/.jarvis/config.json`:
```json
{"embedding": {"model_name": "gte-tiny"}}
```

| Model | Layers | MTEB | Expected Latency | Quality Loss |
|-------|--------|------|------------------|--------------|
| `bge-small` | 12 | ~62 | 100-150ms | **Default** |
| `gte-tiny` | 6 | ~57 | ~50-70ms | ~8% |
| `minilm-l6` | 6 | ~56 | ~50-70ms | ~10% |
| `bge-micro` | 3 | ~54 | ~30-40ms | ~13% |

All output 384 dimensions - index structure compatible.

### What Was Implemented

1. ✅ Model registry in `jarvis/embedding_adapter.py`
2. ✅ Per-model artifact storage: `~/.jarvis/embeddings/{model_name}/`
3. ✅ Per-model FAISS indexes: `~/.jarvis/indexes/{model_name}/`
4. ✅ Config-driven model selection
5. ✅ MLX service supports all models (downloads on first use)
6. ✅ Removed CPU fallback (MLX-only for simplicity)

### Switching Models

```bash
# 1. Update config
vim ~/.jarvis/config.json  # Set model_name

# 2. Restart MLX service
pkill -f "mlx-embed-service"
cd ~/.jarvis/mlx-embed-service && uv run python server.py &

# 3. Retrain classifiers
uv run python -m scripts.train_trigger_classifier --save-best
uv run python -m scripts.train_response_classifier --save-best

# 4. Rebuild FAISS index
uv run python -m jarvis.index build
```

**Recommendation**: Try `gte-tiny` first - best speed/quality balance.

---

## 1A. Domain-Specific Fine-Tuning

**Goal**: Fine-tune bge-small on iMessage-style text.

### Implementation

1. Extract ~50k trigger-response pairs from jarvis.db
2. Create contrastive pairs: (trigger, similar, dissimilar)
3. Fine-tune with sentence-transformers

```python
from sentence_transformers import SentenceTransformer, losses
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
train_loss = losses.MultipleNegativesRankingLoss(model)
```

**Expected Impact**: Trigger similarity correlation 0.56 → 0.70+
**Effort**: Medium (2-3 weeks)

---

## 1B. Contrastive Learning on Conversations

**Goal**: Learn embeddings that group similar conversation flows.

- Positive pairs: Messages in same conversation thread
- Negative pairs: Messages from different conversations
- Loss: InfoNCE / NT-Xent

**Expected Impact**: Same-thread similarity 0.45 → 0.65+
**Effort**: High (3-4 weeks)
**Dependency**: Conversation threading data

---

## 1C. Multi-Task Embeddings

**Goal**: Joint optimization for trigger matching + response quality.

```python
class MultiTaskEmbedder(nn.Module):
    def __init__(self):
        self.encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.quality_head = nn.Linear(384, 1)

    def forward(self, text):
        embedding = self.encoder.encode(text)
        quality = self.quality_head(embedding)
        return embedding, quality
```

**Expected Impact**: Quality signal embedded in representation
**Effort**: High (4+ weeks)
**Dependency**: Quality scoring pipeline
