# Embedding Improvements

## 1D. Faster Embedding Models (HIGH PRIORITY)

**Goal**: Reduce embedding latency from ~100-150ms to ~50-70ms.

**Current Bottleneck**:
```
├─ Embed query text     → ~100-150ms  ← THE BOTTLENECK
├─ FAISS search         → ~5-10ms
├─ Classification       → ~15-30ms
└─ Generation           → ~200-500ms
```

### Available Models

| Model | Layers | MTEB | Expected Latency | Quality Loss |
|-------|--------|------|------------------|--------------|
| `bge-small` | 12 | ~62 | 100-150ms | **Baseline** |
| `gte-tiny` | 6 | ~57 | ~50-70ms | ~8% |
| `minilm-l6` | 6 | ~56 | ~50-70ms | ~10% |
| `bge-micro` | 3 | ~54 | ~30-40ms | ~13% |

All output 384 dimensions - index structure compatible.

### Implementation

1. Add embedding model versioning to config
2. Namespace artifacts by model: `~/.jarvis/embeddings/{model_name}/`
3. Retrain classifiers with new embeddings
4. Compare F1 scores and latency

**Effort**: Low-Medium (1-2 days + training)
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
