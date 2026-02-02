# Retrieval Improvements

## 4A. Hybrid Retrieval (Semantic + BM25)

**Goal**: Combine semantic similarity with keyword matching.

### Implementation

```python
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def search(self, query, k=10, alpha=0.7):
        semantic_results = self.faiss.search(query, k=k*2)
        keyword_results = self.bm25.search(query, k=k*2)

        # Reciprocal rank fusion
        combined = self._rrf_merge(semantic_results, keyword_results)
        return combined[:k]
```

### Performance Impact

| Metric | Before | After |
|--------|--------|-------|
| Retrieval latency | 5-50ms | 10-70ms (+40%) |
| Recall@10 | 0.65 | 0.80+ |
| Keyword match | 0% | 100% |

**Pros**: Catches names/places, better rare term recall, robust to embedding failures
**Cons**: Higher latency, need to tune alpha
**Effort**: Low (1 week)

---

## 4B. Re-Ranking with Cross-Encoder

**Goal**: Re-rank top FAISS results for better precision.

### Implementation

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, candidates, k=5):
    pairs = [(query, c.trigger_text) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return [c for c, s in ranked[:k]]
```

### Performance Impact

| Metric | Before | After |
|--------|--------|-------|
| Retrieval latency | 5-50ms | 50-150ms (+3x) |
| Precision@5 | 0.70 | 0.85+ |
| Memory | 200MB | +150MB |

**Pros**: Significantly better precision, catches semantic nuances
**Cons**: Significant latency increase, additional model
**Effort**: Low (1 week)

---

## 4C. Temporal Weighting

**Goal**: Weight recent messages higher in retrieval.

### Current State

```python
# 10% decay per year
decay = max(0.5, 1.0 - (age_days / 365) * 0.1)
```

### Enhanced Temporal Model

```python
def temporal_weight(timestamp, query_time):
    age_days = (query_time - timestamp).days

    if age_days < 7:
        return 1.0
    elif age_days < 30:
        return 0.9
    elif age_days < 90:
        return 0.8
    elif age_days < 365:
        return 0.6
    else:
        return 0.4
```

**Contact-specific recency:**
- Active contacts: weight recent heavily
- Dormant contacts: weight historical patterns

**Pros**: Simple, no latency impact, prioritizes relevant patterns
**Cons**: May miss useful old patterns
**Effort**: Low (3-5 days)
