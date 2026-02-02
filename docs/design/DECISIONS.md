# Design Decisions & Lessons Learned

## Decision 1: Local-First Architecture

**Options:**
1. Cloud API (GPT-4, Claude) - Best quality, privacy concerns
2. Local small model (1-3B) - Good privacy, acceptable quality
3. Hybrid - Complex, still has privacy issues

**Decision:** Local-only with MLX on Apple Silicon

**Rationale:**
- Messages are deeply personal - privacy non-negotiable
- Apple Silicon has excellent ML performance via MLX
- 1-3B models good enough for short messages
- Works offline, no latency spikes

## Decision 2: FAISS for Similarity Search

**Options:**
1. FAISS flat - Simple, exact search
2. FAISS IVF - Faster, approximate
3. Hnswlib - Very fast, approximate
4. Chroma/Pinecone - Managed, features

**Decision:** FAISS flat index

**Rationale:**
- Dataset (10K-50K pairs) small enough for exact search
- No approximation error
- No external dependencies
- Can upgrade to IVF if needed

## Decision 3: Hybrid Classifiers

**Options:**
1. Pure ML (embeddings + kNN)
2. Pure rules (regex)
3. Hybrid (rules first, ML fallback)

**Decision:** 3-layer hybrid (structural → centroid → SVM)

**Rationale:**
- Structural patterns are fast and interpretable
- But have edge cases ("No way!" = excitement, not decline)
- Centroid verification catches edge cases
- SVM handles truly ambiguous cases

## Decision 4: Retrieval for Examples, Not Direct Responses

**Why not return cached responses?**

Evaluated on 26K holdout pairs:

| Trigger Similarity | Response Similarity |
|--------------------|---------------------|
| 0.95+ | 0.61 |
| 0.90-0.95 | 0.56 |
| 0.80-0.90 | 0.52 |

**Conclusion:** Context matters more than trigger similarity. Use retrieval for few-shot examples.

## Decision 5: Unix Sockets Over HTTP

**Decision:** Unix socket with JSON-RPC for desktop, keep HTTP for CLI

**Rationale:**
- HTTP overhead significant for frequent local calls
- Unix sockets 10-50x faster for local IPC
- JSON-RPC is simple and well-understood

---

# What We Tried and Why It Failed

## Failed: Direct Response Retrieval

**Hypothesis:** If trigger similarity > 0.85, return cached response.

**Result:** Users complained responses were inappropriate.

**Why:** Same question from different people needs different responses. Context matters.

**Lesson:** Use retrieval for examples, not direct responses.

## Failed: Pure Embedding Classification

**Hypothesis:** Use kNN on embeddings to classify.

**Result:** 72% accuracy - not good enough.

**Why:** Embeddings don't capture structural patterns well. "ok" and "okay" have different embeddings.

**Lesson:** Structural patterns first, embeddings for verification.

## Failed: Single Global Confidence Threshold

**Hypothesis:** Use 0.5 threshold for all classes.

**Result:** COMMITMENT questions often misclassified.

**Why:** Different classes have different natural confidence levels.

**Lesson:** Per-class thresholds tuned on validation data.

## Failed: HTTP Polling for New Messages

**Hypothesis:** Poll every 5 seconds.

**Result:** High latency, battery drain, race conditions.

**Lesson:** File watcher + push notifications for instant updates.

## Abandoned: Fine-Tuning for Personalization

**Hypothesis:** LoRA fine-tune on user's history.

**Why abandoned:**
- Expensive compute per user
- Risk of overfitting to old patterns
- Hard to update as style evolves

**Lesson:** RAG + few-shot is more flexible and cheaper.
