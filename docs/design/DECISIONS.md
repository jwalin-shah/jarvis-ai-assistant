# Architecture Decision Records

> **Last Updated:** 2026-02-10

This document captures key architectural decisions, their rationale, and lessons learned.

---

## Core Decisions

### 1. Local-First Architecture

**Options considered:**
1. Cloud API (GPT-4, Claude) - Best quality, privacy concerns
2. Local small model (1-3B) - Good privacy, acceptable quality
3. Hybrid - Complex, still has privacy issues

**Decision:** Local-only with MLX on Apple Silicon

**Rationale:**
- Messages are deeply personal - privacy is non-negotiable
- Apple Silicon has excellent ML performance via MLX
- 1-3B models are good enough for short messages
- Works offline, no latency spikes

---

### 2. Default Model: LFM-2.5-1.2B-Instruct-4bit

**Decision:** Use LFM-2.5-1.2B-Instruct-4bit as default model.

**Rationale:**
- Small footprint (~1GB) works on 8GB RAM
- 4-bit quantization balances quality vs memory
- Sufficient quality for reply suggestions

**Consequence:** Users with more RAM can select larger models via config.

---

### 3. MLX Over Ollama

**Decision:** Use MLX framework directly instead of Ollama wrapper.

**Rationale:** Better memory control on Apple Silicon, explicit Metal cache clearing.

**Consequence:** More code complexity but precise memory management.

---

### 4. Three-Tier Memory Modes (FULL/LITE/MINIMAL)

**Decision:** Support automatic mode detection based on available memory.

**Rationale:** 8GB is aspirational but math doesn't work for full functionality on all machines.

**Consequence:** More complex code paths but viable across hardware range.

---

### 5. Template-First Architecture

**Decision:** Match requests to templates first, generate only when no match.

**Rationale:** Generation is expensive (memory, latency) and risky (hallucination).

**Consequence:** Better latency and quality for common cases. Requires template coverage investment.

---

### 6. No Fine-Tuning, Use RAG + Few-Shot

**Decision:** Use RAG and few-shot prompting only, no fine-tuning.

**Rationale:** Gekhman et al. (EMNLP 2024) shows fine-tuning on new knowledge increases hallucinations.

**Consequence:** Style matching is less precise but hallucination risk is not increased.

---

### 7. FAISS for Similarity Search

**Options considered:**
1. FAISS flat - Simple, exact search
2. FAISS IVF - Faster, approximate
3. Hnswlib - Very fast, approximate
4. Chroma/Pinecone - Managed, features

**Decision:** FAISS flat index (upgraded to IVFPQ for compression)

**Rationale:**
- Dataset (10K-50K pairs) small enough for exact search
- No approximation error
- No external dependencies
- IVFPQ added later for 3.8x compression with 92% recall

---

### 8. Three-Layer Hybrid Classifiers

**Options considered:**
1. Pure ML (embeddings + kNN)
2. Pure rules (regex)
3. Hybrid (rules first, ML fallback)

**Decision:** Structural patterns → Centroid verification → SVM fallback

**Rationale:**
- Structural patterns (regex) catch high-confidence cases instantly (~11%)
- Centroid verification adds semantic check without full classification
- SVM handles ambiguous cases with 82% accuracy

---

### 9. SVM over k-NN for Classification

**Decision:** Use SVM classifiers instead of k-NN for both trigger and response classification.

**Rationale:**
- SVM achieves 82% macro F1 vs ~70% for k-NN
- Faster inference (single model forward pass vs k-nearest search)
- Per-class thresholds enable precision/recall tuning

---

### 10. Sentence-Transformers for Embeddings

**Decision:** Use all-MiniLM-L6-v2 instead of heavier models like GLiNER.

**Rationale:** ~100MB vs 800MB-1.2GB footprint.

**Consequence:** Sufficient for template matching with much lower memory.

---

### 11. Unix Sockets Over HTTP for Desktop

**Decision:** Unix socket with JSON-RPC for desktop, keep HTTP for CLI.

**Rationale:**
- HTTP overhead significant for frequent local calls
- Unix sockets 10-50x faster for local IPC
- JSON-RPC is simple and well-understood

---

### 12. Read-Only iMessage Access

**Decision:** Read-only access with schema detection and fallback.

**Rationale:** chat.db access is fragile; Apple changes schema between releases.

**Consequence:** Cannot write to iMessage (acceptable for v1). Resilient to schema changes.

---

### 13. K-means over HDBSCAN for Topic Clustering

**Decision:** Use K-means for topic clustering, remove HDBSCAN from core dependencies.

**Rationale:**
- Topics are informational only (not critical to reply generation)
- K-means is simpler and faster
- HDBSCAN adds heavy dependency for minimal benefit

---

## What We Tried and Failed

### ❌ Direct Response Retrieval

**Hypothesis:** If trigger similarity > 0.85, return cached response.

**Result:** Users complained responses were inappropriate.

**Why:** Same question from different people needs different responses. Context matters.

**Lesson:** Use retrieval for examples, not direct responses.

---

### ❌ Pure Embedding Classification

**Hypothesis:** Use kNN on embeddings to classify.

**Result:** 72% accuracy - not good enough.

**Why:** Embeddings don't capture structural patterns well. "ok" and "okay" have different embeddings.

**Lesson:** Structural patterns first, embeddings for verification.

---

### ❌ Single Global Confidence Threshold

**Hypothesis:** Use 0.5 threshold for all classes.

**Result:** COMMITMENT questions often misclassified.

**Why:** Different classes have different natural confidence levels.

**Lesson:** Per-class thresholds tuned on validation data.

---

### ❌ HTTP Polling for New Messages

**Hypothesis:** Poll every 5 seconds.

**Result:** High latency, battery drain, race conditions.

**Lesson:** File watcher + push notifications for instant updates.

---

### ❌ Fine-Tuning for Personalization

**Hypothesis:** LoRA fine-tune on user's history.

**Why abandoned:**
- Expensive compute per user
- Risk of overfitting to old patterns
- Hard to update as style evolves

**Lesson:** RAG + few-shot is more flexible and cheaper.

---

## Pending Decisions

### P1: iMessageSender Deprecation

**Context:** Apple's AppleScript restrictions make it fragile.

| Option | Pros | Cons |
|--------|------|------|
| Remove entirely | Clean codebase | Lose write capability |
| Keep but hidden | Available if needed | Technical debt |

**Recommendation:** Keep but mark deprecated, remove in v2.

---

### P2: Template Count Expansion

**Context:** Currently ~75 templates. Target is 50-100.

**Recommendation:** Run coverage benchmark first, then decide on expansion.
