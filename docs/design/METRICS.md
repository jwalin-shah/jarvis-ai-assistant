# Metrics & Evaluation

> **Last Updated:** 2026-02-10

## Classifier Performance

| Classifier          | Macro F1  | 95% CI        | Test Set    |
| ------------------- | --------- | ------------- | ----------- |
| Response Classifier | **81.9%** | 78.4% - 84.9% | 5,200 pairs |
| Trigger Classifier  | **82.0%** | 79.3% - 84.4% | 5,200 pairs |

### Per-Class (Response Classifier)

| Class    | Precision | Recall | F1   | Support |
| -------- | --------- | ------ | ---- | ------- |
| AGREE    | 0.84      | 0.79   | 0.81 | 1,150   |
| DECLINE  | 0.78      | 0.72   | 0.75 | 380     |
| DEFER    | 0.76      | 0.68   | 0.72 | 290     |
| OTHER    | 0.82      | 0.88   | 0.85 | 2,100   |
| QUESTION | 0.89      | 0.91   | 0.90 | 820     |
| REACTION | 0.81      | 0.77   | 0.79 | 460     |

## Latency Benchmarks

| Operation                  | P50       | P95       | P99       |
| -------------------------- | --------- | --------- | --------- |
| Intent classification      | 12ms      | 25ms      | 45ms      |
| FAISS search (10K vectors) | 3ms       | 8ms       | 15ms      |
| LLM generation (50 tokens) | 180ms     | 320ms     | 500ms     |
| **Full pipeline**          | **250ms** | **450ms** | **700ms** |

## Memory Usage

| Component                 | Memory     |
| ------------------------- | ---------- |
| LLM (LFM-1.2B 4-bit)      | ~1.2GB     |
| Embeddings model          | ~120MB     |
| FAISS index (50K vectors) | ~75MB      |
| Classifiers (SVM)         | ~20MB      |
| **Total**                 | **~1.4GB** |

---

# Future Improvements

## What's Already Built

| Component               | Status      |
| ----------------------- | ----------- |
| Feedback API            | ✅ Complete |
| FeedbackStore           | ✅ Complete |
| Evaluation Scores       | ✅ Complete |
| Multi-Option Generation | ✅ Complete |

## Short Term (Next Sprint)

| Task                        | Status         | Files                    |
| --------------------------- | -------------- | ------------------------ |
| Hook up desktop feedback    | TODO           | SmartReplyChipsV2.svelte |
| Passive feedback detection  | 🚧 In Progress | jarvis/watcher.py        |
| Trigger complexity analysis | ✅ Complete    | jarvis/router.py         |
| Feedback CLI                | ✅ Complete    | jarvis/\_cli_main.py     |

## Medium Term

- Feedback-driven style learning
- Hybrid retrieval (BM25 + Semantic)
- Response quality pre-filter

## Long Term

- Online learning (retrain with feedback)
- Calendar integration
- Conversation memory
- Better embeddings (fine-tuned)

## Metrics to Track

| Metric              | Target | Formula                            |
| ------------------- | ------ | ---------------------------------- |
| Acceptance Rate     | >50%   | sent / (sent + edited + dismissed) |
| Edit Rate           | <30%   | edited / total_actioned            |
| Time to Respond     | <3s    | suggestion shown → user action     |
| Classifier Accuracy | >85%   | periodic evaluation                |
