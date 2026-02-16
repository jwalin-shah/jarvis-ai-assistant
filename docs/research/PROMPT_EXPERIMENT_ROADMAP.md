# Prompt Experiment Roadmap

> **Last Updated:** 2026-02-16  
> **Status:** Phase 1 Complete - Categories Hurt Quality

---

## Completed Experiments

### ✅ 1. Categorization Ablation (COMPLETE - Feb 2026)

**Question:** Is the 6-category system actually helping or hurting?

**Result:** Categories HURT reply quality. Universal prompt wins.

| Variant | Judge Avg | Anti-AI Rate | Verdict |
|---------|-----------|--------------|---------|
| universal | **3.97** | **0%** | ✅ **WINNER** |
| categorized | 3.63 | 3.3% | ❌ Loses |
| category_hint | 3.53 | 5.0% | ❌ Loses |

**Key Findings:**
- Category-specific prompts over-constrain the model
- Misclassification cascades (wrong category → wrong prompt → worse reply)
- Universal prompt allows natural adaptation to context

**Action Taken:**
- Removed category-specific system prompts from production
- Kept `skip_slm` categories (acknowledge/closing use templates)
- See [Categorization Ablation Findings](./CATEGORIZATION_ABLATION_FINDINGS.md)

---

### ✅ 2. Universal Prompt Optimization (COMPLETE - Feb 2026)

**Question:** What's the best universal prompt?

**Result:** Baseline prompt wins over creative variants.

| Prompt | Score | Notes |
|--------|-------|-------|
| **baseline** | **6.27** | Clear identity + constraints |
| minimal | 5.82 | Too vague |
| negative | 5.75 | Too many NO rules |
| persona | 5.72 | iPhone context not helpful |

**Winning Prompt:**
```
You are NOT an AI assistant. You are texting from your phone.
Reply naturally, matching the conversation style.
Be brief (1-2 sentences), casual, and sound like a real person.
```

---

## Active Experiments

### 🔄 3. LLM-as-Judge Infrastructure (PRODUCTION)

**Status:** Operational  
**Model:** Llama 3.3 70B via Cerebras (FREE tier)  
**Docs:** [LLM Judge Evaluation](./LLM_JUDGE_EVALUATION.md)

**Capabilities:**
- 0-10 quality scoring
- Anti-AI phrase detection
- Consistent evaluation (~0.78 correlation with human ratings)

**Usage:**
```bash
# Enable in any eval
uv run python evals/ablation_categorization.py --judge
uv run python evals/optimize_universal_prompt.py --judge
```

---

## Planned Experiments

### 📋 4. Context Window Optimization (NEXT)

**Question:** How much conversation history is optimal?

**Test:**
```python
context_depths = [0, 3, 5, 10, 15, 20]  # 15 is optimal for texting
formats = [
    "[timestamp] sender: message",  # Current
    "sender: message",               # No timestamps
    "- message",                     # Just messages
]
```

**Hypothesis:** 5-10 messages is optimal; more adds noise.

---

### 📋 5. Best-of-N Enhancement (PLANNED)

**Current:** 2 candidates (temp 0.1, 0.3), heuristic selection  
**Opportunities:**

1. **Better Selection Heuristics:**
```python
def score_candidate(reply, context):
    scores = {
        'brevity': max(0, 1 - len(reply) / 100),
        'lowercase_bonus': 0.1 if reply[0].islower() else 0,
        'no_ai_phrases': -0.5 if has_ai_phrases(reply) else 0,
        'style_match': style_similarity(reply, context),
    }
    return sum(scores.values())
```

2. **Adaptive N:**
```python
# Generate until confident or max N
for temp in [0.1, 0.2, 0.3, 0.4, 0.5]:
    reply = generate(temperature=temp)
    confidence = score_candidate(reply, context)
    if confidence >= 0.8:
        break
```

---

### 📋 6. RAG + Dynamic Examples (PLANNED)

**Current:** Static few-shot examples  
**Experiment:** Retrieve similar historical exchanges:

```python
def get_dynamic_examples(last_message: str, k: int = 3) -> list:
    similar = vector_search(last_message, limit=k)
    return [
        f"They said: {ex.trigger}\nYou replied: {ex.response}"
        for ex in similar
    ]
```

**Test:** Static vs dynamic vs hybrid

---

### 📋 7. Temperature/Sampling Grid Search (PLANNED)

**Current:** Fixed temperature=0.1  
**Test:**
```python
params = {
    'temperature': [0.1, 0.3, 0.5, 0.7],
    'top_p': [0.8, 0.9, 0.95],
    'top_k': [20, 40, 60],
    'repetition_penalty': [1.0, 1.05, 1.1, 1.15, 1.2],
}
```

**Goal:** Find optimal sampling for:
- Speed (lower temp = faster)
- Quality (higher temp = more diverse)
- Anti-AI rate (balance needed)

---

## Experiment Log

| Date | Experiment | Result | Action |
|------|------------|--------|--------|
| 2026-02-16 | Categorization ablation | Universal wins | Removed category prompts |
| 2026-02-16 | Universal prompt variants | Baseline wins | Using baseline in prod |
| 2026-02-16 | LLM judge setup | Operational | Judge available for all evals |
| TBD | Context window | Pending | - |
| TBD | Best-of-N | Pending | - |
| TBD | Dynamic examples | Pending | - |

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Judge Score (avg) | 6.27 | 7.0+ |
| Anti-AI Violations | 0% | <5% |
| Latency | 300ms | <500ms |
| User acceptance* | ? | >75% |

*Requires production A/B test

---

## Decision Framework

When running experiments:

```
1. Did it improve judge scores significantly (>0.3 pts)?
   → YES: Consider adoption
   → NO: Document and move on

2. Did it hurt latency beyond 10% increase?
   → YES: Optimize or reject
   → NO: Acceptable tradeoff if quality improved

3. Does it generalize across message types?
   → Test on all categories before merging

4. Is it robust to edge cases?
   → Test with short/long/ambiguous messages
```

---

## Related Documents

- [Categorization Ablation Findings](./CATEGORIZATION_ABLATION_FINDINGS.md) - Complete study
- [LLM Judge Evaluation](./LLM_JUDGE_EVALUATION.md) - Evaluation framework
- [Reply Pipeline Guide](../REPLY_PIPELINE_GUIDE.md) - Production system
