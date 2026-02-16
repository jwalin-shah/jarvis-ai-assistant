# Categorization Ablation Study: Findings & Recommendations

> **Last Updated:** 2026-02-16  
> **Status:** Complete - Results Analyzed  
> **Experiment:** `evals/ablation_categorization.py`

---

## Executive Summary

**Key Finding: Category-specific prompts HURT reply quality compared to a universal prompt.**

Our hypothesis that categorization (question, request, emotion, statement, etc.) would improve replies by providing targeted guidance was **incorrect**. The data shows that removing category-specific instructions and using a single universal prompt produces better results.

---

## Experiment Design

### Variants Tested

| Variant | Description | System Prompt Approach |
|---------|-------------|------------------------|
| `categorized` | Current system - 6 category-specific prompts | Different prompt per category |
| `universal` | Single prompt for all messages | One universal instruction |
| `category_hint` | Category mentioned but not prescriptive | Category as context, not instruction |

### Dataset

- **60 evaluation examples** across all 6 categories
- Balanced distribution: question, request, emotion, statement, acknowledge, closing
- Real iMessage conversations with ideal human responses

### Judge

- **Model:** Llama 3.3 70B via Cerebras API
- **Scoring:** 0-10 scale on naturalness, appropriateness, match to ideal
- **Criteria:** Does it sound like a real text? Is it appropriate? Does it match intent?

---

## Results

### Overall Performance

| Variant | Judge Avg | Judge Median | Anti-AI Rate | Latency |
|---------|-----------|--------------|--------------|---------|
| **universal** | **3.97** | **4.0** | **0%** | **264ms** |
| categorized | 3.63 | 2.0 | 3.3% | 286ms |
| category_hint | 3.53 | 2.0 | 5.0% | 313ms |

**Winner: Universal prompt** - higher average score, lower anti-AI rate, faster.

### Per-Category Breakdown

| Category | Categorized | Universal | Category Hint | Winner |
|----------|-------------|-----------|---------------|--------|
| question | 2.2 | **2.8** | **3.8** | category_hint (but all low) |
| request | 4.4 | **5.6** | 5.2 | universal |
| emotion | 3.8 | **3.8** | 3.2 | tie / universal |
| statement | 2.8 | **3.4** | **3.6** | category_hint |
| acknowledge | 3.4 | **3.4** | 1.8 | tie / universal |
| closing | **5.2** | 4.8 | (low sample) | categorized |

**Key insight:** Universal wins on 4/6 categories. Category_hint occasionally wins but with higher variance.

---

## Why Categories Hurt

### 1. Over-Specification

Category-specific prompts like:
```
"They're asking you to do something. Say yes, no, or ask a follow-up."
```

Force the model into a rigid response pattern that may not match the actual conversation flow.

### 2. Misclassification Cascades

When the category classifier is wrong (even 15% error rate), the wrong prompt is used → worse reply.

Example:
- Message: "That sounds amazing! I'm so happy for you"
- Classified as: `statement`
- Actual need: `emotion` response
- Result: Flat reply instead of celebratory

### 3. Loss of Natural Flow

Real texting doesn't fit into 6 buckets. People mix:
- Questions with emotion ("Are you serious?! 😱")
- Requests with statements ("Can you pick up milk? Running low")

Category forcing creates artificial boundaries.

---

## Universal Prompt Optimization

Following the ablation, we tested 6 universal prompt variations:

### Results

| Prompt | Avg Score | Anti-AI Rate | Notes |
|--------|-----------|--------------|-------|
| **baseline** | **6.27** | **0%** | Clear identity + constraints |
| minimal | 5.82 | 1.7% | Too vague |
| negative | 5.75 | 0% | Too many "NO" rules |
| persona | 5.72 | 0% | iPhone context not helpful |
| style_focused | 5.70 | 0% | Style matching alone insufficient |

### Winning Baseline Prompt

```
You are NOT an AI assistant. You are texting from your phone.
Reply naturally, matching the conversation style.
Be brief (1-2 sentences), casual, and sound like a real person.
```

**Why it works:**
1. Clear identity (not AI)
2. Simple constraint (brief, casual)
3. No rigid category rules
4. Allows model to adapt to context naturally

---

## Recommendations

### Immediate Actions

1. **Remove category-specific system prompts**
   - Keep category classification for analytics
   - Don't use it to select prompts

2. **Adopt universal baseline prompt**
   - Replace per-category configs in `CATEGORY_CONFIGS`
   - Single `SYSTEM_PREFIX` for all generation

3. **Keep `skip_slm` categories**
   - `acknowledge` and `closing` still use templates (fast, appropriate)
   - These don't need LLM generation

### Code Changes

```python
# BEFORE (in jarvis/prompts/constants.py)
CATEGORY_CONFIGS = {
    "question": CategoryConfig(
        system_prompt="They asked a question. Just answer it, keep it short."
    ),
    "request": CategoryConfig(
        system_prompt="They're asking you to do something. Say yes, no, or ask a follow-up."
    ),
    # ... etc
}

# AFTER
UNIVERSAL_SYSTEM_PROMPT = (
    "You are NOT an AI assistant. You are texting from your phone. "
    "Reply naturally, matching the conversation style. "
    "Be brief (1-2 sentences), casual, and sound like a real person."
)

CATEGORY_CONFIGS = {
    "question": CategoryConfig(system_prompt=None),  # Use universal
    "request": CategoryConfig(system_prompt=None),   # Use universal
    # ... etc
}
```

### Future Work

1. **Confidence-based routing**
   - If category confidence < 0.7, use universal
   - If confidence > 0.9, could try category hint (experimental)

2. **Dynamic examples**
   - Instead of category-specific prompts, use category to select few-shot examples
   - May preserve benefits without rigid instructions

3. **Per-contact style adaptation**
   - Replace category with contact-specific style analysis
   - More relevant than message category

---

## Appendix: Raw Data

Full results saved to:
- `results/ablation_categorization.json` - Full ablation results
- `results/universal_prompt_optimization.json` - Prompt variant results

### Reproducing

```bash
# Run full ablation
uv run python evals/ablation_categorization.py --variant all --judge

# Test universal prompt variants
uv run python evals/optimize_universal_prompt.py --judge
```

---

## Related Documents

- [LLM Judge Evaluation](./LLM_JUDGE_EVALUATION.md) - How we evaluate with LLM-as-judge
- [Prompt Experiment Roadmap](./PROMPT_EXPERIMENT_ROADMAP.md) - Ongoing experiments
- [Reply Pipeline Guide](../REPLY_PIPELINE_GUIDE.md) - Updated pipeline documentation
