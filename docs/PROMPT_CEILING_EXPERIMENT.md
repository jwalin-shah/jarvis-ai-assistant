# Prompt Ceiling Experiment - Results

**Date:** 2026-02-22  
**Status:** Complete  

---

## Executive Summary

**Prompting ceiling on 1.2B model: ~1-2/10**

After extensive testing with preprocessing, constraints, and simplified prompts, the 1.2B model cannot generate coherent, context-appropriate replies reliably. The model falls back to:
- Generic greetings: "hey", "hey!"
- Pattern echoing: "bc", "btw"
- Meta-commentary: "it sounds like you're..."
- Emojis and stage directions (👋, 😊)

---

## Tested Configurations

| Model | Prompt | Preprocessing | Constraints | Score | Output Examples |
|-------|--------|---------------|-------------|-------|-----------------|
| 0.7B | default | none | none | 0.5/10 | `bc, wanna chat more`, numbers |
| 1.2B Base | bare | none | none | 0.17/10 | `1 2 3 4...`, gibberish |
| 1.2B Instruct | bare | none | none | 0.0/10 | `[response here]`, meta |
| 1.2B Instruct | default | none | none | ~2/10 | `gonna clarify, btw` |
| **1.2B Instruct** | **clean** | **yes** | **yes** | **~1/10** | **"hey", "hey!"** |

### Preprocessing Applied
- Strip phone numbers → "Them"
- Limit context to 3 messages
- Remove reactions/attachments
- Clean unicode

### Constraints Applied
- Stop sequences: `:`, `\n`, `Them:`
- Post-process: strip prefixes, emojis, meta-commentary
- Lower repetition penalty (1.1)
- Max 20 tokens

### Prompts Tested
```
clean: "You are texting a friend. Reply briefly and casually."
ultra: "Reply:"
```

---

## Key Findings

### 1. Model Capacity is the Bottleneck

At 1.2B parameters, the model cannot:
- Maintain conversation context
- Generate varied, appropriate responses
- Follow negative instructions ("no emoji")
- Distinguish input format from output generation

### 2. Preprocessing Helps But Doesn't Fix

Cleaning inputs reduces noise but the model still defaults to:
- Generic safe responses: "hey"
- Echoing patterns from training
- Short, non-committal replies

### 3. Stop Sequences Work

The `:` stop sequence successfully prevents `Them:` from being generated. Post-processing cleans up remaining artifacts. But the underlying generation quality is still poor.

---

## Conclusion

**The 1.2B model is too small for this task.**

Prompt engineering, preprocessing, and output constraints can prevent the worst failures (gibberish, meta-commentary, emojis), but cannot make the model generate contextually appropriate, varied responses.

### Recommendation: Abandon Generation

| Approach | Expected Quality | Effort |
|----------|-----------------|--------|
| Better prompting | 1 → 2/10 | High |
| Larger model (7B+) | Unknown | High (hardware) |
| **Semantic retrieval** | **6-8/10** | **Medium** |
| **Template matching** | **5-7/10** | **Low** |

**Pivot to retrieval:** Use the 1.2B model for embeddings, find similar past conversations, return your actual replies.

---

## Files

- `evals/prompt_ceiling_eval.py` - Evaluation script (cleaned version)
- `results/prompt_ceiling/` - Output results
