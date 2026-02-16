# Category Classification: Current Architecture

## Decision Summary

**We use category classification for routing and configuration, but NOT for prompt selection.**

Research showed that category-specific system prompts **hurt** reply quality compared to a universal prompt. However, we still classify messages for:

1. **Routing** (`skip_slm`) - Fast template responses for acknowledge/closing
2. **Context depth** - How much conversation history to include
3. **Analytics** - Understanding message types

---

## What Changed

### Before (Categorized Prompts)
```python
CATEGORY_CONFIGS = {
    "question": CategoryConfig(
        system_prompt="They asked a question. Just answer it, keep it short."
    ),
    "request": CategoryConfig(
        system_prompt="They're asking you to do something. Say yes, no, or ask a follow-up."
    ),
    # ... different prompt per category
}
```

**Problem:** Over-specification forced rigid response patterns. When classifier was wrong (15% of time), wrong prompt → worse reply.

### After (Universal Prompt)
```python
# Single universal system prompt for ALL categories
SYSTEM_PREFIX = (
    "You are texting from your phone. Reply naturally, matching their style. "
    "Be brief (1-2 sentences), casual, and sound like a real person.\n"
)

CATEGORY_CONFIGS = {
    "question": CategoryConfig(
        system_prompt=None,  # Uses universal SYSTEM_PREFIX
        context_depth=15,    # But still varies by category
        skip_slm=False,
    ),
    "acknowledge": CategoryConfig(
        system_prompt=None,  # Uses universal SYSTEM_PREFIX
        context_depth=0,
        skip_slm=True,       # Use template, not LLM
    ),
    # ... system_prompt=None for all
}
```

**Benefits:**
- More natural responses (not forced into patterns)
- No misclassification cascades
- Simpler codebase
- Better scores on evaluation (6.27 vs 5.20)

---

## What We Keep

### 1. Category Classification (The Label)

We still run the category classifier to label each message:
```python
classification = classify_message(text)  # "question", "request", etc.
```

**Used for:**
- **Analytics** - Track message type distribution
- **Context depth** - `CATEGORY_CONFIGS[category].context_depth`
- **Routing** - `CATEGORY_CONFIGS[category].skip_slm`

### 2. Context Depth per Category

Different categories need different amounts of context:
| Category | Context Depth | Why |
|----------|---------------|-----|
| acknowledge | 0 | Don't need context to say "ok" |
| closing | 0 | Don't need context to say "bye" |
| question | 15 | Need context to answer properly |
| request | 15 | Need context to respond |
| emotion | 15 | Need context to empathize |
| statement | 15 | Need context to react |

### 3. Skip-SLM Routing

Some categories use fast templates instead of LLM:
```python
if category in ("acknowledge", "closing"):
    return template_response(category)  # Fast, 0ms
else:
    return generate_with_llm(prompt)    # Slow, 200-500ms
```

---

## Research Results

### Ablation Study Results

| Approach | Judge Score | Anti-AI Rate |
|----------|-------------|--------------|
| **Universal Prompt** | **5.58/10** | **0%** |
| Categorized Prompts | 5.20/10 | 3.3% |
| Category Hint | 5.51/10 | 5.0% |

**Conclusion:** Universal prompt wins on quality and reliability.

### Why Categories Hurt

1. **Over-specification** - Forced rigid patterns
2. **Misclassification cascades** - Wrong category → wrong prompt → bad reply
3. **Artificial boundaries** - Real texting doesn't fit into 6 buckets

See full analysis: [CATEGORIZATION_ABLATION_FINDINGS.md](./CATEGORIZATION_ABLATION_FINDINGS.md)

---

## Code Architecture

### Flow

```
Incoming Message
    ↓
Category Classifier → Label: "question" (for analytics/routing)
    ↓
Get CategoryConfig:
  - context_depth: 15 (from config)
  - skip_slm: False (from config)
  - system_prompt: None → use universal SYSTEM_PREFIX
    ↓
Build Prompt with:
  - Universal SYSTEM_PREFIX
  - {context_depth} messages of context
    ↓
Generate Reply
```

### Key Files

| File | Role |
|------|------|
| `jarvis/prompts/constants.py` | `SYSTEM_PREFIX` (universal), `CATEGORY_CONFIGS` |
| `jarvis/classifiers/category_classifier.py` | Classification model |
| `jarvis/reply_service_generation.py` | Uses `context_depth` and `skip_slm` from config |

---

## Future Possibilities

### 1. Confidence-Based Routing
```python
if category_confidence < 0.7:
    # Use universal (safe fallback)
else:
    # Could try category-specific (experimental)
```

### 2. Category for Examples (Not Prompts)
Use category to select few-shot examples instead of system prompts:
```python
examples = get_examples_for_category(category)  # Similar past exchanges
prompt = build_universal_prompt(examples=examples)  # Same system prompt
```

### 3. Per-Contact Style (Better Than Category)
Replace category with contact-specific style analysis:
```python
style = analyze_contact_style(contact_id)  # "formal", "emoji-heavy", etc.
prompt = adapt_universal_prompt(style)  # Same base, style tweaks
```

---

## Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| Category Classification | ✅ Active | Used for routing & analytics |
| Category-Specific Prompts | ❌ Removed | Universal prompt works better |
| Context Depth per Category | ✅ Active | Different categories need different context |
| Skip-SLM Routing | ✅ Active | Fast templates for ack/closing |
| Universal Prompt | ✅ Active | Single `SYSTEM_PREFIX` for all |

---

*Document Version: 1.0*
*Last Updated: 2025-01-20*
*Decision: Use universal prompts, keep categorization for routing*
