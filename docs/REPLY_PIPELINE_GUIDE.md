# Reply Pipeline Architecture Guide

> **Last Updated:** 2026-02-17  
> **Status:** Production - Direct-to-LLM

---

## Overview

This guide covers the complete architecture for generating text message replies that match your personal style.

**Key Design Decision:** We use a **direct-to-LLM** path - no classification at request time. The LLM decides IF and HOW to reply. Classification runs in **background (prefetch)** for analytics only.

See [Categorization Ablation Findings](./research/CATEGORIZATION_ABLATION_FINDINGS.md) for why category-specific prompts hurt quality.

---

## Architecture Components

### 1. Direct-to-LLM (Request Time)

**Purpose:** LLM decides IF and HOW to reply - no classification or RAG at request time

```python
# Fast path: skip obvious non-responses (reactions, acknowledgments)
if is_reaction(incoming) or is_acknowledgment_only(incoming):
    return ""  # Don't respond

# Build prompt: universal system prompt + conversation context + category examples
# NO RAG - research showed it causes hallucinations
prompt = build_prompt(conversation)

# LLM generates reply
reply = llm.generate(prompt)
```

**Benefits:**
- Faster (no ML classification, no RAG search at request time)
- Simpler (LLM handles all routing logic)
- Better quality (no RAG hallucinations, same as chatting directly with Ollama)

### 2. Classification (Background - Prefetch)

**Purpose:** Run classification in background for analytics, NOT at request time

Classification runs via the prefetch system and stores results in:
- `message_update.last_category` - category per message
- Analytics tables for dashboard/metrics

Categories (6 - for analytics only):

- `acknowledge` - "ok", "sure", "thanks" → uses template
- `closing` - "bye", "talk later" → uses template
- `question` - "What time?", "Where are you?" → universal prompt
- `request` - "Can you pick me up?" → universal prompt
- `emotion` - "omg!", "so excited!" → universal prompt
- `statement` - "Running late", "Just got home" → universal prompt

**Mobilization** (4 levels):

- `HIGH` - Question/request that needs answer
- `MEDIUM` - Emotional content, casual reply expected
- `LOW` - Statement, optional acknowledgment
- `NONE` - No reply needed

---

### 2. Context Assembly

**What to include in the prompt:**

```python
context_parts = {
    # Recent conversation (essential)
    "conversation_history": get_last_n_messages(10),

    # Contact knowledge (from fact DB)
    "facts": get_relevant_facts(contact_id, query=incoming_text, k=5),

    # Similar past replies (RAG)
    "similar_examples": search_similar_replies(incoming_text, k=3),

    # Relationship type
    "relationship": classify_relationship(contact_id),

    # Time context
    "time_since_last": time_since_last_message(),
}
```

---

### 3. Prompt Engineering

**Universal System Prompt (Current):**

```
You are NOT an AI assistant. You are texting from your phone.
Reply naturally, matching the conversation style.
Be brief (1-2 sentences), casual, and sound like a real person.
```

**Why this works:**
1. Clear identity (not AI)
2. Simple constraints (brief, casual)
3. No rigid category rules
4. Allows model to adapt naturally

See [Categorization Ablation Findings](./research/CATEGORIZATION_ABLATION_FINDINGS.md) for the study that proved categories hurt quality.

**Full Prompt Template:**

```
<|im_start|>system
You are NOT an AI assistant. You are texting from your phone.
Reply naturally, matching the conversation style.
Be brief (1-2 sentences), casual, and sound like a real person.
<|im_end|>
<|im_start|>user
{context_str}

Reply to: {last_message}<|im_end|>
<|im_start|>assistant
```

---

### 4. LLM Selection

| Model | Recommendation |
|-------|----------------|
| **LFM-1.2b-ft** | Default - best speed/quality balance |
| **LFM-0.3b-ft** | Testing only - very fast, lower quality |

**Configuration** (in model registry):

```python
{
    "name": "lfm-1.2b-ft",
    "max_tokens": 25,      # Keep replies brief (1-2 sentences)
    "temperature": 0.1,     # Deterministic but natural
    "top_p": 0.9,
    "repetition_penalty": 1.05,  # Prevents echoing input
}
```

---

### 5. RAG Implementation

**Setup:**

```python
from jarvis.search.hybrid_search import get_hybrid_searcher

# Get hybrid searcher (combines semantic + keyword)
searcher = get_hybrid_searcher()

# Search for similar conversations
results = searcher.search(
    query=incoming_text,
    limit=5,
    rerank=True  # Uses cross-encoder reranker
)

# Use as examples in prompt
examples = []
for r in results:
    examples.append({
        "they_said": r.trigger_text,
        "you_replied": r.response_text
    })
```

**Quality Filtering for RAG:**

Only index high-quality pairs:

- Response length: 3-100 characters
- Not acknowledgments only (avoid "ok" → "ok" training)
- Quality score > 0.7

---

### 6. Generation Pipeline

```python
from jarvis.reply_service import get_reply_service

reply_service = get_reply_service()

result = reply_service.generate_reply(
    context=MessageContext(
        message_text="Want to grab lunch?",
        chat_id="chat123",
    )
)

print(result.response)      # "sure what time?"
print(result.confidence)    # 0.85
print(result.metadata)      # {similarity_score, category, ...}
```

**Pipeline Steps:**

1. **Check health** - Can we use LLM?
2. **Classify** - Mobilization level (not for prompt selection)
3. **Search** - Hybrid RAG for similar examples
4. **Build prompt** - Universal system + context + examples
5. **Generate** - MLX inference
6. **Log** - Persist for analysis

---

### 7. Template Shortcuts (Fast Path)

For `acknowledge` and `closing` categories, skip LLM entirely:

```python
ACKNOWLEDGE_TEMPLATES = [
    "ok", "sounds good", "got it", "thanks", "np",
    "👍", "for sure", "alright", "bet", "cool"
]

CLOSING_TEMPLATES = [
    "bye!", "see ya", "later!", "talk soon",
    "ttyl", "peace", "catch you later", "gn"
]
```

**Why templates work for these categories:**
- Acknowledgments are formulaic
- Closings are ritualistic
- LLM would produce similar results slower

---

## Performance Targets

| Operation | P50 | Target |
|-----------|-----|--------|
| Classification | 12ms | <50ms ✅ |
| Context search | 3ms | <50ms ✅ |
| LLM generation | 180ms/token | <2s total ✅ |
| **Full pipeline** | **~300ms** | **<500ms** ✅ |

---

## Evaluation

### LLM-as-Judge

We use Llama 3.3 70B (Cerebras) to evaluate reply quality:

```bash
# Run evaluation with judge
uv run python evals/batch_eval.py --judge --input data/eval/test.jsonl
```

**Metrics:**
- Judge Score: 0-10 (target >6.0)
- Anti-AI Rate: % with AI-sounding phrases (target <5%)

See [LLM Judge Evaluation](./research/LLM_JUDGE_EVALUATION.md) for details.

### Current Baseline

| Metric | Value |
|--------|-------|
| Judge Score (avg) | 6.27/10 |
| Anti-AI Rate | 0% |
| Latency | ~300ms |

---

## Key Implementation Files

| File | Purpose |
|------|---------|
| `jarvis/reply_service.py` | Main reply generation service |
| `jarvis/prompts/constants.py` | Universal prompt, templates |
| `jarvis/prompts/builders.py` | Prompt assembly logic |
| `jarvis/search/hybrid_search.py` | RAG search |
| `evals/ablation_categorization.py` | Category ablation study |
| `evals/optimize_universal_prompt.py` | Prompt optimization |

---

## Next Steps

### Immediate

1. ✅ Removed category-specific prompts (done)
2. ✅ Adopted universal baseline prompt (done)
3. ✅ LLM judge operational (done)

### Planned Experiments

See [Prompt Experiment Roadmap](./research/PROMPT_EXPERIMENT_ROADMAP.md):

1. **Context window optimization** - How much history is optimal?
2. **Best-of-N enhancement** - Better candidate selection
3. **Dynamic examples** - Retrieve similar exchanges
4. **Temperature grid search** - Optimal sampling parameters

---

## Related Documents

- [Categorization Ablation Findings](./research/CATEGORIZATION_ABLATION_FINDINGS.md) - Why we removed categories
- [LLM Judge Evaluation](./research/LLM_JUDGE_EVALUATION.md) - Evaluation framework
- [Prompt Experiment Roadmap](./research/PROMPT_EXPERIMENT_ROADMAP.md) - Ongoing experiments
- [HOW_IT_WORKS.md](./HOW_IT_WORKS.md) - Overall system architecture
