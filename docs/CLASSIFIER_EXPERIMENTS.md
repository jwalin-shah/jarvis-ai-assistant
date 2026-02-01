# Classifier Experiments & Findings

**Date:** 2026-02-01
**Goal:** Find the best approach for reply suggestions

---

## What We Tried

### 1. Multiple Classifiers (Original Approach)

**Setup:** 5 separate classifiers
- IntentClassifier (user intent)
- MessageClassifier (incoming message type)
- HybridResponseClassifier (response dialogue act)
- DialogueActClassifier (kNN backbone)
- RelationshipClassifier (contact relationship)

**Result:** ❌ Too complex, redundant, confusing
- MessageClassifier and ResponseClassifier overlap significantly
- 54.5% accuracy on validation set
- DECLINE overpredicted (33% accuracy)
- AGREE overpredicted (17% accuracy)

---

### 2. Pure Regex Patterns

**Setup:** Structural patterns only (`jarvis/reply_suggester.py`)

```python
INVITATION: r"\b(want to|wanna|down to)\b.*\?"
QUESTION: r"\?\s*$"
GREETING: r"^(hey|hi|hello)\s*$"
etc.
```

**Result:** ❌ Too many false negatives
- 85% of messages fell to STATEMENT catch-all
- Only 15% got specific classification
- Missed informal patterns ("yo what mac u got" = question without ?)

---

### 3. Hybrid (Regex + Centroid Verification)

**Setup:** `jarvis/trigger_classifier.py`
1. Structural patterns (high precision regex)
2. Centroid similarity (light ML verification)
3. Fallback to STATEMENT

**Result:** ⚠️ Better distribution, but still errors
- STATEMENT: 30% (down from 85%)
- Other types: better spread
- But centroid made mistakes:
  - "Ok. Prob won't be back til tuesday" → classified as INVITATION (wrong)
  - "boy do i have some tea for yall" → classified as INVITATION (wrong)

---

### 4. Pure Retrieval (No Classification)

**Setup:** Just FAISS search, return user's past responses

```python
similar = faiss_search("Want to grab lunch?", k=5)
return [r.response for r in similar]
```

**Result:** ❌ High similarity ≠ appropriate response
- "I got the job!" retrieved "Can we cancel sorry" (wrong type!)
- "Can you pick me up?" retrieved "Costco socks..." (random)
- Similar TRIGGER doesn't mean similar RESPONSE is appropriate

---

### 5. Classification + Retrieval Combined

**Concept:** Use classification to filter retrieval results

```python
classification = classify_trigger(message)  # → GOOD_NEWS
valid_types = ["REACT_POSITIVE", "CONGRATS"]
retrieved = faiss_search(message)
filtered = [r for r in retrieved if r.type in valid_types]
```

**Result:** 🤔 Could work but complex
- Need to classify both triggers AND responses
- Still depends on classification accuracy
- Added complexity may not be worth it

---

## Key Insights

### 1. Classification is Hard for Informal Text
- "yo what mac u got" = question (no ?)
- "lmk" = request (looks like acronym)
- Users don't follow grammar rules

### 2. High Similarity ≠ Right Response Type
- FAISS finds similar-LOOKING messages
- But context determines appropriate response
- "Want to grab lunch?" vs "Want to grab my jacket?" → different response types

### 3. What Actually Matters for Suggestions
For a good suggestion, we need:
1. **Right type** (yes/no for invitations, info for questions)
2. **User's style** (how THEY talk)
3. **Diversity** (offer different options)

---

## Recommended Approach

### Simple 2-Tier System

```
Message comes in
       ↓
┌─────────────────────────────────────────┐
│ TIER 1: Simple Patterns (Templates)     │
│                                         │
│ Detect: greetings, acks, simple y/n     │
│ Return: Personalized templates          │
│ (based on user's common responses)      │
│                                         │
│ Example:                                │
│   "hey" → ["hey!", "what's up", "yo"]   │
│   (pulled from user's actual greetings) │
└─────────────────────────────────────────┘
       ↓ (if not simple)
┌─────────────────────────────────────────┐
│ TIER 2: Retrieval + LLM                 │
│                                         │
│ 1. Retrieve similar triggers from FAISS │
│ 2. Get user's responses as examples     │
│ 3. LLM generates in user's style        │
│ 4. Return 3 diverse options             │
│                                         │
│ Example:                                │
│   "Want to grab lunch tomorrow?"        │
│   Retrieved examples of user saying:    │
│     - "yeah down"                       │
│     - "can't tomorrow"                  │
│     - "where at?"                       │
│   LLM generates similar style options   │
└─────────────────────────────────────────┘
```

### Why This Works

1. **Simple cases are fast** - No ML needed for "hey" → "hey!"
2. **Complex cases use LLM** - It understands context better than classification
3. **Retrieval provides style** - User's actual words, not generic templates
4. **Diversity is natural** - LLM can generate varied options

### What We Need

1. **User's common responses** - Mine from their history
   - Greetings they use
   - Acknowledgments they use
   - Reactions they use

2. **Relationship profiles** - Already have these!
   - Style (casual/formal)
   - Common topics
   - Message patterns

3. **LLM with style prompt** - Use relationship profile + retrieved examples

---

## Files Created/Modified

| File | Purpose | Status |
|------|---------|--------|
| `jarvis/reply_suggester.py` | Simple regex-based suggester | Created, needs improvement |
| `jarvis/trigger_classifier.py` | Hybrid trigger classifier | Created, experimental |
| `jarvis/response_classifier.py` | Response DA classifier | Existing, thresholds tuned |
| `tests/unit/test_reply_suggester.py` | Tests for suggester | Created |

---

## Next Steps

1. [ ] Mine user's common responses by type (greetings, acks, reactions)
2. [ ] Build simple pattern detector for Tier 1
3. [ ] Integrate LLM generation for Tier 2
4. [ ] Use relationship profiles for style
5. [ ] Test end-to-end on real conversations

---

## Metrics to Track

| Metric | How to Measure |
|--------|----------------|
| Suggestion relevance | User picks suggested vs types custom |
| Style match | Embedding similarity to user's actual responses |
| Diversity | Are 3 options actually different? |
| Latency | < 500ms for Tier 1, < 2s for Tier 2 |
