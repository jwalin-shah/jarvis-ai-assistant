# Classification & Routing Pipeline

## Message Response Flow

When a new message arrives:

```
1. MESSAGE DETECTION
   chat.db → File Watcher → Push via Unix Socket (instant)

2. PARALLEL CLASSIFICATION (~15-30ms)
   ┌─────────────────────┐    ┌─────────────────────┐
   │ MESSAGE CLASSIFIER  │    │ TRIGGER CLASSIFIER  │
   │ (what message IS)   │    │ (what response NEEDS)│
   │ QUESTION/STATEMENT/ │    │ COMMITMENT/QUESTION/ │
   │ ACKNOWLEDGMENT/etc  │    │ REACTION/SOCIAL/etc │
   └─────────────────────┘    └─────────────────────┘

3. FAISS SIMILARITY SEARCH (~5-10ms)
   Query embedding → Top-K similar (trigger, response) pairs

4. ROUTING DECISION
   Similarity >= 0.95  → QUICK_REPLY (cached response)
   0.65 - 0.95         → GENERATE with good few-shot examples
   0.45 - 0.65         → GENERATE cautiously (fewer examples)
   < 0.45              → CLARIFY (ask user for context)

5. RESPONSE GENERATION (~200-500ms if needed)
   MLX LLM with: context + few-shot examples + relationship profile

6. DELIVERY
   Stream tokens via Unix socket → Desktop app
```

## Three-Layer Hybrid Classification

Both classifiers use the same pattern:

```
LAYER 1: STRUCTURAL PATTERNS (Regex, <1ms)
├─ r"^(yes|yeah|yep)[\\s!.]*$" → AGREE (0.95)
├─ r"^(no|nope|nah)[\\s!.]*$" → DECLINE (0.95)
└─ r"\\?\\s*$" → QUESTION (0.60)

LAYER 2: CENTROID VERIFICATION (Semantic check)
├─ Compute embedding distance to ALL class centroids
├─ If hint_similarity >= 0.65: CONFIRM structural hint
└─ If other class significantly closer: OVERRIDE hint

LAYER 3: SVM FALLBACK (For ambiguous cases)
└─ Per-class confidence thresholds (tuned on validation)
```

### Why Centroids?

Structural patterns have edge cases:

| Text | Structural Hint | Reality | Centroid Override |
|------|-----------------|---------|-------------------|
| "No way!" | DECLINE | REACT_POSITIVE | Yes |
| "Yeah right" | AGREE | Sarcastic DECLINE | Yes |
| "Sure..." | AGREE | Reluctant DEFER | Yes |

### Per-Class SVM Thresholds

```python
SVM_THRESHOLDS = {
    TriggerType.COMMITMENT: 0.50,  # High stakes
    TriggerType.QUESTION: 0.35,    # Clear from structure
    TriggerType.REACTION: 0.40,    # Moderate
    TriggerType.SOCIAL: 0.25,      # Strong patterns
    TriggerType.STATEMENT: 0.30,   # Catch-all
}
```

## Routing Thresholds Explained

**Why 0.95 for quick reply?**
- Same question from different people needs different responses
- "Want to grab lunch?" from boss vs friend = different response
- Only near-exact matches (same person, same context) are safe

**Why 0.65 for generation with context?**
- At 0.65+, similar examples are good few-shot prompts
- Below 0.65, examples might mislead the model

## Coherence Scoring

Before returning a quick reply:

```python
def score_response_coherence(trigger: str, response: str) -> float:
    """Score how well a response fits a trigger."""
    # Filters out responses that matched by trigger but don't fit

COHERENCE_THRESHOLD = 0.6
```

## Key Finding from Evaluation

> On 26K holdout pairs: High trigger similarity does NOT guarantee appropriate response.
> Even at 0.9+ trigger match, response similarity was only 0.56.

**Conclusion:** Use retrieval for **few-shot examples**, let LLM adapt to current context.
