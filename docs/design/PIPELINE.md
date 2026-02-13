# Classification & Routing Pipeline

> **Last Updated:** 2026-02-12

## Message Response Flow

When a new message arrives:

```
1. MESSAGE DETECTION
   chat.db → File Watcher → Push via Unix Socket (instant)

2. CATEGORY CLASSIFICATION (~15-30ms)
   ┌──────────────────────────────────────────┐
   │ LightGBM CATEGORY CLASSIFIER             │
   │ (BERT embeddings + hand-crafted features) │
   │ Categories: acknowledge/closing/emotion/   │
   │             question/request/statement    │
   └──────────────────────────────────────────┘

3. sqlite-vec SIMILARITY SEARCH (~5-10ms)
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

## Text Normalization

Before classification, all text is normalized:
```python
text = unicodedata.normalize("NFKC", text)  # Smart quotes → ASCII
text = " ".join(text.split())                # Collapse whitespace/newlines
```

See [TEXT_NORMALIZATION.md](./TEXT_NORMALIZATION.md) for details.

## Two-Layer Hybrid Classification

The category classifier uses a cascade of classifiers with heuristic post-processing:

```
LAYER 1: CASCADE CLASSIFIER (~10-20ms)
├─ Fast path: Pattern-based (reactions, acknowledgments)
├─ LightGBM classifier (BERT → BGE embeddings + hand-crafted features)
│   ├─ Input: BGE embedding (384d) + context BGE (384d) + hand-crafted (147d) = 915 features
│   ├─ OneVsRestClassifier(LGBMClassifier) for multi-label
│   └─ Categories: acknowledge, closing, emotion, question, request, statement
└─ Context BGE zeroed at inference (auxiliary supervision during training only)

LAYER 2: HEURISTIC POST-PROCESSING
├─ Fast path: reactions/acknowledgments → acknowledge
├─ Reaction messages ("Laughed at", "Loved") → emotion
├─ Imperative verbs at start → request
└─ Brief agreements → acknowledge
```

### Categories

| Category | Description | Example |
|----------|-------------|---------|
| **acknowledge** | Short acknowledgment/agreement | "Got it", "Sounds good", "OK" |
| **closing** | Conversation ending | "Talk later", "Bye", "See you" |
| **emotion** | Emotional/expressive response | "lol", "I'm so sorry", reactions |
| **question** | Questions requiring answers | "Want to grab lunch?", "How are you?" |
| **request** | Action requests/commands | "Can you send that?", "Let me know" |
| **statement** | Informational statements | "I'm on my way", "That was great" |

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
