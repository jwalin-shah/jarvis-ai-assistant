# Classifier System

This document describes the classification systems used for trigger and response classification.

## Trigger Classifier

Located in `jarvis/trigger_classifier.py`. Classifies **incoming messages** to determine what type of response is needed.

### Trigger Types (5 labels)

| Type | Description | Examples |
|------|-------------|----------|
| COMMITMENT | Invitations/requests | "Want to grab lunch?", "Can you pick me up?" |
| QUESTION | Yes/no or info questions | "What time?", "Did you finish?" |
| REACTION | Emotional content | "That's crazy!", "I got the job!" |
| SOCIAL | Greetings/acks/tapbacks | "hey", "ok", "Loved 'message'" |
| STATEMENT | Neutral info sharing | "I'm on my way", "Meeting at 3pm" |

### Architecture

```
Input Message
    │
    ▼
┌─────────────────────────────────────┐
│ LAYER 1: Structural Patterns        │  Regex for tapbacks, greetings,
│ (high precision, fast)              │  WH-questions, invitations
└─────────────────────────────────────┘
    │ no high-confidence match
    ▼
┌─────────────────────────────────────┐
│ LAYER 2: Trained SVM Classifier     │  Embedding-based, 82% macro F1
│ (C=10, gamma=scale, 6000 examples)  │  Uses bge-small embeddings
└─────────────────────────────────────┘
    │ low confidence
    ▼
┌─────────────────────────────────────┐
│ FALLBACK: STATEMENT                 │
└─────────────────────────────────────┘
```

### Accuracy

**82.0% macro F1** [95% CI: 79.3% - 84.4%] on held-out test set (973 examples)

| Class | F1 Score | Key Signals |
|-------|----------|-------------|
| QUESTION | 86.6% | 37% end with "?", WH-words |
| STATEMENT | 86.2% | "I" statements, neutral info |
| SOCIAL | 85.3% | Tapbacks (32%), greetings, acks |
| COMMITMENT | 76.8% | "wanna", "can you", "let's" |
| REACTION | 75.2% | Emotional words (damn, bro, crazy) |

Training: `experiments/trigger/` (coarse_search → final_eval)

### Usage

```python
from jarvis.trigger_classifier import classify_trigger, TriggerType

result = classify_trigger("Want to grab lunch?")
print(result.trigger_type)  # TriggerType.COMMITMENT
print(result.confidence)    # 0.95
print(result.is_commitment) # True
```

### Training

```bash
# Train with multiple sampling strategies, save best model
uv run python -m scripts.train_trigger_classifier --save-best

# Analyze patterns in labeled data
uv run python -m scripts.analyze_trigger_patterns
```

### File Locations

| File | Purpose |
|------|---------|
| `jarvis/trigger_classifier.py` | Hybrid trigger classifier |
| `data/trigger_labeling.jsonl` | 4,865 labeled examples |
| `data/trigger_new_batch_3000.jsonl` | 3,000 auto-labeled examples |
| `~/.jarvis/trigger_classifier_model/` | Trained SVM model |
| `experiments/trigger/` | Training experiment scripts |

---

## Response Classifier

Located in `jarvis/response_classifier.py`. Classifies **response messages** into dialogue act types.

## Overview

The system classifies response messages into dialogue act types (AGREE, DECLINE, DEFER, ANSWER, QUESTION, etc.) to enable:
1. **Multi-option generation**: Generate diverse responses for commitment questions
2. **DA-filtered retrieval**: Find examples of specific response types
3. **Better reply routing**: Route based on expected response type

## Architecture

### Three-Layer Hybrid Classifier

Located in `jarvis/response_classifier.py`:

```
Input Text
    │
    ▼
┌─────────────────────────────────────┐
│ LAYER 1: Structural Patterns        │  ~11% of cases
│ (regex: "?", "yes", "no", etc.)     │  HIGH precision
└─────────────────────────────────────┘
    │ no match
    ▼
┌─────────────────────────────────────┐
│ LAYER 2: Centroid Verification      │  Verifies structural hints
│ (is embedding close to class mean?) │  semantically
└─────────────────────────────────────┘
    │ no structural hint
    ▼
┌─────────────────────────────────────┐
│ LAYER 3: SVM Classifier             │  ~61% of cases
│ + Confidence threshold filtering    │  81.9% macro F1
└─────────────────────────────────────┘
```

### Response Types

| Type | Description | Examples |
|------|-------------|----------|
| AGREE | Positive acceptance | "Yes", "Sure", "I'm down" |
| DECLINE | Rejection | "No", "Can't", "Sorry" |
| DEFER | Non-committal | "Maybe", "Let me check" |
| ANSWER | Provides information | "2pm works", "At the mall" |
| QUESTION | Asks for info | "What time?", "Where?" |
| ACKNOWLEDGE | Simple confirmation | "Ok", "Got it", "Cool" |
| REACT_POSITIVE | Positive reaction | "Congrats!", "Nice!" |
| REACT_SYMPATHY | Sympathetic reaction | "I'm sorry", "That sucks" |
| GREETING | Greeting | "Hey", "Hi" |
| STATEMENT | General statement | Catch-all |

## Key Improvements

### 1. Confidence Threshold (Option A)
If DA classifier confidence < 0.5, default to ANSWER. This prevents over-prediction of DECLINE/DEFER/AGREE when the classifier isn't confident.

### 2. Trigger Filtering (Option B)
Commitment responses (AGREE/DECLINE/DEFER) are only valid for commitment triggers (INVITATION, REQUEST, YN_QUESTION). For other triggers, these are filtered out.

### 3. Better Exemplars (Option C)
Mined 3,300+ clear exemplars from user data using structural patterns. These high-precision examples improve the DA classifier.

### 4. Proportional Sampling
Instead of equal 500/class sampling (which over-predicted minority classes), use proportional targets:

| Class | Target | Rationale |
|-------|--------|-----------|
| ANSWER | 2000 | Most responses are explanations/info |
| STATEMENT | 800 | Opinions, status updates |
| ACKNOWLEDGE | 600 | Common acknowledgments |
| QUESTION | 600 | Clarifications |
| REACT_POSITIVE | 500 | Positive reactions |
| AGREE | 400 | Commitment acceptance |
| DECLINE | 400 | Commitment rejection |
| DEFER | 300 | Non-committal |
| REACT_SYMPATHY | 200 | Sympathy |
| GREETING | 200 | Greetings |

## Accuracy

**81.9% macro F1** [95% CI: 78.4% - 84.9%] on held-out test set (971 examples)

| Class | F1 Score |
|-------|----------|
| QUESTION | 89.6% |
| STATEMENT | 87.2% |
| ACKNOWLEDGE | 83.3% |
| DECLINE | 80.0% |
| AGREE | 79.2% |
| DEFER | 71.8% |

Training: `scripts/train_response_classifier.py`

### Known Issues
- DEFER is weakest class (71.8% F1) - often confused with STATEMENT
- AGREE has precision issues (72.4%) - some STATEMENT misclassified as AGREE

## Usage

### Response Classifier

```python
from jarvis.response_classifier import get_response_classifier

classifier = get_response_classifier()
result = classifier.classify("Yeah I'm down!")

print(result.label)       # ResponseType.AGREE
print(result.confidence)  # 0.95
print(result.method)      # "structural_verified"
```

### Multi-Option Generation

```python
from jarvis.multi_option import generate_response_options

result = generate_response_options("Want to grab lunch?")

for option in result.options:
    print(f"{option.response_type}: {option.text}")
# AGREE: Yeah I'm down!
# DECLINE: Can't today, sorry
# DEFER: Let me check my schedule
```

### Typed Retrieval

```python
from jarvis.retrieval import get_typed_retriever
from jarvis.response_classifier import ResponseType

retriever = get_typed_retriever()
examples = retriever.get_typed_examples(
    trigger="Want to hang out?",
    target_response_type=ResponseType.AGREE,
    k=5
)

for ex in examples:
    print(f"{ex.response_text} (similarity: {ex.similarity:.2f})")
```

## Scripts

### Train Classifier
```bash
# Train response classifier with multiple sampling strategies
uv run python -m scripts.train_response_classifier --save-best

# Train trigger classifier
uv run python -m scripts.train_trigger_classifier --save-best
```

### Evaluate Classifier
```bash
# Full dataset distribution
uv run python -m scripts.eval_full_classifier

# Sample for manual validation
uv run python -m scripts.eval_full_classifier --validate 200
```

## File Locations

| File | Purpose |
|------|---------|
| `jarvis/response_classifier.py` | Hybrid response classifier |
| `jarvis/multi_option.py` | Multi-option generation |
| `jarvis/retrieval.py` | DA-filtered retrieval |
| `scripts/train_response_classifier.py` | Train SVM response classifier |
| `scripts/eval_full_classifier.py` | Evaluate classifier |
| `~/.jarvis/response_classifier_model/` | Trained SVM model |

## Future Improvements

1. **Better DECLINE/DEFER exemplars**: Mine more clear examples of actual declines/deferrals
2. **Context-aware classification**: Use trigger text to inform response classification
3. **Per-contact style**: Learn individual communication styles
4. **Active learning**: Use user feedback to improve classifications
