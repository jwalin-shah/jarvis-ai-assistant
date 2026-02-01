# Response Classifier System

This document describes the dialogue act classification system used for multi-option reply generation.

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
│ LAYER 3: DA Classifier (k-NN)       │  ~61% of cases
│ + Confidence threshold filtering    │  ~28% filtered to ANSWER
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

Validated on 135 manually-labeled samples:

| Category | Accuracy |
|----------|----------|
| REACT_POSITIVE | 86.4% |
| ANSWER | 77.3% |
| QUESTION | 72.7% |
| ACKNOWLEDGE | 72.7% |
| AGREE | 50.0% |
| DEFER | 45.5% |
| DECLINE | 40.9% |
| **OVERALL** | **64.4%** |

### Known Issues
- DECLINE is still over-predicted for negative-sentiment statements
- DEFER catches uncertain statements that aren't actually deferring
- AGREE sometimes matches agreement with statements, not just commitments

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

### Build DA Classifier
```bash
# Build with proportional sampling (recommended)
uv run python -m scripts.build_da_classifier --build --proportional

# Build with equal sampling
uv run python -m scripts.build_da_classifier --build

# Validate on test examples
uv run python -m scripts.build_da_classifier --validate
```

### Evaluate Classifier
```bash
# Full dataset distribution
uv run python -m scripts.eval_full_classifier

# Sample for manual validation
uv run python -m scripts.eval_full_classifier --validate 200

# Score manual validation
uv run python -m scripts.eval_full_classifier --score
```

### Mine Exemplars
```bash
# Mine exemplars from high-purity clusters
uv run python -m scripts.mine_da_exemplars --extract
```

## File Locations

| File | Purpose |
|------|---------|
| `jarvis/response_classifier.py` | Hybrid response classifier |
| `jarvis/multi_option.py` | Multi-option generation |
| `jarvis/retrieval.py` | DA-filtered retrieval |
| `scripts/build_da_classifier.py` | Build DA classifier indices |
| `scripts/eval_full_classifier.py` | Evaluate classifier |
| `scripts/mine_da_exemplars.py` | Mine exemplars from data |
| `~/.jarvis/da_classifiers/` | Classifier indices |
| `~/.jarvis/da_exemplars/` | Mined exemplars |
| `~/.jarvis/classifier_validation.json` | Manual validation samples |

## Future Improvements

1. **Better DECLINE/DEFER exemplars**: Mine more clear examples of actual declines/deferrals
2. **Context-aware classification**: Use trigger text to inform response classification
3. **Per-contact style**: Learn individual communication styles
4. **Active learning**: Use user feedback to improve classifications
