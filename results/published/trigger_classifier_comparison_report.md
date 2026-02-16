# Trigger Classifier Comparison Report

## Summary

Trained and evaluated trigger classifiers on two label schemes:
1. **Corrected (10 labels)**: Fine-grained categories
2. **Consolidated (7 labels)**: Merged related categories

## Results

### Training Data Only (1,273 samples)

| Metric | Corrected (10) | Consolidated (7) |
|--------|----------------|------------------|
| Accuracy | 73.3% | 79.2% |
| Macro F1 | 74.1% | 79.0% |
| Weighted F1 | 73.3% | 79.2% |

### Training + Candidates Data (4,273 samples)

| Metric | Corrected (10) | Consolidated (7) |
|--------|----------------|------------------|
| Accuracy | 80.0% | **87.0%** |
| Macro F1 | 69.8% | 76.1% |
| Weighted F1 | 79.7% | **86.8%** |

## Key Findings

### 1. Consolidated Labels Perform Better
The 7-label model achieves **87% accuracy** vs 80% for the 10-label model with the larger dataset. This makes sense - fewer classes to distinguish.

### 2. Per-Class Performance (10-label model with 4273 samples)
| Label | F1 Score | Issues |
|-------|----------|--------|
| statement | 87% | Dominant class |
| request | 82% | Good |
| ack | 78% | Good |
| yn_question | 73% | Good |
| info_question | 71% | Confused with yn_question |
| invitation | 61% | Confused with request, yn_question |
| reaction | 59% | Confused with statement |
| bad_news | 58% | Confused with statement |
| good_news | 49% | **Worst** - confused with statement |

### 3. Per-Class Performance (7-label model with 4273 samples)
| Label | F1 Score | Issues |
|-------|----------|--------|
| statement | 93% | Dominant class |
| ack | 85% | Good |
| greeting | 81% | Good |
| question | 77% | Good (merged yn/info) |
| request | 70% | Some confusion with statement |
| commitment | 66% | Confused with statement |
| reaction | 62% | Confused with statement |

## Structural Pattern Analysis

Current structural patterns cover only **30.5%** of samples.

### Coverage by Label (on 1273 training samples)
| Label | Coverage | Precision | Issue |
|-------|----------|-----------|-------|
| yn_question | 57.1% | 99% | Fallback `?$` catches too much |
| request | 51.9% | 76% | Good |
| ack | 48.7% | 95% | Good |
| info_question | 38.6% | 39% | **BAD** - routed to yn_question |
| invitation | 35.2% | 62% | Many routed to yn_question |
| greeting | 32.1% | 70% | Missing casual variants |
| bad_news | 19.7% | 100% | Low coverage |
| good_news | 13.3% | 94% | Low coverage |
| reaction | 9.0% | 11% | **VERY BAD** - routed to bad_news |
| statement | 4.8% | 0% | Expected (fallback) |

### Structural Pattern Improvements Needed

**1. Fix info_question misclassification**
The fallback `?$` pattern routes all questions to yn_question. Add more specific info_question patterns:
```regex
# How questions (not just "how" at start)
\b(how|what|why)\b.*\?$

# Add "how's" pattern
^how('?s)?\s+\w+.*\?
```

**2. Add more greeting patterns**
Missing: "heyyy", "Hello?😭", "happy thanksgiving", "what's good"
```regex
# Extended greetings
^(hey+|hi+|hello+|yo+)[\s!?😭]*$
^(happy|merry)\s+(thanksgiving|christmas|birthday|holiday)
```

**3. Fix reaction misclassification**
Reactions like "Ugh thats terrible" get classified as bad_news. Add reaction patterns:
```regex
# Positive reactions
\b(that'?s|thats)\s+(dope|sick|crazy|wild|insane|fire|lit)\b
\b(lmao|bruh|omg)\b

# Generic reactions
^(bruh|bro|dude)\b.+(!|😂|💀|🤣)
```

**4. Expand ack patterns**
Missing: "Thanks I'll check", "ik hahah", "Swoop"
```regex
# Acknowledgment with continuation
^(thanks|thx)\s+.{0,30}$
^(ik|i know)\s*(haha+)?
```

## Recommendations

1. **Use the consolidated 7-label model** for production - 87% accuracy is solid
2. **Improve structural patterns** to increase coverage from 30% to 50%+
3. **Consider merging `reaction` into `statement`** - hard to distinguish
4. **The `commitment` class (was `invitation`)** could use more training data

## Model Files

Models saved to:
- `~/.jarvis/trigger_classifier_models/corrected_10label/`
- `~/.jarvis/trigger_classifier_models/consolidated_7label/`

Each contains:
- `svm.pkl` - Trained SVM model
- `config.json` - Labels and metadata
