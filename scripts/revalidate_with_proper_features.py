#!/usr/bin/env python3
"""Re-validate LightGBM with PROPER feature extraction.

Fixes bugs:
1. Text normalization (expand slang)
2. Actual spacy feature extraction (not dummy zeros)

Uses saved messages + LLM predictions, only re-runs LightGBM.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import spacy
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.text_normalizer import normalize_text

print("=" * 70)
print("Re-validating LightGBM with Proper Features")
print("=" * 70)
print()

# Load saved data
input_file = PROJECT_ROOT / "production_validation_with_claude.json"
with open(input_file) as f:
    data = json.load(f)

messages = data["messages"]
labels = ["commissive", "directive", "inform", "question"]

print(f"📂 Loaded {len(messages)} messages with saved LLM + Claude predictions")
print()

# Load model
model_path = PROJECT_ROOT / "models" / "lightgbm_category_final.joblib"
model = joblib.load(model_path)
print("✓ Model loaded")
print()

# =============================================================================
# Proper Feature Extraction
# =============================================================================

print("🔮 Extracting features properly...")
print("   1. Normalizing text (expand slang)")
print("   2. Getting BERT embeddings")
print("   3. Extracting hand-crafted features")
print("   4. Extracting SpaCy features")
print()

# 1. Normalize texts
normalized_texts = []
for msg in messages:
    text = msg["text"]
    normalized = normalize_text(text, expand_slang=True, spell_check=False)
    normalized_texts.append(normalized if normalized else text)

print(f"✓ Normalized {len(normalized_texts)} texts")

# 2. Get embeddings
from models.bert_embedder import get_in_process_embedder

print("   Loading BERT model...")
embedder = get_in_process_embedder(model_name="bge-small")
embeddings = embedder.encode(normalized_texts, batch_size=32)
print(f"✓ Embeddings extracted: {embeddings.shape}")

# 3. Hand-crafted features (19 features)
import re

EMOJI_RE = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
    r"\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]"
)

ABBREVIATION_RE = re.compile(
    r"\b(lol|lmao|omg|wtf|brb|btw|smh|tbh|imo|idk|ngl|fr|rn|ong|nvm|wya|hmu|"
    r"fyi|asap|dm|irl|fomo|goat|sus|bet|cap|no cap)\b",
    re.IGNORECASE,
)

def extract_hand_crafted(text: str) -> np.ndarray:
    """Extract 19 hand-crafted features."""
    features = []

    # Message structure (5)
    features.append(float(len(text)))
    features.append(float(len(text.split())))
    features.append(float(text.count("?")))
    features.append(float(text.count("!")))
    features.append(float(len(EMOJI_RE.findall(text))))

    # Response mobilization placeholders (7) - we don't have context
    features.extend([0.0] * 7)

    # Tone flags (2)
    features.append(0.0)  # professional_keywords
    features.append(1.0 if ABBREVIATION_RE.search(text) else 0.0)

    # Context features (3) - no context
    features.extend([0.0, 0.0, 1.0])

    # Style features (2)
    words = text.split()
    total_words = len(words)
    abbr_count = len(ABBREVIATION_RE.findall(text))
    features.append(abbr_count / max(total_words, 1))
    capitalized = sum(1 for w in words[1:] if w[0].isupper()) if len(words) > 1 else 0
    features.append(capitalized / max(len(words) - 1, 1))

    return np.array(features, dtype=np.float32)

hand_crafted_features = np.array([extract_hand_crafted(text) for text in normalized_texts])
print(f"✓ Hand-crafted features: {hand_crafted_features.shape}")

# 4. SpaCy features (14 features)
print("   Loading SpaCy model...")
nlp = spacy.load("en_core_web_sm")

def extract_spacy_features(text: str) -> np.ndarray:
    """Extract 14 SpaCy linguistic features."""
    doc = nlp(text)
    features = []

    # 1. has_imperative: Check for imperative verbs (VB at start)
    has_imperative = 0.0
    if len(doc) > 0 and doc[0].pos_ == "VERB" and doc[0].tag_ == "VB":
        has_imperative = 1.0
    features.append(has_imperative)

    # 2. you_modal: "can you", "could you", "would you", "will you"
    text_lower = text.lower()
    you_modal = 1.0 if any(p in text_lower for p in ["can you", "could you", "would you", "will you", "should you"]) else 0.0
    features.append(you_modal)

    # 3. request_verb: Common request verbs
    request_verbs = {"send", "give", "help", "tell", "show", "let", "call", "get", "make", "take"}
    has_request = 1.0 if any(token.lemma_ in request_verbs for token in doc) else 0.0
    features.append(has_request)

    # 4. starts_modal: Starts with modal verb
    starts_modal = 0.0
    if len(doc) > 0 and doc[0].tag_ in ("MD", "VB"):
        starts_modal = 1.0
    features.append(starts_modal)

    # 5. directive_question: Questions that are really directives
    directive_q = 1.0 if you_modal and "?" in text else 0.0
    features.append(directive_q)

    # 6. i_will: "I'll", "I will", "I'm gonna"
    i_will = 1.0 if any(p in text_lower for p in ["i'll", "i will", "i'm gonna", "ima", "imma"]) else 0.0
    features.append(i_will)

    # 7. promise_verb: Promise/commitment verbs
    promise_verbs = {"promise", "guarantee", "commit", "swear"}
    has_promise = 1.0 if any(token.lemma_ in promise_verbs for token in doc) else 0.0
    features.append(has_promise)

    # 8. first_person_count
    first_person = sum(1 for token in doc if token.text.lower() in ("i", "me", "my", "mine", "myself"))
    features.append(float(first_person))

    # 9. agreement: Agreement words
    agreement_words = {"sure", "okay", "ok", "yes", "yeah", "yep", "yup", "sounds good", "bet", "fs"}
    has_agreement = 1.0 if any(word in text_lower for word in agreement_words) else 0.0
    features.append(has_agreement)

    # 10. modal_count
    modal_count = sum(1 for token in doc if token.tag_ == "MD")
    features.append(float(modal_count))

    # 11. verb_count
    verb_count = sum(1 for token in doc if token.pos_ == "VERB")
    features.append(float(verb_count))

    # 12. second_person_count
    second_person = sum(1 for token in doc if token.text.lower() in ("you", "your", "yours", "yourself"))
    features.append(float(second_person))

    # 13. has_negation
    has_neg = 1.0 if any(token.dep_ == "neg" for token in doc) else 0.0
    features.append(has_neg)

    # 14. is_interrogative: Question indicators
    is_question = 1.0 if "?" in text or any(token.tag_ in ("WDT", "WP", "WP$", "WRB") for token in doc) else 0.0
    features.append(is_question)

    return np.array(features, dtype=np.float32)

spacy_features = np.array([extract_spacy_features(text) for text in normalized_texts])
print(f"✓ SpaCy features: {spacy_features.shape}")

# Combine all features
X = np.hstack([embeddings, hand_crafted_features, spacy_features])
print(f"✓ Total features: {X.shape} (384 + 19 + 14 = 417)")
print()

# =============================================================================
# Get NEW LightGBM Predictions
# =============================================================================

print("🔮 Running LightGBM with proper features...")
lgbm_pred_raw = model.predict(X)

# Handle both string labels and numeric indices
if isinstance(lgbm_pred_raw[0], (int, np.integer)):
    lgbm_preds_new = [labels[i] for i in lgbm_pred_raw]
else:
    lgbm_preds_new = list(lgbm_pred_raw)

print(f"✓ New LightGBM predictions complete")
print()

# =============================================================================
# Compare: New LightGBM vs Old LightGBM vs LLM vs Claude
# =============================================================================

# Extract old predictions
lgbm_preds_old = [msg["lgbm_pred"] for msg in messages]
llm_preds = [msg["llm_pred"] for msg in messages]
claude_preds = [msg["claude_pred"] for msg in messages]

# Agreement rates (Claude as ground truth)
from collections import Counter

lgbm_old_agreement = sum(1 for c, l in zip(claude_preds, lgbm_preds_old) if c == l)
lgbm_new_agreement = sum(1 for c, l in zip(claude_preds, lgbm_preds_new) if c == l)
llm_agreement = sum(1 for c, l in zip(claude_preds, llm_preds) if c == l)

lgbm_old_rate = lgbm_old_agreement / len(messages)
lgbm_new_rate = lgbm_new_agreement / len(messages)
llm_rate = llm_agreement / len(messages)

print("=" * 70)
print("📊 Results: Agreement with Claude's Labels")
print("=" * 70)
print()
print(f"LightGBM (OLD - buggy features): {lgbm_old_rate:.1%} ({lgbm_old_agreement}/200)")
print(f"LightGBM (NEW - proper features): {lgbm_new_rate:.1%} ({lgbm_new_agreement}/200)")
print(f"LLM (Qwen 235B):                  {llm_rate:.1%} ({llm_agreement}/200)")
print()
print(f"Improvement: {(lgbm_new_rate - lgbm_old_rate) * 100:+.1f} percentage points")
print()

# Distributions
lgbm_new_dist = Counter(lgbm_preds_new)
print("=" * 70)
print("New LightGBM Predictions")
print("=" * 70)
for label in labels:
    count = lgbm_new_dist[label]
    pct = 100 * count / len(messages)
    print(f"  {label:12s}: {count:3d} ({pct:5.1f}%)")
print()

# Confusion matrix
print("=" * 70)
print("New LightGBM Confusion Matrix (Claude=rows, LightGBM=cols)")
print("=" * 70)
cm = confusion_matrix(claude_preds, lgbm_preds_new, labels=labels)
print(f"{'':12s}", end="")
for label in labels:
    print(f"{label:12s}", end="")
print()
for i, label in enumerate(labels):
    print(f"{label:12s}", end="")
    for j in range(len(labels)):
        print(f"{cm[i][j]:12d}", end="")
    print()
print()

# Per-class performance
print("=" * 70)
print("New LightGBM Per-Class (Claude as ground truth)")
print("=" * 70)
print(classification_report(claude_preds, lgbm_preds_new, labels=labels, target_names=labels, digits=3, zero_division=0))

# Winner
print("=" * 70)
print("🏆 Final Verdict")
print("=" * 70)
print()

if lgbm_new_rate > llm_rate + 0.05:
    print(f"✅ LightGBM (FIXED) WINS ({lgbm_new_rate:.1%} vs {llm_rate:.1%})")
    print(f"   → Use LightGBM for production")
elif llm_rate > lgbm_new_rate + 0.05:
    print(f"✅ LLM STILL WINS ({llm_rate:.1%} vs {lgbm_new_rate:.1%})")
    print(f"   → Need more improvements to LightGBM")
else:
    print(f"🤝 TIE ({lgbm_new_rate:.1%} vs {llm_rate:.1%})")
    print(f"   → Either model is acceptable")

print()

# Error analysis
errors = [
    (msg["text"], c, new)
    for msg, c, new in zip(messages, claude_preds, lgbm_preds_new)
    if c != new
]

print("=" * 70)
print("New LightGBM Errors (first 15)")
print("=" * 70)
for i, (text, claude, lgbm) in enumerate(errors[:15]):
    print(f"{i+1}. {text[:65]}")
    print(f"   Claude: {claude:12s}  LightGBM: {lgbm:12s}")
    print()

# Save updated results
for msg, new_pred in zip(messages, lgbm_preds_new):
    msg["lgbm_pred_fixed"] = new_pred

output_file = PROJECT_ROOT / "production_validation_fixed.json"
with open(output_file, "w") as f:
    json.dump(data, f, indent=2)

print(f"💾 Results saved to: {output_file}")
