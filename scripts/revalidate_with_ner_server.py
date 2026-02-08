#!/usr/bin/env python3
"""Re-validate using the ACTUAL NER server for syntactic features.

This matches the training data preparation exactly.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.text_normalizer import normalize_text
from jarvis.nlp.ner_client import get_syntactic_features_batch, is_service_running

print("=" * 70)
print("Re-validating with NER Server (Exact Match to Training)")
print("=" * 70)
print()

# Check NER server
if not is_service_running():
    print("❌ NER server is not running!")
    print("   Start it with: uv run python scripts/ner_server.py &")
    sys.exit(1)

print("✓ NER server is running")
print()

# Load saved data
input_file = PROJECT_ROOT / "production_validation_with_claude.json"
with open(input_file) as f:
    data = json.load(f)

messages = data["messages"]
labels = ["commissive", "directive", "inform", "question"]

print(f"📂 Loaded {len(messages)} messages")
print()

# Load model
model_path = PROJECT_ROOT / "models" / "lightgbm_category_final.joblib"
model = joblib.load(model_path)
print("✓ Model loaded")
print()

# Extract features using EXACT same pipeline as training
print("🔮 Extracting features (matching training pipeline)...")

# 1. Normalize texts
texts = [msg["text"] for msg in messages]
normalized_texts = []
for text in texts:
    normalized = normalize_text(text, expand_slang=True, spell_check=False)
    normalized_texts.append(normalized if normalized else text)

print(f"✓ Normalized {len(normalized_texts)} texts")

# 2. Get embeddings
from models.bert_embedder import get_in_process_embedder

print("   Loading BERT model...")
embedder = get_in_process_embedder(model_name="bge-small")
embeddings = embedder.encode(normalized_texts, batch_size=32)
print(f"✓ Embeddings: {embeddings.shape}")

# 3. Hand-crafted features (copy from prepare script)
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
    features = []
    features.append(float(len(text)))
    features.append(float(len(text.split())))
    features.append(float(text.count("?")))
    features.append(float(text.count("!")))
    features.append(float(len(EMOJI_RE.findall(text))))
    features.extend([0.0] * 7)  # mobilization placeholders
    features.append(0.0)  # professional
    features.append(1.0 if ABBREVIATION_RE.search(text) else 0.0)
    features.extend([0.0, 0.0, 1.0])  # context placeholders
    words = text.split()
    total_words = len(words)
    abbr_count = len(ABBREVIATION_RE.findall(text))
    features.append(abbr_count / max(total_words, 1))
    capitalized = sum(1 for w in words[1:] if w[0].isupper()) if len(words) > 1 else 0
    features.append(capitalized / max(len(words) - 1, 1))
    return np.array(features, dtype=np.float32)

hand_crafted = np.array([extract_hand_crafted(text) for text in texts])
print(f"✓ Hand-crafted: {hand_crafted.shape}")

# 4. Syntactic features via NER server (EXACT match to training)
print("   Getting syntactic features from NER server...")
syntactic = get_syntactic_features_batch(texts)
syntactic = np.array(syntactic, dtype=np.float32)
print(f"✓ Syntactic: {syntactic.shape}")

# Combine
X = np.hstack([embeddings, hand_crafted, syntactic])
print(f"✓ Total: {X.shape}")
print()

# Get predictions
print("🔮 Running LightGBM...")
lgbm_pred_raw = model.predict(X)

if isinstance(lgbm_pred_raw[0], (int, np.integer)):
    lgbm_preds = [labels[i] for i in lgbm_pred_raw]
else:
    lgbm_preds = list(lgbm_pred_raw)

print("✓ Predictions complete")
print()

# Compare
lgbm_old = [msg["lgbm_pred"] for msg in messages]
llm = [msg["llm_pred"] for msg in messages]
claude = [msg["claude_pred"] for msg in messages]

lgbm_old_agree = sum(1 for c, l in zip(claude, lgbm_old) if c == l)
lgbm_new_agree = sum(1 for c, l in zip(claude, lgbm_preds) if c == l)
llm_agree = sum(1 for c, l in zip(claude, llm) if c == l)

lgbm_old_rate = lgbm_old_agree / 200
lgbm_new_rate = lgbm_new_agree / 200
llm_rate = llm_agree / 200

print("=" * 70)
print("📊 Results (Claude as Ground Truth)")
print("=" * 70)
print()
print(f"LightGBM (buggy - zero spacy):     {lgbm_old_rate:.1%} ({lgbm_old_agree}/200)")
print(f"LightGBM (fixed - real spacy):     {lgbm_new_rate:.1%} ({lgbm_new_agree}/200)")
print(f"LLM:                                {llm_rate:.1%} ({llm_agree}/200)")
print()
print(f"Improvement: {(lgbm_new_rate - lgbm_old_rate) * 100:+.1f} percentage points")
print()

# Distribution
from collections import Counter
dist = Counter(lgbm_preds)
print("New LightGBM Distribution:")
for label in labels:
    print(f"  {label:12s}: {dist[label]:3d} ({100 * dist[label] / 200:5.1f}%)")
print()

# Confusion matrix
print("=" * 70)
print("Confusion Matrix (Claude=rows, LightGBM=cols)")
print("=" * 70)
cm = confusion_matrix(claude, lgbm_preds, labels=labels)
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

# Per-class
print("=" * 70)
print("Per-Class Performance")
print("=" * 70)
print(classification_report(claude, lgbm_preds, labels=labels, target_names=labels, digits=3, zero_division=0))

# Verdict
print("=" * 70)
print("🏆 Verdict")
print("=" * 70)
print()
if lgbm_new_rate > llm_rate + 0.05:
    print(f"✅ LightGBM WINS ({lgbm_new_rate:.1%} vs {llm_rate:.1%})")
elif llm_rate > lgbm_new_rate + 0.05:
    print(f"✅ LLM WINS ({llm_rate:.1%} vs {lgbm_new_rate:.1%})")
else:
    print(f"🤝 TIE ({lgbm_new_rate:.1%} vs {llm_rate:.1%})")
