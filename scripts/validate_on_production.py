#!/usr/bin/env python3
"""Validate LightGBM classifier on real iMessage production data.

Samples N messages from chat.db, compares LightGBM predictions vs Cerebras
Qwen 235B labels, and analyzes agreement/disagreement patterns.

Usage:
    uv run python scripts/validate_on_production.py --n-samples 200
    uv run python scripts/validate_on_production.py --n-samples 500 --skip-llm  # LightGBM only
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
from openai import OpenAI
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# =============================================================================
# Sample Messages from iMessage DB
# =============================================================================

def sample_messages(n_samples: int = 200, min_length: int = 3) -> list[dict]:
    """Sample N messages from iMessage database.

    Returns list of dicts with keys: text, chat_id, message_date
    """
    # iMessage DB path
    home = Path.home()
    db_path = home / "Library" / "Messages" / "chat.db"

    if not db_path.exists():
        print(f"❌ iMessage database not found: {db_path}")
        sys.exit(1)

    print(f"📂 Sampling {n_samples} messages from {db_path}...")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    # Sample recent messages from various chats
    # Exclude group chats, attachments, reactions
    query = """
    SELECT
        m.text,
        m.ROWID,
        m.date
    FROM message m
    WHERE m.text IS NOT NULL
        AND m.text != ''
        AND length(m.text) >= ?
        AND m.is_from_me = 0  -- Only received messages
        AND m.associated_message_guid IS NULL  -- Exclude reactions
        AND m.cache_has_attachments = 0  -- Exclude attachments
    ORDER BY RANDOM()
    LIMIT ?
    """

    cursor.execute(query, (min_length, n_samples * 2))  # Sample 2x, filter later
    rows = cursor.fetchall()
    conn.close()

    messages = []
    seen_texts = set()

    for text, message_id, date in rows:
        # Clean text
        text = text.strip()

        # Skip duplicates
        if text.lower() in seen_texts:
            continue

        # Skip messages with URLs, emails (likely spam/automated)
        if re.search(r'https?://|www\.|@.*\.(com|org|net)', text, re.I):
            continue

        # Skip very long messages (likely forwards/articles)
        if len(text) > 500:
            continue

        seen_texts.add(text.lower())
        messages.append({
            "text": text,
            "message_id": message_id,
            "message_date": date,
        })

        if len(messages) >= n_samples:
            break

    print(f"✓ Sampled {len(messages)} unique messages")
    return messages


# =============================================================================
# Feature Extraction (matches prepare_dailydialog_data.py)
# =============================================================================

EMOJI_RE = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
    r"\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]"
)


def extract_hand_crafted_features(text: str) -> np.ndarray:
    """Extract 19 hand-crafted features from text."""
    features = []

    # Message structure (5)
    features.append(float(len(text)))
    features.append(float(len(text.split())))
    features.append(float(text.count("?")))
    features.append(float(text.count("!")))
    features.append(float(len(EMOJI_RE.findall(text))))

    # Response mobilization placeholders (7) - we don't have context
    features.extend([0.0] * 7)

    # Message characteristics (7)
    features.append(1.0 if text.isupper() else 0.0)
    features.append(1.0 if any(c.isdigit() for c in text) else 0.0)
    features.append(1.0 if text.startswith(("i ", "i'm ", "i've ")) else 0.0)
    features.append(1.0 if text.startswith(("you ", "your ", "you're ")) else 0.0)
    features.append(float(text.count("!")))
    features.append(1.0 if re.search(r'\b(lol|lmao|haha)\b', text, re.I) else 0.0)
    features.append(1.0 if text.strip().endswith("?") else 0.0)

    return np.array(features, dtype=np.float32)


def get_embeddings(texts: list[str]) -> np.ndarray:
    """Get embeddings using BERT embedder (loaded on-demand)."""
    from models.bert_embedder import get_in_process_embedder

    # Lazy load embedder (singleton)
    if not hasattr(get_embeddings, '_embedder'):
        print(f"   Loading BERT model...")
        get_embeddings._embedder = get_in_process_embedder(model_name="bge-small")

    embedder = get_embeddings._embedder

    # Encode in batches
    embeddings = embedder.encode(texts, batch_size=32)
    return embeddings


def extract_features_batch(messages: list[dict]) -> np.ndarray:
    """Extract features for a batch of messages."""
    texts = [msg["text"] for msg in messages]

    print(f"🔮 Extracting embeddings (BERT)...")
    embeddings = get_embeddings(texts)

    print(f"🔧 Extracting hand-crafted features...")
    hand_crafted = np.array([extract_hand_crafted_features(text) for text in texts])

    # Combine: [embeddings (384) | hand_crafted (19)]
    features = np.hstack([embeddings, hand_crafted])

    # Add dummy spacy features (14 zeros) if model expects them
    if features.shape[1] == 403:  # 384 + 19 = 403, need 417
        spacy_dummy = np.zeros((len(messages), 14), dtype=np.float32)
        features = np.hstack([features, spacy_dummy])

    print(f"✓ Features extracted: {features.shape}")
    return features


# =============================================================================
# LLM Labeling (Cerebras Qwen 235B)
# =============================================================================

def get_llm_labels(messages: list[dict], labels: list[str]) -> list[str]:
    """Get labels from Cerebras Qwen 235B."""
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        print("❌ CEREBRAS_API_KEY not found in environment")
        sys.exit(1)

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.cerebras.ai/v1",
    )

    # Construct prompt
    labels_desc = ", ".join(labels)
    system_prompt = f"""You are a text classification expert. Classify each message into one of these categories:
- commissive: promises, commitments ("I'll do it", "sure, I can help")
- directive: requests, commands ("can you help?", "send me the file")
- inform: statements, facts ("I'm at the store", "the meeting is at 3pm")
- question: asking for information ("where are you?", "what time?")

Respond with ONLY the category name, nothing else."""

    predictions = []
    print(f"🤖 Getting LLM labels (Qwen 235B)...")
    print(f"   Cost: ~${len(messages) * 0.0001:.4f}")

    for i, msg in enumerate(messages):
        if (i + 1) % 50 == 0:
            print(f"   Progress: {i+1}/{len(messages)}", flush=True)

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b",  # Fast, cheap model for classification
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": msg["text"]},
                ],
                temperature=0.0,
                max_tokens=10,
            )

            label = response.choices[0].message.content.strip().lower()

            # Validate label
            if label not in labels:
                print(f"   ⚠️  Invalid label '{label}' for: {msg['text'][:50]}...")
                label = "inform"  # Default to inform

            predictions.append(label)

        except Exception as e:
            print(f"   ❌ Error on message {i}: {e}")
            predictions.append("inform")  # Default

    print(f"✓ LLM labeling complete")
    return predictions


# =============================================================================
# Analysis
# =============================================================================

def analyze_results(messages, lgbm_preds, llm_preds, labels):
    """Analyze agreement and disagreement patterns."""
    print()
    print("=" * 70)
    print("📊 Results Analysis")
    print("=" * 70)
    print()

    # Agreement rate
    agreements = sum(1 for l, g in zip(llm_preds, lgbm_preds) if l == g)
    agreement_rate = agreements / len(messages)

    print(f"Agreement Rate: {agreement_rate:.1%} ({agreements}/{len(messages)})")
    print()

    # Confusion matrix
    print("Confusion Matrix (LLM=rows, LightGBM=cols):")
    cm = confusion_matrix(llm_preds, lgbm_preds, labels=labels)

    # Print header
    print(f"{'':12s}", end="")
    for label in labels:
        print(f"{label:12s}", end="")
    print()

    # Print matrix
    for i, label in enumerate(labels):
        print(f"{label:12s}", end="")
        for j in range(len(labels)):
            print(f"{cm[i][j]:12d}", end="")
        print()
    print()

    # Per-class metrics (using LLM as ground truth)
    print("Per-Class Performance (LLM as ground truth):")
    report = classification_report(
        llm_preds, lgbm_preds,
        labels=labels,
        target_names=labels,
        digits=3,
        zero_division=0,
    )
    print(report)

    # Disagreement examples
    print("=" * 70)
    print("📝 Disagreement Examples (first 20)")
    print("=" * 70)
    print()

    disagreements = [
        (msg, llm, lgbm)
        for msg, llm, lgbm in zip(messages, llm_preds, lgbm_preds)
        if llm != lgbm
    ]

    for i, (msg, llm, lgbm) in enumerate(disagreements[:20]):
        print(f"{i+1}. Text: {msg['text'][:70]}")
        print(f"   LLM: {llm:12s}  LightGBM: {lgbm:12s}")
        print()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate classifier on production data")
    parser.add_argument("--n-samples", type=int, default=200, help="Number of messages to sample")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM labeling (LightGBM only)")
    parser.add_argument("--output", type=Path, help="Save results to JSON file")
    args = parser.parse_args()

    print("=" * 70)
    print("Production Validation: LightGBM vs Qwen 235B")
    print("=" * 70)
    print()

    # Load model
    model_path = PROJECT_ROOT / "models" / "lightgbm_category_final.joblib"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print(f"   Run: uv run python scripts/train_final_lightgbm.py")
        sys.exit(1)

    print(f"📦 Loading model from {model_path}...")
    model = joblib.load(model_path)

    metadata_path = PROJECT_ROOT / "models" / "lightgbm_category_final.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    labels = metadata["labels"]
    print(f"✓ Model loaded (labels: {', '.join(labels)})")
    print()

    # Sample messages
    messages = sample_messages(n_samples=args.n_samples)

    # Extract features
    features = extract_features_batch(messages)

    # Get LightGBM predictions
    print(f"🔮 Running LightGBM classifier...")
    lgbm_pred_raw = model.predict(features)

    # Handle both string labels and numeric indices
    if isinstance(lgbm_pred_raw[0], (int, np.integer)):
        lgbm_preds = [labels[i] for i in lgbm_pred_raw]
    else:
        lgbm_preds = list(lgbm_pred_raw)

    print(f"✓ LightGBM predictions complete")
    print()

    # Distribution
    from collections import Counter
    lgbm_dist = Counter(lgbm_preds)
    print("LightGBM Predictions:")
    for label in labels:
        count = lgbm_dist[label]
        pct = 100 * count / len(messages)
        print(f"  {label:12s}: {count:3d} ({pct:5.1f}%)")
    print()

    if args.skip_llm:
        print("Skipping LLM labeling (--skip-llm flag)")
        return

    # Get LLM predictions
    llm_preds = get_llm_labels(messages, labels)

    # LLM distribution
    llm_dist = Counter(llm_preds)
    print("LLM Predictions:")
    for label in labels:
        count = llm_dist[label]
        pct = 100 * count / len(messages)
        print(f"  {label:12s}: {count:3d} ({pct:5.1f}%)")
    print()

    # Analyze
    analyze_results(messages, lgbm_preds, llm_preds, labels)

    # Save results
    if args.output:
        results = {
            "n_samples": len(messages),
            "agreement_rate": sum(1 for l, g in zip(llm_preds, lgbm_preds) if l == g) / len(messages),
            "messages": [
                {
                    "text": msg["text"],
                    "lgbm_pred": lgbm,
                    "llm_pred": llm,
                    "agree": lgbm == llm,
                }
                for msg, lgbm, llm in zip(messages, lgbm_preds, llm_preds)
            ],
        }

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        print(f"💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
