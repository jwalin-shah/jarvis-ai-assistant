#!/usr/bin/env python3
"""Prepare DailyDialog data with native dialog act labels - CPU VERSION.

Uses sentence-transformers (CPU-based) instead of MLX to avoid 5GB memory issue.
Identical output to MLX version, but uses ~500MB instead of 5GB.

Usage:
    uv run python scripts/prepare_dailydialog_data_cpu.py --dry-run
    uv run python scripts/prepare_dailydialog_data_cpu.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Load .env
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# DailyDialog dialog act labels
DIALOG_ACT_MAP = {
    1: "inform",
    2: "question",
    3: "directive",
    4: "commissive",
}

OUTPUT_DIR = PROJECT_ROOT / "data" / "dailydialog_native"

# Hand-crafted feature extraction (same as MLX version)
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

PROFESSIONAL_KEYWORDS_RE = re.compile(
    r"\b(meeting|deadline|project|report|schedule|conference|presentation|"
    r"budget|client|invoice|proposal)\b",
    re.IGNORECASE,
)


def log_memory(label: str) -> None:
    """Log current memory usage."""
    process = psutil.Process()
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / 1024 / 1024
    print(f"[MEMORY] {label}: {rss_mb:.1f} MB")


def extract_hand_crafted_features(
    text: str,
    context_messages: list[str],
    mobilization_pressure: str,
    mobilization_type: str,
) -> np.ndarray:
    """Extract 19 hand-crafted features from a message + context."""
    features: list[float] = []

    features.append(float(len(text)))
    features.append(float(len(text.split())))
    features.append(float(text.count("?")))
    features.append(float(text.count("!")))
    features.append(float(len(EMOJI_RE.findall(text))))

    for level in ("high", "medium", "low", "none"):
        features.append(1.0 if mobilization_pressure == level else 0.0)
    for rtype in ("commitment", "answer", "emotional"):
        features.append(1.0 if mobilization_type == rtype else 0.0)

    features.append(1.0 if PROFESSIONAL_KEYWORDS_RE.search(text) else 0.0)
    features.append(1.0 if ABBREVIATION_RE.search(text) else 0.0)

    features.append(float(len(context_messages)))
    avg_ctx_len = (
        float(np.mean([len(m) for m in context_messages])) if context_messages else 0.0
    )
    features.append(avg_ctx_len)
    features.append(1.0 if len(context_messages) == 0 else 0.0)

    words = text.split()
    total_words = len(words)
    abbr_count = len(ABBREVIATION_RE.findall(text))
    features.append(abbr_count / max(total_words, 1))
    capitalized = sum(1 for w in words[1:] if w[0].isupper()) if len(words) > 1 else 0
    features.append(capitalized / max(len(words) - 1, 1))

    return np.array(features, dtype=np.float32)


def load_dailydialog() -> list[dict]:
    """Load DailyDialog and extract per-utterance examples with native labels."""
    from datasets import load_dataset

    print("Loading DailyDialog...")
    ds = load_dataset("OpenRL/daily_dialog", split="train")
    print(f"  {len(ds)} dialogues loaded")

    examples: list[dict] = []

    for dialogue in ds:
        utterances = dialogue["dialog"]
        acts = dialogue["act"]

        if len(utterances) < 2:
            continue

        for i in range(1, len(utterances)):
            text = utterances[i].strip()
            if len(text) < 2:
                continue

            act = acts[i]
            label = DIALOG_ACT_MAP.get(act, "inform")

            context = [u.strip() for u in utterances[max(0, i - 5):i]]
            last_msg = utterances[i - 1].strip()

            examples.append({
                "text": text,
                "last_message": last_msg,
                "label": label,
                "context": context,
                "source": "dailydialog",
            })

    print(f"  {len(examples)} per-utterance examples extracted")
    return examples


def prepare_data(seed: int = 42, dry_run: bool = False) -> dict:
    """Load, extract features, and save DailyDialog training data (CPU version)."""
    from jarvis.classifiers.response_mobilization import classify_response_pressure

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    log_memory("START")

    examples = load_dailydialog()
    total_raw = len(examples)
    log_memory("After loading DailyDialog")

    print(f"\nTotal raw examples: {total_raw}")

    raw_counts = Counter(ex["label"] for ex in examples)
    print("\nRaw label distribution:")
    for label, count in sorted(raw_counts.items(), key=lambda x: -x[1]):
        pct = count / total_raw * 100
        print(f"  {label:12s} {count:6d} ({pct:.1f}%)")

    min_count = min(raw_counts.values())
    max_count = max(raw_counts.values())
    balance_ratio = max_count / max(min_count, 1)
    print(f"\nBalance ratio: {balance_ratio:.1f}x (max/min)")

    if dry_run:
        print("\n--- DRY RUN: stopping before feature extraction ---")
        return {
            "total_raw": total_raw,
            "raw_distribution": dict(raw_counts),
            "balance_ratio": balance_ratio,
        }

    filtered = [ex for ex in examples if len(ex["text"].strip()) >= 3]
    total_filtered = len(filtered)
    print(f"\nAfter filtering (len >= 3): {total_filtered}")

    del examples
    log_memory("After deleting examples")

    # Extract hand-crafted features
    print("\nExtracting hand-crafted features...")
    hc_matrix = np.zeros((total_filtered, 19), dtype=np.float32)

    for i, ex in enumerate(filtered):
        mob = classify_response_pressure(ex["last_message"])
        hc = extract_hand_crafted_features(
            text=ex["text"],
            context_messages=ex["context"],
            mobilization_pressure=mob.pressure.value,
            mobilization_type=mob.response_type.value,
        )
        hc_matrix[i] = hc

        if (i + 1) % 10000 == 0 or (i + 1) == total_filtered:
            print(f"  {i + 1}/{total_filtered} features extracted")

    hand_crafted_dims = hc_matrix.shape[1]
    log_memory("After hand-crafted features")

    # Extract texts and labels, then delete filtered
    print("\nExtracting texts and labels...")
    all_texts = [ex["last_message"] for ex in filtered]
    all_labels = [ex["label"] for ex in filtered]
    del filtered
    log_memory("After deleting filtered")

    # Load sentence-transformers (CPU-based)
    print("\nLoading sentence-transformers model (CPU)...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')
    log_memory("After loading sentence-transformers")

    # Compute embeddings in batches
    print("Computing embeddings in batches...")
    batch_size = 5000
    embedding_list = []

    for i in range(0, len(all_texts), batch_size):
        batch_end = min(i + batch_size, len(all_texts))
        batch_texts = all_texts[i:batch_end]

        batch_emb = model.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        embedding_list.append(batch_emb)

        if batch_end % 10000 == 0 or batch_end == len(all_texts):
            print(f"  {batch_end}/{len(all_texts)} embeddings computed")
            log_memory(f"After {batch_end} embeddings")

    embeddings = np.vstack(embedding_list)
    del all_texts, embedding_list, model
    log_memory("After combining embeddings")

    embedding_dims = embeddings.shape[1]
    print(f"  Embeddings shape: {embeddings.shape}")

    # Build feature matrix
    y = np.array(all_labels)
    del all_labels

    labels = sorted(set(y))
    feature_dims = embedding_dims + hand_crafted_dims

    X = np.hstack([embeddings, hc_matrix])
    del embeddings, hc_matrix
    log_memory("After combining features")

    print(f"\nFeature matrix: {X.shape}, Labels: {y.shape}")

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y,
    )

    del X, y

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    train_counts = Counter(y_train)
    test_counts = Counter(y_test)

    np.savez(output_dir / "train.npz", X=X_train, y=y_train)
    np.savez(output_dir / "test.npz", X=X_test, y=y_test)

    label_map = {label: i for i, label in enumerate(labels)}

    metadata = {
        "source": "OpenRL/daily_dialog",
        "labels": labels,
        "label_map": label_map,
        "total_raw": total_raw,
        "total_filtered": total_filtered,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_dims": feature_dims,
        "embedding_dims": embedding_dims,
        "hand_crafted_dims": hand_crafted_dims,
        "label_distribution_raw": dict(raw_counts),
        "label_distribution_train": {k: int(v) for k, v in train_counts.items()},
        "label_distribution_test": {k: int(v) for k, v in test_counts.items()},
        "balance_ratio": balance_ratio,
        "seed": seed,
        "embedder": "sentence-transformers/BAAI/bge-small-en-v1.5 (CPU)",
    }

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    elapsed = time.perf_counter() - t0
    print(f"\nSaved to {output_dir}/")
    print(f"Elapsed: {elapsed:.1f}s")
    log_memory("FINAL")
    print(json.dumps(metadata, indent=2))

    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare DailyDialog data (CPU version)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview label distribution without feature extraction",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    prepare_data(seed=args.seed, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
