#!/usr/bin/env python3
"""Prepare Gemini training data WITH BERT embeddings (to match original format).

Extracts: 384 BERT embeddings + 147 hand-crafted = 531 total features
(vs original which had 384 BERT + 19 hand-crafted = 403)

This allows merging with original training data.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
LABELED_DATA = ROOT / "data" / "gemini_training" / "labeled_examples.jsonl"


def load_labeled_examples() -> list[dict]:
    """Load Gemini-labeled examples."""
    if not LABELED_DATA.exists():
        logger.error("Labeled data not found: %s", LABELED_DATA)
        sys.exit(1)

    examples = []
    for line in LABELED_DATA.open():
        line = line.strip()
        if line:
            examples.append(json.loads(line))

    logger.info(f"Loaded {len(examples)} labeled examples")
    return examples


def extract_features_with_embeddings(examples: list[dict]) -> tuple[np.ndarray, list[str], list[str]]:
    """Extract BERT embeddings + hand-crafted features."""
    from jarvis.classifiers.mixins import EmbedderMixin
    from jarvis.features.category_features import CategoryFeatureExtractor

    logger.info("Initializing embedder and feature extractor...")
    embedder_mixin = EmbedderMixin()
    feature_extractor = CategoryFeatureExtractor()

    features_list = []
    categories = []
    ids = []

    logger.info(f"Extracting BERT embeddings + features for {len(examples)} examples...")

    for i, ex in enumerate(examples):
        try:
            # Get BERT embedding (384 dims)
            embedding = embedder_mixin.embedder.encode(ex["text"])

            # Get hand-crafted features (147 dims)
            hand_crafted = feature_extractor.extract_all(text=ex["text"], context=[])

            # Combine: embedding (384) + hand_crafted (147) = 531 dims
            combined = np.concatenate([embedding, hand_crafted])
            features_list.append(combined)

            categories.append(ex["category"])
            ids.append(ex["id"])

            if (i + 1) % 500 == 0:
                logger.info(f"  {i + 1}/{len(examples)} features extracted")

        except Exception as e:
            logger.warning(f"Failed to extract features for {ex['id']}: {e}")
            continue

    X = np.array(features_list)
    y = np.array(categories)

    logger.info(f"Extracted {X.shape[0]} examples with {X.shape[1]} features")
    logger.info(f"Feature breakdown: 384 BERT + {X.shape[1] - 384} hand-crafted = {X.shape[1]} total")

    return X, y, ids


def create_splits(
    X: np.ndarray,
    y: np.ndarray,
    ids: list[str],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple:
    """Create stratified train/test splits."""
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X,
        y,
        ids,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    from collections import Counter

    logger.info("Train distribution:")
    train_dist = Counter(y_train)
    for cat, count in sorted(train_dist.items()):
        logger.info(f"  {cat:15s}: {count:4d}")

    logger.info("Test distribution:")
    test_dist = Counter(y_test)
    for cat, count in sorted(test_dist.items()):
        logger.info(f"  {cat:15s}: {count:4d}")

    return X_train, X_test, y_train, y_test, ids_train, ids_test


def save_training_data(
    output_dir: Path,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Save training data."""
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(output_dir / "train.npz", X=X_train, y=y_train)
    np.savez(output_dir / "test.npz", X=X_test, y=y_test)

    labels = sorted(set(y_train) | set(y_test))
    metadata = {
        "source": "gemini_labeled_with_embeddings",
        "categories": labels,
        "label_map": {label: i for i, label in enumerate(labels)},
        "feature_dims": int(X_train.shape[1]),
        "embedding_dims": 384,
        "hand_crafted_dims": X_train.shape[1] - 384,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    logger.info(f"✓ Training data saved to {output_dir}")
    logger.info(f"  - train.npz: {X_train.shape}")
    logger.info(f"  - test.npz: {X_test.shape}")


def main() -> None:
    examples = load_labeled_examples()
    X, y, ids = extract_features_with_embeddings(examples)
    X_train, X_test, y_train, y_test, ids_train, ids_test = create_splits(X, y, ids)

    output_dir = ROOT / "data" / "gemini_training_with_embeddings"
    save_training_data(output_dir, X_train, X_test, y_train, y_test)

    logger.info(f"\nNext: Update merge script to use {output_dir}")


if __name__ == "__main__":
    main()
