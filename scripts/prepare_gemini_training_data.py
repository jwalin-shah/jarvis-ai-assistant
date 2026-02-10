#!/usr/bin/env python3
"""Prepare training data from Gemini-labeled examples.

Extracts features (BERT embeddings + hand-crafted) and creates
train/test splits for retraining category and mobilization classifiers.

Usage:
    uv run python scripts/prepare_gemini_training_data.py
    uv run python scripts/prepare_gemini_training_data.py --output-dir data/gemini_features
"""

from __future__ import annotations

import argparse
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
        logger.error("Run scripts/eval_and_retrain_gemini.py first")
        sys.exit(1)

    examples = []
    for line in LABELED_DATA.open():
        line = line.strip()
        if line:
            examples.append(json.loads(line))

    logger.info(f"Loaded {len(examples)} labeled examples")
    return examples


def extract_features(examples: list[dict]) -> tuple[np.ndarray, list[str], list[str]]:
    """Extract BERT + hand-crafted features for all examples."""
    from jarvis.features.category_features import CategoryFeatureExtractor

    logger.info("Initializing feature extractor...")
    extractor = CategoryFeatureExtractor()

    features_list = []
    categories = []
    ids = []

    logger.info(f"Extracting features for {len(examples)} examples...")

    for i, ex in enumerate(examples):
        try:
            # Extract features
            feature_vec = extractor.extract_all(text=ex["text"], context=[])
            features_list.append(feature_vec)
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
    logger.info(f"Feature shape: {X.shape}")
    logger.info(f"Labels: {sorted(set(y))}")

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
    logger.info(f"Train labels: {sorted(set(y_train))}")
    logger.info(f"Test labels: {sorted(set(y_test))}")

    # Distribution
    logger.info("Train distribution:")
    from collections import Counter
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
    ids_train: list[str],
    ids_test: list[str],
) -> None:
    """Save training data in expected format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save numpy arrays
    np.savez(output_dir / "train.npz", X=X_train, y=y_train)
    np.savez(output_dir / "test.npz", X=X_test, y=y_test)

    # Save metadata
    labels = sorted(set(y_train) | set(y_test))
    metadata = {
        "source": "gemini_labeled",
        "categories": labels,
        "label_map": {label: i for i, label in enumerate(labels)},
        "feature_dims": int(X_train.shape[1]),
        "embedding_dims": 384,
        "hand_crafted_dims": X_train.shape[1] - 384,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "total_size": len(X_train) + len(X_test),
        "train_ids": ids_train,
        "test_ids": ids_test,
    }

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    logger.info(f"✓ Training data saved to {output_dir}")
    logger.info(f"  - train.npz: {X_train.shape}")
    logger.info(f"  - test.npz: {X_test.shape}")
    logger.info(f"  - metadata.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Gemini training data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "gemini_features",
        help="Output directory for training data",
    )
    args = parser.parse_args()

    # Load examples
    examples = load_labeled_examples()

    # Extract features
    X, y, ids = extract_features(examples)

    # Create splits
    X_train, X_test, y_train, y_test, ids_train, ids_test = create_splits(X, y, ids)

    # Save
    save_training_data(args.output_dir, X_train, X_test, y_train, y_test, ids_train, ids_test)

    logger.info(f"\n✓ Training data prepared in {args.output_dir}")
    logger.info("Next: uv run python scripts/train_category_svm.py --data-dir data/gemini_features")


if __name__ == "__main__":
    main()
