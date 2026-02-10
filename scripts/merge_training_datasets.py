#!/usr/bin/env python3
"""Merge original training data with Gemini-labeled iMessage data.

Combines:
- 18,496 original examples (DailyDialog/SAMSum, old 5-category)
- 2,430 Gemini-labeled examples (iMessage, new 6-category)
= 20,926 total examples for retraining

Maps old 5-category to new 6-category:
  ack → acknowledge
  clarify → question
  emotional → emotion
  info → statement
  social → acknowledge/statement (heuristic)

Usage:
    uv run python scripts/merge_training_datasets.py
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
ORIGINAL_DIR = ROOT / "data" / "category_training"
GEMINI_DIR = ROOT / "data" / "gemini_features"
OUTPUT_DIR = ROOT / "data" / "combined_training"


def load_original_data() -> tuple[np.ndarray, np.ndarray]:
    """Load original DailyDialog/SAMSum training data."""
    train_file = ORIGINAL_DIR / "train.npz"
    if not train_file.exists():
        logger.error(f"Original training data not found: {train_file}")
        sys.exit(1)

    data = np.load(train_file, allow_pickle=True)
    X = data["X"]
    y = data["y"]

    logger.info(f"Loaded original training data: {X.shape}")
    logger.info(f"Original categories: {sorted(set(y))}")
    return X, y


def load_gemini_data() -> tuple[np.ndarray, np.ndarray]:
    """Load Gemini-labeled iMessage data."""
    train_file = GEMINI_DIR / "train.npz"
    if not train_file.exists():
        logger.error(f"Gemini training data not found: {train_file}")
        logger.error("Run: uv run python scripts/prepare_gemini_training_data.py")
        sys.exit(1)

    data = np.load(train_file, allow_pickle=True)
    X = data["X"]
    y = data["y"]

    logger.info(f"Loaded Gemini training data: {X.shape}")
    logger.info(f"Gemini categories: {sorted(set(y))}")
    return X, y


def map_old_to_new(old_label: str) -> str:
    """Map old 5-category to new 6-category system."""
    mapping = {
        "ack": "acknowledge",
        "clarify": "question",
        "emotional": "emotion",
        "info": "statement",
        "social": "acknowledge",  # Conservative: treat social as acknowledge
    }
    return mapping.get(old_label, "statement")


def merge_datasets(
    X_orig: np.ndarray, y_orig: np.ndarray, X_gemini: np.ndarray, y_gemini: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Merge original and Gemini datasets."""
    logger.info("\n" + "=" * 70)
    logger.info("MERGING DATASETS")
    logger.info("=" * 70)

    # Map original categories to new system
    y_orig_mapped = np.array([map_old_to_new(label) for label in y_orig])

    logger.info(f"\nOriginal (mapped to new 6-category):")
    from collections import Counter

    orig_dist = Counter(y_orig_mapped)
    for cat, count in sorted(orig_dist.items()):
        logger.info(f"  {cat:15s}: {count:5d}")

    logger.info(f"\nGemini (native 6-category):")
    gemini_dist = Counter(y_gemini)
    for cat, count in sorted(gemini_dist.items()):
        logger.info(f"  {cat:15s}: {count:5d}")

    # Concatenate
    X_combined = np.vstack([X_orig, X_gemini])
    y_combined = np.concatenate([y_orig_mapped, y_gemini])

    logger.info(f"\nCombined dataset: {X_combined.shape}")
    logger.info(f"Total examples: {len(y_combined)}")

    combined_dist = Counter(y_combined)
    logger.info(f"\nCombined distribution:")
    for cat, count in sorted(combined_dist.items()):
        logger.info(f"  {cat:15s}: {count:5d}")

    return X_combined, y_combined


def create_splits(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42
) -> tuple:
    """Create stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    logger.info(f"\n" + "=" * 70)
    logger.info("TRAIN/TEST SPLIT")
    logger.info("=" * 70)
    logger.info(f"Train: {X_train.shape}")
    logger.info(f"Test: {X_test.shape}")

    logger.info(f"\nTrain distribution:")
    from collections import Counter

    train_dist = Counter(y_train)
    for cat, count in sorted(train_dist.items()):
        logger.info(f"  {cat:15s}: {count:5d}")

    logger.info(f"\nTest distribution:")
    test_dist = Counter(y_test)
    for cat, count in sorted(test_dist.items()):
        logger.info(f"  {cat:15s}: {count:5d}")

    return X_train, X_test, y_train, y_test


def save_combined_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Save combined training data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(OUTPUT_DIR / "train.npz", X=X_train, y=y_train)
    np.savez(OUTPUT_DIR / "test.npz", X=X_test, y=y_test)

    # Metadata
    labels = sorted(set(y_train) | set(y_test))
    metadata = {
        "source": "combined (original + gemini)",
        "original_size": 18496,
        "gemini_size": 2430,
        "combined_size": len(y_train) + len(y_test),
        "categories": labels,
        "label_map": {label: i for i, label in enumerate(labels)},
        "feature_dims": X_train.shape[1],
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    logger.info(f"\n✓ Combined training data saved to {OUTPUT_DIR}")
    logger.info(f"  - train.npz: {X_train.shape}")
    logger.info(f"  - test.npz: {X_test.shape}")
    logger.info(f"  - metadata.json")


def main() -> None:
    # Load both datasets
    X_orig, y_orig = load_original_data()
    X_gemini, y_gemini = load_gemini_data()

    # Merge
    X_combined, y_combined = merge_datasets(X_orig, y_orig, X_gemini, y_gemini)

    # Split
    X_train, X_test, y_train, y_test = create_splits(X_combined, y_combined)

    # Save
    save_combined_data(X_train, X_test, y_train, y_test)

    logger.info("\n" + "=" * 70)
    logger.info("READY TO RETRAIN")
    logger.info("=" * 70)
    logger.info("Next: uv run python scripts/retrain_on_combined.py")


if __name__ == "__main__":
    main()
