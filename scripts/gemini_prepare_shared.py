"""Shared helpers for Gemini training data preparation scripts.

Extracted from prepare_gemini_training_with_embeddings.py and
prepare_gemini_with_full_features.py to eliminate duplication.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np


def load_labeled_examples(labeled_data: Path, logger: logging.Logger) -> list[dict]:
    """Load Gemini-labeled examples from JSONL file.

    Args:
        labeled_data: Path to labeled_examples.jsonl.
        logger: Logger instance.

    Returns:
        List of example dicts.
    """
    if not labeled_data.exists():
        logger.error("Labeled data not found: %s", labeled_data)
        sys.exit(1)

    examples = []
    try:
        with labeled_data.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
    except OSError as exc:
        logger.error("Failed to read labeled data %s: %s", labeled_data, exc)
        raise SystemExit(1) from exc

    logger.info(f"Loaded {len(examples)} labeled examples")
    return examples


def create_splits(
    X: np.ndarray,
    y: np.ndarray,
    ids: list[str],
    test_size: float = 0.2,
    seed: int = 42,
    logger: logging.Logger | None = None,
) -> tuple:
    """Create stratified train/test splits.

    Args:
        X: Feature matrix.
        y: Label array.
        ids: Example IDs.
        test_size: Fraction of data in test split.
        seed: Random seed.
        logger: Optional logger for distribution reporting.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, ids_train, ids_test).
    """
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X,
        y,
        ids,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    if logger:
        logger.info("Train: %s, Test: %s", X_train.shape, X_test.shape)

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
    logger: logging.Logger,
    source: str = "gemini_labeled",
) -> None:
    """Save training data as NPZ files with metadata.

    Args:
        output_dir: Directory to write files into.
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train: Training labels.
        y_test: Test labels.
        logger: Logger instance.
        source: Source identifier for metadata.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("Failed to create output directory %s: %s", output_dir, exc)
        raise SystemExit(1) from exc

    try:
        np.savez(output_dir / "train.npz", X=X_train, y=y_train)
        np.savez(output_dir / "test.npz", X=X_test, y=y_test)
    except OSError as exc:
        logger.error("Failed to write NPZ training data in %s: %s", output_dir, exc)
        raise SystemExit(1) from exc

    labels = sorted(set(y_train) | set(y_test))
    metadata = {
        "source": source,
        "categories": labels,
        "label_map": {label: i for i, label in enumerate(labels)},
        "feature_dims": int(X_train.shape[1]),
        "embedding_dims": 384,
        "hand_crafted_dims": X_train.shape[1] - 384,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    try:
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    except OSError as exc:
        logger.error("Failed to write metadata in %s: %s", output_dir, exc)
        raise SystemExit(1) from exc

    logger.info(f"Training data saved to {output_dir}")
    logger.info(f"  - train.npz: {X_train.shape}")
    logger.info(f"  - test.npz: {X_test.shape}")
    logger.info("  - metadata.json")
