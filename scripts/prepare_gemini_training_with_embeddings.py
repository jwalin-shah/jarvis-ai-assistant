#!/usr/bin/env python3
"""Prepare Gemini training data WITH BERT embeddings (to match original format).

Extracts: 384 BERT embeddings + 147 hand-crafted = 531 total features
(vs original which had 384 BERT + 19 hand-crafted = 403)

This allows merging with original training data.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELED_DATA = ROOT / "data" / "gemini_training" / "labeled_examples.jsonl"


def setup_logging() -> logging.Logger:
    """Setup logging with file and stream handlers."""
    log_file = Path("prepare_gemini_training_with_embeddings.log")
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labeled-data",
        type=Path,
        default=LABELED_DATA,
        help="Path to labeled_examples.jsonl (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "gemini_training_with_embeddings",
        help="Directory to write extracted features (default: %(default)s).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data in test split (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split (default: %(default)s).",
    )
    return parser.parse_args(argv)


def load_labeled_examples(labeled_data: Path, logger: logging.Logger) -> list[dict]:
    """Load Gemini-labeled examples."""
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


def extract_features_with_embeddings(
    examples: list[dict], logger: logging.Logger
) -> tuple[np.ndarray, list[str], list[str]]:
    """Extract BERT embeddings + hand-crafted features."""
    import numpy as np
    from tqdm import tqdm

    from jarvis.classifiers.mixins import EmbedderMixin
    from jarvis.features.category_features import CategoryFeatureExtractor

    logger.info("Initializing embedder and feature extractor...")
    embedder_mixin = EmbedderMixin()
    feature_extractor = CategoryFeatureExtractor()

    features_list = []
    categories = []
    ids = []

    logger.info(f"Extracting BERT embeddings + features for {len(examples)} examples...")

    for i, ex in enumerate(tqdm(examples, desc="Extracting features", unit="ex")):
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
    logger.info(
        f"Feature breakdown: 384 BERT + {X.shape[1] - 384} hand-crafted = {X.shape[1]} total"
    )

    return X, y, ids


def create_splits(
    X: np.ndarray,
    y: np.ndarray,
    ids: list[str],
    test_size: float = 0.2,
    seed: int = 42,
    logger: logging.Logger | None = None,
) -> tuple:
    """Create stratified train/test splits."""
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
    logger: logging.Logger,
) -> None:
    """Save training data."""
    import numpy as np

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
        "source": "gemini_labeled_with_embeddings",
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

    logger.info(f"✓ Training data saved to {output_dir}")
    logger.info(f"  - train.npz: {X_train.shape}")
    logger.info(f"  - test.npz: {X_test.shape}")


def main(argv: Sequence[str] | None = None) -> None:
    logger = setup_logging()
    args = parse_args(argv)
    examples = load_labeled_examples(args.labeled_data, logger)
    X, y, ids = extract_features_with_embeddings(examples, logger)
    X_train, X_test, y_train, y_test, ids_train, ids_test = create_splits(
        X,
        y,
        ids,
        test_size=args.test_size,
        seed=args.seed,
        logger=logger,
    )

    save_training_data(args.output_dir, X_train, X_test, y_train, y_test, logger)

    logger.info(f"\nNext: Update merge script to use {args.output_dir}")


if __name__ == "__main__":
    main()
