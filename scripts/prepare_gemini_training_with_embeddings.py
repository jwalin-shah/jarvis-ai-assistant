#!/usr/bin/env python3
"""Prepare Gemini training data WITH BERT embeddings (to match original format).

Extracts: 384 BERT embeddings + 147 hand-crafted = 531 total features
(vs original which had 384 BERT + 19 hand-crafted = 403)

This allows merging with original training data.
"""

from __future__ import annotations

import argparse
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

    # Batch encode all texts at once (10-40x faster than one-at-a-time)
    texts = [ex["text"] for ex in examples]
    logger.info(f"Batch encoding {len(texts)} texts...")
    all_embeddings = embedder_mixin.embedder.encode(texts)
    logger.info("Batch encoding complete")

    for i, (ex, embedding) in enumerate(
        tqdm(
            zip(examples, all_embeddings),
            desc="Extracting features",
            total=len(examples),
            unit="ex",
        )
    ):
        try:
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


def main(argv: Sequence[str] | None = None) -> None:
    from scripts.gemini_prepare_shared import (
        create_splits,
        load_labeled_examples,
        save_training_data,
    )

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

    save_training_data(
        args.output_dir, X_train, X_test, y_train, y_test, logger,
        source="gemini_labeled_with_embeddings",
    )

    logger.info(f"\nNext: Update merge script to use {args.output_dir}")


if __name__ == "__main__":
    main()
