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

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_DIR = ROOT / "data" / "category_training"
GEMINI_DIR = ROOT / "data" / "gemini_features"
OUTPUT_DIR = ROOT / "data" / "combined_training"


def setup_logging() -> logging.Logger:
    """Setup logging with file and stream handlers."""
    log_file = Path("merge_training_datasets.log")
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
        "--original-dir",
        type=Path,
        default=ORIGINAL_DIR,
        help="Directory containing original train.npz (default: %(default)s).",
    )
    parser.add_argument(
        "--gemini-dir",
        type=Path,
        default=GEMINI_DIR,
        help="Directory containing Gemini train.npz (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to write combined datasets (default: %(default)s).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio for train/test split (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split (default: %(default)s).",
    )
    return parser.parse_args(argv)


def load_original_data(original_dir: Path, logger: logging.Logger) -> tuple[np.ndarray, np.ndarray]:
    """Load original DailyDialog/SAMSum training data."""
    train_file = original_dir / "train.npz"
    if not train_file.exists():
        logger.error(f"Original training data not found: {train_file}")
        sys.exit(1)

    try:
        data = np.load(train_file, allow_pickle=True)
    except OSError as exc:
        logger.error("Failed to load original training data %s: %s", train_file, exc)
        raise SystemExit(1) from exc
    X = data["X"]  # noqa: N806
    y = data["y"]

    logger.info(f"Loaded original training data: {X.shape}")
    logger.info(f"Original categories: {sorted(set(y))}")
    return X, y


def load_gemini_data(gemini_dir: Path, logger: logging.Logger) -> tuple[np.ndarray, np.ndarray]:
    """Load Gemini-labeled iMessage data."""
    train_file = gemini_dir / "train.npz"
    if not train_file.exists():
        logger.error(f"Gemini training data not found: {train_file}")
        logger.error("Run: uv run python scripts/prepare_gemini_training_data.py")
        sys.exit(1)

    try:
        data = np.load(train_file, allow_pickle=True)
    except OSError as exc:
        logger.error("Failed to load Gemini training data %s: %s", train_file, exc)
        raise SystemExit(1) from exc
    X = data["X"]  # noqa: N806
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
    X_orig: np.ndarray,  # noqa: N803
    y_orig: np.ndarray,
    X_gemini: np.ndarray,  # noqa: N803
    y_gemini: np.ndarray,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge original and Gemini datasets."""
    from collections import Counter

    logger.info("\n" + "=" * 70)
    logger.info("MERGING DATASETS")
    logger.info("=" * 70)

    # Map original categories to new system
    y_orig_mapped = np.array([map_old_to_new(label) for label in y_orig])

    logger.info("\nOriginal (mapped to new 6-category):")

    orig_dist = Counter(y_orig_mapped)
    for cat, count in sorted(orig_dist.items()):
        logger.info(f"  {cat:15s}: {count:5d}")

    logger.info("\nGemini (native 6-category):")
    gemini_dist = Counter(y_gemini)
    for cat, count in sorted(gemini_dist.items()):
        logger.info(f"  {cat:15s}: {count:5d}")

    # Concatenate
    X_combined = np.vstack([X_orig, X_gemini])  # noqa: N806
    y_combined = np.concatenate([y_orig_mapped, y_gemini])

    logger.info(f"\nCombined dataset: {X_combined.shape}")
    logger.info(f"Total examples: {len(y_combined)}")

    combined_dist = Counter(y_combined)
    logger.info("\nCombined distribution:")
    for cat, count in sorted(combined_dist.items()):
        logger.info(f"  {cat:15s}: {count:5d}")

    return X_combined, y_combined


def create_splits(
    X: np.ndarray,  # noqa: N803
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
    logger: logging.Logger | None = None,
) -> tuple:
    """Create stratified train/test split."""
    from collections import Counter

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    if logger:
        logger.info("\n" + "=" * 70)
        logger.info("TRAIN/TEST SPLIT")
        logger.info("=" * 70)
        logger.info(f"Train: {X_train.shape}")
        logger.info(f"Test: {X_test.shape}")

        logger.info("\nTrain distribution:")
        train_dist = Counter(y_train)
        for cat, count in sorted(train_dist.items()):
            logger.info(f"  {cat:15s}: {count:5d}")

        logger.info("\nTest distribution:")
        test_dist = Counter(y_test)
        for cat, count in sorted(test_dist.items()):
            logger.info(f"  {cat:15s}: {count:5d}")

    return X_train, X_test, y_train, y_test


def save_combined_data(
    X_train: np.ndarray,  # noqa: N803
    X_test: np.ndarray,  # noqa: N803
    y_train: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """Save combined training data."""

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("Failed to create output directory %s: %s", output_dir, exc)
        raise SystemExit(1) from exc

    try:
        np.savez(output_dir / "train.npz", X=X_train, y=y_train)
        np.savez(output_dir / "test.npz", X=X_test, y=y_test)
    except OSError as exc:
        logger.error("Failed to write combined npz files in %s: %s", output_dir, exc)
        raise SystemExit(1) from exc

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

    try:
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    except OSError as exc:
        logger.error("Failed to write metadata file in %s: %s", output_dir, exc)
        raise SystemExit(1) from exc

    logger.info(f"\n✓ Combined training data saved to {output_dir}")
    logger.info(f"  - train.npz: {X_train.shape}")
    logger.info(f"  - test.npz: {X_test.shape}")
    logger.info("  - metadata.json")


def main(argv: Sequence[str] | None = None) -> None:
    logger = setup_logging()
    args = parse_args(argv)
    # Load both datasets
    X_orig, y_orig = load_original_data(args.original_dir, logger)  # noqa: N806
    X_gemini, y_gemini = load_gemini_data(args.gemini_dir, logger)  # noqa: N806

    # Merge
    X_combined, y_combined = merge_datasets(  # noqa: N806
        X_orig, y_orig, X_gemini, y_gemini, logger
    )

    # Split
    X_train, X_test, y_train, y_test = create_splits(  # noqa: N806
        X_combined,
        y_combined,
        test_size=args.test_size,
        seed=args.seed,
        logger=logger,
    )

    # Save
    save_combined_data(
        X_train, X_test, y_train, y_test,
        output_dir=args.output_dir, logger=logger,
    )

    logger.info("\n" + "=" * 70)
    logger.info("READY TO RETRAIN")
    logger.info("=" * 70)
    logger.info("Next: uv run python scripts/retrain_on_combined.py")


if __name__ == "__main__":
    main()
