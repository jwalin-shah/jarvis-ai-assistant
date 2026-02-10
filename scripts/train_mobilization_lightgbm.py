#!/usr/bin/env python3
"""Train LightGBM for mobilization classification on Gemini labels.

Usage:
    uv run python scripts/train_mobilization_lightgbm.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure logging with both file and stream handlers."""
    log_file = Path(__file__).resolve().parent.parent / "logs" / "train_mobilization_lightgbm.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger.info("Logging to %s", log_file)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "mobilization_gemini"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing mobilization train.npz and test.npz (default: %(default)s).",
    )
    return parser.parse_args(argv)


def load_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load training data."""
    import numpy as np

    try:
        train_data = np.load(data_dir / "train.npz", allow_pickle=True)
        test_data = np.load(data_dir / "test.npz", allow_pickle=True)
    except OSError as exc:
        logger.error("Failed to load mobilization data from %s: %s", data_dir, exc)
        raise SystemExit(1) from exc

    X_train, y_train = train_data["X"], train_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Train LightGBM model."""
    from lightgbm import LGBMClassifier
    from sklearn.metrics import classification_report, confusion_matrix

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING LightGBM")
    logger.info("=" * 70)

    # Train model
    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(set(y_train)),
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=100,
        random_state=42,
        verbose=1,
    )

    logger.info("Training model...")
    model.fit(X_train, y_train)

    # Evaluate
    logger.info("\n" + "=" * 70)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 70)

    y_pred = model.predict(X_test)

    print("\nClassification Report:", flush=True)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred), flush=True)

    # Confusion matrix
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    logger.info("\nConfusion Matrix:")
    header = "          " + "  ".join(f"{l:>10}" for l in labels)
    logger.info(header)
    for i, label in enumerate(labels):
        row = f"{label:>10}" + "  ".join(f"{cm[i][j]:>10}" for j in range(len(labels)))
        logger.info(row)

    # Results
    macro_f1 = report["macro avg"]["f1-score"]
    accuracy = report["accuracy"]

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info(f"Macro F1:  {macro_f1:.4f}")
    logger.info(f"Accuracy:  {accuracy:.4f}")

    logger.info("\nPer-class F1:")
    for label in labels:
        f1 = report[label]["f1-score"]
        logger.info(f"  {label:10s}: {f1:.4f}")

    return {
        "model_type": "lightgbm",
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy),
        "per_class_f1": {label: float(report[label]["f1-score"]) for label in labels},
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _setup_logging()
    X_train, y_train, X_test, y_test = load_data(args.data_dir)
    results = train_lightgbm(X_train, y_train, X_test, y_test)

    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON: Current Rule-Based vs LightGBM")
    logger.info("=" * 70)
    logger.info(f"Rule-based (current):  F1 = 0.3420 (baseline)")
    logger.info(f"LightGBM:              F1 = {results['macro_f1']:.4f}")

    improvement = ((results["macro_f1"] - 0.342) / 0.342) * 100
    logger.info(f"Improvement: {improvement:+.1f}%")

    if results["macro_f1"] > 0.342:
        logger.info("\n✓ LightGBM is BETTER than rule-based!")
    else:
        logger.info("\n✗ LightGBM is worse - check what's wrong")


if __name__ == "__main__":
    main()
