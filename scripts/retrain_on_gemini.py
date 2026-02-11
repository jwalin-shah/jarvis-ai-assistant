#!/usr/bin/env python3
"""Retrain LightGBM category classifier on Gemini labels.

Steps:
1. Load Gemini-labeled features (4,629 train / 1,158 test)
2. Train LightGBM with GridSearchCV for hyperparameter tuning
3. Evaluate on test set
4. Save new model
5. Compare old vs new performance

Usage:
    uv run python scripts/retrain_on_gemini.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

def _setup_logging() -> None:
    """Configure logging with FileHandler + StreamHandler."""
    log_file = Path("retrain_on_gemini.log")
    file_handler = logging.FileHandler(log_file, mode="a")
    stream_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[file_handler, stream_handler],
    )


logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DATA_DIR = ROOT / "data" / "gemini_features"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-data-dir",
        type=Path,
        default=TRAIN_DATA_DIR,
        help="Directory containing train.npz and test.npz (default: %(default)s).",
    )
    return parser.parse_args(argv)


def load_training_data(train_data_dir: Path) -> tuple:
    """Load Gemini training data."""
    import numpy as np

    if not train_data_dir.exists():
        logger.error("Training data not found: %s", train_data_dir)
        logger.error("Run: uv run python scripts/prepare_gemini_training_data.py")
        sys.exit(1)

    try:
        train_data = np.load(train_data_dir / "train.npz", allow_pickle=True)
        test_data = np.load(train_data_dir / "test.npz", allow_pickle=True)
    except OSError as exc:
        logger.error("Failed to load training data from %s: %s", train_data_dir, exc)
        raise SystemExit(1) from exc

    X_train, y_train = train_data["X"], train_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Features: {X_train.shape[1]}")
    logger.info(f"Train labels: {sorted(set(y_train))}")

    return X_train, y_train, X_test, y_test


def train_model(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> dict:
    """Train LightGBM with GridSearchCV."""
    from lightgbm import LGBMClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import GridSearchCV

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING LightGBM on Gemini Labels")
    logger.info("=" * 70)

    # LightGBM hyperparameters to tune
    param_grid = {
        "num_leaves": [31, 50, 100],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 200],
    }

    base_model = LGBMClassifier(
        objective="multiclass",
        num_class=len(set(y_train)),
        random_state=42,
        verbose=-1,
    )

    logger.info(f"Hyperparameter grid: {param_grid}")
    logger.info("Running GridSearchCV...")

    search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=1,  # CRITICAL: 8GB RAM, must use n_jobs=1
        verbose=2,
    )

    search.fit(X_train, y_train)

    logger.info(f"\nBest params: {search.best_params_}")
    logger.info(f"Best CV F1: {search.best_score_:.4f}")

    # Evaluate on test set
    logger.info("\n" + "=" * 70)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 70)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\nClassification Report:", flush=True)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred), flush=True)

    # Confusion matrix
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    logger.info("\nConfusion Matrix:")
    header = "          " + "  ".join(f"{l:>12}" for l in labels)
    logger.info(header)
    for i, label in enumerate(labels):
        row = f"{label:>12}" + "  ".join(f"{cm[i][j]:>12}" for j in range(len(labels)))
        logger.info(row)

    # Summary
    test_f1 = report["macro avg"]["f1-score"]
    test_accuracy = report["accuracy"]

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Test F1 (macro): {test_f1:.4f}")
    logger.info(f"Test accuracy:   {test_accuracy:.4f}")

    # Per-class F1
    logger.info("\nPer-class F1:")
    for label in labels:
        f1 = report[label]["f1-score"]
        logger.info(f"  {label:15s}: {f1:.4f}")

    return {
        "best_params": search.best_params_,
        "best_cv_f1": float(search.best_score_),
        "test_f1": float(test_f1),
        "test_accuracy": float(test_accuracy),
        "per_class_f1": {label: float(report[label]["f1-score"]) for label in labels},
        "model": best_model,
    }


def main(argv: Sequence[str] | None = None) -> None:
    _setup_logging()
    logging.info("Starting retrain_on_gemini.py")
    args = parse_args(argv)
    # Load data
    X_train, y_train, X_test, y_test = load_training_data(args.train_data_dir)

    # Train
    results = train_model(X_train, y_train, X_test, y_test)

    # Compare with baseline
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON: OLD vs NEW")
    logger.info("=" * 70)
    logger.info(f"Baseline (on Gemini labels): F1 = 0.4020 (old model)")
    logger.info(f"Retrained on Gemini:         F1 = {results['test_f1']:.4f} (new model)")

    improvement = ((results["test_f1"] - 0.402) / 0.402) * 100
    logger.info(f"Improvement: {improvement:+.1f}%")

    if results["test_f1"] > 0.402:
        logger.info("\n✓ NEW MODEL IS BETTER - Ready to deploy")
    else:
        logger.info("\n✗ New model is worse - investigate")

    logger.info("\nNext step: Review model performance and optionally deploy")
    logging.info("Finished retrain_on_gemini.py")


if __name__ == "__main__":
    main()
