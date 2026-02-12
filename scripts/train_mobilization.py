#!/usr/bin/env python3
"""Train mobilization classifier (LightGBM or Logistic Regression) on Gemini labels.

Usage:
    uv run python scripts/train_mobilization.py --model-type lightgbm
    uv run python scripts/train_mobilization.py --model-type logistic
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from jarvis.utils.logging import setup_script_logging

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "mobilization_gemini"
RULE_BASED_BASELINE_F1 = 0.342


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-type",
        choices=["lightgbm", "logistic"],
        default="lightgbm",
        help="Model type to train (default: lightgbm).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing mobilization train.npz and test.npz (default: %(default)s).",
    )
    return parser.parse_args(argv)


def load_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load training data."""
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


def _build_model(model_type: str) -> tuple:
    """Build model and optional scaler based on model type.

    Returns:
        (model, scaler_or_None)
    """
    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            objective="multiclass",
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=100,
            random_state=42,
            verbose=1,
        )
        return model, None
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        model = LogisticRegression(max_iter=1000, random_state=42, verbose=1)
        return model, StandardScaler()


def train_and_evaluate(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Train model and evaluate on test set."""
    model, scaler = _build_model(model_type)
    model_name = "LightGBM" if model_type == "lightgbm" else "Logistic Regression"

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING %s", model_name.upper())
    logger.info("=" * 70)

    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    logger.info("Training model...")
    model.fit(X_train, y_train)

    logger.info("\n" + "=" * 70)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 70)

    y_pred = model.predict(X_test)

    print("\nClassification Report:", flush=True)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred), flush=True)

    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    logger.info("\nConfusion Matrix:")
    header = "          " + "  ".join(f"{l:>10}" for l in labels)
    logger.info(header)
    for i, label in enumerate(labels):
        row = f"{label:>10}" + "  ".join(f"{cm[i][j]:>10}" for j in range(len(labels)))
        logger.info(row)

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
        "model_type": model_type,
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy),
        "per_class_f1": {label: float(report[label]["f1-score"]) for label in labels},
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    setup_script_logging(f"train_mobilization_{args.model_type}")
    X_train, y_train, X_test, y_test = load_data(args.data_dir)
    results = train_and_evaluate(args.model_type, X_train, y_train, X_test, y_test)

    model_name = "LightGBM" if args.model_type == "lightgbm" else "Logistic Regression"
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON: Current Rule-Based vs %s", model_name)
    logger.info("=" * 70)
    logger.info("Rule-based (current):  F1 = %.4f (baseline)", RULE_BASED_BASELINE_F1)
    logger.info(f"{model_name + ':':23s}F1 = {results['macro_f1']:.4f}")

    improvement = ((results["macro_f1"] - RULE_BASED_BASELINE_F1) / RULE_BASED_BASELINE_F1) * 100
    logger.info(f"Improvement: {improvement:+.1f}%")

    if results["macro_f1"] > RULE_BASED_BASELINE_F1:
        logger.info("\n%s is BETTER than rule-based!", model_name)
    else:
        logger.info("\n%s is worse - check what's wrong", model_name)


if __name__ == "__main__":
    main()
