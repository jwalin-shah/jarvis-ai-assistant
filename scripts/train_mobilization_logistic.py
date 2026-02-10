#!/usr/bin/env python3
"""Train Logistic Regression for mobilization classification on Gemini labels.

Usage:
    uv run python scripts/train_mobilization_logistic.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "mobilization_gemini"


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load training data."""
    train_data = np.load(DATA_DIR / "train.npz", allow_pickle=True)
    test_data = np.load(DATA_DIR / "test.npz", allow_pickle=True)

    X_train, y_train = train_data["X"], train_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Train logistic regression model."""
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING LOGISTIC REGRESSION")
    logger.info("=" * 70)

    # Scale features (important for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        verbose=1,
    )

    logger.info("Training model...")
    model.fit(X_train_scaled, y_train)

    # Evaluate
    logger.info("\n" + "=" * 70)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 70)

    y_pred = model.predict(X_test_scaled)

    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

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
        "model_type": "logistic_regression",
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy),
        "per_class_f1": {label: float(report[label]["f1-score"]) for label in labels},
    }


def main() -> None:
    X_train, y_train, X_test, y_test = load_data()
    results = train_logistic_regression(X_train, y_train, X_test, y_test)

    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON: Current Rule-Based vs Logistic Regression")
    logger.info("=" * 70)
    logger.info(f"Rule-based (current):      F1 = 0.3420 (baseline)")
    logger.info(f"Logistic Regression:       F1 = {results['macro_f1']:.4f}")

    improvement = ((results["macro_f1"] - 0.342) / 0.342) * 100
    logger.info(f"Improvement: {improvement:+.1f}%")

    if results["macro_f1"] > 0.342:
        logger.info("\n✓ Logistic Regression is BETTER than rule-based!")
    else:
        logger.info("\n✗ Logistic Regression is worse - check what's wrong")


if __name__ == "__main__":
    main()
