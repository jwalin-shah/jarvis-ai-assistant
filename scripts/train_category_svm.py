#!/usr/bin/env python3
"""Train LinearSVC category classifier on labeled SOC-2508 data.

Loads labeled features from data/soc_categories/{train,test}.npz,
trains a LinearSVC with GridSearchCV, computes per-class centroids,
and serializes the model to ~/.jarvis/embeddings/{model}/category_classifier_model/.

Usage:
    uv run python scripts/train_category_svm.py
    uv run python scripts/train_category_svm.py --data-dir data/soc_categories
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def train(
    data_dir: Path | None = None,
    seed: int = 42,
) -> dict:
    """Train LinearSVC on labeled category data.

    Returns:
        Dict with training metrics.
    """
    from jarvis.config import get_category_classifier_path

    if data_dir is None:
        data_dir = PROJECT_ROOT / "data" / "category_training"

    # Load data
    print(f"Loading data from {data_dir}/...")
    train_data = np.load(data_dir / "train.npz", allow_pickle=True)
    test_data = np.load(data_dir / "test.npz", allow_pickle=True)
    metadata = json.loads((data_dir / "metadata.json").read_text())

    X_train, y_train = train_data["X"], train_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    embedding_dims = metadata["embedding_dims"]
    hand_crafted_dims = metadata["hand_crafted_dims"]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Embedding dims: {embedding_dims}, Hand-crafted dims: {hand_crafted_dims}")
    print(f"Labels: {sorted(set(y_train))}")

    # Build pipeline: scale hand-crafted features, passthrough embeddings
    embedding_cols = list(range(embedding_dims))
    hc_cols = list(range(embedding_dims, embedding_dims + hand_crafted_dims))

    preprocessor = ColumnTransformer(
        transformers=[
            ("embeddings", "passthrough", embedding_cols),
            ("hand_crafted", StandardScaler(), hc_cols),
        ],
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("svm", LinearSVC(class_weight="balanced", max_iter=5000, random_state=seed)),
    ])

    # GridSearchCV over C values
    param_grid = {"svm__C": [0.01, 0.1, 1.0, 10.0]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    print("\nRunning GridSearchCV (5-fold, 4 C values)...")
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"\nBest C: {search.best_params_['svm__C']}")
    print(f"Best CV macro F1: {search.best_score_:.4f}")

    # Evaluate on test set
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\nTest set classification report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    # Compute per-class centroids from training embeddings
    print("Computing per-class centroids...")
    labels = sorted(set(y_train))
    centroids = {}
    for label in labels:
        mask = y_train == label
        # Use only the embedding columns for centroids
        label_embeddings = X_train[mask, :embedding_dims]
        centroid = label_embeddings.mean(axis=0)
        # Normalize centroid for cosine similarity
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids[label] = centroid.astype(np.float32)

    # Save model artifacts
    model_dir = get_category_classifier_path()
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving model to {model_dir}/...")

    # SVM model
    joblib.dump(best_model, model_dir / "svm_model.joblib")

    # Centroids
    np.savez(model_dir / "centroids.npz", **centroids)

    # Metadata
    label_map = {label: i for i, label in enumerate(labels)}
    model_metadata = {
        "label_map": label_map,
        "labels": labels,
        "best_C": search.best_params_["svm__C"],
        "cv_macro_f1": float(search.best_score_),
        "test_macro_f1": float(report["macro avg"]["f1-score"]),
        "test_accuracy": float(report["accuracy"]),
        "per_class_f1": {
            label: float(report[label]["f1-score"])
            for label in labels
            if label in report
        },
        "feature_dims": int(X_train.shape[1]),
        "embedding_dims": embedding_dims,
        "hand_crafted_dims": hand_crafted_dims,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    (model_dir / "metadata.json").write_text(json.dumps(model_metadata, indent=2))

    print(f"Model saved. Test macro F1: {report['macro avg']['f1-score']:.4f}")
    print(json.dumps(model_metadata, indent=2))

    return model_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Train category SVM classifier")
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Path to labeled data directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    train(data_dir=args.data_dir, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
