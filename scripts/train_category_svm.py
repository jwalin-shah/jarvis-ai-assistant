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
import threading
import time
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

from jarvis.utils.memory import MemoryMonitor, get_swap_info, get_top_memory_processes

# Setup logging to file for real-time progress tracking
LOG_FILE = PROJECT_ROOT / "training_progress.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def train(
    data_dir: Path | None = None,
    seed: int = 42,
    label_map: str = "4class",
) -> dict:
    """Train LinearSVC on labeled category data.

    Args:
        data_dir: Path to labeled data directory
        seed: Random seed
        label_map: "4class" for native labels, "3class" to merge directive+commissive → action

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

    # Apply label mapping if 3-class
    if label_map == "3class":
        label_mapping = {
            "inform": "inform",
            "question": "question",
            "directive": "action",
            "commissive": "action",
        }
        y_train = np.array([label_mapping.get(label, label) for label in y_train])
        y_test = np.array([label_mapping.get(label, label) for label in y_test])
        print("Applied 3-class label mapping (directive+commissive → action)")

    embedding_dims = metadata["embedding_dims"]
    hand_crafted_dims = metadata["hand_crafted_dims"]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Embedding dims: {embedding_dims}, Hand-crafted dims: {hand_crafted_dims}")
    print(f"Labels: {sorted(set(y_train))}")

    # Initial memory check
    swap_info = get_swap_info()
    print(
        f"\nInitial swap: {swap_info['used_mb']:.1f}MB / {swap_info['total_mb']:.1f}MB "
        f"({swap_info['percent']:.1f}%)"
    )
    if swap_info["used_mb"] > 500:
        print("⚠️  WARNING: Already using significant swap before training started")
        print("Consider closing other applications to free up memory")

    # Build pipeline: scale hand-crafted features, passthrough embeddings
    embedding_cols = list(range(embedding_dims))
    hc_cols = list(range(embedding_dims, embedding_dims + hand_crafted_dims))

    preprocessor = ColumnTransformer(
        transformers=[
            ("embeddings", "passthrough", embedding_cols),
            ("hand_crafted", StandardScaler(), hc_cols),
        ],
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("svm", LinearSVC(class_weight="balanced", max_iter=5000, random_state=seed)),
        ]
    )

    # GridSearchCV over C values
    param_grid = {"svm__C": [0.01, 0.1, 1.0, 10.0]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    print("\nRunning GridSearchCV (5-fold, 4 C values)...")
    print("Progress will show after each fold completes...\n")

    # Adaptive n_jobs based on current swap state
    # Each worker: ~350MB data + 200-300MB optimizer buffers = 550MB
    # n_jobs=2 would use 1.1GB + main process = risk of swapping
    current_swap = get_swap_info()["used_mb"]
    if current_swap < 200 and X_train.shape[0] < 100000:  # Low swap + reasonable dataset size
        n_jobs = 2
        print(f"✓ Low swap ({current_swap:.0f}MB) - using n_jobs=2 for faster training")
    else:
        n_jobs = 1
        print(f"Using n_jobs=1 to avoid swap thrashing (current swap: {current_swap:.0f}MB)")

    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=n_jobs,
        verbose=2,  # Show detailed progress for each fold
    )

    # Start memory monitoring
    monitor = MemoryMonitor(interval_sec=15.0, swap_threshold_mb=500.0)
    monitor.start("GridSearchCV")

    # Background thread to periodically check memory
    stop_monitoring = threading.Event()

    def memory_checker():
        while not stop_monitoring.is_set():
            monitor.check()
            stop_monitoring.wait(timeout=15.0)  # Check every 15 seconds

    checker_thread = threading.Thread(target=memory_checker, daemon=True)
    checker_thread.start()

    total_fits = len(param_grid["svm__C"]) * cv.get_n_splits()
    print(
        f"Starting training at {time.strftime('%H:%M:%S')} "
        f"({total_fits} fits: {len(param_grid['svm__C'])} C values x "
        f"{cv.get_n_splits()} folds)...",
        flush=True,
    )
    train_start = time.time()
    try:
        search.fit(X_train, y_train)
    finally:
        # Stop monitoring
        stop_monitoring.set()
        checker_thread.join(timeout=2.0)
        final_info = monitor.stop()

        elapsed = time.time() - train_start
        print(
            f"\nTraining completed at {time.strftime('%H:%M:%S')} "
            f"(elapsed: {elapsed:.1f}s, {elapsed / total_fits:.1f}s/fit)",
            flush=True,
        )
        print(f"Peak RAM: {monitor.peak_rss_mb:.1f}MB")
        print(f"Peak swap: {monitor.peak_swap_mb:.1f}MB")

        # Show top memory processes if swap was significant
        if monitor.peak_swap_mb > 500:
            print("\n⚠️  HIGH SWAP USAGE DETECTED")
            swap_info = get_swap_info()
            print(
                f"System swap: {swap_info['used_mb']:.1f}MB / {swap_info['total_mb']:.1f}MB "
                f"({swap_info['percent']:.1f}%)"
            )
            print("\nTop memory-consuming processes:")
            for proc in get_top_memory_processes(limit=5):
                print(
                    f"  PID {proc['pid']}: {proc['name']:20s} - "
                    f"RSS: {proc['rss_mb']:8.1f}MB, VMS: {proc['vms_mb']:8.1f}MB"
                )
            print()

    print(f"\nBest C: {search.best_params_['svm__C']}")
    print(f"Best CV macro F1: {search.best_score_:.4f}")

    # Show all C values and their scores
    print("\nAll C values tested:")
    for i in range(len(search.cv_results_["params"])):
        c_val = search.cv_results_["params"][i]["svm__C"]
        mean_score = search.cv_results_["mean_test_score"][i]
        std_score = search.cv_results_["std_test_score"][i]
        print(f"  C={c_val:6.2f}: {mean_score:.4f} (+/- {std_score:.4f})")

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

    # Save all C value results for comparison
    cv_results = {
        str(search.cv_results_["params"][i]["svm__C"]): {
            "mean_score": float(search.cv_results_["mean_test_score"][i]),
            "std_score": float(search.cv_results_["std_test_score"][i]),
        }
        for i in range(len(search.cv_results_["params"]))
    }

    model_metadata = {
        "label_map": label_map,
        "labels": labels,
        "best_C": search.best_params_["svm__C"],
        "cv_macro_f1": float(search.best_score_),
        "test_macro_f1": float(report["macro avg"]["f1-score"]),
        "test_accuracy": float(report["accuracy"]),
        "per_class_f1": {
            label: float(report[label]["f1-score"]) for label in labels if label in report
        },
        "cv_results_all_C": cv_results,  # All C values tested
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
    parser.add_argument(
        "--label-map",
        choices=["4class", "3class"],
        default="4class",
        help="3class merges directive+commissive → action",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    train(data_dir=args.data_dir, seed=args.seed, label_map=args.label_map)
    return 0


if __name__ == "__main__":
    sys.exit(main())
