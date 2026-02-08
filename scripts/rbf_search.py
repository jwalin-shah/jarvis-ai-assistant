#!/usr/bin/env python3
"""RBF kernel grid search with class imbalance controls.

Tests RBF SVM with different C values and class balancing strategies.

Usage:
    # Baseline RBF
    uv run python scripts/rbf_search.py

    # Oversample minority classes
    uv run python scripts/rbf_search.py --oversample clarify=3.0 brief=1.5

    # Custom C values
    uv run python scripts/rbf_search.py --c-values 10 50 100 --gamma scale auto
"""

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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.utils.memory import MemoryMonitor, get_swap_info, get_memory_info, get_top_memory_processes

import psutil

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


def parse_resample_args(resample_list):
    """Parse resample arguments like ['commissive=2.6', 'inform=0.7']"""
    resample_map = {}
    if resample_list:
        for item in resample_list:
            if not item or item.strip() == "":  # Skip empty strings
                continue
            label, factor = item.split("=")
            resample_map[label] = float(factor)
    return resample_map


def apply_resampling(X, y, resample_map, seed=42):
    """Resample classes: oversample (factor>1) or downsample (factor<1)."""
    if not resample_map:
        return X, y

    print(f"\nApplying resampling:")
    X_list = []
    y_list = []

    for label in sorted(set(y)):
        mask = y == label
        label_X = X[mask]
        label_y = y[mask]
        n_original = len(label_y)

        factor = resample_map.get(label, 1.0)  # Default: keep as-is

        if factor > 1.0:
            # Oversample: add duplicates
            n_to_add = int(n_original * (factor - 1.0))
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(label_y), size=n_to_add, replace=True)
            X_list.append(label_X)
            X_list.append(label_X[indices])
            y_list.append(label_y)
            y_list.append(label_y[indices])
            print(f"  {label:12s}: {n_original:5d} → {n_original + n_to_add:5d} ({factor:.2f}x oversample)")

        elif factor < 1.0:
            # Downsample: remove samples
            n_keep = int(n_original * factor)
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(label_y), size=n_keep, replace=False)
            X_list.append(label_X[indices])
            y_list.append(label_y[indices])
            print(f"  {label:12s}: {n_original:5d} → {n_keep:5d} ({factor:.2f}x downsample)")

        else:
            # Keep as-is
            X_list.append(label_X)
            y_list.append(label_y)
            print(f"  {label:12s}: {n_original:5d} (unchanged)")

    X_resampled = np.vstack(X_list)
    y_resampled = np.concatenate(y_list)

    # Shuffle
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(y_resampled))

    return X_resampled[indices], y_resampled[indices]


def main():
    parser = argparse.ArgumentParser(description="RBF kernel grid search with class balancing")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/dailydialog_native"),
        help="Data directory",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Use subset for quick testing (default: None = use full dataset)",
    )
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=[7.0],
        help="C values to test (default: 7.0)",
    )
    parser.add_argument(
        "--gamma",
        type=str,
        nargs="+",
        default=["scale"],
        help="Gamma values to test: 'scale', 'auto', or float (default: scale)",
    )
    parser.add_argument(
        "--resample",
        type=str,
        nargs="+",
        default=["commissive=2.6", "directive=3.0"],
        help="Resample classes: >1.0=oversample, <1.0=downsample (default: commissive=2.6 directive=3.0)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel jobs (default: auto-detect based on memory)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("RBF Kernel Grid Search")
    print("=" * 60)
    print(f"C values: {args.c_values}")
    print(f"Gamma values: {args.gamma}")

    # Check initial memory
    swap_info = get_swap_info()
    print(f"\nInitial swap: {swap_info['used_mb']:.1f}MB ({swap_info['percent']:.1f}%)")

    # Load pre-split data (no need to split again!)
    print(f"\nLoading pre-split data from {args.data_dir}/...")
    train_data = np.load(args.data_dir / "train.npz", allow_pickle=True)
    test_data = np.load(args.data_dir / "test.npz", allow_pickle=True)
    metadata = json.loads((args.data_dir / "metadata.json").read_text())

    X_train, y_train = train_data["X"], train_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    embedding_dims = metadata["embedding_dims"]
    hand_crafted_dims = metadata["hand_crafted_dims"]

    print(f"Train: {len(X_train):,} samples")
    print(f"Test:  {len(X_test):,} samples")
    print(f"Total: {len(X_train) + len(X_test):,} samples")

    # Use subset if specified (for quick testing)
    if args.subset_size is not None and args.subset_size < len(X_train):
        from sklearn.model_selection import train_test_split
        print(f"\nUsing subset: {args.subset_size:,} samples (for testing)")
        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=args.subset_size,
            stratify=y_train,
            random_state=args.seed,
        )

    # Apply resampling ONLY to training data
    resample_map = parse_resample_args(args.resample)
    X_train_resampled, y_train_resampled = apply_resampling(
        X_train, y_train, resample_map, args.seed
    )

    print(f"\nAfter resampling: {len(X_train_resampled):,} training samples")

    # Show class distribution (training data after resampling)
    print(f"\nTraining class distribution (after resampling):")
    for label in sorted(set(y_train_resampled)):
        count = (y_train_resampled == label).sum()
        pct = 100 * count / len(y_train_resampled)
        print(f"  {label:12s}: {count:5d} ({pct:5.1f}%)")

    # Build preprocessor
    # Features: [embeddings (384) | hand_crafted (19) | spacy (14)]
    embedding_cols = list(range(embedding_dims))
    hc_cols = list(range(embedding_dims, embedding_dims + hand_crafted_dims))

    # Check if we have SpaCy features in the data
    spacy_dims = metadata.get("spacy_dims", 0)
    if spacy_dims > 0:
        spacy_cols = list(range(embedding_dims + hand_crafted_dims, embedding_dims + hand_crafted_dims + spacy_dims))
        print(f"\nFeature columns:")
        print(f"  Embeddings: {len(embedding_cols)} features (normalized, passthrough)")
        print(f"  Hand-crafted: {len(hc_cols)} features (scaled)")
        print(f"  SpaCy: {len(spacy_cols)} features (scaled)")

        preprocessor = ColumnTransformer(
            transformers=[
                ("embeddings", "passthrough", embedding_cols),
                ("hand_crafted", StandardScaler(), hc_cols),
                ("spacy", StandardScaler(), spacy_cols),
            ],
        )
    else:
        print(f"\nFeature columns:")
        print(f"  Embeddings: {len(embedding_cols)} features (normalized, passthrough)")
        print(f"  Hand-crafted: {len(hc_cols)} features (scaled)")

        preprocessor = ColumnTransformer(
            transformers=[
                ("embeddings", "passthrough", embedding_cols),
                ("hand_crafted", StandardScaler(), hc_cols),
            ],
        )

    # Set n_jobs
    if args.n_jobs is not None:
        n_jobs = args.n_jobs
        print(f"\nUsing n_jobs={n_jobs} (manual override)")
    else:
        # Auto-detect n_jobs (conservative for RBF)
        mem_info = get_memory_info()
        if mem_info.macos_pressure and mem_info.macos_pressure.pressure_level > 0:
            n_jobs = 1
            print(f"\nUsing n_jobs=1 (memory pressure: {mem_info.macos_pressure.pressure_level})")
        else:
            vm = psutil.virtual_memory()
            available_gb = (vm.available / 1024**3) - 1.5
            dataset_size_mb = (len(X_train_resampled) * X_train_resampled.shape[1] * 8) / 1024**2
            memory_per_worker_mb = dataset_size_mb * 2.0 + 200  # RBF needs more memory
            max_jobs = max(1, min(int((available_gb * 1024) / memory_per_worker_mb), 3))
            n_jobs = max_jobs
            print(f"\nUsing n_jobs={n_jobs} (auto-detected)")

    # Parse gamma values (handle 'scale'/'auto' and numeric)
    gamma_values = []
    for g in args.gamma:
        if g in ["scale", "auto"]:
            gamma_values.append(g)
        else:
            gamma_values.append(float(g))

    # Build pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("svm", SVC(
            kernel="rbf",
            class_weight="balanced",
            max_iter=50000,
            cache_size=1000,  # 1GB kernel cache
            tol=1e-3,
            random_state=args.seed,
            verbose=True,  # Show convergence progress
        )),
    ])

    # GridSearchCV over C and gamma
    param_grid = {
        "svm__C": args.c_values,
        "svm__gamma": gamma_values,
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    print(f"\nTesting {len(args.c_values)} C values × {len(gamma_values)} gamma values = {len(args.c_values) * len(gamma_values)} configs")
    print("Progress will show after each fold completes...\n")

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
    monitor.start("RBF GridSearchCV")

    # Background thread to periodically check memory
    stop_monitoring = threading.Event()

    def memory_checker():
        while not stop_monitoring.is_set():
            monitor.check()
            stop_monitoring.wait(timeout=15.0)  # Check every 15 seconds

    checker_thread = threading.Thread(target=memory_checker, daemon=True)
    checker_thread.start()

    print(f"Starting training at {time.strftime('%H:%M:%S')}...")
    logger.info("GridSearchCV started")

    start_time = time.time()
    try:
        search.fit(X_train_resampled, y_train_resampled)
    finally:
        # Stop monitoring
        stop_monitoring.set()
        checker_thread.join(timeout=2.0)
        final_info = monitor.stop()

        elapsed = time.time() - start_time
        print(f"\nTraining completed at {time.strftime('%H:%M:%S')}")
        print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"Peak RAM: {monitor.peak_rss_mb:.1f}MB")
        print(f"Peak swap: {monitor.peak_swap_mb:.1f}MB")

        # Show top memory processes if swap was significant
        if monitor.peak_swap_mb > 500:
            print("\n⚠️  HIGH SWAP USAGE DETECTED")
            swap_info = get_swap_info()
            print(f"System swap: {swap_info['used_mb']:.1f}MB / {swap_info['total_mb']:.1f}MB "
                  f"({swap_info['percent']:.1f}%)")
            print("\nTop memory-consuming processes:")
            for proc in get_top_memory_processes(limit=5):
                print(f"  PID {proc['pid']}: {proc['name']:20s} - "
                      f"RSS: {proc['rss_mb']:8.1f}MB, VMS: {proc['vms_mb']:8.1f}MB")
            print()

    # Extract results from GridSearchCV
    results = []
    print(f"\n{'C':>8s}  {'Gamma':>8s}  {'Mean F1':>8s}  {'Std':>6s}")
    print("-" * 45)
    for i in range(len(search.cv_results_['params'])):
        params = search.cv_results_['params'][i]
        mean_f1 = search.cv_results_['mean_test_score'][i]
        std_f1 = search.cv_results_['std_test_score'][i]

        gamma_str = str(params['svm__gamma']) if isinstance(params['svm__gamma'], str) else f"{params['svm__gamma']:.4f}"
        print(f"{params['svm__C']:8.1f}  {gamma_str:>8s}  {mean_f1:8.4f}  {std_f1:6.4f}")

        results.append({
            "C": params['svm__C'],
            "gamma": params['svm__gamma'],
            "mean_f1": mean_f1,
            "std_f1": std_f1,
        })

    # Best configuration
    best_result = {
        "C": search.best_params_['svm__C'],
        "gamma": search.best_params_['svm__gamma'],
        "mean_f1": search.best_score_,
        "std_f1": search.cv_results_['std_test_score'][search.best_index_],
    }

    print(f"\n{'=' * 60}")
    print(f"Best config: C={best_result['C']}, gamma={best_result['gamma']}")
    print(f"Best F1: {best_result['mean_f1']:.4f} ± {best_result['std_f1']:.4f}")
    print(f"{'=' * 60}")

    # Save best model
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "category_svm_rbf_best.pkl"

    joblib.dump(search.best_estimator_, model_path)
    print(f"\n✓ Best model saved to: {model_path.name}")
    logger.info(f"Model saved: {model_path}")

    # Evaluate on test set
    print(f"\n{'=' * 60}")
    print("TEST SET EVALUATION")
    print(f"{'=' * 60}")

    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = search.best_estimator_.predict(X_test)

    print("\nPer-class metrics:")
    print(classification_report(y_test, y_pred, digits=4))

    print("\nConfusion Matrix:")
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Print header
    print(f"{'True \\ Pred':>15s}", end="")
    for label in labels:
        print(f"{label[:10]:>12s}", end="")
    print()
    print("-" * 60)

    # Print rows
    for i, true_label in enumerate(labels):
        print(f"{true_label[:15]:>15s}", end="")
        for j in range(len(labels)):
            print(f"{cm[i, j]:>12d}", end="")
        print()

    # Save per-class results
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    best_result["test_set_metrics"] = report_dict
    best_result["confusion_matrix"] = cm.tolist()

    # Save test set for later evaluation
    test_file = model_dir / "test_data.npz"
    np.savez(test_file, X=X_test, y=y_test)
    print(f"\n✓ Test set saved to: {test_file.name} ({len(X_test):,} samples)")

    # Save results
    output_file = PROJECT_ROOT / f"rbf_search_results_{int(time.time())}.json"
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "train_size_original": len(train_data["y"]),
            "test_size_original": len(test_data["y"]),
            "train_size_after_resample": len(y_train_resampled),
            "c_values": args.c_values,
            "gamma_values": [str(g) for g in gamma_values],
            "resample": resample_map,
            "n_jobs": n_jobs,
            "seed": args.seed,
        },
        "class_distribution_train": {
            label: int((y_train_resampled == label).sum())
            for label in sorted(set(y_train_resampled))
        },
        "class_distribution_test": {
            label: int((y_test == label).sum())
            for label in sorted(set(y_test))
        },
        "results": results,
        "best_config": best_result,
    }
    output_file.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved to: {output_file.name}")

    # Recommendations
    print(f"\nNext steps:")
    if best_result['mean_f1'] < 0.65:
        print(f"  ⚠️  F1 is still low ({best_result['mean_f1']:.4f})")
        print(f"  → Try different oversampling rates")
    elif best_result['mean_f1'] > 0.75:
        print(f"  ✅ Good F1 score ({best_result['mean_f1']:.4f})!")
        print(f"  → Evaluate on test set with show_per_class_scores.py")
    else:
        print(f"  → Moderate F1 ({best_result['mean_f1']:.4f})")
        print(f"  → Try adjusting class balance")

    # Check final memory
    swap_info = get_swap_info()
    print(f"\nFinal swap: {swap_info['used_mb']:.1f}MB ({swap_info['percent']:.1f}%)")


if __name__ == "__main__":
    main()
