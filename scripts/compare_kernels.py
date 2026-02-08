#!/usr/bin/env python3
"""Compare Linear vs RBF kernels for SVM category classification.

Tests both kernels with the same C values to see if RBF can capture
non-linear patterns that linear misses.

Usage:
    uv run python scripts/compare_kernels.py
    uv run python scripts/compare_kernels.py --c-values 10 20 50
    uv run python scripts/compare_kernels.py --subset-size 20000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.utils.memory import get_swap_info, get_memory_info

import psutil


def main():
    parser = argparse.ArgumentParser(description="Compare Linear vs RBF SVM kernels")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/dailydialog_native"),
        help="Data directory",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=10000,
        help="Number of samples to use (default: 10000 for speed)",
    )
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=[10.0, 50.0, 100.0],
        help="C values to test (default: 10 50 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Linear vs RBF Kernel Comparison")
    print("=" * 60)
    print(f"Subset size: {args.subset_size:,} samples")
    print(f"C values to test: {args.c_values}")

    # Check initial memory
    swap_info = get_swap_info()
    print(f"\nInitial swap: {swap_info['used_mb']:.1f}MB ({swap_info['percent']:.1f}%)")

    # Load data
    print(f"\nLoading data from {args.data_dir}/...")
    train_data = np.load(args.data_dir / "train.npz", allow_pickle=True)
    metadata = json.loads((args.data_dir / "metadata.json").read_text())

    X_full, y_full = train_data["X"], train_data["y"]
    embedding_dims = metadata["embedding_dims"]
    hand_crafted_dims = metadata["hand_crafted_dims"]

    print(f"Full dataset: {X_full.shape}, Labels: {sorted(set(y_full))}")

    # Create stratified subset
    print(f"\nCreating stratified subset of {args.subset_size:,} samples...")
    from sklearn.model_selection import train_test_split

    X_subset, _, y_subset, _ = train_test_split(
        X_full,
        y_full,
        train_size=args.subset_size,
        stratify=y_full,
        random_state=args.seed,
    )

    # Show class distribution
    print(f"\nClass distribution:")
    for label in sorted(set(y_subset)):
        count = (y_subset == label).sum()
        pct = 100 * count / len(y_subset)
        print(f"  {label:12s}: {count:5d} ({pct:5.1f}%)")

    # Build preprocessor
    embedding_cols = list(range(embedding_dims))
    hc_cols = list(range(embedding_dims, embedding_dims + hand_crafted_dims))

    preprocessor = ColumnTransformer(
        transformers=[
            ("embeddings", "passthrough", embedding_cols),
            ("hand_crafted", StandardScaler(), hc_cols),
        ],
    )

    # Auto-detect n_jobs (conservative for RBF - it's slower and more memory-intensive)
    mem_info = get_memory_info()
    if mem_info.macos_pressure and mem_info.macos_pressure.pressure_level > 0:
        n_jobs = 1
        print(f"\nUsing n_jobs=1 (memory pressure: {mem_info.macos_pressure.pressure_level})")
    else:
        vm = psutil.virtual_memory()
        available_gb = (vm.available / 1024**3) - 1.5  # Reserve for OS
        dataset_size_mb = (len(X_subset) * X_subset.shape[1] * 8) / 1024**2

        # RBF needs more memory (kernel matrix), so be conservative
        memory_per_worker_mb = dataset_size_mb * 2.0 + 200
        max_jobs = max(1, min(int((available_gb * 1024) / memory_per_worker_mb), 3))
        n_jobs = max_jobs
        print(f"\nUsing n_jobs={n_jobs} (conservative for RBF kernel)")

    # Test both kernels
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    all_results = {"linear": [], "rbf": []}

    # Test Linear kernel
    print("\n" + "=" * 60)
    print("LINEAR KERNEL")
    print("=" * 60)
    print(f"{'C':>8s}  {'Mean F1':>8s}  {'Std':>6s}  {'Time':>8s}")
    print("-" * 40)

    for c_val in args.c_values:
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("svm", LinearSVC(
                C=c_val,
                class_weight="balanced",
                max_iter=5000,
                random_state=args.seed
            )),
        ])

        start = time.time()
        scores = cross_val_score(
            pipeline, X_subset, y_subset,
            cv=cv, scoring="f1_macro", n_jobs=n_jobs
        )
        elapsed = time.time() - start

        mean_f1 = scores.mean()
        std_f1 = scores.std()

        print(f"{c_val:8.1f}  {mean_f1:8.4f}  {std_f1:6.4f}  {elapsed:7.1f}s", flush=True)

        all_results["linear"].append({
            "C": c_val,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "time_sec": elapsed,
        })

    # Test RBF kernel (gamma='scale' is sklearn default)
    print("\n" + "=" * 60)
    print("RBF KERNEL (gamma='scale')")
    print("=" * 60)
    print(f"{'C':>8s}  {'Mean F1':>8s}  {'Std':>6s}  {'Time':>8s}")
    print("-" * 40)

    for c_val in args.c_values:
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("svm", SVC(
                C=c_val,
                kernel="rbf",
                gamma="scale",  # 1 / (n_features * X.var())
                class_weight="balanced",
                max_iter=5000,
                random_state=args.seed
            )),
        ])

        start = time.time()
        scores = cross_val_score(
            pipeline, X_subset, y_subset,
            cv=cv, scoring="f1_macro", n_jobs=n_jobs
        )
        elapsed = time.time() - start

        mean_f1 = scores.mean()
        std_f1 = scores.std()

        print(f"{c_val:8.1f}  {mean_f1:8.4f}  {std_f1:6.4f}  {elapsed:7.1f}s", flush=True)

        all_results["rbf"].append({
            "C": c_val,
            "gamma": "scale",
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "time_sec": elapsed,
        })

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)

    best_linear = max(all_results["linear"], key=lambda x: x["mean_f1"])
    best_rbf = max(all_results["rbf"], key=lambda x: x["mean_f1"])

    print(f"\nBest Linear: C={best_linear['C']:.1f}")
    print(f"  F1: {best_linear['mean_f1']:.4f} ± {best_linear['std_f1']:.4f}")
    print(f"  Time: {best_linear['time_sec']:.1f}s")

    print(f"\nBest RBF: C={best_rbf['C']:.1f}, gamma='scale'")
    print(f"  F1: {best_rbf['mean_f1']:.4f} ± {best_rbf['std_f1']:.4f}")
    print(f"  Time: {best_rbf['time_sec']:.1f}s")

    # Calculate improvement
    improvement = best_rbf['mean_f1'] - best_linear['mean_f1']
    improvement_pct = 100 * improvement / best_linear['mean_f1']

    print(f"\n{'=' * 60}")
    if improvement > 0.01:  # >1% improvement
        print(f"✅ RBF wins by {improvement:.4f} ({improvement_pct:.2f}%)")
        print(f"\nRBF can capture non-linear patterns that linear misses!")
        print(f"→ Use RBF kernel for production")
    elif improvement < -0.01:  # Linear wins
        print(f"✅ Linear wins by {-improvement:.4f} ({-improvement_pct:.2f}%)")
        print(f"\nLinear is sufficient - data is already separable!")
        print(f"→ Stick with LinearSVC (faster, simpler)")
    else:  # Negligible difference
        print(f"⚖️  Tie: difference = {improvement:.4f} ({improvement_pct:.2f}%)")
        print(f"\nBoth kernels perform equally!")
        print(f"→ Use Linear (3-5x faster, lower memory)")
    print(f"{'=' * 60}")

    # Save results
    output_file = PROJECT_ROOT / f"kernel_comparison_{int(time.time())}.json"
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "subset_size": args.subset_size,
            "c_values": args.c_values,
            "seed": args.seed,
        },
        "results": all_results,
        "best_linear": best_linear,
        "best_rbf": best_rbf,
        "improvement": improvement,
    }
    output_file.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved to: {output_file.name}")

    # Check final memory
    swap_info = get_swap_info()
    print(f"\nFinal swap: {swap_info['used_mb']:.1f}MB ({swap_info['percent']:.1f}%)")


if __name__ == "__main__":
    main()
