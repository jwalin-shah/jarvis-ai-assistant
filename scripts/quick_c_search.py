#!/usr/bin/env python3
"""Quick C value search on a subset of data for fast iteration.

Tests a range of C values on a smaller dataset to find the ballpark optimal C,
then you can run the full dataset with a narrower range.

Usage:
    uv run python scripts/quick_c_search.py --subset-size 10000 --c-values 10 20 50 100
    uv run python scripts/quick_c_search.py --subset-size 20000 --oversample-directive 2.0
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
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.utils.memory import get_swap_info, get_memory_info

import psutil


def main():
    parser = argparse.ArgumentParser(description="Quick C value search on subset")
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
        default=[10.0, 20.0, 50.0, 100.0],
        help="C values to test (default: 10 20 50 100)",
    )
    parser.add_argument(
        "--oversample-directive",
        type=float,
        default=1.0,
        help="Multiply directive samples by this factor (default: 1.0, no oversampling)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel jobs (default: auto-detect based on dataset size)",
    )

    args = parser.parse_args()

    print(f"Quick C value search")
    print(f"Subset size: {args.subset_size:,} samples")
    print(f"C values to test: {args.c_values}")
    print(f"Directive oversample factor: {args.oversample_directive}x")

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

    # Optionally oversample directive
    if args.oversample_directive > 1.0:
        print(f"\nOversampling directive by {args.oversample_directive}x...")
        directive_mask = y_subset == "directive"
        directive_X = X_subset[directive_mask]
        directive_y = y_subset[directive_mask]

        n_directive_original = len(directive_y)
        n_to_add = int(n_directive_original * (args.oversample_directive - 1.0))

        # Randomly sample with replacement
        indices = np.random.choice(len(directive_y), size=n_to_add, replace=True)
        X_subset = np.vstack([X_subset, directive_X[indices]])
        y_subset = np.concatenate([y_subset, directive_y[indices]])

        print(f"Added {n_to_add} directive samples ({n_directive_original} → {len(directive_y) + n_to_add})")

    # Show class distribution
    print(f"\nSubset class distribution:")
    for label in sorted(set(y_subset)):
        count = (y_subset == label).sum()
        pct = 100 * count / len(y_subset)
        print(f"  {label:12s}: {count:5d} ({pct:5.1f}%)")

    # Build pipeline
    embedding_cols = list(range(embedding_dims))
    hc_cols = list(range(embedding_dims, embedding_dims + hand_crafted_dims))

    preprocessor = ColumnTransformer(
        transformers=[
            ("embeddings", "passthrough", embedding_cols),
            ("hand_crafted", StandardScaler(), hc_cols),
        ],
    )

    # Smart adaptive n_jobs calculation based on available memory
    if args.n_jobs is not None:
        n_jobs = args.n_jobs
        print(f"Using n_jobs={n_jobs} (manual override)")
    else:
        mem_info = get_memory_info()

        # DEBUG: Print what we're actually reading
        if mem_info.macos_pressure:
            print(f"[DEBUG] Memory pressure object: {mem_info.macos_pressure}")
            print(f"[DEBUG] Pressure level: {mem_info.macos_pressure.pressure_level}")
        else:
            print("[DEBUG] No macOS pressure info available")

        # Check if already under memory pressure - if so, stay conservative
        if mem_info.macos_pressure and mem_info.macos_pressure.pressure_level > 0:
            n_jobs = 1
            print(f"Using n_jobs=1 (memory pressure: {mem_info.macos_pressure.pressure_level})")
        else:
            # Calculate available memory
            vm = psutil.virtual_memory()
            total_ram_gb = vm.total / 1024**3
            used_ram_gb = vm.used / 1024**3

            # Reserve for OS and main process
            reserved_gb = 1.5  # OS (1 GB) + main process (0.5 GB)
            available_gb = max(0, total_ram_gb - used_ram_gb - reserved_gb)

            # Estimate memory per worker based on dataset size
            # Each worker needs: data copy + CV split + optimizer buffers + Python overhead
            dataset_size_mb = (len(X_subset) * X_subset.shape[1] * 8) / 1024**2
            memory_per_worker_mb = (
                dataset_size_mb * 1.5  # Data + CV split overhead (80% of data per fold)
                + 100  # Base optimizer buffers
                + (dataset_size_mb / 100)  # Scale with dataset (more data = more support vectors)
                + 50  # Python worker process overhead
            )

            # Safety margin: overestimate by 50% to be conservative
            memory_per_worker_mb *= 1.5

            # Calculate max safe n_jobs
            available_mb = available_gb * 1024
            max_jobs = int(available_mb / memory_per_worker_mb)
            max_jobs = max(1, min(max_jobs, 4))  # Clamp between 1-4 (diminishing returns after 4)

            n_jobs = max_jobs
            print(
                f"Auto-detected n_jobs={n_jobs} "
                f"(available: {available_gb:.1f}GB, ~{memory_per_worker_mb:.0f}MB/worker, "
                f"pressure: {mem_info.macos_pressure.pressure_level if mem_info.macos_pressure else 'N/A'})"
            )

    # Test each C value with cross-validation
    print(f"\nTesting {len(args.c_values)} C values with 5-fold CV...")
    print(f"{'C':>8s}  {'Mean F1':>8s}  {'Std':>6s}  {'Time':>8s}")
    print("-" * 40)

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    for c_val in args.c_values:
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("svm", LinearSVC(C=c_val, class_weight="balanced", max_iter=5000, random_state=args.seed)),
        ])

        start = time.time()
        scores = cross_val_score(pipeline, X_subset, y_subset, cv=cv, scoring="f1_macro", n_jobs=n_jobs)
        elapsed = time.time() - start

        mean_f1 = scores.mean()
        std_f1 = scores.std()

        print(f"{c_val:8.1f}  {mean_f1:8.4f}  {std_f1:6.4f}  {elapsed:7.1f}s", flush=True)

        results.append({
            "C": c_val,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "time_sec": elapsed,
        })

    # Find best C
    best_result = max(results, key=lambda x: x["mean_f1"])
    print(f"\n{'=' * 40}")
    print(f"Best C: {best_result['C']}")
    print(f"Best mean F1: {best_result['mean_f1']:.4f} (+/- {best_result['std_f1']:.4f})")
    print(f"{'=' * 40}")

    # Save results to JSON
    output_file = PROJECT_ROOT / f"c_search_results_{int(time.time())}.json"
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "subset_size": args.subset_size,
            "c_values": args.c_values,
            "oversample_directive": args.oversample_directive,
            "seed": args.seed,
        },
        "class_distribution": {
            label: int((y_subset == label).sum())
            for label in sorted(set(y_subset))
        },
        "results": results,
        "best_C": best_result["C"],
        "best_mean_f1": best_result["mean_f1"],
    }
    output_file.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved to: {output_file.name}")

    # Recommendations
    print(f"\nRecommendations:")
    if best_result["C"] == max(args.c_values):
        print(f"  ⚠️  Best C is at the upper bound ({best_result['C']})")
        higher_vals = [int(c * 2) for c in args.c_values[-2:]]
        print(f"  → Test higher values: {higher_vals}")
        print(f"  → Run: python scripts/quick_c_search.py --c-values {' '.join(map(str, higher_vals))}")
    elif best_result["C"] == min(args.c_values):
        print(f"  ⚠️  Best C is at the lower bound ({best_result['C']})")
        lower_vals = [c / 2 for c in args.c_values[:2]]
        print(f"  → Test lower values: {lower_vals}")
        print(f"  → Run: python scripts/quick_c_search.py --c-values {' '.join(map(str, lower_vals))}")
    else:
        # Find neighbors - check if there's a big gap
        c_idx = [r["C"] for r in results].index(best_result["C"])

        # Check gap with next value
        if c_idx < len(results) - 1:
            next_c = results[c_idx + 1]["C"]
            gap_ratio = next_c / best_result["C"]

            if gap_ratio > 1.5:  # Gap > 50%
                # Suggest filling the gap
                mid_point = int((best_result["C"] + next_c) / 2)
                refined_range = [
                    int(best_result["C"]),
                    mid_point,
                    int(next_c),
                ]
                print(f"  ⚠️  Large gap between C={best_result['C']} and C={next_c} (ratio: {gap_ratio:.1f}x)")
                print(f"  → Refine search in this range: {refined_range}")
                print(f"  → Run: python scripts/quick_c_search.py --c-values {' '.join(map(str, refined_range))}")
            else:
                neighbors = []
                if c_idx > 0:
                    neighbors.append(results[c_idx - 1]["C"])
                neighbors.append(best_result["C"])
                neighbors.append(results[c_idx + 1]["C"])

                print(f"  ✓ Best C is in the middle of tested range")
                print(f"  → Run full dataset with C={neighbors} for final tuning")
        else:
            print(f"  ✓ Best C found")
            print(f"  → Run full dataset with C={best_result['C']}")

    # Check final memory
    swap_info = get_swap_info()
    print(f"\nFinal swap: {swap_info['used_mb']:.1f}MB ({swap_info['percent']:.1f}%)")


if __name__ == "__main__":
    main()
