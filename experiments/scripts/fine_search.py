#!/usr/bin/env python3
"""Phase 3: Fine-grained hyperparameter search.

Based on coarse search results, does finer-grained search in the best region:
- Tests sizes at 1k increments within best range
- Tests same (C, gamma) grid for precision

Output:
- experiments/results/fine_search.json

Usage:
    uv run python -m experiments.scripts.fine_search
    uv run python -m experiments.scripts.fine_search --size-range 15000 25000
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from experiments.scripts.utils import (
    DATA_DIR,
    RESULTS_DIR,
    LabeledExample,
    SearchResult,
    get_label_distribution,
    load_labeled_data,
    load_results,
    save_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_training_data() -> tuple[list[LabeledExample], list[LabeledExample], np.ndarray]:
    """Load training data and cached embeddings."""
    train_seed = load_labeled_data(DATA_DIR / "train_seed.jsonl")
    auto_labeled = load_labeled_data(DATA_DIR / "auto_labeled_90pct.jsonl")

    data = np.load(DATA_DIR / "embeddings_cache.npz")
    embeddings = data["embeddings"]

    return train_seed, auto_labeled, embeddings


def presort_auto_labeled_by_class(
    auto_labeled: list[LabeledExample],
) -> dict[str, list[int]]:
    """Pre-sort auto-labeled examples by class, then by confidence within each class."""
    from collections import defaultdict

    by_class: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for idx, ex in enumerate(auto_labeled):
        by_class[ex.label].append((idx, ex.confidence))

    sorted_by_class: dict[str, list[int]] = {}
    for label, items in by_class.items():
        items.sort(key=lambda x: x[1], reverse=True)
        sorted_by_class[label] = [idx for idx, _ in items]

    return sorted_by_class


def select_training_subset(
    train_seed: list[LabeledExample],
    embeddings: np.ndarray,
    target_size: int,
    n_seed: int,
    auto_sorted_by_class: dict[str, list[int]],
    minority_boost: float = 1.5,
) -> tuple[np.ndarray, list[str]]:
    """Select a training subset using stratified sampling with minority class boost."""
    from collections import Counter

    MINORITY_CLASSES = {"AGREE", "DECLINE", "DEFER"}

    if target_size <= n_seed:
        rng = np.random.default_rng(42)
        indices = rng.choice(n_seed, size=target_size, replace=False)
        X = embeddings[indices]  # noqa: N806
        y = [train_seed[i].label for i in indices]
        return X, y

    seed_y = [e.label for e in train_seed]
    seed_counts = Counter(seed_y)
    n_auto_needed = target_size - n_seed

    # Calculate target per class with minority boost
    total_seed = sum(seed_counts.values())
    raw_targets: dict[str, float] = {}
    for label, count in seed_counts.items():
        proportion = count / total_seed
        if label in MINORITY_CLASSES:
            raw_targets[label] = n_auto_needed * proportion * minority_boost
        else:
            raw_targets[label] = n_auto_needed * proportion

    # Normalize to fit within n_auto_needed
    total_raw = sum(raw_targets.values())
    target_per_class: dict[str, int] = {}
    for label, raw in raw_targets.items():
        available = len(auto_sorted_by_class.get(label, []))
        target_per_class[label] = min(int(raw * n_auto_needed / total_raw), available)

    # Sample from each class
    auto_indices: list[int] = []
    auto_labels: list[str] = []
    for label, n_take in target_per_class.items():
        available = auto_sorted_by_class.get(label, [])
        taken = available[:n_take]
        auto_indices.extend(n_seed + idx for idx in taken)
        auto_labels.extend([label] * len(taken))

    all_indices = list(range(n_seed)) + auto_indices
    X = embeddings[all_indices]  # noqa: N806
    y = seed_y + auto_labels

    return X, y


def find_best_range_from_coarse(coarse_results: dict) -> tuple[int, int, float, str]:
    """Analyze coarse results to find best size range.

    Returns:
        (min_size, max_size, best_C, best_gamma)
    """
    results = coarse_results.get("results", [])
    if not results:
        raise ValueError("No coarse results found")

    # Group by size and find best config per size
    by_size: dict[int, dict] = {}
    for r in results:
        size = r["size"]
        if size not in by_size or r["cv_mean"] > by_size[size]["cv_mean"]:
            by_size[size] = r

    # Find best overall
    best = max(results, key=lambda r: r["cv_mean"])
    best_size = best["size"]
    best_C = best["C"]  # noqa: N806
    best_gamma = best["gamma"]

    logger.info(
        "Best from coarse: size=%d, C=%.1f, gamma=%s, F1=%.3f",
        best_size,
        best_C,
        best_gamma,
        best["cv_mean"],
    )

    # Define range around best
    sizes = sorted(by_size.keys())
    best_idx = sizes.index(best_size) if best_size in sizes else len(sizes) // 2

    # Take +/- 1 step from best (with bounds)
    min_idx = max(0, best_idx - 1)
    max_idx = min(len(sizes) - 1, best_idx + 1)

    min_size = sizes[min_idx]
    max_size = sizes[max_idx]

    logger.info("Fine search range: [%d, %d]", min_size, max_size)

    return min_size, max_size, best_C, best_gamma


def run_fine_search(
    train_seed: list[LabeledExample],
    auto_labeled: list[LabeledExample],
    embeddings: np.ndarray,
    min_size: int,
    max_size: int,
    step: int,
    c_values: list[float],
    gamma_values: list[str],
    n_folds: int = 5,
    n_jobs: int = 1,
) -> list[SearchResult]:
    """Run fine-grained hyperparameter search with parallel grid search.

    Args:
        train_seed: Human-labeled training examples
        auto_labeled: Auto-labeled examples
        embeddings: Pre-computed embeddings
        min_size: Minimum dataset size
        max_size: Maximum dataset size
        step: Step size between sizes
        c_values: C parameter values to try
        gamma_values: Gamma parameter values to try
        n_folds: Number of CV folds
        n_jobs: Number of parallel jobs (-1 = all cores)

    Returns:
        List of SearchResults for all configurations
    """
    from experiments.scripts.utils import grid_search_cv

    results: list[SearchResult] = []

    max_possible = len(train_seed) + len(auto_labeled)
    sizes = list(range(min_size, min(max_size + 1, max_possible + 1), step))
    n_seed = len(train_seed)

    total_configs = len(c_values) * len(gamma_values)
    logger.info(
        "Running fine search: %d sizes × %d configs = %d experiments",
        len(sizes),
        total_configs,
        len(sizes) * total_configs,
    )

    # Pre-sort auto-labeled by class ONCE
    logger.info("Pre-sorting %d auto-labeled examples by class...", len(auto_labeled))
    auto_sorted_by_class = presort_auto_labeled_by_class(auto_labeled)
    for label, indices in auto_sorted_by_class.items():
        logger.info("  %s: %d available", label, len(indices))

    for size_idx, size in enumerate(sizes):
        logger.info("")
        logger.info("[%d/%d] Size: %d", size_idx + 1, len(sizes), size)

        X, y = select_training_subset(  # noqa: N806
            train_seed,
            embeddings,
            size,
            n_seed,
            auto_sorted_by_class,
        )
        logger.info("Distribution: %s", get_label_distribution(y))

        # Run parallel grid search
        grid_result = grid_search_cv(
            X,
            y,
            c_values=c_values,
            gamma_values=gamma_values,
            n_folds=n_folds,
            n_jobs=n_jobs,
        )

        logger.info(
            "Best: C=%.1f, gamma=%s, F1=%.3f",
            grid_result["best_params"]["C"],
            grid_result["best_params"]["gamma"],
            grid_result["best_score"],
        )

        # Convert to SearchResults
        cv_results = grid_result["cv_results"]
        for params, mean_score, std_score in zip(
            cv_results["params"],
            cv_results["mean_test_score"],
            cv_results["std_test_score"],
        ):
            results.append(
                SearchResult(
                    size=size,
                    C=params["C"],
                    gamma=params["gamma"],
                    cv_mean=mean_score,
                    cv_std=std_score,
                    per_class_f1={},
                )
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Fine hyperparameter search")
    parser.add_argument(
        "--size-range",
        type=int,
        nargs=2,
        default=None,
        help="Min and max sizes to search (default: auto from coarse)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1000,
        help="Step size between dataset sizes (default: 1000)",
    )
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=[1, 2, 5, 10, 20, 50],
        help="C parameter values to try",
    )
    parser.add_argument(
        "--gamma-values",
        type=str,
        nargs="+",
        default=["scale", "auto"],
        help="Gamma parameter values to try",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 = all cores, default: -1)",
    )
    args = parser.parse_args()

    # Load coarse results if size range not specified
    if args.size_range is None:
        coarse_path = RESULTS_DIR / "coarse_search.json"
        coarse_results = load_results(coarse_path)
        if coarse_results is None:
            logger.error("Coarse search results not found. Run coarse_search.py first.")
            logger.error("Or specify --size-range manually.")
            return

        min_size, max_size, _, _ = find_best_range_from_coarse(coarse_results)
    else:
        min_size, max_size = args.size_range

    # Load training data
    train_seed, auto_labeled, embeddings = load_training_data()

    # Run fine search
    results = run_fine_search(
        train_seed=train_seed,
        auto_labeled=auto_labeled,
        embeddings=embeddings,
        min_size=min_size,
        max_size=max_size,
        step=args.step,
        c_values=args.c_values,
        gamma_values=args.gamma_values,
        n_folds=args.n_folds,
        n_jobs=args.n_jobs,
    )

    # Sort by CV mean F1
    results.sort(key=lambda r: r.cv_mean, reverse=True)

    # Print top 10
    logger.info("")
    logger.info("=" * 70)
    logger.info("TOP 10 CONFIGURATIONS (Fine Search)")
    logger.info("=" * 70)
    logger.info(
        "%6s %8s %8s %8s %8s",
        "Size",
        "C",
        "Gamma",
        "CV Mean",
        "CV Std",
    )
    logger.info("-" * 50)
    for r in results[:10]:
        logger.info(
            "%6d %8.1f %8s %8.3f %8.3f",
            r.size,
            r.C,
            r.gamma,
            r.cv_mean,
            r.cv_std,
        )

    # Best overall
    best = results[0]
    logger.info("")
    logger.info("=" * 70)
    logger.info("BEST CONFIGURATION")
    logger.info("=" * 70)
    logger.info("Size: %d", best.size)
    logger.info("C: %.1f", best.C)
    logger.info("Gamma: %s", best.gamma)
    logger.info("CV F1: %.3f ± %.3f", best.cv_mean, best.cv_std)
    logger.info("")
    logger.info("Per-class F1:")
    for label, f1 in sorted(best.per_class_f1.items()):
        logger.info("  %s: %.3f", label, f1)

    # Save results
    output_path = RESULTS_DIR / "fine_search.json"

    output_data = {
        "config": {
            "min_size": min_size,
            "max_size": max_size,
            "step": args.step,
            "c_values": args.c_values,
            "gamma_values": args.gamma_values,
            "n_folds": args.n_folds,
        },
        "best": {
            "size": best.size,
            "C": best.C,
            "gamma": best.gamma,
            "cv_mean": best.cv_mean,
            "cv_std": best.cv_std,
            "per_class_f1": best.per_class_f1,
        },
        "results": [
            {
                "size": r.size,
                "C": r.C,
                "gamma": r.gamma,
                "cv_mean": r.cv_mean,
                "cv_std": r.cv_std,
                "per_class_f1": r.per_class_f1,
            }
            for r in results
        ],
    }

    save_results(output_data, output_path)
    logger.info("")
    logger.info("Results saved to %s", output_path)
    logger.info("")
    logger.info("Next: Run final_eval.py to evaluate on the held-out test set")


if __name__ == "__main__":
    main()
