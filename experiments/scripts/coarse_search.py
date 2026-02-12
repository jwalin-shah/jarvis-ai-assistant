#!/usr/bin/env python3
"""Phase 2: Coarse hyperparameter search.

Tests a grid of dataset sizes and hyperparameters using 5-fold cross-validation.
Only uses training data (train_seed + auto_labeled) - NEVER touches test set.

Grid:
- Sizes: [3k, 8k, 13k, 18k, 23k, 28k]
- C values: [1, 2, 5, 10, 20, 50]
- Gamma: ['scale', 'auto']

Output:
- experiments/results/coarse_search.json

Usage:
    uv run python -m experiments.scripts.coarse_search
    uv run python -m experiments.scripts.coarse_search --sizes 3000 8000 13000
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
    save_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_training_data() -> tuple[list[LabeledExample], list[LabeledExample], np.ndarray]:
    """Load training data and cached embeddings.

    Returns:
        (train_seed, auto_labeled, embeddings)
    """
    # Load labeled data
    train_seed = load_labeled_data(DATA_DIR / "train_seed.jsonl")
    auto_labeled = load_labeled_data(DATA_DIR / "auto_labeled_90pct.jsonl")

    logger.info("Loaded %d train_seed examples", len(train_seed))
    logger.info("Loaded %d auto_labeled examples", len(auto_labeled))

    # Load embeddings
    cache_path = DATA_DIR / "embeddings_cache.npz"
    if not cache_path.exists():
        logger.error("Embeddings not found. Run prepare_data.py first.")
        raise FileNotFoundError(cache_path)

    data = np.load(cache_path)
    embeddings = data["embeddings"]
    n_seed = int(data["n_seed"])
    n_auto = int(data["n_auto"])

    logger.info(
        "Loaded embeddings: shape=%s, n_seed=%d, n_auto=%d",
        embeddings.shape,
        n_seed,
        n_auto,
    )

    # Verify counts match
    if n_seed != len(train_seed) or n_auto != len(auto_labeled):
        logger.warning(
            "Embedding counts don't match! Expected seed=%d, auto=%d, got seed=%d, auto=%d",
            len(train_seed),
            len(auto_labeled),
            n_seed,
            n_auto,
        )

    return train_seed, auto_labeled, embeddings


def presort_auto_labeled_by_class(
    auto_labeled: list[LabeledExample],
) -> dict[str, list[int]]:
    """Pre-sort auto-labeled examples by class, then by confidence within each class.

    Call this ONCE before the search loop, then reuse the sorted indices.

    Returns:
        Dict mapping label -> list of indices sorted by confidence (highest first)
    """
    from collections import defaultdict

    by_class: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for idx, ex in enumerate(auto_labeled):
        by_class[ex.label].append((idx, ex.confidence))

    # Sort each class by confidence (highest first)
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
    """Select a training subset using stratified sampling with minority class boost.

    Maintains rough seed distribution but gives minority classes a boost.

    Args:
        train_seed: Human-labeled training examples
        embeddings: Pre-computed embeddings (train_seed first, then auto_labeled)
        target_size: Target number of samples
        n_seed: Number of seed examples
        auto_sorted_by_class: Dict mapping label -> sorted indices (by confidence)
        minority_boost: Multiplier for minority classes (AGREE, DECLINE, DEFER)

    Returns:
        (X, y) - embeddings and labels for the subset
    """
    from collections import Counter

    MINORITY_CLASSES = {"AGREE", "DECLINE", "DEFER"}

    # If target is smaller than seed, sample from seed
    if target_size <= n_seed:
        rng = np.random.default_rng(42)
        indices = rng.choice(n_seed, size=target_size, replace=False)
        X = embeddings[indices]  # noqa: N806
        y = [train_seed[i].label for i in indices]
        return X, y

    # Include all seed examples
    seed_y = [e.label for e in train_seed]
    seed_counts = Counter(seed_y)

    # Calculate how many auto-labeled we need
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

    # Normalize to fit within n_auto_needed (boosting minorities reduces majorities)
    total_raw = sum(raw_targets.values())
    target_per_class: dict[str, int] = {}
    for label, raw in raw_targets.items():
        # Cap by what's available in auto-labeled
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

    # Use efficient numpy indexing
    all_indices = list(range(n_seed)) + auto_indices
    X = embeddings[all_indices]  # noqa: N806
    y = seed_y + auto_labels

    return X, y


def run_coarse_search(
    train_seed: list[LabeledExample],
    auto_labeled: list[LabeledExample],
    embeddings: np.ndarray,
    sizes: list[int],
    c_values: list[float],
    gamma_values: list[str],
    n_folds: int = 5,
    n_jobs: int = 1,
) -> list[SearchResult]:
    """Run coarse hyperparameter search with parallel grid search.

    Uses sklearn's GridSearchCV for each dataset size, which parallelizes
    across both parameter combinations AND folds for much faster execution.

    Args:
        train_seed: Human-labeled training examples
        auto_labeled: Auto-labeled examples
        embeddings: Pre-computed embeddings
        sizes: Dataset sizes to try
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
    sizes = [s for s in sizes if s <= max_possible]
    n_seed = len(train_seed)

    total_configs = len(c_values) * len(gamma_values)
    logger.info(
        "Running grid search: %d sizes × %d configs = %d total experiments",
        len(sizes),
        total_configs,
        len(sizes) * total_configs,
    )
    logger.info("Using %s parallel jobs", n_jobs if n_jobs > 0 else "all cores")

    # Pre-sort auto-labeled by class ONCE (not every iteration!)
    logger.info(
        "Pre-sorting %d auto-labeled examples by class and confidence...", len(auto_labeled)
    )
    auto_sorted_by_class = presort_auto_labeled_by_class(auto_labeled)
    for label, indices in auto_sorted_by_class.items():
        logger.info("  %s: %d available", label, len(indices))

    for size_idx, size in enumerate(sizes):
        logger.info("")
        logger.info("=" * 50)
        logger.info("[%d/%d] Dataset size: %d", size_idx + 1, len(sizes), size)
        logger.info("=" * 50)

        # Get subset using stratified sampling (maintains seed distribution)
        X, y = select_training_subset(  # noqa: N806
            train_seed,
            embeddings,
            size,
            n_seed,
            auto_sorted_by_class,
        )
        logger.info("Distribution: %s", get_label_distribution(y))

        # Run parallel grid search for this size
        logger.info("Running GridSearchCV with %d configs...", total_configs)
        grid_result = grid_search_cv(
            X,
            y,
            c_values=c_values,
            gamma_values=gamma_values,
            n_folds=n_folds,
            n_jobs=n_jobs,
        )

        logger.info(
            "Best for size %d: C=%.1f, gamma=%s, F1=%.3f",
            size,
            grid_result["best_params"]["C"],
            grid_result["best_params"]["gamma"],
            grid_result["best_score"],
        )

        # Convert grid results to SearchResults
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
                    per_class_f1={},  # GridSearchCV doesn't give per-class easily
                )
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Coarse hyperparameter search")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[3000, 5000, 7000, 9000, 11000, 13000],
        help="Dataset sizes to try (max effective ~13k with stratified sampling)",
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

    # Load data
    train_seed, auto_labeled, embeddings = load_training_data()

    # Run search
    results = run_coarse_search(
        train_seed=train_seed,
        auto_labeled=auto_labeled,
        embeddings=embeddings,
        sizes=args.sizes,
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
    logger.info("TOP 10 CONFIGURATIONS")
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

    # Analyze best size range
    logger.info("")
    logger.info("=" * 70)
    logger.info("BEST SIZE ANALYSIS")
    logger.info("=" * 70)

    by_size: dict[int, list[SearchResult]] = {}
    for r in results:
        if r.size not in by_size:
            by_size[r.size] = []
        by_size[r.size].append(r)

    logger.info("%6s %12s %12s %12s", "Size", "Best CV F1", "Avg CV F1", "Best (C, γ)")
    logger.info("-" * 50)
    for size in sorted(by_size.keys()):
        size_results = by_size[size]
        best = max(size_results, key=lambda r: r.cv_mean)
        avg = sum(r.cv_mean for r in size_results) / len(size_results)
        logger.info(
            "%6d %12.3f %12.3f %12s",
            size,
            best.cv_mean,
            avg,
            f"({best.C}, {best.gamma})",
        )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "coarse_search.json"

    # Format results for JSON
    output_data = {
        "config": {
            "sizes": args.sizes,
            "c_values": args.c_values,
            "gamma_values": args.gamma_values,
            "n_folds": args.n_folds,
            "train_seed_count": len(train_seed),
            "auto_labeled_count": len(auto_labeled),
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
        "top_10": [
            {"size": r.size, "C": r.C, "gamma": r.gamma, "cv_mean": r.cv_mean} for r in results[:10]
        ],
    }

    save_results(output_data, output_path)
    logger.info("")
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
