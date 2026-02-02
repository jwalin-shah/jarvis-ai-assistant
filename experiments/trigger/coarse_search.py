#!/usr/bin/env python3
"""Coarse grid search for trigger classifier.

Searches over:
- Dataset size (using auto-labeled data)
- SVM hyperparameters (C, gamma)

Usage:
    uv run python -m experiments.trigger.coarse_search
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

# Trigger labels
LABELS = ["commitment", "question", "reaction", "social", "statement"]

# Hyperparameter grid
C_VALUES = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
GAMMA_VALUES = ["scale", "auto"]

# Dataset sizes to try (will be capped at available data)
DATASET_SIZES = [3000, 4000, 5000, 6000, 7000, 7865]


def load_data():
    """Load train seed and auto-labeled data with embeddings."""
    # Load examples
    train_seed = []
    with open(DATA_DIR / "train_seed.jsonl") as f:
        for line in f:
            if line.strip():
                train_seed.append(json.loads(line))

    auto_labeled = []
    with open(DATA_DIR / "auto_labeled.jsonl") as f:
        for line in f:
            if line.strip():
                auto_labeled.append(json.loads(line))

    # Load embeddings
    cache = np.load(DATA_DIR / "embeddings_cache.npz")
    embeddings = cache["embeddings"]
    n_seed = int(cache["n_seed"])
    n_auto = int(cache["n_auto"])

    return train_seed, auto_labeled, embeddings, n_seed, n_auto


def sample_dataset(
    train_seed: list[dict],
    auto_labeled: list[dict],
    embeddings: np.ndarray,
    n_seed: int,
    target_size: int,
) -> tuple[np.ndarray, list[str]]:
    """Sample a dataset of target_size, preserving class distribution.

    Always includes all seed data, samples from auto-labeled to reach target.
    """
    # Start with all seed data
    seed_embeddings = embeddings[:n_seed]
    seed_labels = [e["label"] for e in train_seed]

    if target_size <= n_seed:
        # Just use seed data (shouldn't happen with our sizes)
        return seed_embeddings[:target_size], seed_labels[:target_size]

    # Need to sample from auto-labeled
    n_auto_needed = target_size - n_seed

    # Sort auto-labeled by class and confidence for stratified sampling
    auto_by_class: dict[str, list[tuple[int, float]]] = {label: [] for label in LABELS}
    for i, ex in enumerate(auto_labeled):
        label = ex["label"]
        conf = ex.get("confidence", 1.0)
        auto_by_class[label].append((i, conf))

    # Sort each class by confidence (highest first)
    for label in LABELS:
        auto_by_class[label].sort(key=lambda x: -x[1])

    # Sample proportionally to class distribution in seed
    seed_dist = Counter(seed_labels)
    total_seed = sum(seed_dist.values())

    selected_auto_indices = []
    for label in LABELS:
        # Target proportion for this class
        proportion = seed_dist.get(label, 0) / total_seed
        n_needed = int(n_auto_needed * proportion)

        # Take top-confidence examples
        available = auto_by_class[label][:n_needed]
        selected_auto_indices.extend([idx for idx, _ in available])

    # If we didn't get enough, fill from remaining
    while len(selected_auto_indices) < n_auto_needed:
        for label in LABELS:
            available = auto_by_class[label]
            for idx, _ in available:
                if idx not in selected_auto_indices:
                    selected_auto_indices.append(idx)
                    if len(selected_auto_indices) >= n_auto_needed:
                        break
            if len(selected_auto_indices) >= n_auto_needed:
                break

    # Combine seed + selected auto
    auto_embeddings = embeddings[n_seed:][selected_auto_indices]
    auto_labels = [auto_labeled[i]["label"] for i in selected_auto_indices]

    X = np.vstack([seed_embeddings, auto_embeddings])
    y = seed_labels + auto_labels

    return X, y


def run_grid_search(X: np.ndarray, y: list[str], n_folds: int = 5) -> list[dict]:
    """Run grid search over C and gamma values."""
    results = []

    for C in C_VALUES:
        for gamma in GAMMA_VALUES:
            svm = SVC(
                C=C,
                kernel="rbf",
                gamma=gamma,
                class_weight="balanced",
                probability=True,
                random_state=42,
            )

            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            scores = cross_val_score(svm, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)

            results.append(
                {
                    "C": C,
                    "gamma": gamma,
                    "cv_mean": float(np.mean(scores)),
                    "cv_std": float(np.std(scores)),
                }
            )

    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    train_seed, auto_labeled, embeddings, n_seed, n_auto = load_data()
    logger.info("Loaded: n_seed=%d, n_auto=%d", n_seed, n_auto)

    max_available = n_seed + n_auto
    logger.info("Maximum available: %d", max_available)

    all_results = []

    for size in DATASET_SIZES:
        if size > max_available:
            size = max_available

        logger.info("")
        logger.info("=" * 50)
        logger.info("Dataset size: %d", size)
        logger.info("=" * 50)

        X, y = sample_dataset(train_seed, auto_labeled, embeddings, n_seed, size)
        logger.info("Sampled: X=%s, distribution=%s", X.shape, dict(Counter(y)))

        results = run_grid_search(X, y)

        best = max(results, key=lambda r: r["cv_mean"])
        logger.info(
            "Best: C=%.1f, gamma=%s, F1=%.3f ± %.3f",
            best["C"],
            best["gamma"],
            best["cv_mean"],
            best["cv_std"],
        )

        for r in results:
            r["size"] = size
            all_results.append(r)

    # Sort by CV mean
    all_results.sort(key=lambda r: -r["cv_mean"])

    logger.info("")
    logger.info("=" * 70)
    logger.info("TOP 10 CONFIGURATIONS")
    logger.info("=" * 70)
    logger.info("  Size        C    Gamma  CV Mean   CV Std")
    logger.info("-" * 50)
    for r in all_results[:10]:
        logger.info(
            "  %5d  %6.1f  %7s    %.3f    %.3f",
            r["size"],
            r["C"],
            r["gamma"],
            r["cv_mean"],
            r["cv_std"],
        )

    # Save results
    output_path = RESULTS_DIR / "coarse_search.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("")
    logger.info("Saved results to %s", output_path)


if __name__ == "__main__":
    main()
