#!/usr/bin/env python3
"""Systematic experiment sweep for DailyDialog dialog act classifier.

Searches over 144 configurations:
- Category configs: 4-class, 3-class (merge directive+commissive → action)
- Feature sets: embeddings-only, hand-crafted-only, combined
- Class balancing: natural, balanced, moderate (max 2x imbalance)
- SVM hyperparameters: C ∈ {0.1, 1.0, 10.0, 50.0}, class_weight ∈ {balanced, None}

Memory strategy (8GB RAM constraint):
- Load data once, reuse across all experiments
- GridSearchCV with n_jobs=1 (avoid worker overhead)
- Peak memory per experiment: ~400MB

Output: experiments/results/dailydialog_sweep.json

Usage:
    uv run python experiments/scripts/dailydialog_sweep.py
    uv run python experiments/scripts/dailydialog_sweep.py --quick  # 3-class combined only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "dailydialog_native"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

# Experiment dimensions
CATEGORY_CONFIGS = ["4class", "3class"]  # 3class merges directive+commissive
FEATURE_SETS = ["embeddings", "handcrafted", "combined"]
BALANCING_STRATEGIES = ["natural", "balanced", "moderate"]  # moderate = max 2x
C_VALUES = [0.1, 1.0, 10.0, 50.0]
CLASS_WEIGHTS = ["balanced", None]

# Label mapping for 3-class
LABEL_MAP_3CLASS = {
    "inform": "inform",
    "question": "question",
    "directive": "action",
    "commissive": "action",
}


def load_data_once() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load data once, return embeddings, hand-crafted, labels, metadata.

    Returns views (not copies) to minimize memory usage.
    """
    print("Loading data from", DATA_DIR)
    train_data = np.load(DATA_DIR / "train.npz", allow_pickle=True)
    test_data = np.load(DATA_DIR / "test.npz", allow_pickle=True)
    metadata = json.loads((DATA_DIR / "metadata.json").read_text())

    X_train, y_train = train_data["X"], train_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    embedding_dims = metadata["embedding_dims"]
    hand_crafted_dims = metadata["hand_crafted_dims"]

    # Split into embeddings and hand-crafted features
    X_train_emb = X_train[:, :embedding_dims]
    X_train_hc = X_train[:, embedding_dims:]
    X_test_emb = X_test[:, :embedding_dims]
    X_test_hc = X_test[:, embedding_dims:]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Embeddings: {embedding_dims}d, Hand-crafted: {hand_crafted_dims}d")
    print(f"Labels: {sorted(set(y_train))}")

    return (X_train_emb, X_train_hc, y_train, X_test_emb, X_test_hc, y_test, metadata)


def apply_label_mapping(
    y: np.ndarray,
    config: str,
) -> np.ndarray:
    """Apply label mapping for category config."""
    if config == "3class":
        return np.array([LABEL_MAP_3CLASS[label] for label in y])
    return y


def apply_balancing(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply class balancing strategy.

    Args:
        X: Feature matrix (N, D)
        y: Labels (N,)
        strategy: "natural", "balanced", or "moderate"
        seed: Random seed

    Returns:
        Balanced X, y
    """
    if strategy == "natural":
        return X, y

    counts = Counter(y)
    min_count = min(counts.values())

    if strategy == "balanced":
        max_per_class = min_count
    elif strategy == "moderate":
        max_per_class = min_count * 2
    else:
        raise ValueError(f"Unknown balancing strategy: {strategy}")

    rng = np.random.default_rng(seed)
    indices = []

    for label in sorted(set(y)):
        label_indices = np.where(y == label)[0]
        if len(label_indices) > max_per_class:
            sampled = rng.choice(label_indices, max_per_class, replace=False)
            indices.extend(sampled)
        else:
            indices.extend(label_indices)

    indices = np.array(indices)
    rng.shuffle(indices)

    return X[indices], y[indices]


def select_features(
    X_emb: np.ndarray,
    X_hc: np.ndarray,
    feature_set: str,
) -> np.ndarray:
    """Select feature subset."""
    if feature_set == "embeddings":
        return X_emb
    elif feature_set == "handcrafted":
        return X_hc
    elif feature_set == "combined":
        return np.hstack([X_emb, X_hc])
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")


def run_single_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C: float,
    class_weight: str | None,
    seed: int = 42,
) -> dict:
    """Run a single SVM experiment with cross-validation.

    Returns dict with CV and test metrics.
    """
    svm = LinearSVC(
        C=C,
        class_weight=class_weight,
        max_iter=5000,
        random_state=seed,
    )

    # 5-fold cross-validation (n_jobs=1 for 8GB RAM constraint)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(
        svm,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,
    )

    # Train on full training set and evaluate on test
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average="macro")

    # Per-class F1 on test set
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    labels = sorted(set(y_test))
    per_class_f1 = {label: report[label]["f1-score"] for label in labels if label in report}

    return {
        "cv_mean_f1": float(np.mean(cv_scores)),
        "cv_std_f1": float(np.std(cv_scores)),
        "test_f1": float(test_f1),
        "per_class_f1": per_class_f1,
    }


def run_sweep(quick: bool = False) -> list[dict]:
    """Run full experiment sweep.

    Returns list of experiment results.
    """
    # Load data once
    (X_train_emb, X_train_hc, y_train_raw, X_test_emb, X_test_hc, y_test_raw, metadata) = (
        load_data_once()
    )

    # Generate all configurations
    experiments = []

    category_configs = ["3class"] if quick else CATEGORY_CONFIGS
    feature_sets = ["combined"] if quick else FEATURE_SETS
    balancing_strategies = ["moderate"] if quick else BALANCING_STRATEGIES

    for cat_config in category_configs:
        for feat_set in feature_sets:
            for balance_strat in balancing_strategies:
                for C in C_VALUES:
                    for class_weight in CLASS_WEIGHTS:
                        experiments.append(
                            {
                                "category_config": cat_config,
                                "feature_set": feat_set,
                                "balancing": balance_strat,
                                "C": C,
                                "class_weight": class_weight,
                            }
                        )

    total = len(experiments)
    logger.info("Generated %d experiment configurations", total)
    logger.info("")

    results = []
    start_time = time.perf_counter()

    for i, config in enumerate(experiments):
        exp_start = time.perf_counter()

        # Apply label mapping
        y_train = apply_label_mapping(y_train_raw, config["category_config"])
        y_test = apply_label_mapping(y_test_raw, config["category_config"])

        # Select features
        X_train_features = select_features(X_train_emb, X_train_hc, config["feature_set"])
        X_test_features = select_features(X_test_emb, X_test_hc, config["feature_set"])

        # Apply balancing
        X_train_balanced, y_train_balanced = apply_balancing(
            X_train_features, y_train, config["balancing"]
        )

        # Run experiment
        metrics = run_single_experiment(
            X_train_balanced,
            y_train_balanced,
            X_test_features,
            y_test,
            C=config["C"],
            class_weight=config["class_weight"],
        )

        # Merge config + metrics
        result = {**config, **metrics}

        # Add dataset info
        result["train_size"] = len(X_train_balanced)
        result["test_size"] = len(X_test_features)
        result["train_distribution"] = dict(Counter(y_train_balanced))

        results.append(result)

        exp_elapsed = time.perf_counter() - exp_start
        total_elapsed = time.perf_counter() - start_time
        avg_time = total_elapsed / (i + 1)
        remaining = avg_time * (total - i - 1)

        logger.info(
            "[%3d/%3d] %s | CV F1: %.3f ± %.3f | Test F1: %.3f | %.1fs (ETA: %.0fm)",
            i + 1,
            total,
            _config_str(config),
            metrics["cv_mean_f1"],
            metrics["cv_std_f1"],
            metrics["test_f1"],
            exp_elapsed,
            remaining / 60,
        )

        # Save intermediate results every 10 experiments
        if (i + 1) % 10 == 0:
            _save_results(results, intermediate=True)

    # Save final results
    _save_results(results, intermediate=False)

    total_elapsed = time.perf_counter() - start_time
    logger.info("")
    logger.info("Sweep complete! Total time: %.1f minutes", total_elapsed / 60)

    return results


def _config_str(config: dict) -> str:
    """Format config as compact string."""
    return (
        f"{config['category_config']:6s} | "
        f"{config['feature_set']:10s} | "
        f"{config['balancing']:8s} | "
        f"C={config['C']:5.1f} | "
        f"wt={str(config['class_weight']):8s}"
    )


def _save_results(results: list[dict], intermediate: bool = False) -> None:
    """Save results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Sort by CV mean F1
    sorted_results = sorted(results, key=lambda r: -r["cv_mean_f1"])

    output = {
        "total_experiments": len(results),
        "top_10": sorted_results[:10],
        "experiments": sorted_results,
    }

    filename = "dailydialog_sweep_partial.json" if intermediate else "dailydialog_sweep.json"
    output_path = RESULTS_DIR / filename

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    if not intermediate:
        logger.info("Saved final results to %s", output_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run systematic experiment sweep for DailyDialog classifier"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 3-class combined features only (32 experiments)",
    )
    args = parser.parse_args()

    results = run_sweep(quick=args.quick)

    # Print top 10
    logger.info("")
    logger.info("=" * 120)
    logger.info("TOP 10 CONFIGURATIONS")
    logger.info("=" * 120)
    logger.info(
        "%-6s  %-10s  %-8s  %6s  %-8s  %8s  %8s  %8s",
        "Cat",
        "Features",
        "Balance",
        "C",
        "Weight",
        "CV F1",
        "CV Std",
        "Test F1",
    )
    logger.info("-" * 120)

    for r in results[:10]:
        logger.info(
            "%-6s  %-10s  %-8s  %6.1f  %-8s  %8.3f  %8.3f  %8.3f",
            r["category_config"],
            r["feature_set"],
            r["balancing"],
            r["C"],
            str(r["class_weight"]),
            r["cv_mean_f1"],
            r["cv_std_f1"],
            r["test_f1"],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
