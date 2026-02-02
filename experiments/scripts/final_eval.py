#!/usr/bin/env python3
"""Phase 4: Final evaluation on held-out test set.

This is the ONLY script that touches test_human.jsonl.
It trains the final model with the best config and evaluates on the test set.

Output:
- experiments/models/response_v2/svm.pkl - Trained SVM
- experiments/models/response_v2/config.json - Model config
- experiments/results/final_evaluation.json - Test results

Usage:
    uv run python -m experiments.scripts.final_eval
    uv run python -m experiments.scripts.final_eval --size 18000 --C 10 --gamma scale
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from experiments.scripts.utils import (
    DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    LabeledExample,
    get_label_distribution,
    load_labeled_data,
    load_results,
    save_results,
    train_svm,
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


def load_test_data() -> tuple[list[LabeledExample], np.ndarray]:
    """Load held-out test data and embeddings."""
    test_examples = load_labeled_data(DATA_DIR / "test_human.jsonl")

    data = np.load(DATA_DIR / "test_embeddings.npz")
    embeddings = data["embeddings"]

    return test_examples, embeddings


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


def get_best_config_from_fine_search() -> dict:
    """Load best config from fine search results."""
    fine_path = RESULTS_DIR / "fine_search.json"
    fine_results = load_results(fine_path)

    if fine_results is None:
        # Fall back to coarse search
        coarse_path = RESULTS_DIR / "coarse_search.json"
        coarse_results = load_results(coarse_path)
        if coarse_results is None:
            raise FileNotFoundError("No search results found. Run coarse_search.py first.")
        best = coarse_results["top_10"][0]
    else:
        best = fine_results["best"]

    return best


def compute_confidence_interval(
    y_true: list[str],
    y_pred: list[str],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for macro F1.

    Optimized to use numpy arrays and pre-generate all indices.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95)

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(42)
    n = len(y_true)

    # Convert to numpy arrays for efficient indexing
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # Pre-generate ALL bootstrap indices at once (much faster)
    all_indices = rng.choice(n, size=(n_bootstrap, n), replace=True)

    # Compute F1 for each bootstrap sample
    bootstrap_f1s = [
        f1_score(
            y_true_arr[all_indices[i]], y_pred_arr[all_indices[i]], average="macro", zero_division=0
        )
        for i in range(n_bootstrap)
    ]

    point = f1_score(y_true, y_pred, average="macro", zero_division=0)
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_f1s, alpha * 100)
    upper = np.percentile(bootstrap_f1s, (1 - alpha) * 100)

    return float(point), float(lower), float(upper)


def main():
    parser = argparse.ArgumentParser(description="Final evaluation on test set")
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Training size (default: from fine_search results)",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=None,
        help="SVM C parameter (default: from fine_search results)",
    )
    parser.add_argument(
        "--gamma",
        type=str,
        default=None,
        help="SVM gamma parameter (default: from fine_search results)",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="Save the trained model (default: True)",
    )
    parser.add_argument(
        "--no-save-model",
        action="store_false",
        dest="save_model",
        help="Don't save the trained model",
    )
    args = parser.parse_args()

    # Get best config
    if args.size is None or args.C is None or args.gamma is None:
        logger.info("Loading best config from search results...")
        best_config = get_best_config_from_fine_search()
        size = args.size or best_config["size"]
        C = args.C or best_config["C"]  # noqa: N806
        gamma = args.gamma or best_config["gamma"]
    else:
        size = args.size
        C = args.C  # noqa: N806
        gamma = args.gamma

    logger.info("Final config: size=%d, C=%.1f, gamma=%s", size, C, gamma)

    # Load training data
    logger.info("")
    logger.info("=" * 70)
    logger.info("LOADING TRAINING DATA")
    logger.info("=" * 70)
    train_seed, auto_labeled, train_embeddings = load_training_data()
    logger.info("Train seed: %d, Auto-labeled: %d", len(train_seed), len(auto_labeled))

    # Pre-sort auto-labeled by class once
    n_seed = len(train_seed)
    auto_sorted_by_class = presort_auto_labeled_by_class(auto_labeled)
    for label, indices in auto_sorted_by_class.items():
        logger.info("  %s: %d available", label, len(indices))

    # Select training subset (stratified with minority boost)
    X_train, y_train = select_training_subset(  # noqa: N806
        train_seed,
        train_embeddings,
        size,
        n_seed,
        auto_sorted_by_class,
    )
    logger.info("Training set size: %d", len(y_train))
    logger.info("Training distribution: %s", get_label_distribution(y_train))

    # Train final model
    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING FINAL MODEL")
    logger.info("=" * 70)
    clf = train_svm(X_train, y_train, C=C, gamma=gamma)
    logger.info("Model trained!")

    # Load test data
    logger.info("")
    logger.info("=" * 70)
    logger.info("LOADING TEST DATA (FIRST TIME ACCESSING)")
    logger.info("=" * 70)
    test_examples, X_test = load_test_data()  # noqa: N806
    y_test = [e.label for e in test_examples]
    logger.info("Test set size: %d", len(y_test))
    logger.info("Test distribution: %s", get_label_distribution(y_test))

    # Evaluate
    logger.info("")
    logger.info("=" * 70)
    logger.info("EVALUATING ON TEST SET")
    logger.info("=" * 70)

    y_pred = clf.predict(X_test).tolist()

    # Metrics
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Confidence interval
    point, lower, upper = compute_confidence_interval(y_test, y_pred)
    logger.info("Macro F1: %.3f [95%% CI: %.3f - %.3f]", point, lower, upper)
    logger.info("Weighted F1: %.3f", weighted_f1)

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    logger.info("")
    logger.info("Per-class metrics:")
    logger.info("%12s %10s %10s %10s %10s", "Class", "Precision", "Recall", "F1", "Support")
    logger.info("-" * 55)

    for label in sorted(set(y_test)):
        if label in report:
            m = report[label]
            logger.info(
                "%12s %10.3f %10.3f %10.3f %10d",
                label,
                m["precision"],
                m["recall"],
                m["f1-score"],
                int(m["support"]),
            )

    # Confusion matrix
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    logger.info("")
    logger.info("Confusion Matrix:")
    logger.info("Predicted -> " + " ".join(f"{lbl[:6]:>8}" for lbl in labels))
    for i, row_label in enumerate(labels):
        row_str = " ".join(f"{cm[i, j]:>8}" for j in range(len(labels)))
        logger.info(f"{row_label:<12} {row_str}")

    # Save model
    if args.save_model:
        logger.info("")
        logger.info("=" * 70)
        logger.info("SAVING MODEL")
        logger.info("=" * 70)

        model_dir = MODELS_DIR / "response_v2"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save SVM
        svm_path = model_dir / "svm.pkl"
        with open(svm_path, "wb") as f:
            pickle.dump(clf, f)
        logger.info("Saved SVM to %s", svm_path)

        # Save config
        config = {
            "labels": sorted(set(y_train)),
            "size": size,
            "C": C,
            "gamma": gamma,
            "macro_f1": macro_f1,
            "macro_f1_ci_lower": lower,
            "macro_f1_ci_upper": upper,
            "weighted_f1": weighted_f1,
            "train_size": len(y_train),
            "test_size": len(y_test),
        }
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info("Saved config to %s", config_path)

    # Save evaluation results
    logger.info("")
    logger.info("=" * 70)
    logger.info("SAVING RESULTS")
    logger.info("=" * 70)

    results = {
        "config": {
            "size": size,
            "C": C,
            "gamma": gamma,
        },
        "metrics": {
            "macro_f1": macro_f1,
            "macro_f1_ci_lower": lower,
            "macro_f1_ci_upper": upper,
            "weighted_f1": weighted_f1,
        },
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": int(report[label]["support"]),
            }
            for label in sorted(set(y_test))
            if label in report
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
        "train_distribution": get_label_distribution(y_train),
        "test_distribution": get_label_distribution(y_test),
    }

    results_path = RESULTS_DIR / "final_evaluation.json"
    save_results(results, results_path)
    logger.info("Results saved to %s", results_path)

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info("Best config: size=%d, C=%.1f, gamma=%s", size, C, gamma)
    logger.info("Test Macro F1: %.3f [95%% CI: %.3f - %.3f]", macro_f1, lower, upper)
    logger.info("")
    logger.info("This is the HONEST estimate on held-out human-labeled data.")


if __name__ == "__main__":
    main()
