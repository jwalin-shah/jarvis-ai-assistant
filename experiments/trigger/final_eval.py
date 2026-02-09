#!/usr/bin/env python3
"""Final evaluation of trigger classifier on held-out test set.

Trains with best config from coarse search and evaluates on test_human.

Usage:
    uv run python -m experiments.trigger.final_eval
"""

from __future__ import annotations

import json
import logging
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.svm import SVC
from sklearn.utils import resample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
MODELS_DIR = Path(__file__).parent / "models"

LABELS = ["commitment", "question", "reaction", "social", "statement"]


def load_best_config() -> dict:
    """Load best config from coarse search."""
    path = RESULTS_DIR / "coarse_search.json"
    with open(path) as f:
        results = json.load(f)
    # Results are sorted by cv_mean descending
    return results[0]


def load_train_data(target_size: int | None = None):
    """Load training data (seed + auto)."""
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

    cache = np.load(DATA_DIR / "embeddings_cache.npz")
    embeddings = cache["embeddings"]
    n_seed = int(cache["n_seed"])

    # Combine all
    all_examples = train_seed + auto_labeled
    all_labels = [e["label"] for e in all_examples]

    if target_size and target_size < len(all_examples):
        # Sample to target size, preserving class distribution
        seed_embeddings = embeddings[:n_seed]
        seed_labels = [e["label"] for e in train_seed]

        n_auto_needed = target_size - n_seed
        auto_by_class: dict[str, list[tuple[int, float]]] = {label: [] for label in LABELS}
        for i, ex in enumerate(auto_labeled):
            label = ex["label"]
            conf = ex.get("confidence", 1.0)
            auto_by_class[label].append((i, conf))

        for label in LABELS:
            auto_by_class[label].sort(key=lambda x: -x[1])

        seed_dist = Counter(seed_labels)
        total_seed = sum(seed_dist.values())

        selected_auto_indices = []
        for label in LABELS:
            proportion = seed_dist.get(label, 0) / total_seed
            n_needed = int(n_auto_needed * proportion)
            available = auto_by_class[label][:n_needed]
            selected_auto_indices.extend([idx for idx, _ in available])

        while len(selected_auto_indices) < n_auto_needed:
            for label in LABELS:
                for idx, _ in auto_by_class[label]:
                    if idx not in selected_auto_indices:
                        selected_auto_indices.append(idx)
                        if len(selected_auto_indices) >= n_auto_needed:
                            break
                if len(selected_auto_indices) >= n_auto_needed:
                    break

        auto_embeddings = embeddings[n_seed:][selected_auto_indices]
        auto_labels = [auto_labeled[i]["label"] for i in selected_auto_indices]

        X = np.vstack([seed_embeddings, auto_embeddings])
        y = seed_labels + auto_labels
    else:
        X = embeddings
        y = all_labels

    return X, y


def load_test_data():
    """Load held-out test data."""
    test_examples = []
    with open(DATA_DIR / "test_human.jsonl") as f:
        for line in f:
            if line.strip():
                test_examples.append(json.loads(line))

    cache = np.load(DATA_DIR / "test_embeddings.npz")
    embeddings = cache["embeddings"]

    labels = [e["label"] for e in test_examples]
    return embeddings, labels


def bootstrap_ci(y_true, y_pred, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42):
    """Compute bootstrap confidence interval for macro F1."""
    rng = np.random.RandomState(seed)
    scores = []
    n = len(y_true)

    for i in range(n_bootstrap):
        indices = resample(range(n), n_samples=n, random_state=rng)
        y_true_boot = [y_true[i] for i in indices]
        y_pred_boot = [y_pred[i] for i in indices]
        score = f1_score(y_true_boot, y_pred_boot, average="macro")
        scores.append(score)

    alpha = (1 - ci) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)

    return lower, upper


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load best config
    logger.info("Loading best config from coarse search...")
    best = load_best_config()
    logger.info(
        "Best config: size=%d, C=%.1f, gamma=%s, CV F1=%.3f",
        best["size"],
        best["C"],
        best["gamma"],
        best["cv_mean"],
    )

    # Load training data
    logger.info("")
    logger.info("=" * 70)
    logger.info("LOADING TRAINING DATA")
    logger.info("=" * 70)

    X_train, y_train = load_train_data(target_size=best["size"])
    logger.info("Training data: X=%s", X_train.shape)
    logger.info("Distribution: %s", dict(Counter(y_train)))

    # Train final model
    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING FINAL MODEL")
    logger.info("=" * 70)

    svm = SVC(
        C=best["C"],
        kernel="rbf",
        gamma=best["gamma"],
        class_weight="balanced",
        probability=True,
        random_state=42,
    )
    svm.fit(X_train, y_train)
    logger.info("Model trained!")

    # Load test data
    logger.info("")
    logger.info("=" * 70)
    logger.info("LOADING TEST DATA (FIRST TIME ACCESSING)")
    logger.info("=" * 70)

    X_test, y_test = load_test_data()
    logger.info("Test data: X=%s", X_test.shape)
    logger.info("Distribution: %s", dict(Counter(y_test)))

    # Evaluate
    logger.info("")
    logger.info("=" * 70)
    logger.info("EVALUATING ON TEST SET")
    logger.info("=" * 70)

    y_pred = svm.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    ci_lower, ci_upper = bootstrap_ci(y_test, y_pred)

    logger.info("Macro F1: %.3f [95%% CI: %.3f - %.3f]", macro_f1, ci_lower, ci_upper)
    logger.info("Weighted F1: %.3f", weighted_f1)

    # Per-class metrics
    logger.info("")
    logger.info("Per-class metrics:")
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info("     Class  Precision     Recall         F1    Support")
    logger.info("-" * 55)
    for label in LABELS:
        if label in report:
            r = report[label]
            logger.info(
                "%10s      %.3f      %.3f      %.3f        %d",
                label,
                r["precision"],
                r["recall"],
                r["f1-score"],
                int(r["support"]),
            )

    # Confusion matrix
    logger.info("")
    logger.info("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    header = "Predicted ->  " + "  ".join(f"{l[:6]:>6}" for l in LABELS)
    logger.info(header)
    for i, label in enumerate(LABELS):
        row = f"{label[:10]:10}" + "  ".join(f"{cm[i, j]:>6}" for j in range(len(LABELS)))
        logger.info(row)

    # Save model
    logger.info("")
    logger.info("=" * 70)
    logger.info("SAVING MODEL")
    logger.info("=" * 70)

    with open(MODELS_DIR / "svm.pkl", "wb") as f:
        pickle.dump(svm, f)
    logger.info("Saved SVM to %s", MODELS_DIR / "svm.pkl")

    config = {
        "labels": LABELS,
        "size": best["size"],
        "C": best["C"],
        "gamma": best["gamma"],
        "macro_f1": macro_f1,
        "macro_f1_ci_lower": ci_lower,
        "macro_f1_ci_upper": ci_upper,
        "weighted_f1": weighted_f1,
        "train_size": len(y_train),
        "test_size": len(y_test),
    }
    with open(MODELS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Saved config to %s", MODELS_DIR / "config.json")

    # Save results
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info("Best config: size=%d, C=%.1f, gamma=%s", best["size"], best["C"], best["gamma"])
    logger.info("Test Macro F1: %.3f [95%% CI: %.3f - %.3f]", macro_f1, ci_lower, ci_upper)
    logger.info("")
    logger.info("This is the HONEST estimate on held-out human-labeled data.")

    results = {
        "config": best,
        "test_macro_f1": macro_f1,
        "test_macro_f1_ci_lower": ci_lower,
        "test_macro_f1_ci_upper": ci_upper,
        "test_weighted_f1": weighted_f1,
        "per_class": {label: report[label] for label in LABELS if label in report},
    }
    with open(RESULTS_DIR / "final_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", RESULTS_DIR / "final_evaluation.json")


if __name__ == "__main__":
    main()
