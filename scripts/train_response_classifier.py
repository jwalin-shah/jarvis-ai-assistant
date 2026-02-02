#!/usr/bin/env python3
"""Train and evaluate response classifier with different configurations.

Tries multiple sampling strategies and hyperparameters to find the best setup.

Usage:
    uv run python -m scripts.train_response_classifier
    uv run python -m scripts.train_response_classifier --save-best
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from jarvis.config import get_config, get_response_classifier_path
from jarvis.embedding_adapter import get_embedder


@dataclass
class ExperimentResult:
    """Result from a single experiment."""

    name: str
    sampling: str
    C: float
    gamma: str | float
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]]
    train_size: int
    test_size: int
    train_distribution: dict[str, int]


def load_data(path: Path) -> tuple[list[str], list[str]]:
    """Load labeled response data."""
    texts = []
    labels = []

    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("response", "").strip()
            label = row.get("label")
            if text and label:
                labels.append(label.upper())
                texts.append(text)

    return texts, labels


def downsample_to_target(
    texts: list[str],
    labels: list[str],
    target_total: int = 3000,
    minority_classes: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Downsample majority classes to reach target total while keeping minority classes intact.

    Args:
        texts: Input texts
        labels: Input labels
        target_total: Target total number of samples
        minority_classes: Classes to keep fully (not downsample)
    """
    if minority_classes is None:
        minority_classes = ["AGREE", "DEFER", "DECLINE"]

    counts = Counter(labels)

    # Group by label
    by_label: dict[str, list[str]] = {l: [] for l in set(labels)}
    for text, label in zip(texts, labels):
        by_label[label].append(text)

    # Calculate minority total
    minority_total = sum(counts[l] for l in minority_classes if l in counts)
    majority_classes = [l for l in counts if l not in minority_classes]
    majority_total = sum(counts[l] for l in majority_classes)

    # Budget for majority classes
    majority_budget = target_total - minority_total

    if majority_budget <= 0:
        # Not enough budget, just return as-is
        return texts, labels

    # Calculate proportional reduction for majority classes
    rng = np.random.default_rng(42)
    new_texts = []
    new_labels = []

    for label in minority_classes:
        if label in by_label:
            new_texts.extend(by_label[label])
            new_labels.extend([label] * len(by_label[label]))

    for label in majority_classes:
        target_count = int(counts[label] / majority_total * majority_budget)
        label_texts = by_label[label]

        if len(label_texts) <= target_count:
            sampled = label_texts
        else:
            indices = rng.choice(len(label_texts), size=target_count, replace=False)
            sampled = [label_texts[i] for i in indices]

        new_texts.extend(sampled)
        new_labels.extend([label] * len(sampled))

    return new_texts, new_labels


def undersample_majority(
    texts: list[str],
    labels: list[str],
    max_per_class: int | None = None,
    target_ratio: float | None = None,
) -> tuple[list[str], list[str]]:
    """Undersample majority class(es).

    Args:
        texts: Input texts
        labels: Input labels
        max_per_class: Hard cap per class
        target_ratio: Max ratio of largest to smallest class
    """
    counts = Counter(labels)
    min_count = min(counts.values())

    if max_per_class:
        cap = max_per_class
    elif target_ratio:
        cap = int(min_count * target_ratio)
    else:
        cap = min_count  # Fully balanced

    # Group by label
    by_label: dict[str, list[str]] = {l: [] for l in set(labels)}
    for text, label in zip(texts, labels):
        by_label[label].append(text)

    # Sample from each
    new_texts = []
    new_labels = []
    rng = np.random.default_rng(42)

    for label, label_texts in by_label.items():
        if len(label_texts) <= cap:
            sampled = label_texts
        else:
            indices = rng.choice(len(label_texts), size=cap, replace=False)
            sampled = [label_texts[i] for i in indices]

        new_texts.extend(sampled)
        new_labels.extend([label] * len(sampled))

    return new_texts, new_labels


def run_experiment(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: list[str],
    y_test: list[str],
    C: float,
    gamma: str | float,
    name: str,
    sampling: str,
) -> ExperimentResult:
    """Run a single training experiment."""
    clf = SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        class_weight="balanced",
        probability=True,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Get per-class metrics
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    per_class = {
        k: v for k, v in report.items() if k not in ["accuracy", "macro avg", "weighted avg"]
    }

    return ExperimentResult(
        name=name,
        sampling=sampling,
        C=C,
        gamma=gamma,
        accuracy=accuracy_score(y_test, y_pred),
        macro_f1=f1_score(y_test, y_pred, average="macro", zero_division=0),
        weighted_f1=f1_score(y_test, y_pred, average="weighted", zero_division=0),
        per_class=per_class,
        train_size=len(y_train),
        test_size=len(y_test),
        train_distribution=dict(Counter(y_train)),
    )


def main():
    parser = argparse.ArgumentParser(description="Train response classifier")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/response_labeling.jsonl"),
        help="Input JSONL file with labeled responses",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="Save the best model to ~/.jarvis/response_classifier_model/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/response_classifier_experiments.json"),
        help="Output JSON with all experiment results",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=3000,
        help="Target number of samples after downsampling (default: 3000)",
    )
    args = parser.parse_args()

    # Load data
    print("Loading data...", flush=True)
    texts, labels = load_data(args.input)
    print(f"Loaded {len(texts)} labeled examples", flush=True)
    print(f"Distribution: {dict(Counter(labels))}", flush=True)

    # Downsample to target (keep minority, reduce majority)
    print(f"\nDownsampling to {args.target_samples} samples...", flush=True)
    texts, labels = downsample_to_target(texts, labels, args.target_samples)
    print(f"After downsampling: {len(texts)} examples", flush=True)
    print(f"Distribution: {dict(Counter(labels))}", flush=True)

    # Get embeddings for all texts
    print("\nGenerating embeddings...", flush=True)
    embedder = get_embedder()
    print(f"  Embedding {len(texts)} texts (this may take 30-60 seconds)...", flush=True)
    all_embeddings = embedder.encode(texts, normalize=True)
    print(f"  Done! Embeddings shape: {all_embeddings.shape}", flush=True)

    # Sampling strategies to try
    sampling_configs = [
        ("natural", None, None),  # Use natural distribution
        ("balanced", None, 1.0),  # Equal samples per class (ratio=1.0 means min_count)
        ("cap_500", 500, None),  # Cap at 500 per class
        ("cap_400", 400, None),  # Cap at 400 per class
        ("cap_300", 300, None),  # Cap at 300 per class
        ("ratio_2x", None, 2.0),  # Max 2x the minority class
        ("ratio_3x", None, 3.0),  # Max 3x the minority class
    ]

    # Hyperparameters to try
    C_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    gamma_values = ["scale", "auto", 0.1, 0.5]

    results: list[ExperimentResult] = []
    best_result: ExperimentResult | None = None
    total_experiments = len(sampling_configs) * len(C_values) * len(gamma_values)
    exp_count = 0

    print("\n" + "=" * 70, flush=True)
    print(f"Running {total_experiments} experiments...", flush=True)
    print("=" * 70, flush=True)

    for sampling_name, max_per_class, target_ratio in sampling_configs:
        print(f"\n--- Sampling: {sampling_name} ---", flush=True)

        # Apply sampling
        if sampling_name == "natural":
            sampled_texts, sampled_labels = texts, labels
        else:
            sampled_texts, sampled_labels = undersample_majority(
                texts, labels, max_per_class, target_ratio
            )

        # Create index mapping for embeddings
        text_to_idx = {t: i for i, t in enumerate(texts)}
        sampled_indices = [text_to_idx[t] for t in sampled_texts]
        sampled_embeddings = all_embeddings[sampled_indices]

        print(f"  Samples: {len(sampled_texts)}, Distribution: {dict(Counter(sampled_labels))}")

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            sampled_embeddings,
            sampled_labels,
            test_size=0.2,
            random_state=42,
            stratify=sampled_labels,
        )

        # Try different hyperparameters
        for C in C_values:
            for gamma in gamma_values:
                exp_count += 1
                name = f"{sampling_name}_C{C}_g{gamma}"
                print(f"  [{exp_count}/{total_experiments}] {name}...", end=" ", flush=True)

                result = run_experiment(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    C=C,
                    gamma=gamma,
                    name=name,
                    sampling=sampling_name,
                )
                results.append(result)
                print(f"acc={result.accuracy:.3f} f1={result.macro_f1:.3f}", flush=True)

                # Track best
                if best_result is None or result.macro_f1 > best_result.macro_f1:
                    best_result = result

    # Sort results by macro F1
    results.sort(key=lambda r: r.macro_f1, reverse=True)

    # Print top 10
    print("\n" + "=" * 70)
    print("TOP 10 CONFIGURATIONS (by macro F1)")
    print("=" * 70)
    print(f"{'Rank':<5} {'Name':<30} {'Acc':>7} {'MacroF1':>8} {'WeightF1':>8} {'Train':>6}")
    print("-" * 70)

    for i, r in enumerate(results[:10], 1):
        print(
            f"{i:<5} {r.name:<30} {r.accuracy:>7.3f} {r.macro_f1:>8.3f} "
            f"{r.weighted_f1:>8.3f} {r.train_size:>6}"
        )

    # Print best result details
    if best_result:
        print("\n" + "=" * 70)
        print("BEST CONFIGURATION DETAILS")
        print("=" * 70)
        print(f"Name: {best_result.name}")
        print(f"Sampling: {best_result.sampling}")
        print(f"Hyperparams: C={best_result.C}, gamma={best_result.gamma}")
        print(f"Train size: {best_result.train_size}, Test size: {best_result.test_size}")
        print(f"Train distribution: {best_result.train_distribution}")
        print("\nOverall Metrics:")
        print(f"  Accuracy:    {best_result.accuracy:.3f}")
        print(f"  Macro F1:    {best_result.macro_f1:.3f}")
        print(f"  Weighted F1: {best_result.weighted_f1:.3f}")
        print("\nPer-Class Performance:")
        print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("  " + "-" * 52)
        for cls, metrics in sorted(best_result.per_class.items()):
            if isinstance(metrics, dict):
                prec = metrics.get("precision", 0)
                rec = metrics.get("recall", 0)
                f1 = metrics.get("f1-score", 0)
                sup = metrics.get("support", 0)
                print(f"  {cls:<12} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {sup:>10.0f}")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "total_samples": len(texts),
        "distribution": dict(Counter(labels)),
        "experiments": [
            {
                "name": r.name,
                "sampling": r.sampling,
                "C": r.C,
                "gamma": r.gamma,
                "accuracy": r.accuracy,
                "macro_f1": r.macro_f1,
                "weighted_f1": r.weighted_f1,
                "train_size": r.train_size,
                "test_size": r.test_size,
                "train_distribution": r.train_distribution,
                "per_class": r.per_class,
            }
            for r in results
        ],
    }
    args.output.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved to {args.output}")

    # Save best model if requested
    if args.save_best and best_result:
        print("\n" + "=" * 70)
        print("SAVING BEST MODEL")
        print("=" * 70)

        # Retrain on full training data with best config
        sampling_name = best_result.sampling
        if sampling_name == "natural":
            final_texts, final_labels = texts, labels
        else:
            # Find the config
            for sname, max_pc, ratio in sampling_configs:
                if sname == sampling_name:
                    final_texts, final_labels = undersample_majority(texts, labels, max_pc, ratio)
                    break
            else:
                final_texts, final_labels = texts, labels

        text_to_idx = {t: i for i, t in enumerate(texts)}
        final_indices = [text_to_idx[t] for t in final_texts]
        final_embeddings = all_embeddings[final_indices]

        # Train final model
        final_clf = SVC(
            kernel="rbf",
            C=best_result.C,
            gamma=best_result.gamma,
            class_weight="balanced",
            probability=True,
            random_state=42,
        )
        final_clf.fit(final_embeddings, final_labels)

        # Save to versioned path based on configured embedding model
        model_path = get_response_classifier_path()
        model_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving model to: {model_path} (embedding model: {get_config().embedding.model_name})")

        with open(model_path / "svm.pkl", "wb") as f:
            pickle.dump(final_clf, f)

        config = {
            "labels": sorted(set(final_labels)),
            "sampling": best_result.sampling,
            "C": best_result.C,
            "gamma": best_result.gamma,
            "accuracy": best_result.accuracy,
            "macro_f1": best_result.macro_f1,
            "train_size": len(final_labels),
        }
        with open(model_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"Model saved to {model_path}")
        print(f"Config: {config}")


if __name__ == "__main__":
    main()
