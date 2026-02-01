#!/usr/bin/env python3
"""Compare trigger classifiers trained on different label schemes.

Trains and evaluates SVM classifiers on:
1. Corrected data (10 fine-grained labels)
2. Consolidated data (7 merged labels)

Uses proper stratified train/test splits with holdout for fair comparison.

Usage:
    uv run python -m scripts.compare_trigger_classifiers
    uv run python -m scripts.compare_trigger_classifiers --test-structural
    uv run python -m scripts.compare_trigger_classifiers --save-models
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC

from jarvis.embedding_adapter import get_embedder

# =============================================================================
# Structural Patterns (copied from trigger_classifier.py for testing)
# =============================================================================

STRUCTURAL_PATTERNS = [
    (re.compile(r'^(Liked|Loved|Laughed at|Emphasized|Questioned|Disliked)\s+["\u201c\u201d]', re.I), "ack", 0.95),
    (re.compile(r"^(hey|hi|hello|yo|sup|what'?s up|wassup|hiya|howdy|hola|yooo*)[\s!?]*$", re.I), "greeting", 0.95),
    (re.compile(r"^how are you[\s?!]*$", re.I), "greeting", 0.95),
    (re.compile(r"^good\s*(night|morning|evening)['\s!?]*$", re.I), "greeting", 0.95),
    (re.compile(r"^(ok|okay|k|kk|sure|bet|got it|sounds good|cool|alright|aight|word)[\s!.]*$", re.I), "ack", 0.95),
    (re.compile(r"^(yes|yea|yeah|yup|yep|nah|nope|true|for sure|all\s*right|could be)[\s!.]*$", re.I), "ack", 0.95),
    (re.compile(r"^(thanks|thank you|thx|ty|appreciate it)[\s!.]*$", re.I), "ack", 0.95),
    (re.compile(r"^(lol|lmao|haha+|hehe+|😂|🤣|💀)+[\s!]*$", re.I), "ack", 0.90),
    (re.compile(r"\b(wanna|want to|down to|dtf|tryna|trying to)\s+.*(hang|chill|go|come|grab|get|play|watch|do)\b.*\?", re.I), "invitation", 0.95),
    (re.compile(r"^(wanna|want to|down to)\s+\w+.*\?", re.I), "invitation", 0.90),
    (re.compile(r"\b(you|u)\s+(free|available|busy|down)\s*(today|tonight|tomorrow|tmrw|later|this weekend|rn)?\s*\?", re.I), "invitation", 0.95),
    (re.compile(r"^(let'?s|lets)\s+(go|hang|chill|grab|get|do|play|watch)\b", re.I), "invitation", 0.85),
    (re.compile(r"\bcome (over|through|thru|hang|chill)\b.*\?", re.I), "invitation", 0.90),
    (re.compile(r"^(can|could|would|will)\s+(you|u)\s+(please|pls|plz)\b", re.I), "request", 0.90),
    (re.compile(r"^(please|pls|plz)\s+\w+", re.I), "request", 0.85),
    (re.compile(r"\b(pick me up|drop me off|send me|get me|help me)\b", re.I), "request", 0.90),
    (re.compile(r"\b(lmk|let me know)\s+(if|when|what)\b", re.I), "request", 0.80),
    (re.compile(r"^(what|what'?s)\s+(time|day|the plan|up|going on|happening)\b", re.I), "info_question", 0.95),
    (re.compile(r"^(when|where|who|which|how)\s+.+\?$", re.I), "info_question", 0.90),
    (re.compile(r"\b(what time|how long|how much|how many)\b.*\?", re.I), "info_question", 0.95),
    (re.compile(r"^(do|does|did|is|are|was|were|have|has|can|could|will|would|should)\s+(you|u|we|they|i|it|he|she)\b.*\?", re.I), "yn_question", 0.85),
    (re.compile(r"\b(i got (the job|accepted|promoted|in|hired)|i passed|i made it|we won|i'm engaged|i'm pregnant)\b", re.I), "good_news", 0.85),
    (re.compile(r"^(great news|good news|finally|so excited|so happy)[!:\s]", re.I), "good_news", 0.80),
    (re.compile(r"\b(i lost my (wallet|keys|phone|job)|i failed|i got fired|i'm sick|someone died|passed away)\b", re.I), "bad_news", 0.85),
    (re.compile(r"^(so sad|so upset|terrible news|awful news|bad news|unfortunately)\b", re.I), "bad_news", 0.80),
    (re.compile(r"^(omg|oh my god)\b.+(\?|!{2,}|wtf|crazy|insane)", re.I), "reaction", 0.85),
    (re.compile(r"\b(did you (see|hear|watch)|have you seen)\b.*\?", re.I), "reaction", 0.85),
    (re.compile(r"\b(can you believe|isn't that|wasn't that)\b.*\?", re.I), "reaction", 0.85),
    (re.compile(r"^(fuck+|shit+|damn+|ugh+|omfg|fml)\b", re.I), "bad_news", 0.80),
    (re.compile(r"\?\s*$"), "yn_question", 0.60),
]


def match_structural(text: str) -> tuple[str | None, float]:
    """Match text against structural patterns."""
    text = text.strip()
    for pattern, label, conf in STRUCTURAL_PATTERNS:
        if pattern.search(text):
            return label, conf
    return None, 0.0


# =============================================================================
# Data Loading
# =============================================================================

@dataclass
class DataConfig:
    """Configuration for a dataset."""
    name: str
    path: Path
    text_field: str = "text"
    label_field: str = "label"


def load_data(config: DataConfig) -> tuple[list[str], list[str]]:
    """Load texts and labels from a JSONL file."""
    texts = []
    labels = []

    with config.path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get(config.text_field, "").strip()
            label = row.get(config.label_field, "").strip().lower()
            if text and label:
                texts.append(text)
                labels.append(label)

    return texts, labels


# =============================================================================
# Training and Evaluation
# =============================================================================

@dataclass
class ClassifierResult:
    """Results from training and evaluating a classifier."""
    name: str
    labels: list[str]
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_metrics: dict[str, dict[str, float]]
    confusion_matrix: np.ndarray
    train_size: int
    test_size: int
    model: SVC | None = None


def train_and_evaluate(
    texts: list[str],
    labels: list[str],
    name: str,
    embedder,
    test_size: float = 0.2,
    seed: int = 42,
    min_samples_per_class: int = 5,
) -> ClassifierResult:
    """Train SVM and evaluate with stratified split."""

    # Filter classes with too few samples
    label_counts = Counter(labels)
    valid_labels = {l for l, c in label_counts.items() if c >= min_samples_per_class}

    filtered_texts = []
    filtered_labels = []
    for t, l in zip(texts, labels):
        if l in valid_labels:
            filtered_texts.append(t)
            filtered_labels.append(l)

    if len(filtered_texts) < 10:
        raise ValueError(f"Too few samples after filtering: {len(filtered_texts)}")

    print(f"\n=== Training {name} ===")
    print(f"Total samples: {len(filtered_texts)}")
    print(f"Classes: {sorted(set(filtered_labels))}")
    print(f"Distribution: {Counter(filtered_labels)}")

    # Embed all texts
    print("Embedding texts...")
    embeddings = embedder.encode(filtered_texts, normalize=True)

    # Stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    indices = np.arange(len(filtered_labels))
    train_idx, test_idx = next(splitter.split(indices, filtered_labels))

    X_train = embeddings[train_idx]
    X_test = embeddings[test_idx]
    y_train = [filtered_labels[i] for i in train_idx]
    y_test = [filtered_labels[i] for i in test_idx]

    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    # Train SVM
    print("Training SVM...")
    clf = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=seed,
    )
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Metrics
    unique_labels = sorted(set(y_test))
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    per_class = {}
    for label in unique_labels:
        mask = [l == label for l in y_test]
        if sum(mask) == 0:
            continue
        y_true_binary = [1 if l == label else 0 for l in y_test]
        y_pred_binary = [1 if l == label else 0 for l in y_pred]
        per_class[label] = {
            "precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
            "recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
            "f1": f1_score(y_true_binary, y_pred_binary, zero_division=0),
            "support": sum(mask),
        }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

    return ClassifierResult(
        name=name,
        labels=unique_labels,
        accuracy=acc,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class_metrics=per_class,
        confusion_matrix=cm,
        train_size=len(y_train),
        test_size=len(y_test),
        model=clf,
    )


def print_results(result: ClassifierResult) -> None:
    """Print classification results."""
    print(f"\n{'='*60}")
    print(f"RESULTS: {result.name}")
    print(f"{'='*60}")
    print(f"Train size: {result.train_size}, Test size: {result.test_size}")
    print(f"Accuracy:    {result.accuracy:.3f}")
    print(f"Macro F1:    {result.macro_f1:.3f}")
    print(f"Weighted F1: {result.weighted_f1:.3f}")

    print(f"\nPer-Class Metrics:")
    print(f"{'Label':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 55)
    for label in sorted(result.per_class_metrics.keys()):
        m = result.per_class_metrics[label]
        print(f"{label:<15} {m['precision']:>10.2f} {m['recall']:>10.2f} {m['f1']:>10.2f} {m['support']:>10}")

    print(f"\nConfusion Matrix:")
    print(f"Labels: {result.labels}")
    print(result.confusion_matrix)


def evaluate_structural(texts: list[str], labels: list[str], name: str) -> dict:
    """Evaluate structural pattern matching."""
    print(f"\n=== Structural Patterns on {name} ===")

    correct = 0
    covered = 0
    total = len(texts)

    per_label = defaultdict(lambda: {"correct": 0, "wrong": 0, "missed": 0, "total": 0})

    for text, true_label in zip(texts, labels):
        per_label[true_label]["total"] += 1
        pred_label, conf = match_structural(text)

        if pred_label:
            covered += 1
            if pred_label == true_label:
                correct += 1
                per_label[true_label]["correct"] += 1
            else:
                per_label[true_label]["wrong"] += 1
        else:
            per_label[true_label]["missed"] += 1

    coverage = covered / total if total > 0 else 0
    accuracy_covered = correct / covered if covered > 0 else 0
    accuracy_total = correct / total if total > 0 else 0

    print(f"Total samples: {total}")
    print(f"Covered: {covered} ({coverage:.1%})")
    print(f"Accuracy (on covered): {accuracy_covered:.3f}")
    print(f"Accuracy (on total): {accuracy_total:.3f}")

    print(f"\nPer-Label Coverage:")
    print(f"{'Label':<15} {'Correct':>8} {'Wrong':>8} {'Missed':>8} {'Coverage':>10} {'Precision':>10}")
    print("-" * 65)
    for label in sorted(per_label.keys()):
        m = per_label[label]
        cov = (m["correct"] + m["wrong"]) / m["total"] if m["total"] > 0 else 0
        prec = m["correct"] / (m["correct"] + m["wrong"]) if (m["correct"] + m["wrong"]) > 0 else 0
        print(f"{label:<15} {m['correct']:>8} {m['wrong']:>8} {m['missed']:>8} {cov:>10.1%} {prec:>10.2f}")

    return {
        "coverage": coverage,
        "accuracy_covered": accuracy_covered,
        "accuracy_total": accuracy_total,
        "per_label": dict(per_label),
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare trigger classifiers")
    parser.add_argument(
        "--corrected",
        type=Path,
        default=Path("results/trigger_training_full_corrected.jsonl"),
        help="Path to corrected (10-label) data",
    )
    parser.add_argument(
        "--consolidated",
        type=Path,
        default=Path("results/trigger_training_full_consolidated.jsonl"),
        help="Path to consolidated (7-label) data",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--test-structural",
        action="store_true",
        help="Also test structural pattern coverage",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save trained models to ~/.jarvis/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/trigger_classifier_comparison.json"),
        help="Output JSON report",
    )
    parser.add_argument(
        "--include-candidates",
        action="store_true",
        help="Include candidates file in training data",
    )
    args = parser.parse_args()

    # Check files exist
    if not args.corrected.exists():
        print(f"Error: {args.corrected} not found")
        return
    if not args.consolidated.exists():
        print(f"Error: {args.consolidated} not found")
        return

    # Load embedder
    print("Loading embedder...")
    embedder = get_embedder()

    results = {}

    # Load corrected data
    corrected_config = DataConfig(name="corrected", path=args.corrected)
    texts_corr, labels_corr = load_data(corrected_config)

    # Load consolidated data
    consolidated_config = DataConfig(name="consolidated", path=args.consolidated)
    texts_cons, labels_cons = load_data(consolidated_config)

    # Optionally include candidates
    if args.include_candidates:
        candidates_corr = Path("results/trigger_candidates_labeled_corrected.jsonl")
        candidates_cons = Path("results/trigger_candidates_labeled_consolidated.jsonl")

        if candidates_corr.exists():
            # Candidates use trigger_text instead of text
            t, l = load_data(DataConfig("cand_corr", candidates_corr, text_field="trigger_text"))
            texts_corr.extend(t)
            labels_corr.extend(l)
            print(f"Added {len(t)} candidates to corrected data")

        if candidates_cons.exists():
            t, l = load_data(DataConfig("cand_cons", candidates_cons, text_field="trigger_text"))
            texts_cons.extend(t)
            labels_cons.extend(l)
            print(f"Added {len(t)} candidates to consolidated data")

    # Test structural patterns
    if args.test_structural:
        results["structural_corrected"] = evaluate_structural(texts_corr, labels_corr, "Corrected")
        results["structural_consolidated"] = evaluate_structural(texts_cons, labels_cons, "Consolidated")

    # Train and evaluate on corrected data (10 labels)
    result_corr = train_and_evaluate(
        texts_corr,
        labels_corr,
        "Corrected (10 labels)",
        embedder,
        test_size=args.test_size,
        seed=args.seed,
    )
    print_results(result_corr)

    # Train and evaluate on consolidated data (7 labels)
    result_cons = train_and_evaluate(
        texts_cons,
        labels_cons,
        "Consolidated (7 labels)",
        embedder,
        test_size=args.test_size,
        seed=args.seed,
    )
    print_results(result_cons)

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'Corrected (10)':<20} {'Consolidated (7)':<20}")
    print("-" * 60)
    print(f"{'Accuracy':<20} {result_corr.accuracy:<20.3f} {result_cons.accuracy:<20.3f}")
    print(f"{'Macro F1':<20} {result_corr.macro_f1:<20.3f} {result_cons.macro_f1:<20.3f}")
    print(f"{'Weighted F1':<20} {result_corr.weighted_f1:<20.3f} {result_cons.weighted_f1:<20.3f}")
    print(f"{'Num Classes':<20} {len(result_corr.labels):<20} {len(result_cons.labels):<20}")

    # Save results
    results["corrected"] = {
        "name": result_corr.name,
        "labels": result_corr.labels,
        "accuracy": result_corr.accuracy,
        "macro_f1": result_corr.macro_f1,
        "weighted_f1": result_corr.weighted_f1,
        "per_class_metrics": result_corr.per_class_metrics,
        "train_size": result_corr.train_size,
        "test_size": result_corr.test_size,
    }
    results["consolidated"] = {
        "name": result_cons.name,
        "labels": result_cons.labels,
        "accuracy": result_cons.accuracy,
        "macro_f1": result_cons.macro_f1,
        "weighted_f1": result_cons.weighted_f1,
        "per_class_metrics": result_cons.per_class_metrics,
        "train_size": result_cons.train_size,
        "test_size": result_cons.test_size,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to: {args.output}")

    # Save models if requested
    if args.save_models:
        model_dir = Path.home() / ".jarvis" / "trigger_classifier_models"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save corrected model
        corr_dir = model_dir / "corrected_10label"
        corr_dir.mkdir(parents=True, exist_ok=True)
        with open(corr_dir / "svm.pkl", "wb") as f:
            pickle.dump(result_corr.model, f)
        (corr_dir / "config.json").write_text(json.dumps({
            "labels": result_corr.labels,
            "accuracy": result_corr.accuracy,
            "embedder": "bge-small-en-v1.5",
        }, indent=2))
        print(f"Saved corrected model to: {corr_dir}")

        # Save consolidated model
        cons_dir = model_dir / "consolidated_7label"
        cons_dir.mkdir(parents=True, exist_ok=True)
        with open(cons_dir / "svm.pkl", "wb") as f:
            pickle.dump(result_cons.model, f)
        (cons_dir / "config.json").write_text(json.dumps({
            "labels": result_cons.labels,
            "accuracy": result_cons.accuracy,
            "embedder": "bge-small-en-v1.5",
        }, indent=2))
        print(f"Saved consolidated model to: {cons_dir}")


if __name__ == "__main__":
    main()
