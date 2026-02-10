#!/usr/bin/env python3
"""Full evaluation pipeline: baseline → Gemini comparison → retraining.

Steps:
1. Evaluate current models on Gemini-labeled data (baseline)
2. Compare Gemini vs auto labels (agreement analysis)
3. Generate training data from Gemini labels
4. Retrain classifiers
5. Compare old vs new performance

Usage:
    uv run python scripts/eval_and_retrain_gemini.py
    uv run python scripts/eval_and_retrain_gemini.py --skip-retrain
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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = ROOT / "evals" / "data" / "pipeline_eval_labeled.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"


def load_eval_data() -> list[dict]:
    """Load Gemini-labeled eval dataset."""
    if not EVAL_PATH.exists():
        logger.error("Labeled eval dataset not found: %s", EVAL_PATH)
        sys.exit(1)

    examples = []
    for line in EVAL_PATH.open():
        line = line.strip()
        if line:
            examples.append(json.loads(line))
    return examples


def analyze_label_agreement(examples: list[dict]) -> dict:
    """Compare Gemini labels vs auto labels and confidence distribution."""
    gemini_labels = [e for e in examples if e.get("label_confidence") == "gemini"]
    auto_labels = [e for e in examples if e.get("label_confidence") == "auto"]
    needs_review = [e for e in examples if e.get("label_confidence") == "needs_review"]

    logger.info("\n" + "=" * 60)
    logger.info("LABEL CONFIDENCE DISTRIBUTION")
    logger.info("=" * 60)
    logger.info(f"Gemini labels:  {len(gemini_labels):5d} ({100*len(gemini_labels)//len(examples):3d}%)")
    logger.info(f"Auto labels:    {len(auto_labels):5d} ({100*len(auto_labels)//len(examples):3d}%)")
    logger.info(f"Needs review:   {len(needs_review):5d} ({100*len(needs_review)//len(examples):3d}%)")
    logger.info(f"Total:          {len(examples):5d}")

    # Category distribution
    cat_dist = Counter(e.get("category") for e in examples if e.get("category"))
    logger.info("\nCATEGORY DISTRIBUTION (all):")
    for cat, count in sorted(cat_dist.items()):
        logger.info(f"  {cat:15s}: {count:4d} ({100*count//len(examples):3d}%)")

    # Mobilization distribution
    mob_dist = Counter(e.get("mobilization") for e in examples if e.get("mobilization"))
    logger.info("\nMOBILIZATION DISTRIBUTION (all):")
    for mob, count in sorted(mob_dist.items()):
        logger.info(f"  {mob:15s}: {count:4d} ({100*count//len(examples):3d}%)")

    return {
        "total": len(examples),
        "gemini": len(gemini_labels),
        "auto": len(auto_labels),
        "needs_review": len(needs_review),
        "category_dist": dict(cat_dist),
        "mobilization_dist": dict(mob_dist),
    }


def eval_mobilization_baseline(examples: list[dict]) -> dict:
    """Evaluate mobilization classifier on Gemini-labeled data."""
    from jarvis.classifiers.response_mobilization import classify_response_pressure

    labeled = [e for e in examples if e.get("mobilization")]
    if not labeled:
        logger.warning("No mobilization labels, skipping")
        return {"error": "no_labels"}

    logger.info("\n" + "=" * 60)
    logger.info("MOBILIZATION BASELINE (Current Model)")
    logger.info("=" * 60)
    logger.info(f"Evaluating on {len(labeled)} examples...")

    y_true = []
    y_pred = []
    latencies = []

    for i, ex in enumerate(labeled):
        start = time.perf_counter()
        result = classify_response_pressure(ex["text"])
        latencies.append((time.perf_counter() - start) * 1000)

        y_true.append(ex["mobilization"].upper())
        y_pred.append(result.pressure.value.upper())

        if (i + 1) % 500 == 0:
            logger.info(f"  {i+1}/{len(labeled)} evaluated")

    return _compute_metrics(y_true, y_pred, latencies, "mobilization")


def eval_category_baseline(examples: list[dict]) -> dict:
    """Evaluate category classifier on Gemini-labeled data."""
    from jarvis.classifiers.category_classifier import classify_category

    labeled = [e for e in examples if e.get("category")]
    if not labeled:
        logger.warning("No category labels, skipping")
        return {"error": "no_labels"}

    logger.info("\n" + "=" * 60)
    logger.info("CATEGORY BASELINE (Current Model)")
    logger.info("=" * 60)
    logger.info(f"Evaluating on {len(labeled)} examples...")

    y_true = []
    y_pred = []
    latencies = []

    for i, ex in enumerate(labeled):
        start = time.perf_counter()
        result = classify_category(ex["text"], context=ex.get("thread", []))
        latencies.append((time.perf_counter() - start) * 1000)

        y_true.append(ex["category"])
        y_pred.append(result.category)

        if (i + 1) % 500 == 0:
            logger.info(f"  {i+1}/{len(labeled)} evaluated")

    return _compute_metrics(y_true, y_pred, latencies, "category")


def _compute_metrics(
    y_true: list[str], y_pred: list[str], latencies: list[float], stage: str
) -> dict:
    """Compute per-class metrics."""
    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"\n{'=' * 60}")
    print(f"  {stage.upper()} Classification Report")
    print(f"{'=' * 60}")
    print(classification_report(y_true, y_pred, labels=labels))

    print("Confusion Matrix:")
    header = "          " + "  ".join(f"{l:>10}" for l in labels)
    print(header)
    for i, label in enumerate(labels):
        row = f"{label:>10}" + "  ".join(f"{cm[i][j]:>10}" for j in range(len(labels)))
        print(row)
    print()

    return {
        "stage": stage,
        "n_examples": len(y_true),
        "accuracy": round(report["accuracy"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "per_class": {
            label: {
                "precision": round(report[label]["precision"], 4),
                "recall": round(report[label]["recall"], 4),
                "f1": round(report[label]["f1-score"], 4),
                "support": report[label]["support"],
            }
            for label in labels
            if label in report
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
        "latency_ms": {
            "mean": round(float(np.mean(latencies)), 2),
            "p50": round(float(np.median(latencies)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
        },
    }


def compare_label_sources(examples: list[dict]) -> dict:
    """Compare Gemini labels vs auto labels for agreement."""
    logger.info("\n" + "=" * 60)
    logger.info("LABEL SOURCE COMPARISON")
    logger.info("=" * 60)

    # Examples with both auto and gemini labels
    paired = [
        e
        for e in examples
        if e.get("category") and e.get("old_label")  # old_label = auto classification
    ]

    if not paired:
        logger.warning("No paired labels for comparison")
        return {"error": "no_paired_labels"}

    # Category agreement
    cat_agreement = sum(
        1
        for e in paired
        if e.get("category") and e.get("old_label") == e.get("category")
    )
    cat_accuracy = cat_agreement / len(paired)

    logger.info(f"\nCategory Agreement (Gemini vs Auto):")
    logger.info(f"  Examples with both labels: {len(paired)}")
    logger.info(f"  Agreement: {cat_agreement}/{len(paired)} ({100*cat_accuracy:.1f}%)")

    # Confusion between labels
    mismatches = [e for e in paired if e.get("old_label") != e.get("category")]
    if mismatches:
        logger.info(f"\n  Top Disagreements (Auto → Gemini):")
        mismatch_pairs = Counter(
            (e.get("old_label"), e.get("category")) for e in mismatches
        )
        for (auto, gemini), count in mismatch_pairs.most_common(10):
            logger.info(f"    {auto:15s} → {gemini:15s}: {count:3d} cases")

    # Mobilization agreement (if available)
    mob_paired = [e for e in examples if e.get("mobilization")]
    if mob_paired:
        # Check if we can infer old mobilization from somewhere
        logger.info(f"\nMobilization labels: {len(mob_paired)} examples")

    return {
        "paired_examples": len(paired),
        "category_agreement": round(cat_accuracy, 4),
        "mismatches": len(mismatches),
    }


def generate_training_data(
    examples: list[dict], output_dir: Path | None = None
) -> Path:
    """Generate training data from Gemini labels.

    Extracts text+features and creates train/test split.
    """
    if output_dir is None:
        output_dir = ROOT / "data" / "gemini_training"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to high-confidence labels (gemini + auto)
    labeled = [
        e
        for e in examples
        if e.get("label_confidence") in ("gemini", "auto")
        and e.get("category")
        and e.get("mobilization")
    ]

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING DATA GENERATION")
    logger.info("=" * 60)
    logger.info(f"Examples with complete labels: {len(labeled)}")

    # Save training data with labels
    output_file = output_dir / "labeled_examples.jsonl"
    with output_file.open("w") as f:
        for ex in labeled:
            record = {
                "id": ex.get("id"),
                "text": ex.get("text"),
                "category": ex.get("category"),
                "mobilization": ex.get("mobilization"),
                "source": ex.get("label_confidence"),  # gemini or auto
            }
            f.write(json.dumps(record) + "\n")

    logger.info(f"Saved {len(labeled)} examples to {output_file}")
    logger.info(f"Category distribution:")
    cat_dist = Counter(e.get("category") for e in labeled)
    for cat, count in sorted(cat_dist.items()):
        logger.info(f"  {cat:15s}: {count:4d}")

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and retrain with Gemini labels")
    parser.add_argument(
        "--skip-retrain",
        action="store_true",
        help="Skip retraining, just evaluate baseline",
    )
    args = parser.parse_args()

    # Load data
    examples = load_eval_data()
    logger.info(f"Loaded {len(examples)} labeled examples")

    # Step 1: Analyze label confidence distribution
    confidence_stats = analyze_label_agreement(examples)

    # Step 2: Compare label sources
    comparison = compare_label_sources(examples)

    # Step 3: Evaluate baseline on Gemini labels
    baseline_results = {
        "category": eval_category_baseline(examples),
        "mobilization": eval_mobilization_baseline(examples),
    }

    # Step 4: Generate training data
    train_data_dir = generate_training_data(examples)

    # Compile results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "confidence_stats": confidence_stats,
        "label_comparison": comparison,
        "baseline_results": baseline_results,
        "training_data_dir": str(train_data_dir),
    }

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "gemini_eval_baseline.json"
    with results_file.open("w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Results saved to {results_file}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Gemini labels:           {confidence_stats['gemini']:4d}")
    logger.info(f"Category baseline F1:    {baseline_results['category']['macro_f1']:.4f}")
    logger.info(f"Mobilization baseline F1: {baseline_results['mobilization']['macro_f1']:.4f}")
    logger.info(f"Label agreement (auto):  {100*comparison['category_agreement']:.1f}%")


if __name__ == "__main__":
    main()
