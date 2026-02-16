#!/usr/bin/env python3
"""Evaluate classifier stages against the labeled eval dataset.

Runs mobilization, category, and replyability classifiers independently
and reports per-class precision/recall/F1 + confusion matrices.

Usage:
    uv run python scripts/eval_classifiers.py
    uv run python scripts/eval_classifiers.py --stages mobilization category
    uv run python scripts/eval_classifiers.py --output evals/results/classifier_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from jarvis.utils.logging import setup_script_logging

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "eval_classifiers.log"

logger = logging.getLogger(__name__)

EVAL_PATH = ROOT / "evals" / "data" / "pipeline_eval.jsonl"
DEFAULT_OUTPUT = ROOT / "evals" / "results" / "classifier_eval.json"

ALL_STAGES = ("mobilization", "category", "replyability")


def load_eval_data() -> list[dict]:
    """Load eval dataset, filtering to labeled examples."""
    if not EVAL_PATH.exists():
        logger.error("Eval dataset not found: %s", EVAL_PATH)
        logger.error("Run scripts/build_eval_dataset.py first.")
        sys.exit(1)

    examples = []
    for line in EVAL_PATH.open():
        line = line.strip()
        if line:
            examples.append(json.loads(line))
    return examples


def eval_mobilization(examples: list[dict]) -> dict:
    """Evaluate mobilization classifier."""
    from jarvis.classifiers.response_mobilization import classify_response_pressure

    # Filter to examples with mobilization labels
    labeled = [e for e in examples if e.get("mobilization")]
    if not labeled:
        logger.warning("No mobilization labels found, skipping")
        return {"error": "no_labels"}

    logger.info("Evaluating mobilization on %d examples...", len(labeled))

    y_true = []
    y_pred = []
    latencies: list[float] = []

    for i, ex in enumerate(labeled):
        start = time.perf_counter()
        result = classify_response_pressure(ex["text"])
        latencies.append((time.perf_counter() - start) * 1000)

        y_true.append(ex["mobilization"].upper())
        y_pred.append(result.pressure.value.upper())

        if (i + 1) % 500 == 0:
            logger.info("  %d/%d evaluated", i + 1, len(labeled))

    return _compute_metrics(y_true, y_pred, latencies, "mobilization")


def eval_category(examples: list[dict]) -> dict:
    """Evaluate category classifier."""
    from jarvis.classifiers.category_classifier import classify_category

    labeled = [e for e in examples if e.get("category")]
    if not labeled:
        logger.warning("No category labels found, skipping")
        return {"error": "no_labels"}

    logger.info("Evaluating category on %d examples...", len(labeled))

    y_true = []
    y_pred = []
    latencies: list[float] = []

    for i, ex in enumerate(labeled):
        start = time.perf_counter()
        result = classify_category(ex["text"], context=ex.get("thread", []))
        latencies.append((time.perf_counter() - start) * 1000)

        y_true.append(ex["category"])
        y_pred.append(result.category)

        if (i + 1) % 500 == 0:
            logger.info("  %d/%d evaluated", i + 1, len(labeled))

    return _compute_metrics(y_true, y_pred, latencies, "category")


def eval_replyability(examples: list[dict]) -> dict:
    """Evaluate replyability gate."""
    # Only evaluate on examples with should_reply labels
    labeled = [e for e in examples if e.get("should_reply") is not None]
    if not labeled:
        logger.warning("No replyability labels found, skipping")
        return {"error": "no_labels"}

    logger.info("Evaluating replyability on %d examples...", len(labeled))

    try:
        from jarvis.classifiers.response_mobilization import classify_response_pressure
        from jarvis.reply_service import ReplyService

        service = ReplyService.__new__(ReplyService)
        # Check if _check_replyability exists
        if not hasattr(service, "_check_replyability"):
            logger.warning("ReplyService._check_replyability not found, skipping")
            return {"error": "not_implemented"}
    except ImportError as e:
        logger.warning("Cannot import reply service: %s", e)
        return {"error": str(e)}

    y_true = []
    y_pred = []
    latencies: list[float] = []

    for i, ex in enumerate(labeled):
        start = time.perf_counter()
        mobilization = classify_response_pressure(ex["text"])

        # Build minimal context for replyability check
        has_contact = ex.get("contact_name") is not None
        should, _reason = service._check_replyability(
            incoming=ex["text"],
            mobilization=mobilization,
            has_contact=has_contact,
        )
        latencies.append((time.perf_counter() - start) * 1000)

        y_true.append(ex["should_reply"])
        y_pred.append(should)

        if (i + 1) % 100 == 0:
            logger.info("  %d/%d evaluated", i + 1, len(labeled))

    # Binary metrics
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "stage": "replyability",
        "n_examples": len(labeled),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "latency_ms": {
            "mean": round(float(np.mean(latencies)), 2),
            "p50": round(float(np.median(latencies)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
        },
    }


def _compute_metrics(
    y_true: list[str], y_pred: list[str], latencies: list[float], stage: str
) -> dict:
    """Compute per-class metrics and confusion matrix."""
    from sklearn.metrics import classification_report, confusion_matrix

    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Print summary table
    print(f"\n{'=' * 60}", flush=True)
    print(f"  {stage.upper()} Classification Report", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(classification_report(y_true, y_pred, labels=labels), flush=True)

    print("Confusion Matrix:", flush=True)
    header = "          " + "  ".join(f"{l:>10}" for l in labels)
    print(header, flush=True)
    for i, label in enumerate(labels):
        row = f"{label:>10}" + "  ".join(f"{cm[i][j]:>10}" for j in range(len(labels)))
        print(row, flush=True)
    print(flush=True)

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


def main() -> None:
    setup_script_logging("eval_classifiers")
    logger.info("Starting eval_classifiers.py")
    parser = argparse.ArgumentParser(description="Evaluate classifiers")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=ALL_STAGES,
        default=list(ALL_STAGES),
        help="Which stages to evaluate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path",
    )
    args = parser.parse_args()

    examples = load_eval_data()
    logger.info("Loaded %d eval examples", len(examples))

    results: dict[str, dict] = {}

    stage_funcs = {
        "mobilization": eval_mobilization,
        "category": eval_category,
        "replyability": eval_replyability,
    }

    for stage in args.stages:
        logger.info("--- Evaluating %s ---", stage)
        results[stage] = stage_funcs[stage](examples)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
