#!/usr/bin/env python3
"""Threshold analysis for trigger classifier performance.

Usage:
    uv run python -m scripts.threshold_trigger_eval \
        --input ~/.jarvis/gold_trigger_labels_500.jsonl \
        --thresholds 0.85,0.6,0.5,0.3
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from jarvis.trigger_classifier import TriggerType, get_trigger_classifier


def _normalize_label(label: str | None) -> str | None:
    if label is None:
        return None
    clean = label.strip().lower()
    if clean in {"reaction_prompt", "reaction"}:
        return TriggerType.REACTION_PROMPT.value
    if clean in {"ack", "acknowledge", "acknowledgment"}:
        return TriggerType.ACKNOWLEDGMENT.value
    if clean in {"good news", "good_news"}:
        return TriggerType.GOOD_NEWS.value
    if clean in {"bad news", "bad_news"}:
        return TriggerType.BAD_NEWS.value
    if clean in {"yn", "yn_question", "yes_no_question"}:
        return TriggerType.YN_QUESTION.value
    if clean in {"info", "info_question", "wh_question"}:
        return TriggerType.INFO_QUESTION.value
    if clean in {"invite", "invitation"}:
        return TriggerType.INVITATION.value
    return clean


def _valid_labels() -> list[str]:
    return [t.value for t in TriggerType]


def _load_rows(input_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            label = _normalize_label(row.get("label"))
            text = (row.get("trigger_text") or "").strip()
            if not text or label is None:
                continue
            if label not in _valid_labels():
                continue
            rows.append({"label": label, "text": text})
    return rows


def _compute_metrics(
    rows: list[dict[str, Any]],
    thresholds: list[float],
    mode: str,
) -> list[dict[str, Any]]:
    classifier = get_trigger_classifier()
    labels = _valid_labels()

    predictions = []
    for row in rows:
        result = classifier.classify(row["text"], use_centroid=(mode == "hybrid"))
        predictions.append(
            {
                "label": row["label"],
                "pred": result.trigger_type.value,
                "confidence": result.confidence,
                "method": result.method,
            }
        )

    results = []
    for threshold in thresholds:
        total = len(predictions)
        covered = 0
        correct = 0

        per_label_counts = Counter()
        per_label_correct = Counter()
        per_label_predicted = Counter()

        for item in predictions:
            label = item["label"]
            per_label_counts[label] += 1

            if item["confidence"] < threshold:
                continue
            if mode == "structural" and not item["method"].startswith("structural"):
                continue

            covered += 1
            pred = item["pred"]
            per_label_predicted[pred] += 1
            if pred == label:
                correct += 1
                per_label_correct[label] += 1

        accuracy = correct / covered if covered else 0.0
        coverage = covered / total if total else 0.0

        per_label = {}
        for label in labels:
            tp = per_label_correct[label]
            predicted = per_label_predicted[label]
            support = per_label_counts[label]
            precision = tp / predicted if predicted else 0.0
            recall = tp / support if support else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            per_label[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }

        supports = [per_label[label]["support"] for label in per_label]
        total_support = sum(supports)
        macro_f1 = (
            sum(per_label[label]["f1"] for label in per_label if per_label[label]["support"] > 0)
            / sum(1 for label in per_label if per_label[label]["support"] > 0)
            if any(per_label[label]["support"] > 0 for label in per_label)
            else 0.0
        )
        weighted_f1 = (
            sum(per_label[label]["f1"] * per_label[label]["support"] for label in per_label)
            / total_support
            if total_support
            else 0.0
        )

        results.append(
            {
                "threshold": threshold,
                "coverage": coverage,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "per_label": per_label,
            }
        )

    return results


def _print_summary(results: list[dict[str, Any]]) -> None:
    print("=" * 70)
    print("THRESHOLD SUMMARY")
    print("=" * 70)
    print("Threshold  Coverage  Accuracy  MacroF1  WeightedF1")
    for row in results:
        print(
            f"{row['threshold']:>8.2f}  {row['coverage'] * 100:>7.1f}%"
            f"  {row['accuracy'] * 100:>8.1f}%  {row['macro_f1']:.3f}   {row['weighted_f1']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold analysis for trigger classifier")
    parser.add_argument("--input", type=Path, required=True, help="Gold JSONL file")
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.85,0.6,0.5,0.3",
        help="Comma-separated thresholds (default: 0.85,0.6,0.5,0.3)",
    )
    parser.add_argument(
        "--mode",
        choices=["hybrid", "structural"],
        default="hybrid",
        help="Classifier mode (default: hybrid)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional JSON output for metrics",
    )
    args = parser.parse_args()

    thresholds = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]
    rows = _load_rows(args.input)
    results = _compute_metrics(rows=rows, thresholds=thresholds, mode=args.mode)

    _print_summary(results)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
