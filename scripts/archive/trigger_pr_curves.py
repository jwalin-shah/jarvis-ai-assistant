#!/usr/bin/env python3
"""Compute per-class precision/recall curves for trigger classifier.

Usage:
    uv run python -m scripts.trigger_pr_curves \
        --input ~/.jarvis/gold_trigger_labels_500.jsonl \
        --output results/trigger_pr_curves.json
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
    valid = set(_valid_labels())
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            label = _normalize_label(row.get("label"))
            text = (row.get("trigger_text") or "").strip()
            if not text or label is None or label not in valid:
                continue
            rows.append({"label": label, "text": text})
    return rows


def _compute_curves(
    rows: list[dict[str, Any]],
    thresholds: list[float],
    mode: str,
) -> dict[str, list[dict[str, Any]]]:
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

    support = Counter(item["label"] for item in predictions)

    curves: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
    for threshold in thresholds:
        per_label_tp = Counter()
        per_label_predicted = Counter()

        for item in predictions:
            if item["confidence"] < threshold:
                continue
            if mode == "structural" and not item["method"].startswith("structural"):
                continue
            pred = item["pred"]
            per_label_predicted[pred] += 1
            if pred == item["label"]:
                per_label_tp[pred] += 1

        for label in labels:
            predicted = per_label_predicted[label]
            tp = per_label_tp[label]
            denom = support[label]
            precision = tp / predicted if predicted else 0.0
            recall = tp / denom if denom else 0.0
            curves[label].append(
                {
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "support": denom,
                    "predicted": predicted,
                    "true_positive": tp,
                }
            )

    return curves


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-class precision/recall curves")
    parser.add_argument("--input", type=Path, required=True, help="Gold JSONL file")
    parser.add_argument(
        "--thresholds",
        type=str,
        default=",".join(f"{i / 100:.2f}" for i in range(0, 100, 5)),
        help="Comma-separated thresholds (default: 0.00..0.95 step 0.05)",
    )
    parser.add_argument(
        "--mode",
        choices=["hybrid", "structural"],
        default="hybrid",
        help="Classifier mode (default: hybrid)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file")
    args = parser.parse_args()

    thresholds = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]
    rows = _load_rows(args.input)
    curves = _compute_curves(rows=rows, thresholds=thresholds, mode=args.mode)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(curves, indent=2), encoding="utf-8")
    print(f"Wrote curves to {args.output}")


if __name__ == "__main__":
    main()
