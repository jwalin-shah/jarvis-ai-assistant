#!/usr/bin/env python3
"""Score trigger classifier against a manually labeled gold set.

Usage:
    uv run python -m scripts.score_trigger_gold_set --input gold_trigger_labels.jsonl
    uv run python -m scripts.score_trigger_gold_set --input gold_trigger_labels.jsonl \
        --mode structural --min-confidence 0.9
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

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


def _valid_labels() -> set[str]:
    return {t.value for t in TriggerType}


def _score(
    input_path: Path,
    mode: str,
    min_confidence: float,
    require_label: bool,
) -> dict[str, object]:
    classifier = get_trigger_classifier()
    valid_labels = _valid_labels()

    total = 0
    labeled = 0
    covered = 0
    correct = 0

    per_label_counts = Counter()
    per_label_correct = Counter()
    per_label_predicted = Counter()
    confusions: dict[str, Counter] = defaultdict(Counter)

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            label = _normalize_label(row.get("label"))
            text = (row.get("trigger_text") or "").strip()

            if not text:
                continue

            if label is None:
                if require_label:
                    continue
            elif label not in valid_labels:
                continue
            else:
                labeled += 1

            result = classifier.classify(text, use_centroid=(mode == "hybrid"))

            if result.confidence < min_confidence:
                continue
            if mode == "structural" and not result.method.startswith("structural"):
                continue

            covered += 1
            prediction = result.trigger_type.value
            per_label_predicted[prediction] += 1

            if label is None:
                continue
            per_label_counts[label] += 1
            confusions[label][prediction] += 1

            if prediction == label:
                correct += 1
                per_label_correct[label] += 1

    accuracy = correct / covered if covered else 0.0
    coverage = covered / labeled if labeled else 0.0

    per_label_metrics = {}
    for label in sorted(valid_labels):
        tp = per_label_correct[label]
        total_label = per_label_counts[label]
        predicted_label = per_label_predicted[label]

        precision = tp / predicted_label if predicted_label else 0.0
        recall = tp / total_label if total_label else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_label_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": total_label,
        }

    return {
        "total_rows": total,
        "labeled_rows": labeled,
        "covered_rows": covered,
        "coverage": coverage,
        "accuracy_on_covered": accuracy,
        "per_label": per_label_metrics,
        "confusions": {label: dict(counts) for label, counts in confusions.items()},
    }


def _print_summary(results: dict[str, object]) -> None:
    print("=" * 70)
    print("TRIGGER CLASSIFIER SCORECARD")
    print("=" * 70)
    print(f"Rows total:     {results['total_rows']}")
    print(f"Rows labeled:   {results['labeled_rows']}")
    print(f"Rows covered:   {results['covered_rows']}")
    print(f"Coverage:       {results['coverage']:.3f}")
    print(f"Accuracy@cov:   {results['accuracy_on_covered']:.3f}")

    print("\nPer-label metrics (precision/recall/f1/support):")
    per_label = results["per_label"]
    for label in sorted(per_label.keys()):
        metrics = per_label[label]
        print(
            f"  {label:16}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}"
            f"  F1={metrics['f1']:.3f}  n={metrics['support']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score trigger classifier on gold labels")
    parser.add_argument("--input", type=Path, required=True, help="Gold JSONL file")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "structural"],
        default="hybrid",
        help="Classifier mode (default: hybrid)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.85,
        help="Minimum confidence for coverage (default: 0.85)",
    )
    parser.add_argument(
        "--allow-unlabeled",
        action="store_true",
        help="Include rows without labels in coverage stats",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional JSON output for metrics",
    )
    args = parser.parse_args()

    results = _score(
        input_path=args.input,
        mode=args.mode,
        min_confidence=args.min_confidence,
        require_label=not args.allow_unlabeled,
    )
    _print_summary(results)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
