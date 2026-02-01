#!/usr/bin/env python3
"""Extract misclassified examples for target trigger classes.

Usage:
    uv run python -m scripts.trigger_error_analysis \
        --input ~/.jarvis/gold_trigger_labels_500.jsonl \
        --targets good_news,greeting \
        --limit 25 \
        --output results/trigger_errors.json
"""

from __future__ import annotations

import argparse
import json
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


def _valid_labels() -> set[str]:
    return {t.value for t in TriggerType}


def _load_rows(input_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    valid = _valid_labels()
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            label = _normalize_label(row.get("label"))
            text = (row.get("trigger_text") or "").strip()
            if not text or label is None or label not in valid:
                continue
            rows.append(
                {
                    "pair_id": row.get("pair_id"),
                    "label": label,
                    "text": text,
                }
            )
    return rows


def _collect_errors(
    rows: list[dict[str, Any]],
    targets: set[str],
    mode: str,
    limit: int,
) -> list[dict[str, Any]]:
    classifier = get_trigger_classifier()
    errors = []
    for row in rows:
        if row["label"] not in targets:
            continue
        result = classifier.classify(row["text"], use_centroid=(mode == "hybrid"))
        pred = result.trigger_type.value
        if pred == row["label"]:
            continue
        errors.append(
            {
                "pair_id": row.get("pair_id"),
                "label": row["label"],
                "prediction": pred,
                "confidence": result.confidence,
                "method": result.method,
                "trigger_text": row["text"],
            }
        )

    errors.sort(key=lambda item: item["confidence"], reverse=True)
    return errors[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract misclassified trigger examples")
    parser.add_argument("--input", type=Path, required=True, help="Gold JSONL file")
    parser.add_argument(
        "--targets",
        type=str,
        required=True,
        help="Comma-separated target labels (e.g., good_news,greeting)",
    )
    parser.add_argument(
        "--mode",
        choices=["hybrid", "structural"],
        default="hybrid",
        help="Classifier mode (default: hybrid)",
    )
    parser.add_argument("--limit", type=int, default=25, help="Examples per target set")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file")
    args = parser.parse_args()

    targets = {t.strip() for t in args.targets.split(",") if t.strip()}
    rows = _load_rows(args.input)
    errors = _collect_errors(rows=rows, targets=targets, mode=args.mode, limit=args.limit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(errors, indent=2), encoding="utf-8")
    print(f"Wrote {len(errors)} errors to {args.output}")


if __name__ == "__main__":
    main()
