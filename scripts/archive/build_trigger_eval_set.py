#!/usr/bin/env python3
"""Build a stratified evaluation set from labeled triggers.

Usage:
    uv run python -m scripts.build_trigger_eval_set \
        --input ~/.jarvis/gold_trigger_labels_500.jsonl \
        --output results/trigger_eval_balanced.jsonl \
        --per-label 25
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def _load_rows(path: Path) -> dict[str, list[dict[str, object]]]:
    by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            label = (row.get("label") or "").strip().lower()
            if not label:
                continue
            by_label[label].append(row)
    return by_label


def build_eval_set(
    input_path: Path,
    output_path: Path,
    per_label: int,
    seed: int,
) -> dict[str, int]:
    rng = random.Random(seed)
    by_label = _load_rows(input_path)

    selected: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for label in sorted(by_label.keys()):
        rows = by_label[label]
        take = min(per_label, len(rows))
        counts[label] = take
        selected.extend(rng.sample(rows, take))

    rng.shuffle(selected)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a stratified trigger eval set")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL with labels")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument(
        "--per-label",
        type=int,
        default=25,
        help="Rows per label (default: 25)",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed (default: 13)")
    args = parser.parse_args()

    counts = build_eval_set(
        input_path=args.input,
        output_path=args.output,
        per_label=args.per_label,
        seed=args.seed,
    )

    total = sum(counts.values())
    print(f"Wrote {total} rows to {args.output}")
    print("Per-label counts:")
    for label in sorted(counts.keys()):
        print(f"  {label:16} {counts[label]}")


if __name__ == "__main__":
    main()
