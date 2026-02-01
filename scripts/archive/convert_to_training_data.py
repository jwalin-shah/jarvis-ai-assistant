#!/usr/bin/env python3
"""Convert labeled trigger data to training format.

Converts fine-grained labels to merged classes for SVM training and
exports in the format expected by the trigger classifier.

Usage:
    uv run python -m scripts.convert_to_training_data \
        --input results/trigger_eval_balanced_40.jsonl \
        --output data/trigger_training.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

# Mapping from fine-grained to merged classes (for SVM)
FINE_TO_MERGED = {
    "greeting": "acknowledgment",
    "ack": "acknowledgment",
    "invitation": "action",
    "request": "action",
    "yn_question": "question",
    "info_question": "question",
    "good_news": "emotional",
    "bad_news": "emotional",
    "reaction": "emotional",
    "statement": "statement",
}


def load_labeled_data(paths: list[Path]) -> list[dict[str, Any]]:
    """Load labeled data from multiple JSONL files."""
    data = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if item.get("label"):
                        data.append(item)
    return data


def convert_to_training_format(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert labeled data to training format with merged classes."""
    training_data = []
    
    for item in data:
        fine_label = item["label"]
        merged_label = FINE_TO_MERGED.get(fine_label)
        
        if not merged_label:
            continue
        
        training_item = {
            "text": item["trigger_text"],
            "fine_label": fine_label,
            "merged_label": merged_label,
            "context": item.get("context_text", ""),
            "is_group": item.get("is_group", False),
            "source": item.get("pair_id"),
        }
        training_data.append(training_item)
    
    return training_data


def export_training_data(data: list[dict[str, Any]], output_path: Path) -> None:
    """Export training data to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    
    print(f"Exported {len(data)} training examples to {output_path}")


def print_statistics(data: list[dict[str, Any]]) -> None:
    """Print distribution statistics."""
    fine_dist = Counter(item["fine_label"] for item in data)
    merged_dist = Counter(item["merged_label"] for item in data)
    
    print("\nFine-grained label distribution:")
    for label, count in sorted(fine_dist.items(), key=lambda x: -x[1]):
        pct = count / len(data) * 100
        print(f"  {label:15s}: {count:4d} ({pct:5.1f}%)")
    
    print("\nMerged class distribution (for SVM):")
    for label, count in sorted(merged_dist.items(), key=lambda x: -x[1]):
        pct = count / len(data) * 100
        print(f"  {label:15s}: {count:4d} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        default=[
            Path("results/trigger_eval_balanced.jsonl"),
            Path("results/trigger_eval_balanced_40.jsonl"),
        ],
        help="Input labeled JSONL files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/trigger_training.jsonl"),
        help="Output training data path",
    )
    args = parser.parse_args()
    
    print("Loading labeled data...")
    data = load_labeled_data(args.input)
    print(f"Loaded {len(data)} labeled examples")
    
    print("\nConverting to training format...")
    training_data = convert_to_training_format(data)
    
    print_statistics(training_data)
    
    export_training_data(training_data, args.output)
    print("\nDone!")


if __name__ == "__main__":
    main()
