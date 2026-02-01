#!/usr/bin/env python3
"""Prepare trigger classifier training data.

Converts corrected data to both fine-grained (10 label) and consolidated (7 label) formats.

Usage:
    uv run python -m scripts.prepare_classifier_data
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

# Paths
INPUT_FILE = Path("data/trigger_training_corrected.jsonl")
OUTPUT_FINE = Path("data/trigger_training_10label.jsonl")
OUTPUT_CONSOLIDATED = Path("data/trigger_training_7label.jsonl")

# Mapping from fine-grained to consolidated labels
CONSOLIDATION_MAP = {
    # Fine-grained -> consolidated
    "invitation": "commitment",
    "request": "request",  # Keep separate - different response types
    "yn_question": "question",
    "info_question": "question",
    "good_news": "reaction",
    "bad_news": "reaction",
    "reaction": "reaction",
    "statement": "statement",
    "greeting": "greeting",
    "ack": "ack",
}


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    samples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def save_jsonl(samples: list[dict], path: Path) -> None:
    """Save to JSONL file."""
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def main():
    print(f"Loading data from {INPUT_FILE}...")
    samples = load_jsonl(INPUT_FILE)
    print(f"Loaded {len(samples)} samples")

    # Original distribution
    print("\nOriginal label distribution:")
    orig_dist = Counter(s.get("label") for s in samples)
    for label, count in orig_dist.most_common():
        print(f"  {label}: {count}")

    # Create fine-grained version (same as input, just renamed)
    fine_samples = []
    for sample in samples:
        new_sample = {
            "text": sample.get("text", ""),
            "label": sample.get("label", ""),
        }
        fine_samples.append(new_sample)

    print(f"\nSaving fine-grained (10 label) to {OUTPUT_FINE}...")
    save_jsonl(fine_samples, OUTPUT_FINE)

    # Create consolidated version
    consolidated_samples = []
    for sample in samples:
        orig_label = sample.get("label", "")
        cons_label = CONSOLIDATION_MAP.get(orig_label, orig_label)
        new_sample = {
            "text": sample.get("text", ""),
            "label": cons_label,
            "original_label": orig_label,
        }
        consolidated_samples.append(new_sample)

    print(f"\nSaving consolidated (7 label) to {OUTPUT_CONSOLIDATED}...")
    save_jsonl(consolidated_samples, OUTPUT_CONSOLIDATED)

    # Consolidated distribution
    print("\nConsolidated label distribution:")
    cons_dist = Counter(s.get("label") for s in consolidated_samples)
    for label, count in cons_dist.most_common():
        print(f"  {label}: {count}")

    print(f"\nDone! Files created:")
    print(f"  - {OUTPUT_FINE}")
    print(f"  - {OUTPUT_CONSOLIDATED}")


if __name__ == "__main__":
    main()
