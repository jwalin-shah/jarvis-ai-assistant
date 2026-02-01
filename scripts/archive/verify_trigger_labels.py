#!/usr/bin/env python3
"""Verify quality of labeled trigger training data."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

VALID_LABELS = {
    "invitation", "request", "yn_question", "info_question",
    "good_news", "bad_news", "reaction", "greeting", "ack", "statement"
}

def load_labeled_data(path: Path) -> list[dict[str, Any]]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                if item.get("label"):
                    data.append(item)
    return data

def verify_file(path: Path) -> None:
    print(f"Verifying: {path}")
    print("=" * 60)
    
    data = load_labeled_data(path)
    total = sum(1 for _ in open(path))
    
    if not data:
        print("No labeled data found")
        return
    
    print(f"Total rows: {total}")
    print(f"Labeled: {len(data)}")
    print(f"Unlabeled: {total - len(data)}")
    
    dist = Counter(item["label"] for item in data)
    print("\nLabel Distribution:")
    for label, count in sorted(dist.items()):
        pct = count / len(data) * 100
        print(f"  {label:15s}: {count:4d} ({pct:5.1f}%)")
    
    # Check for invalid labels
    invalid = [item for item in data if item["label"] not in VALID_LABELS]
    if invalid:
        print(f"\nWARNING: {len(invalid)} items with invalid labels")
    
    print("\n✅ Verification complete")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path, help="Labeled JSONL file to verify")
    args = parser.parse_args()
    
    verify_file(args.file)

if __name__ == "__main__":
    main()
