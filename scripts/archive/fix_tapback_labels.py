#!/usr/bin/env python3
"""Fix inconsistent tapback labels in trigger_labeling.jsonl.

Tapbacks (Liked, Loved, Laughed at, Emphasized, Questioned, Disliked)
should all be labeled consistently as SOCIAL.

Usage:
    uv run python -m scripts.fix_tapback_labels --dry-run  # Preview changes
    uv run python -m scripts.fix_tapback_labels --apply    # Apply changes
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


TAPBACK_PATTERN = re.compile(
    r'^(Liked|Loved|Laughed at|Emphasized|Questioned|Disliked)\s+["\u201c\u201d]',
    re.I
)

TARGET_LABEL = "social"


def main():
    parser = argparse.ArgumentParser(description="Fix tapback labels")
    parser.add_argument("--input", type=Path, default=Path("data/trigger_labeling.jsonl"))
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--apply", action="store_true", help="Apply changes")
    args = parser.parse_args()

    if not args.dry_run and not args.apply:
        print("Must specify --dry-run or --apply")
        return

    # Load data
    data = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # Find tapbacks and their current labels
    changes = []
    current_labels = Counter()

    for row in data:
        text = row.get("text", "")
        label = row.get("label")

        if label and TAPBACK_PATTERN.match(text):
            current_labels[label] += 1
            if label != TARGET_LABEL:
                changes.append({
                    "id": row.get("id"),
                    "text": text[:50] + "..." if len(text) > 50 else text,
                    "old_label": label,
                    "new_label": TARGET_LABEL,
                })

    print("=" * 60)
    print("TAPBACK LABEL ANALYSIS")
    print("=" * 60)
    print(f"\nCurrent distribution of tapback labels:")
    for label, count in current_labels.most_common():
        marker = " ✓" if label == TARGET_LABEL else " ← will change"
        print(f"  {label}: {count}{marker}")

    print(f"\nTotal tapbacks: {sum(current_labels.values())}")
    print(f"Changes needed: {len(changes)}")

    if changes:
        print(f"\nSample changes (first 10):")
        for c in changes[:10]:
            print(f"  ID {c['id']}: {c['old_label']} → {c['new_label']}")
            print(f"    \"{c['text']}\"")

    if args.apply and changes:
        print("\n" + "=" * 60)
        print("APPLYING CHANGES")
        print("=" * 60)

        # Apply changes
        id_to_new_label = {c["id"]: c["new_label"] for c in changes}
        for row in data:
            if row.get("id") in id_to_new_label:
                row["label"] = id_to_new_label[row["id"]]

        # Write back
        with open(args.input, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

        print(f"Applied {len(changes)} changes to {args.input}")

        # Verify
        new_labels = Counter()
        for row in data:
            text = row.get("text", "")
            label = row.get("label")
            if label and TAPBACK_PATTERN.match(text):
                new_labels[label] += 1

        print(f"\nNew distribution:")
        for label, count in new_labels.most_common():
            print(f"  {label}: {count}")

    elif args.dry_run:
        print("\n[DRY RUN - no changes made. Use --apply to apply changes]")


if __name__ == "__main__":
    main()
