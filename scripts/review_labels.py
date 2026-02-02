#!/usr/bin/env python3
"""Interactive label review tool for trigger and response data.

Finds confusing examples (where model prediction differs from label)
and lets you review/fix them.

Usage:
    # Review trigger labels (most confused examples first)
    uv run python -m scripts.review_labels --type trigger --limit 50

    # Review response labels
    uv run python -m scripts.review_labels --type response --limit 50

    # Review specific confusion pattern
    uv run python -m scripts.review_labels --type trigger --pattern "statement->reaction"

    # Just show stats, don't review
    uv run python -m scripts.review_labels --type trigger --stats-only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC


def load_trigger_data(path: Path) -> list[dict]:
    """Load trigger data with line numbers."""
    data = []
    with open(path) as f:
        for i, line in enumerate(f):
            if line.strip():
                row = json.loads(line)
                row['_line'] = i
                row['_text'] = row.get('text', '')
                row['_label'] = row.get('label', '').lower()
                data.append(row)
    return data


def load_response_data(path: Path) -> list[dict]:
    """Load response data with line numbers."""
    data = []
    with open(path) as f:
        for i, line in enumerate(f):
            if line.strip():
                row = json.loads(line)
                row['_line'] = i
                row['_text'] = row.get('response', '')
                row['_label'] = row.get('label', '').upper()
                data.append(row)
    return data


def get_predictions(data: list[dict], data_type: str) -> list[str]:
    """Get cross-validated predictions for all examples."""
    from jarvis.embedding_adapter import get_embedder

    texts = [d['_text'] for d in data]
    labels = [d['_label'] for d in data]

    print("Generating embeddings...")
    embedder = get_embedder()
    embeddings = embedder.encode(texts, normalize=True)

    print("Running cross-validation predictions...")
    if data_type == 'trigger':
        clf = SVC(kernel='rbf', C=5.0, gamma='scale', class_weight='balanced',
                  probability=True, random_state=42)
    else:
        clf = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced',
                  probability=True, random_state=42)

    predictions = cross_val_predict(clf, embeddings, labels, cv=5)
    return list(predictions)


def show_stats(data: list[dict], predictions: list[str]) -> dict[str, list[int]]:
    """Show confusion statistics and return indices by pattern."""
    confusion = defaultdict(list)

    for i, (d, pred) in enumerate(zip(data, predictions)):
        true_label = d['_label']
        if true_label != pred:
            pattern = f"{true_label}->{pred}"
            confusion[pattern].append(i)

    total_wrong = sum(len(v) for v in confusion.values())
    print(f"\nMisclassified: {total_wrong} / {len(data)} ({total_wrong/len(data)*100:.1f}%)")

    print(f"\n{'Pattern':<30} {'Count':>6} {'%':>6}")
    print("-" * 45)

    sorted_patterns = sorted(confusion.items(), key=lambda x: -len(x[1]))
    for pattern, indices in sorted_patterns[:15]:
        pct = len(indices) / total_wrong * 100
        print(f"{pattern:<30} {len(indices):>6} {pct:>5.1f}%")

    return confusion


def review_examples(
    data: list[dict],
    indices: list[int],
    predictions: list[str],
    data_type: str,
    limit: int = 50,
) -> list[dict]:
    """Interactively review examples and collect corrections."""

    if data_type == 'trigger':
        valid_labels = ['commitment', 'question', 'reaction', 'social', 'statement']
    else:
        valid_labels = ['AGREE', 'DECLINE', 'DEFER', 'QUESTION', 'REACTION', 'OTHER']

    print("\n" + "=" * 60)
    print("LABEL REVIEW")
    print("=" * 60)
    print(f"Reviewing {min(limit, len(indices))} examples")
    print(f"\nFor each example, enter:")
    print(f"  [Enter] = keep current label")
    print(f"  1-{len(valid_labels)} = change to that label")
    print(f"  s = skip")
    print(f"  q = quit")
    print(f"\nLabels: {', '.join(f'{i+1}={l}' for i, l in enumerate(valid_labels))}")
    print("=" * 60)

    corrections = []

    for count, idx in enumerate(indices[:limit], 1):
        d = data[idx]
        pred = predictions[idx]

        print(f"\n[{count}/{min(limit, len(indices))}] Line {d['_line']}")
        print(f"  Text: \"{d['_text'][:70]}{'...' if len(d['_text']) > 70 else ''}\"")
        print(f"  Current label: {d['_label']}")
        print(f"  Model predicts: {pred}")

        while True:
            response = input("  New label? ").strip().lower()

            if response == '':
                print("  → Keeping current label")
                break
            elif response == 's':
                print("  → Skipped")
                break
            elif response == 'q':
                print("\nQuitting review...")
                return corrections
            elif response.isdigit():
                num = int(response)
                if 1 <= num <= len(valid_labels):
                    new_label = valid_labels[num - 1]
                    if new_label != d['_label']:
                        corrections.append({
                            'line': d['_line'],
                            'text': d['_text'],
                            'old_label': d['_label'],
                            'new_label': new_label,
                        })
                        print(f"  → Changed: {d['_label']} → {new_label}")
                    else:
                        print("  → Same as current, no change")
                    break
                else:
                    print(f"  Invalid number. Enter 1-{len(valid_labels)}")
            else:
                print("  Invalid input. Enter number, s, q, or press Enter")

    return corrections


def save_corrections(corrections: list[dict], output_path: Path) -> None:
    """Save corrections to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(corrections, f, indent=2)
    print(f"\nSaved {len(corrections)} corrections to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Review and fix classifier labels")
    parser.add_argument(
        "--type",
        choices=["trigger", "response"],
        required=True,
        help="Which dataset to review"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max examples to review (default: 50)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Review specific confusion pattern (e.g., 'statement->reaction')"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics, don't review"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for corrections (default: results/<type>_corrections.json)"
    )
    args = parser.parse_args()

    # Load data
    if args.type == "trigger":
        data_path = Path("data/trigger_labeling.jsonl")
        data = load_trigger_data(data_path)
    else:
        data_path = Path("data/response_labeling.jsonl")
        data = load_response_data(data_path)

    print(f"Loaded {len(data)} examples from {data_path}")

    # Get predictions
    predictions = get_predictions(data, args.type)

    # Show stats
    confusion = show_stats(data, predictions)

    if args.stats_only:
        return

    # Get indices to review
    if args.pattern:
        pattern = args.pattern.lower() if args.type == "trigger" else args.pattern.upper()
        if pattern not in confusion:
            print(f"\nPattern '{pattern}' not found. Available patterns:")
            for p in sorted(confusion.keys()):
                print(f"  {p}")
            return
        indices = confusion[pattern]
    else:
        # Review most confused examples first
        all_indices = []
        for indices_list in confusion.values():
            all_indices.extend(indices_list)
        indices = all_indices

    # Review
    corrections = review_examples(data, indices, predictions, args.type, args.limit)

    # Save corrections
    if corrections:
        output_path = args.output or Path(f"results/{args.type}_corrections.json")
        save_corrections(corrections, output_path)

        print("\nTo apply corrections, you'll need to update the original JSONL file.")
        print("The corrections file contains line numbers and new labels.")


if __name__ == "__main__":
    main()
