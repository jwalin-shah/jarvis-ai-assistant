#!/usr/bin/env python3
"""Compare 5-label vs 12-label classification schemes using zero-shot.

This script tests both label schemes on real messages to determine:
1. Which scheme has better class balance (no label dominates)
2. Which produces more intuitive classifications
3. Which has fewer ambiguous/wrong cases

Usage:
    uv run python -m scripts.compare_label_schemes
    uv run python -m scripts.compare_label_schemes --input data/messages.jsonl --limit 500
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from jarvis.classifiers.zeroshot import ZeroShotClassifier

# 5-label trigger scheme (current)
LABELS_5 = ["commitment", "question", "reaction", "social", "statement"]

# 12-label research-backed scheme
LABELS_12 = [
    "question",
    "statement",
    "request",
    "acknowledgment",
    "agreement",
    "disagreement",
    "greeting",
    "closing",
    "emotional",
    "opinion",
    "backchannel",
    "other",
]

# Mapping from 12-label to 5-label (for comparison)
LABEL_12_TO_5 = {
    "question": "question",
    "statement": "statement",
    "request": "commitment",
    "acknowledgment": "social",
    "agreement": "social",
    "disagreement": "statement",  # Could be social, but disagreement has content
    "greeting": "social",
    "closing": "social",
    "emotional": "reaction",
    "opinion": "statement",
    "backchannel": "social",
    "other": "statement",
}


@dataclass
class LabelResult:
    """Result from labeling a text with both schemes."""

    text: str
    label_5: str
    score_5: float
    label_12: str
    score_12: float
    label_12_mapped: str  # 12-label mapped to 5-label


def load_messages(path: Path, limit: int | None = None) -> list[str]:
    """Load messages from a JSONL file."""
    messages = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text", "").strip()
            if text:
                messages.append(text)

    if limit and len(messages) > limit:
        random.seed(42)
        messages = random.sample(messages, limit)

    return messages


def compare_schemes(messages: list[str]) -> list[LabelResult]:
    """Compare both label schemes on a set of messages."""
    print("Loading zero-shot classifiers...")

    clf_5 = ZeroShotClassifier(labels=LABELS_5, use_descriptions=True)
    clf_12 = ZeroShotClassifier(labels=LABELS_12, use_descriptions=True)

    results = []
    total = len(messages)

    print(f"Classifying {total} messages with both schemes...")

    for i, text in enumerate(messages):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{total}")

        label_5, score_5 = clf_5.classify(text)
        label_12, score_12 = clf_12.classify(text)
        label_12_mapped = LABEL_12_TO_5[label_12]

        results.append(
            LabelResult(
                text=text,
                label_5=label_5,
                score_5=score_5,
                label_12=label_12,
                score_12=score_12,
                label_12_mapped=label_12_mapped,
            )
        )

    return results


def analyze_results(results: list[LabelResult]) -> dict:
    """Analyze the comparison results."""
    # Count distributions
    dist_5 = Counter(r.label_5 for r in results)
    dist_12 = Counter(r.label_12 for r in results)
    dist_12_mapped = Counter(r.label_12_mapped for r in results)

    # Count agreements between 5-label and mapped 12-label
    agreements = sum(1 for r in results if r.label_5 == r.label_12_mapped)
    agreement_rate = agreements / len(results)

    # Average confidence scores
    avg_score_5 = sum(r.score_5 for r in results) / len(results)
    avg_score_12 = sum(r.score_12 for r in results) / len(results)

    # Find disagreements for manual review
    disagreements = [r for r in results if r.label_5 != r.label_12_mapped]

    # Calculate entropy (measure of class balance)
    def entropy(dist: Counter) -> float:
        import math

        total = sum(dist.values())
        return -sum(
            (c / total) * math.log2(c / total) for c in dist.values() if c > 0
        )

    entropy_5 = entropy(dist_5)
    entropy_12 = entropy(dist_12)

    return {
        "total": len(results),
        "agreement_rate": agreement_rate,
        "avg_score_5": avg_score_5,
        "avg_score_12": avg_score_12,
        "entropy_5": entropy_5,
        "entropy_12": entropy_12,
        "distribution_5": dict(dist_5.most_common()),
        "distribution_12": dict(dist_12.most_common()),
        "distribution_12_mapped": dict(dist_12_mapped.most_common()),
        "disagreements": disagreements,
    }


def print_analysis(analysis: dict) -> None:
    """Print analysis results in a human-readable format."""
    print("\n" + "=" * 70)
    print("LABEL SCHEME COMPARISON RESULTS")
    print("=" * 70)

    print(f"\nTotal messages: {analysis['total']}")
    print(f"Agreement rate (5-label vs mapped 12-label): {analysis['agreement_rate']:.1%}")

    print("\n--- 5-Label Scheme ---")
    print(f"Average confidence: {analysis['avg_score_5']:.3f}")
    print(f"Entropy (class balance): {analysis['entropy_5']:.3f}")
    print("Distribution:")
    for label, count in analysis["distribution_5"].items():
        pct = count / analysis["total"] * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:<12} {count:>5} ({pct:>5.1f}%) {bar}")

    print("\n--- 12-Label Scheme ---")
    print(f"Average confidence: {analysis['avg_score_12']:.3f}")
    print(f"Entropy (class balance): {analysis['entropy_12']:.3f}")
    print("Distribution:")
    for label, count in analysis["distribution_12"].items():
        pct = count / analysis["total"] * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:<15} {count:>5} ({pct:>5.1f}%) {bar}")

    print("\n--- 12-Label Mapped to 5-Label ---")
    print("Distribution:")
    for label, count in analysis["distribution_12_mapped"].items():
        pct = count / analysis["total"] * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:<12} {count:>5} ({pct:>5.1f}%) {bar}")

    # Show sample disagreements
    disagreements = analysis["disagreements"]
    if disagreements:
        print("\n--- Sample Disagreements (first 20) ---")
        print(f"{'Text':<40} | 5-label    | 12-label (mapped)")
        print("-" * 70)
        for r in disagreements[:20]:
            text = r.text[:38] + ".." if len(r.text) > 40 else r.text
            print(f"{text:<40} | {r.label_5:<10} | {r.label_12} ({r.label_12_mapped})")

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    if analysis["entropy_12"] > analysis["entropy_5"]:
        print("- 12-label scheme has better class balance (higher entropy)")
    else:
        print("- 5-label scheme has better class balance (higher entropy)")

    if analysis["avg_score_12"] > analysis["avg_score_5"]:
        print("- 12-label scheme has higher average confidence")
    else:
        print("- 5-label scheme has higher average confidence")

    if analysis["agreement_rate"] < 0.7:
        print("- Low agreement rate suggests schemes capture different aspects")
        print("- Consider using 12-label internally and mapping to 5 for responses")
    else:
        print("- High agreement rate suggests both schemes work similarly")
        print("- 5-label scheme is simpler and recommended")


def main():
    parser = argparse.ArgumentParser(description="Compare label schemes")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/trigger_labeling.jsonl"),
        help="Input JSONL file with messages",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of messages to process (default: 200)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/label_scheme_comparison.json"),
        help="Output JSON file for results",
    )
    args = parser.parse_args()

    # Load messages
    print(f"Loading messages from {args.input}...")
    messages = load_messages(args.input, limit=args.limit)
    print(f"Loaded {len(messages)} messages")

    # Compare schemes
    results = compare_schemes(messages)

    # Analyze
    analysis = analyze_results(results)
    print_analysis(analysis)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Convert LabelResult objects to dicts for JSON serialization
    output_data = {
        "total": analysis["total"],
        "agreement_rate": analysis["agreement_rate"],
        "avg_score_5": analysis["avg_score_5"],
        "avg_score_12": analysis["avg_score_12"],
        "entropy_5": analysis["entropy_5"],
        "entropy_12": analysis["entropy_12"],
        "distribution_5": analysis["distribution_5"],
        "distribution_12": analysis["distribution_12"],
        "distribution_12_mapped": analysis["distribution_12_mapped"],
        "disagreements": [
            {
                "text": r.text,
                "label_5": r.label_5,
                "score_5": r.score_5,
                "label_12": r.label_12,
                "score_12": r.score_12,
                "label_12_mapped": r.label_12_mapped,
            }
            for r in analysis["disagreements"]
        ],
    }
    args.output.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
