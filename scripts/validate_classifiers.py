#!/usr/bin/env python3
"""Manual validation script for trigger and response classifiers.

Samples random examples, shows predictions with confidence, and prompts
for human verification to calculate real-world accuracy.

Usage:
    # Validate trigger classifier
    uv run python -m scripts.validate_classifiers --classifier trigger --samples 50

    # Validate response classifier
    uv run python -m scripts.validate_classifiers --classifier response --samples 50

    # Use specific data file
    uv run python -m scripts.validate_classifiers --classifier trigger \
        --input data/trigger_labeling.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValidationResult:
    """Result from validating a single example."""

    text: str
    predicted: str
    confidence: float
    ground_truth: str | None
    human_verified: bool | None  # True=correct, False=wrong, None=skipped
    method: str


def load_trigger_data(path: Path) -> list[tuple[str, str]]:
    """Load trigger data (text, label) pairs."""
    data = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text", "").strip()
            label = row.get("label", "").lower()
            if text and label:
                data.append((text, label))
    return data


def load_response_data(path: Path) -> list[tuple[str, str]]:
    """Load response data (response, label) pairs."""
    data = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("response", "").strip()
            label = row.get("label", "").upper()
            if text and label:
                data.append((text, label))
    return data


def validate_trigger(samples: list[tuple[str, str]]) -> list[ValidationResult]:
    """Run trigger classifier on samples and prompt for validation."""
    from jarvis.classifiers.trigger_classifier import classify_trigger

    results = []
    total = len(samples)

    print("\n" + "=" * 70)
    print("TRIGGER CLASSIFIER VALIDATION")
    print("=" * 70)
    print("For each example, enter:")
    print("  y = prediction is correct")
    print("  n = prediction is wrong")
    print("  s = skip (unsure)")
    print("  q = quit early")
    print("=" * 70 + "\n")

    for i, (text, ground_truth) in enumerate(samples, 1):
        result = classify_trigger(text)

        print(f"\n[{i}/{total}] Text: {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"  Ground truth: {ground_truth}")
        print(f"  Predicted:    {result.trigger_type.value}")
        print(f"  Confidence:   {result.confidence:.2f}")
        print(f"  Method:       {result.method}")

        # Check if prediction matches ground truth
        matches = result.trigger_type.value == ground_truth
        print(f"  Auto-check:   {'✓ MATCH' if matches else '✗ MISMATCH'}")

        while True:
            response = input("  Correct? [y/n/s/q]: ").strip().lower()
            if response in ("y", "n", "s", "q"):
                break
            print("  Invalid input. Enter y, n, s, or q.")

        if response == "q":
            print("\nQuitting early...")
            break

        human_verified = True if response == "y" else (False if response == "n" else None)

        results.append(
            ValidationResult(
                text=text,
                predicted=result.trigger_type.value,
                confidence=result.confidence,
                ground_truth=ground_truth,
                human_verified=human_verified,
                method=result.method,
            )
        )

    return results


def validate_response(samples: list[tuple[str, str]]) -> list[ValidationResult]:
    """Run response classifier on samples and prompt for validation."""
    from jarvis.classifiers.response_classifier import get_response_classifier

    classifier = get_response_classifier()
    results = []
    total = len(samples)

    print("\n" + "=" * 70)
    print("RESPONSE CLASSIFIER VALIDATION")
    print("=" * 70)
    print("For each example, enter:")
    print("  y = prediction is correct")
    print("  n = prediction is wrong")
    print("  s = skip (unsure)")
    print("  q = quit early")
    print("=" * 70 + "\n")

    for i, (text, ground_truth) in enumerate(samples, 1):
        result = classifier.classify(text)

        print(f"\n[{i}/{total}] Text: {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"  Ground truth: {ground_truth}")
        print(f"  Predicted:    {result.label.value}")
        print(f"  Confidence:   {result.confidence:.2f}")
        print(f"  Method:       {result.method}")

        # Check if prediction matches ground truth
        matches = result.label.value == ground_truth
        print(f"  Auto-check:   {'✓ MATCH' if matches else '✗ MISMATCH'}")

        while True:
            response = input("  Correct? [y/n/s/q]: ").strip().lower()
            if response in ("y", "n", "s", "q"):
                break
            print("  Invalid input. Enter y, n, s, or q.")

        if response == "q":
            print("\nQuitting early...")
            break

        human_verified = True if response == "y" else (False if response == "n" else None)

        results.append(
            ValidationResult(
                text=text,
                predicted=result.label.value,
                confidence=result.confidence,
                ground_truth=ground_truth,
                human_verified=human_verified,
                method=result.method,
            )
        )

    return results


def print_summary(results: list[ValidationResult], classifier_type: str) -> None:
    """Print validation summary statistics."""
    if not results:
        print("\nNo results to summarize.")
        return

    print("\n" + "=" * 70)
    print(f"{classifier_type.upper()} CLASSIFIER VALIDATION SUMMARY")
    print("=" * 70)

    # Filter out skipped
    verified = [r for r in results if r.human_verified is not None]
    skipped = len(results) - len(verified)

    if not verified:
        print("No verified samples.")
        return

    correct = sum(1 for r in verified if r.human_verified)
    incorrect = len(verified) - correct

    print(f"\nTotal samples:    {len(results)}")
    print(f"Verified:         {len(verified)}")
    print(f"Skipped:          {skipped}")
    print(f"\nCorrect:          {correct}")
    print(f"Incorrect:        {incorrect}")
    print(f"Human-verified accuracy: {correct / len(verified) * 100:.1f}%")

    # Auto-accuracy (matches ground truth label)
    auto_correct = sum(1 for r in results if r.predicted == r.ground_truth)
    print(f"Auto-check accuracy:     {auto_correct / len(results) * 100:.1f}%")

    # Per-class breakdown
    print("\nPer-class performance (human-verified):")
    by_class: dict[str, list[bool]] = {}
    for r in verified:
        label = r.ground_truth or "unknown"
        if label not in by_class:
            by_class[label] = []
        by_class[label].append(r.human_verified)

    print(f"  {'Class':<15} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("  " + "-" * 43)
    for label, verdicts in sorted(by_class.items()):
        cls_correct = sum(verdicts)
        cls_total = len(verdicts)
        accuracy = cls_correct / cls_total * 100 if cls_total > 0 else 0
        print(f"  {label:<15} {cls_correct:>8} {cls_total:>8} {accuracy:>9.1f}%")

    # Method breakdown
    print("\nBy classification method:")
    by_method: dict[str, list[bool]] = {}
    for r in verified:
        if r.method not in by_method:
            by_method[r.method] = []
        by_method[r.method].append(r.human_verified)

    print(f"  {'Method':<25} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("  " + "-" * 53)
    for method, verdicts in sorted(by_method.items()):
        m_correct = sum(verdicts)
        m_total = len(verdicts)
        accuracy = m_correct / m_total * 100 if m_total > 0 else 0
        print(f"  {method:<25} {m_correct:>8} {m_total:>8} {accuracy:>9.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Manual validation for trigger/response classifiers"
    )
    parser.add_argument(
        "--classifier",
        choices=["trigger", "response"],
        required=True,
        help="Which classifier to validate",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples to validate (default: 50)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input JSONL file (defaults to standard data files)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save validation results to JSON file",
    )
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Load data
    if args.classifier == "trigger":
        input_path = args.input or Path("data/trigger_labeling.jsonl")
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)
        data = load_trigger_data(input_path)
        print(f"Loaded {len(data)} trigger examples from {input_path}")
    else:
        input_path = args.input or Path("data/response_labeling.jsonl")
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)
        data = load_response_data(input_path)
        print(f"Loaded {len(data)} response examples from {input_path}")

    # Sample
    n_samples = min(args.samples, len(data))
    samples = random.sample(data, n_samples)

    # Show distribution of sampled data
    labels = [s[1] for s in samples]
    print(f"Sample distribution: {dict(Counter(labels))}")

    # Run validation
    if args.classifier == "trigger":
        results = validate_trigger(samples)
    else:
        results = validate_response(samples)

    # Print summary
    print_summary(results, args.classifier)

    # Save results if requested
    if args.output:
        output_data = {
            "classifier": args.classifier,
            "samples_validated": len(results),
            "results": [
                {
                    "text": r.text,
                    "predicted": r.predicted,
                    "confidence": r.confidence,
                    "ground_truth": r.ground_truth,
                    "human_verified": r.human_verified,
                    "method": r.method,
                }
                for r in results
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output_data, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
