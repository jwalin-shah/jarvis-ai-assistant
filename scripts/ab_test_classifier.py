#!/usr/bin/env python3
"""A/B test classifier configurations and show diffs.

Usage:
    uv run python -m scripts.ab_test_classifier              # Test all configs
    uv run python -m scripts.ab_test_classifier --diff       # Show what changed per sample
    uv run python -m scripts.ab_test_classifier --config X   # Test specific config only
"""

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarvis.response_classifier import (
    HybridResponseClassifier,
    ResponseType,
    STRUCTURAL_PATTERNS,
    _COMPILED_PATTERNS,
    reset_response_classifier,
)


# =============================================================================
# Configuration Variants
# =============================================================================

@dataclass
class ClassifierConfig:
    """A classifier configuration to test."""
    name: str
    description: str
    decline_threshold: float = 0.7
    defer_threshold: float = 0.65
    low_confidence_threshold: float = 0.5
    exclude_patterns: list[tuple[ResponseType, str]] | None = None  # (type, pattern_substring)


# Define test configurations
CONFIGS = [
    ClassifierConfig(
        name="current",
        description="Current config (baseline)",
        decline_threshold=0.7,
        defer_threshold=0.65,
    ),
    ClassifierConfig(
        name="lower_decline",
        description="Lower DECLINE threshold (0.55)",
        decline_threshold=0.55,
        defer_threshold=0.65,
    ),
    ClassifierConfig(
        name="lower_both",
        description="Lower DECLINE (0.55) and DEFER (0.50)",
        decline_threshold=0.55,
        defer_threshold=0.50,
    ),
    ClassifierConfig(
        name="no_aux_verb",
        description="Remove auxiliary verb question pattern",
        decline_threshold=0.7,
        defer_threshold=0.65,
        exclude_patterns=[(ResponseType.QUESTION, "do|does|did|are|is|was")],
    ),
    ClassifierConfig(
        name="lower_decline_no_aux",
        description="Lower DECLINE (0.55) + no aux verb pattern",
        decline_threshold=0.55,
        defer_threshold=0.65,
        exclude_patterns=[(ResponseType.QUESTION, "do|does|did|are|is|was")],
    ),
    ClassifierConfig(
        name="aggressive",
        description="Low thresholds (0.45 DECLINE, 0.40 DEFER)",
        decline_threshold=0.45,
        defer_threshold=0.40,
    ),
]


# =============================================================================
# Custom Classifier for A/B Testing
# =============================================================================

class ConfigurableClassifier(HybridResponseClassifier):
    """Classifier with configurable thresholds and pattern exclusions."""

    def __init__(
        self,
        config: ClassifierConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config

        # Override thresholds
        self.DECLINE_CONFIDENCE_THRESHOLD = config.decline_threshold
        self.DEFER_CONFIDENCE_THRESHOLD = config.defer_threshold
        self.LOW_CONFIDENCE_THRESHOLD = config.low_confidence_threshold

        # Build excluded pattern set
        self._excluded_patterns: set[str] = set()
        if config.exclude_patterns:
            for response_type, pattern_substring in config.exclude_patterns:
                for pattern, is_regex in STRUCTURAL_PATTERNS.get(response_type, []):
                    if pattern_substring in pattern:
                        self._excluded_patterns.add(pattern)

    def _get_structural_hint(self, text: str) -> ResponseType | None:
        """Get structural hint, excluding configured patterns."""
        text_lower = text.lower().strip()

        for response_type, patterns in _COMPILED_PATTERNS.items():
            for compiled_pattern in patterns:
                # Check if this pattern should be excluded
                if compiled_pattern.pattern in self._excluded_patterns:
                    continue

                if compiled_pattern.search(text_lower):
                    return response_type

        return None


# =============================================================================
# Evaluation Logic
# =============================================================================

def load_validation_data() -> list[dict[str, Any]]:
    """Load the validated samples with ground truth labels."""
    validation_file = Path.home() / ".jarvis" / "classifier_validation.json"

    if not validation_file.exists():
        raise FileNotFoundError(
            f"No validation file at {validation_file}\n"
            "Run: uv run python -m scripts.eval_full_classifier --validate 200"
        )

    with open(validation_file) as f:
        samples = json.load(f)

    # Filter to only judged samples
    judged = [s for s in samples if s.get("correct") is not None]

    if not judged:
        raise ValueError(
            "No samples have been judged yet.\n"
            "Edit the JSON file and set 'correct': true or false for each sample."
        )

    return judged


def evaluate_config(
    config: ClassifierConfig,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate a single configuration on the validation samples."""

    classifier = ConfigurableClassifier(config)

    # Classify all samples
    predictions = []
    correct_count = 0
    by_type: dict[str, dict[str, int]] = {}

    for sample in samples:
        response = sample["response"]
        expected_correct = sample["correct"]
        original_pred = sample["predicted"]

        # Get new prediction
        result = classifier.classify(response)
        new_pred = result.label.value

        # A prediction is correct if:
        # - It matches the original AND original was marked correct
        # - OR it's a change that might be better (we'll track these separately)
        is_correct = (new_pred == original_pred) and expected_correct

        # Actually, we should judge based on whether the NEW prediction is correct
        # But we only have ground truth for the original prediction
        # So: if original was correct and we predict same -> correct
        #     if original was wrong and we predict same -> wrong
        #     if we predict different -> unknown (track separately)

        predictions.append({
            "response": response[:60],
            "original_pred": original_pred,
            "new_pred": new_pred,
            "original_correct": expected_correct,
            "changed": new_pred != original_pred,
            "confidence": result.confidence,
            "method": result.method,
        })

        # Track per-type accuracy (for unchanged predictions)
        if new_pred == original_pred:
            if original_pred not in by_type:
                by_type[original_pred] = {"correct": 0, "wrong": 0}
            if expected_correct:
                by_type[original_pred]["correct"] += 1
                correct_count += 1
            else:
                by_type[original_pred]["wrong"] += 1

    # Count changes
    changes = [p for p in predictions if p["changed"]]
    unchanged = [p for p in predictions if not p["changed"]]

    # Calculate metrics
    unchanged_accuracy = correct_count / len(unchanged) * 100 if unchanged else 0

    return {
        "config": config.name,
        "description": config.description,
        "total_samples": len(samples),
        "unchanged": len(unchanged),
        "changed": len(changes),
        "unchanged_accuracy": unchanged_accuracy,
        "correct_unchanged": correct_count,
        "by_type": by_type,
        "predictions": predictions,
        "changes": changes,
    }


def print_results(results: list[dict[str, Any]], show_diff: bool = False):
    """Print evaluation results."""

    print("=" * 80)
    print("A/B TEST RESULTS")
    print("=" * 80)
    print()

    # Summary table
    print(f"{'Config':<25} {'Unchanged':<12} {'Changed':<10} {'Accuracy*':<12}")
    print("-" * 60)

    for r in sorted(results, key=lambda x: -x["unchanged_accuracy"]):
        print(
            f"{r['config']:<25} "
            f"{r['unchanged']:>4}/{r['total_samples']:<6} "
            f"{r['changed']:>4}       "
            f"{r['unchanged_accuracy']:>5.1f}%"
        )

    print()
    print("* Accuracy on unchanged predictions (where we have ground truth)")
    print()

    # Per-type breakdown for each config
    for r in results:
        print("=" * 80)
        print(f"CONFIG: {r['config']} - {r['description']}")
        print("=" * 80)
        print()
        print(f"{'Type':<20} {'Correct':<10} {'Wrong':<10} {'Accuracy':<10}")
        print("-" * 50)

        for label in sorted(r["by_type"].keys()):
            stats = r["by_type"][label]
            total = stats["correct"] + stats["wrong"]
            acc = stats["correct"] / total * 100 if total > 0 else 0
            print(f"{label:<20} {stats['correct']:<10} {stats['wrong']:<10} {acc:>5.1f}%")

        # Show what changed
        if r["changes"]:
            print()
            print(f"CHANGES ({len(r['changes'])} samples):")
            print("-" * 50)

            # Group by change type
            change_types = Counter(
                (c["original_pred"], c["new_pred"]) for c in r["changes"]
            )
            for (old, new), count in sorted(change_types.items(), key=lambda x: -x[1]):
                print(f"  {old} -> {new}: {count}")

            if show_diff:
                print()
                print("DETAILED CHANGES:")
                for c in r["changes"][:20]:  # Limit to 20
                    marker = "✓" if c["original_correct"] else "✗"
                    print(f"  [{marker}] {c['original_pred']} -> {c['new_pred']}")
                    print(f"      \"{c['response']}...\"")
                    print()

        print()


def analyze_regressions(results: list[dict[str, Any]]):
    """Analyze which configs fix which regressions."""

    print("=" * 80)
    print("REGRESSION ANALYSIS")
    print("=" * 80)
    print()

    # Find the baseline (current config)
    baseline = next((r for r in results if r["config"] == "current"), results[0])

    # For each config, show improvements vs regressions
    for r in results:
        if r["config"] == "current":
            continue

        print(f"\n{r['config']} vs current:")
        print("-" * 40)

        # Count improvements (originally wrong -> now different)
        # and regressions (originally right -> now different)
        improvements = []
        regressions = []

        for p in r["predictions"]:
            if p["changed"]:
                if not p["original_correct"]:
                    improvements.append(p)
                else:
                    regressions.append(p)

        print(f"  Potential improvements: {len(improvements)} (original was wrong, now different)")
        print(f"  Potential regressions:  {len(regressions)} (original was right, now different)")

        if improvements:
            print("\n  Sample improvements:")
            for p in improvements[:5]:
                print(f"    {p['original_pred']} -> {p['new_pred']}: \"{p['response']}...\"")

        if regressions:
            print("\n  Sample regressions:")
            for p in regressions[:5]:
                print(f"    {p['original_pred']} -> {p['new_pred']}: \"{p['response']}...\"")


def main():
    parser = argparse.ArgumentParser(description="A/B test classifier configs")
    parser.add_argument("--diff", action="store_true",
                        help="Show detailed per-sample changes")
    parser.add_argument("--config", type=str, metavar="NAME",
                        help="Test only this config (comma-separated for multiple)")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze improvements vs regressions")
    args = parser.parse_args()

    # Load validation data
    try:
        samples = load_validation_data()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1

    print(f"Loaded {len(samples)} judged samples")
    print()

    # Select configs to test
    configs_to_test = CONFIGS
    if args.config:
        config_names = [n.strip() for n in args.config.split(",")]
        configs_to_test = [c for c in CONFIGS if c.name in config_names]
        if not configs_to_test:
            print(f"Unknown config(s): {args.config}")
            print(f"Available: {', '.join(c.name for c in CONFIGS)}")
            return 1

    # Run evaluation for each config
    results = []
    for config in configs_to_test:
        print(f"Testing: {config.name}...")
        reset_response_classifier()
        result = evaluate_config(config, samples)
        results.append(result)

    print()

    # Print results
    print_results(results, show_diff=args.diff)

    if args.analyze:
        analyze_regressions(results)

    # Recommendation
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    best = max(results, key=lambda x: x["unchanged_accuracy"])
    print(f"\nBest config: {best['config']} ({best['unchanged_accuracy']:.1f}% accuracy)")
    print(f"Description: {best['description']}")

    # Check if any config has fewer changes AND higher accuracy
    current = next((r for r in results if r["config"] == "current"), results[0])
    for r in results:
        if r["config"] != "current":
            if r["unchanged_accuracy"] > current["unchanged_accuracy"]:
                # Check if changes are net positive
                improvements = sum(
                    1 for p in r["predictions"]
                    if p["changed"] and not p["original_correct"]
                )
                regressions = sum(
                    1 for p in r["predictions"]
                    if p["changed"] and p["original_correct"]
                )

                if improvements > regressions:
                    print(f"\n{r['config']} shows net improvement:")
                    print(f"  +{improvements} potential fixes, -{regressions} potential regressions")

    return 0


if __name__ == "__main__":
    exit(main())
