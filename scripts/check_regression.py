#!/usr/bin/env python3
"""Check for prompt/model performance regression between baselines.

Compares evaluation results against a baseline and reports regressions.
Can be used in CI to block PRs that introduce unacceptable degradation.

Usage:
    python scripts/check_regression.py --baseline results/baseline.json --current results/current.json
    python scripts/check_regression.py --baseline results/baseline.json --current results/current.json --ci
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Thresholds:
    """Regression thresholds for various metrics."""

    # Format: (warning_threshold, block_threshold)
    # Negative = decrease, Positive = increase (for rates)
    category_accuracy: tuple[float, float] = (-0.05, -0.10)  # -5% warn, -10% block
    anti_ai_clean_rate: tuple[float, float] = (-0.05, -0.10)
    judge_avg: tuple[float, float] = (-0.5, -1.0)  # -0.5 warn, -1.0 block
    avg_latency_ms: tuple[float, float] = (0.10, 0.25)  # +10% warn, +25% block
    similarity_score: tuple[float, float] = (-0.05, -0.10)


def load_results(path: Path) -> dict:
    """Load evaluation results from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    with open(path) as f:
        return json.load(f)


def check_regression(
    baseline: dict,
    current: dict,
    thresholds: Thresholds,
) -> tuple[list[str], list[str]]:
    """Check for regressions and return warnings and errors.

    Returns:
        Tuple of (warnings, errors)
    """
    warnings = []
    errors = []

    # Check category accuracy
    base_acc = baseline.get("category_accuracy", 0)
    curr_acc = current.get("category_accuracy", 0)
    acc_delta = curr_acc - base_acc

    if acc_delta < thresholds.category_accuracy[1]:
        errors.append(
            f"Category accuracy regression: {base_acc:.1%} -> {curr_acc:.1%} "
            f"({acc_delta:+.1%}, threshold: {thresholds.category_accuracy[1]:.1%})"
        )
    elif acc_delta < thresholds.category_accuracy[0]:
        warnings.append(
            f"Category accuracy warning: {base_acc:.1%} -> {curr_acc:.1%} ({acc_delta:+.1%})"
        )

    # Check anti-AI clean rate
    base_clean = baseline.get("anti_ai_clean_rate", 0)
    curr_clean = current.get("anti_ai_clean_rate", 0)
    clean_delta = curr_clean - base_clean

    if clean_delta < thresholds.anti_ai_clean_rate[1]:
        errors.append(
            f"Anti-AI clean rate regression: {base_clean:.1%} -> {curr_clean:.1%} "
            f"({clean_delta:+.1%}, threshold: {thresholds.anti_ai_clean_rate[1]:.1%})"
        )
    elif clean_delta < thresholds.anti_ai_clean_rate[0]:
        warnings.append(
            f"Anti-AI clean rate warning: {base_clean:.1%} -> {curr_clean:.1%} ({clean_delta:+.1%})"
        )

    # Check judge score
    base_judge = baseline.get("judge_avg")
    curr_judge = current.get("judge_avg")

    if base_judge is not None and curr_judge is not None:
        judge_delta = curr_judge - base_judge

        if judge_delta < thresholds.judge_avg[1]:
            errors.append(
                f"Judge score regression: {base_judge:.1f} -> {curr_judge:.1f} "
                f"({judge_delta:+.1f}, threshold: {thresholds.judge_avg[1]:.1f})"
            )
        elif judge_delta < thresholds.judge_avg[0]:
            warnings.append(
                f"Judge score warning: {base_judge:.1f} -> {curr_judge:.1f} ({judge_delta:+.1f})"
            )

    # Check latency
    base_latency = baseline.get("latency", {}).get("avg_ms", 0)
    curr_latency = current.get("latency", {}).get("avg_ms", 0)

    if base_latency > 0:
        latency_delta = (curr_latency - base_latency) / base_latency

        if latency_delta > thresholds.avg_latency_ms[1]:
            errors.append(
                f"Latency regression: {base_latency:.0f}ms -> {curr_latency:.0f}ms "
                f"({latency_delta:+.1%}, threshold: {thresholds.avg_latency_ms[1]:.1%})"
            )
        elif latency_delta > thresholds.avg_latency_ms[0]:
            warnings.append(
                f"Latency warning: {base_latency:.0f}ms -> {curr_latency:.0f}ms "
                f"({latency_delta:+.1%})"
            )

    # Check similarity score
    base_sim = baseline.get("avg_similarity")
    curr_sim = current.get("avg_similarity")

    if base_sim is not None and curr_sim is not None:
        sim_delta = curr_sim - base_sim

        if sim_delta < thresholds.similarity_score[1]:
            errors.append(
                f"Similarity score regression: {base_sim:.3f} -> {curr_sim:.3f} "
                f"({sim_delta:+.3f}, threshold: {thresholds.similarity_score[1]:.3f})"
            )
        elif sim_delta < thresholds.similarity_score[0]:
            warnings.append(
                f"Similarity score warning: {base_sim:.3f} -> {curr_sim:.3f} ({sim_delta:+.3f})"
            )

    # Per-category checks
    base_per_cat = baseline.get("per_category", {})
    curr_per_cat = current.get("per_category", {})

    for category in set(base_per_cat.keys()) | set(curr_per_cat.keys()):
        base_cat = base_per_cat.get(category, {})
        curr_cat = curr_per_cat.get(category, {})

        base_cat_acc = base_cat.get("classify_accuracy", 0)
        curr_cat_acc = curr_cat.get("classify_accuracy", 0)

        if base_cat_acc > 0:
            cat_delta = curr_cat_acc - base_cat_acc

            if cat_delta < -0.15:  # 15% drop is critical
                errors.append(
                    f"Category '{category}' accuracy regression: "
                    f"{base_cat_acc:.1%} -> {curr_cat_acc:.1%} ({cat_delta:+.1%})"
                )
            elif cat_delta < -0.10:  # 10% drop is warning
                warnings.append(
                    f"Category '{category}' accuracy warning: "
                    f"{base_cat_acc:.1%} -> {curr_cat_acc:.1%} ({cat_delta:+.1%})"
                )

    return warnings, errors


def format_report(
    baseline: dict,
    current: dict,
    warnings: list[str],
    errors: list[str],
) -> str:
    """Format a human-readable regression report."""
    lines = []
    lines.append("=" * 70)
    lines.append("REGRESSION CHECK REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Summary table
    lines.append("METRIC COMPARISON:")
    lines.append("-" * 70)
    lines.append(f"{'Metric':<25} {'Baseline':<15} {'Current':<15} {'Delta':<10}")
    lines.append("-" * 70)

    metrics = [
        ("Category Accuracy", "category_accuracy", ".1%"),
        ("Anti-AI Clean Rate", "anti_ai_clean_rate", ".1%"),
        ("Judge Average", "judge_avg", ".1f"),
        ("Avg Similarity", "avg_similarity", ".3f"),
    ]

    for name, key, fmt in metrics:
        base_val = baseline.get(key)
        curr_val = current.get(key)

        if base_val is not None and curr_val is not None:
            delta = curr_val - base_val
            # Apply format string
            base_str = format(base_val, fmt)
            curr_str = format(curr_val, fmt)
            delta_str = f"{delta:+.1%}" if "accuracy" in key or "rate" in key else f"{delta:+.2f}"
            lines.append(f"{name:<25} {base_str:<15} {curr_str:<15} {delta_str:<10}")

    # Latency
    base_lat = baseline.get("latency", {}).get("avg_ms", 0)
    curr_lat = current.get("latency", {}).get("avg_ms", 0)
    if base_lat > 0:
        delta_pct = (curr_lat - base_lat) / base_lat
        lines.append(
            f"{'Avg Latency':<25} {base_lat:.0f}ms{'':<10} {curr_lat:.0f}ms"
            f"{'':<10} {delta_pct:+.1%}"
        )

    lines.append("")

    # Warnings and errors
    if errors:
        lines.append("ERRORS (blocking):")
        lines.append("-" * 70)
        for error in errors:
            lines.append(f"  ❌ {error}")
        lines.append("")

    if warnings:
        lines.append("WARNINGS (non-blocking):")
        lines.append("-" * 70)
        for warning in warnings:
            lines.append(f"  ⚠️  {warning}")
        lines.append("")

    if not errors and not warnings:
        lines.append("✅ No regressions detected!")
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check for performance regression")
    parser.add_argument("--baseline", required=True, type=Path, help="Baseline results JSON")
    parser.add_argument("--current", required=True, type=Path, help="Current results JSON")
    parser.add_argument("--ci", action="store_true", help="CI mode (exit 1 on errors)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    try:
        baseline = load_results(args.baseline)
        current = load_results(args.current)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON: {e}")
        return 1

    thresholds = Thresholds()
    warnings, errors = check_regression(baseline, current, thresholds)

    if args.json:
        output = {
            "passed": len(errors) == 0,
            "warnings": warnings,
            "errors": errors,
            "baseline_version": baseline.get("prompt_version", "unknown"),
            "current_version": current.get("prompt_version", "unknown"),
        }
        print(json.dumps(output, indent=2))
    else:
        print(format_report(baseline, current, warnings, errors))

    if args.ci and errors:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
