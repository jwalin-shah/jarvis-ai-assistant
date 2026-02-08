#!/usr/bin/env python3
"""Analyze DailyDialog experiment sweep results.

Reads experiments/results/dailydialog_sweep.json and provides:
- Top 10 configurations by CV F1
- Best config per dimension (4-class, 3-class, embeddings-only, etc.)
- Per-class F1 breakdown
- Generalization check (CV vs test F1)
- Decision criteria for production model

Usage:
    uv run python experiments/scripts/analyze_dailydialog_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_FILE = PROJECT_ROOT / "experiments" / "results" / "dailydialog_sweep.json"


def analyze_results() -> None:
    """Analyze experiment sweep results and print recommendations."""
    if not RESULTS_FILE.exists():
        print(f"ERROR: Results file not found: {RESULTS_FILE}")
        print("Run experiments/scripts/dailydialog_sweep.py first.")
        sys.exit(1)

    with open(RESULTS_FILE) as f:
        data = json.load(f)

    experiments = data["experiments"]
    total = data["total_experiments"]

    print("=" * 100)
    print(f"DAILYDIALOG EXPERIMENT SWEEP ANALYSIS ({total} experiments)")
    print("=" * 100)
    print()

    # --------------------------------------------------------------------
    # 1. Top 10 overall
    # --------------------------------------------------------------------
    print("TOP 10 CONFIGURATIONS (by CV F1)")
    print("-" * 100)
    print(
        f"{'Rank':<5} {'Cat':<7} {'Features':<12} {'Balance':<9} "
        f"{'C':<6} {'Weight':<9} {'CV F1':<8} {'Test F1':<8} {'Gap':<6}"
    )
    print("-" * 100)

    for i, exp in enumerate(experiments[:10], 1):
        gap = exp["cv_mean_f1"] - exp["test_f1"]
        print(
            f"{i:<5} {exp['category_config']:<7} {exp['feature_set']:<12} "
            f"{exp['balancing']:<9} {exp['C']:<6.1f} {str(exp['class_weight']):<9} "
            f"{exp['cv_mean_f1']:<8.3f} {exp['test_f1']:<8.3f} {gap:>6.3f}"
        )

    print()

    # --------------------------------------------------------------------
    # 2. Best per dimension
    # --------------------------------------------------------------------
    print("BEST CONFIGURATION PER DIMENSION")
    print("-" * 100)

    dimensions = [
        ("category_config", "Category Config"),
        ("feature_set", "Feature Set"),
        ("balancing", "Balancing"),
    ]

    for dim_key, dim_name in dimensions:
        print(f"\n{dim_name}:")
        values = sorted(set(exp[dim_key] for exp in experiments))

        for val in values:
            subset = [exp for exp in experiments if exp[dim_key] == val]
            best = max(subset, key=lambda x: x["cv_mean_f1"])
            print(
                f"  {val:12s}: CV F1={best['cv_mean_f1']:.3f}, "
                f"Test F1={best['test_f1']:.3f} "
                f"(C={best['C']:.1f}, weight={best['class_weight']})"
            )

    print()

    # --------------------------------------------------------------------
    # 3. Per-class F1 breakdown for top config
    # --------------------------------------------------------------------
    best_config = experiments[0]
    print("PER-CLASS F1 (Top Configuration)")
    print("-" * 100)
    print(f"Config: {_format_config(best_config)}")
    print()

    per_class = best_config["per_class_f1"]
    for label in sorted(per_class.keys()):
        f1 = per_class[label]
        print(f"  {label:12s}: {f1:.3f}")

    print()

    # --------------------------------------------------------------------
    # 4. Generalization check
    # --------------------------------------------------------------------
    print("GENERALIZATION CHECK (CV vs Test F1)")
    print("-" * 100)

    gaps = [(exp, exp["cv_mean_f1"] - exp["test_f1"]) for exp in experiments]
    gaps.sort(key=lambda x: abs(x[1]))

    print("Best generalization (smallest CV-Test gap):")
    for exp, gap in gaps[:5]:
        print(
            f"  Gap: {gap:>6.3f} | CV: {exp['cv_mean_f1']:.3f} | "
            f"Test: {exp['test_f1']:.3f} | {_format_config(exp)}"
        )

    print()
    print("Worst generalization (largest CV-Test gap):")
    for exp, gap in gaps[-5:]:
        print(
            f"  Gap: {gap:>6.3f} | CV: {exp['cv_mean_f1']:.3f} | "
            f"Test: {exp['test_f1']:.3f} | {_format_config(exp)}"
        )

    print()

    # --------------------------------------------------------------------
    # 5. Decision criteria and recommendation
    # --------------------------------------------------------------------
    print("=" * 100)
    print("RECOMMENDATION FOR PRODUCTION MODEL")
    print("=" * 100)
    print()

    # Compare 3-class vs 4-class
    best_3class = max(
        [exp for exp in experiments if exp["category_config"] == "3class"],
        key=lambda x: x["cv_mean_f1"]
    )
    best_4class = max(
        [exp for exp in experiments if exp["category_config"] == "4class"],
        key=lambda x: x["cv_mean_f1"]
    )

    f1_diff = best_3class["cv_mean_f1"] - best_4class["cv_mean_f1"]
    print(f"3-class best CV F1: {best_3class['cv_mean_f1']:.3f}")
    print(f"4-class best CV F1: {best_4class['cv_mean_f1']:.3f}")
    print(f"Difference: {f1_diff:+.3f}")

    if f1_diff > 0.03:
        print("→ RECOMMEND: 3-class (merge directive+commissive → action)")
        recommended = best_3class
    else:
        print("→ RECOMMEND: 4-class (keep native labels)")
        recommended = best_4class

    print()

    # Check feature set
    feat_set = recommended["feature_set"]
    print(f"Feature set: {feat_set}")
    if feat_set == "combined":
        # Check if embeddings-only is close
        emb_only = max(
            [exp for exp in experiments
             if exp["feature_set"] == "embeddings" and
             exp["category_config"] == recommended["category_config"]],
            key=lambda x: x["cv_mean_f1"]
        )
        feat_diff = recommended["cv_mean_f1"] - emb_only["cv_mean_f1"]
        if feat_diff < 0.01:
            print(f"  (embeddings-only is within 1%: F1={emb_only['cv_mean_f1']:.3f})")
            print("  → Could simplify to embeddings-only")

    print()

    # Check balancing
    balance = recommended["balancing"]
    print(f"Balancing: {balance}")
    if balance != "natural":
        natural = max(
            [exp for exp in experiments
             if exp["balancing"] == "natural" and
             exp["category_config"] == recommended["category_config"]],
            key=lambda x: x["cv_mean_f1"]
        )
        bal_diff = recommended["cv_mean_f1"] - natural["cv_mean_f1"]
        print(f"  (natural balance F1: {natural['cv_mean_f1']:.3f}, diff: {bal_diff:+.3f})")
        if bal_diff < 0.05:
            print("  → Natural balance within 5%, could use natural")

    print()

    # Generalization check
    gen_gap = recommended["cv_mean_f1"] - recommended["test_f1"]
    print(f"Generalization (CV - Test): {gen_gap:.3f}")
    if gen_gap > 0.02:
        print("  ⚠ Warning: Potential overfitting (gap > 2%)")
    else:
        print("  ✓ Good generalization (gap ≤ 2%)")

    print()

    # Final recommendation
    print("RECOMMENDED CONFIGURATION:")
    print("-" * 100)
    print(f"  Category config: {recommended['category_config']}")
    print(f"  Feature set:     {recommended['feature_set']}")
    print(f"  Balancing:       {recommended['balancing']}")
    print(f"  C:               {recommended['C']}")
    print(f"  Class weight:    {recommended['class_weight']}")
    print()
    print(f"  Expected CV F1:   {recommended['cv_mean_f1']:.3f} ± {recommended['cv_std_f1']:.3f}")
    print(f"  Expected Test F1: {recommended['test_f1']:.3f}")
    print()
    print("TRAINING COMMAND:")
    print("-" * 100)

    if recommended["category_config"] == "3class":
        label_map_flag = "--label-map 3class"
    else:
        label_map_flag = "--label-map 4class"

    print(f"  uv run python scripts/train_category_svm.py \\")
    print(f"    --data-dir data/dailydialog_native \\")
    print(f"    {label_map_flag}")
    print()

    # Compare to baseline
    baseline_f1 = 0.38  # Current custom category classifier
    improvement = recommended["test_f1"] - baseline_f1
    improvement_pct = (improvement / baseline_f1) * 100

    print("EXPECTED IMPROVEMENT:")
    print("-" * 100)
    print(f"  Baseline (custom categories):        {baseline_f1:.3f}")
    print(f"  Recommended (native dialog acts):    {recommended['test_f1']:.3f}")
    print(f"  Absolute improvement:                {improvement:+.3f}")
    print(f"  Relative improvement:                {improvement_pct:+.1f}%")
    print()

    # Human agreement benchmark
    human_agreement = 0.789  # 78.9% inter-annotator agreement
    print(f"  Human inter-annotator agreement:     {human_agreement:.3f}")
    print(f"  Gap to human performance:            {human_agreement - recommended['test_f1']:.3f}")
    print()


def _format_config(exp: dict) -> str:
    """Format experiment config as compact string."""
    return (
        f"{exp['category_config']} | {exp['feature_set']} | {exp['balancing']} | "
        f"C={exp['C']:.1f} | wt={exp['class_weight']}"
    )


def main() -> int:
    analyze_results()
    return 0


if __name__ == "__main__":
    sys.exit(main())
