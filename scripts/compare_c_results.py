#!/usr/bin/env python3
"""Compare C search results from multiple runs and plot the curve.

Usage:
    uv run python scripts/compare_c_results.py c_search_results_*.json
    uv run python scripts/compare_c_results.py --plot c_search_results_*.json
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description="Compare C search results")
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="JSON result files from quick_c_search.py",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate matplotlib plot (requires matplotlib)",
    )

    args = parser.parse_args()

    # Load all results
    all_results = []
    for file in args.files:
        if not file.exists():
            print(f"File not found: {file}", file=sys.stderr)
            continue

        data = json.loads(file.read_text())
        for result in data["results"]:
            all_results.append({
                "C": result["C"],
                "mean_f1": result["mean_f1"],
                "std_f1": result["std_f1"],
                "source": file.name,
                "config": data["config"],
            })

    if not all_results:
        print("No valid result files found", file=sys.stderr)
        return 1

    # Sort by C value
    all_results.sort(key=lambda x: x["C"])

    # Print table
    print(f"{'C':>8s}  {'Mean F1':>8s}  {'Std':>6s}  {'Source':>30s}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['C']:8.1f}  {r['mean_f1']:8.4f}  {r['std_f1']:6.4f}  {r['source']:30s}")

    # Find overall best
    best = max(all_results, key=lambda x: x["mean_f1"])
    print(f"\n{'=' * 60}")
    print(f"Best C across all runs: {best['C']}")
    print(f"Best mean F1: {best['mean_f1']:.4f} (+/- {best['std_f1']:.4f})")
    print(f"From: {best['source']}")
    print(f"{'=' * 60}")

    # Check for gaps
    print(f"\nGap analysis:")
    for i in range(len(all_results) - 1):
        curr = all_results[i]
        next_r = all_results[i + 1]
        gap_ratio = next_r["C"] / curr["C"]
        f1_diff = abs(next_r["mean_f1"] - curr["mean_f1"])

        if gap_ratio > 1.5 and f1_diff > 0.01:
            mid_c = int((curr["C"] + next_r["C"]) / 2)
            print(f"  Large gap: C={curr['C']:.1f} → C={next_r['C']:.1f} (ratio {gap_ratio:.1f}x)")
            print(f"             F1 diff: {f1_diff:.4f}")
            print(f"    → Test C={mid_c} to fill gap")

    # Plot if requested
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            c_values = [r["C"] for r in all_results]
            f1_values = [r["mean_f1"] for r in all_results]
            std_values = [r["std_f1"] for r in all_results]

            plt.figure(figsize=(10, 6))
            plt.errorbar(c_values, f1_values, yerr=std_values, marker='o', capsize=5)
            plt.xscale('log')
            plt.xlabel('C value (log scale)')
            plt.ylabel('Mean F1 Score (5-fold CV)')
            plt.title('LinearSVC C Parameter Search')
            plt.grid(True, alpha=0.3)

            # Mark best C
            best_idx = c_values.index(best["C"])
            plt.axvline(x=best["C"], color='r', linestyle='--', alpha=0.5, label=f'Best C={best["C"]}')
            plt.legend()

            output_plot = PROJECT_ROOT / "c_search_curve.png"
            plt.savefig(output_plot, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {output_plot}")

        except ImportError:
            print("\nmatplotlib not installed - skipping plot", file=sys.stderr)
            print("Install with: uv pip install matplotlib", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main() or 0)
