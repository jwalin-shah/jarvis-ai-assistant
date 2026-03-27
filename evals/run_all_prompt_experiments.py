#!/usr/bin/env python3
"""Master script to run all prompt optimization experiments.

Runs in sequence:
1. Categorization ablation study (categorized vs universal vs hint)
2. DSPy global optimization
3. DSPy per-category optimization
4. Format comparison (ChatML vs XML vs minimal)
5. Compiles results into comparison report

Usage:
    uv run python evals/run_all_prompt_experiments.py --quick  # Fast mode (subset)
    uv run python evals/run_all_prompt_experiments.py --full   # Complete (slow)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class ExperimentResult:
    name: str
    status: str  # "success", "failed", "skipped"
    duration_s: float
    output_path: Path | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str = ""


def run_command(
    cmd: list[str],
    description: str,
    timeout: int = 1800,
) -> tuple[bool, str, float]:
    """Run a command and return success, output, duration."""
    print(f"\n{'=' * 70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")

    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=PROJECT_ROOT,
        )
        duration = time.perf_counter() - start

        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        success = result.returncode == 0
        return success, result.stdout + result.stderr, duration

    except subprocess.TimeoutExpired:
        duration = time.perf_counter() - start
        return False, f"Timeout after {timeout}s", duration
    except Exception as e:
        duration = time.perf_counter() - start
        return False, str(e), duration


def run_ablation_study(quick: bool = False) -> ExperimentResult:
    """Run categorization ablation study."""
    name = "ablation_categorization"
    cmd = [
        sys.executable,
        "evals/ablation_categorization.py",
        "--variant",
        "all",
        "--judge",
    ]

    if quick:
        # Will need to modify script to support --limit
        pass

    success, output, duration = run_command(
        cmd,
        "Categorization Ablation Study (categorized vs universal vs hint)",
        timeout=1800,
    )

    output_path = PROJECT_ROOT / "results" / "ablation_categorization.json"
    metrics = {}

    if success and output_path.exists():
        try:
            data = json.loads(output_path.read_text())
            analysis = data.get("analysis", {})
            for variant, stats in analysis.items():
                metrics[f"{variant}_judge_avg"] = stats.get("judge_avg", 0)
                metrics[f"{variant}_anti_ai_rate"] = stats.get("anti_ai_rate", 0)
        except Exception as e:
            print(f"Warning: Could not parse ablation results: {e}")

    return ExperimentResult(
        name=name,
        status="success" if success else "failed",
        duration_s=duration,
        output_path=output_path if output_path.exists() else None,
        metrics=metrics,
        error="" if success else output[-500:],  # Last 500 chars
    )


def run_dspy_global(quick: bool = False) -> ExperimentResult:
    """Run DSPy global optimization."""
    name = "dspy_global"

    # First run optimization
    opt_cmd = [
        sys.executable,
        "evals/dspy_optimize.py",
        "--optimizer",
        "bootstrap" if quick else "mipro",
    ]

    success, output, duration = run_command(
        opt_cmd,
        "DSPy Global Optimization",
        timeout=3600 if not quick else 600,
    )

    if not success:
        return ExperimentResult(
            name=name,
            status="failed",
            duration_s=duration,
            error=output[-500:],
        )

    # Then evaluate
    eval_cmd = [
        sys.executable,
        "evals/batch_eval.py",
        "--judge",
        "--optimized",
    ]

    success, output, duration_eval = run_command(
        eval_cmd,
        "DSPy Global Evaluation",
        timeout=600,
    )

    return ExperimentResult(
        name=name,
        status="success" if success else "failed",
        duration_s=duration + duration_eval,
        output_path=PROJECT_ROOT / "results" / "batch_eval_latest.json",
        error="" if success else output[-500:],
    )


def run_dspy_per_category(quick: bool = False) -> ExperimentResult:
    """Run DSPy per-category optimization."""
    name = "dspy_per_category"

    # Run optimization
    opt_cmd = [
        sys.executable,
        "evals/dspy_optimize.py",
        "--per-category",
        "--optimizer",
        "bootstrap" if quick else "mipro",
    ]

    success, output, duration = run_command(
        opt_cmd,
        "DSPy Per-Category Optimization",
        timeout=3600 if not quick else 900,
    )

    if not success:
        return ExperimentResult(
            name=name,
            status="failed",
            duration_s=duration,
            error=output[-500:],
        )

    # Evaluate
    eval_cmd = [
        sys.executable,
        "evals/dspy_optimize.py",
        "--eval-only",
        "--per-category",
    ]

    success, output, duration_eval = run_command(
        eval_cmd,
        "DSPy Per-Category Evaluation",
        timeout=600,
    )

    return ExperimentResult(
        name=name,
        status="success" if success else "failed",
        duration_s=duration + duration_eval,
        output_path=PROJECT_ROOT / "evals" / "optimized_categories" / "summary.json",
        error="" if success else output[-500:],
    )


def run_format_comparison() -> ExperimentResult:
    """Compare prompt formats (ChatML vs XML vs minimal)."""
    name = "format_comparison"

    # This would need a dedicated script - for now, placeholder
    print(f"\n{'=' * 70}")
    print("Format Comparison (ChatML vs XML vs Minimal)")
    print("NOTE: Using eval_pipeline as baseline for format comparison")
    print(f"{'=' * 70}\n")

    cmd = [
        sys.executable,
        "evals/eval_pipeline.py",
        "--judge",
    ]

    success, output, duration = run_command(
        cmd,
        "Baseline Eval Pipeline",
        timeout=600,
    )

    return ExperimentResult(
        name=name,
        status="success" if success else "failed",
        duration_s=duration,
        output_path=PROJECT_ROOT / "results" / "eval_pipeline_baseline.json",
        error="" if success else output[-500:],
    )


def compile_report(results: list[ExperimentResult]) -> dict:
    """Compile all results into a summary report."""
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiments": [],
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.status == "success"),
            "failed": sum(1 for r in results if r.status == "failed"),
            "total_duration_s": sum(r.duration_s for r in results),
        },
    }

    for r in results:
        exp_data = {
            "name": r.name,
            "status": r.status,
            "duration_s": round(r.duration_s, 1),
            "output_path": str(r.output_path) if r.output_path else None,
            "metrics": r.metrics,
        }
        if r.error:
            exp_data["error_preview"] = r.error[:200]
        report["experiments"].append(exp_data)

    return report


def print_report(report: dict) -> None:
    """Print formatted report to console."""
    print("\n" + "=" * 70)
    print("PROMPT EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)

    summary = report["summary"]
    print(f"\nTotal Experiments: {summary['total']}")
    print(f"Passed: {summary['passed']} | Failed: {summary['failed']}")
    print(
        f"Total Duration: {summary['total_duration_s']:.1f}s ({summary['total_duration_s'] / 60:.1f} min)"
    )

    print("\n" + "-" * 70)
    print("INDIVIDUAL RESULTS")
    print("-" * 70)

    for exp in report["experiments"]:
        status_icon = "✅" if exp["status"] == "success" else "❌"
        print(f"\n{status_icon} {exp['name']}")
        print(f"   Status: {exp['status']}")
        print(f"   Duration: {exp['duration_s']:.1f}s")
        if exp["output_path"]:
            print(f"   Output: {exp['output_path']}")
        if exp["metrics"]:
            print("   Metrics:")
            for k, v in exp["metrics"].items():
                if isinstance(v, float):
                    print(f"      {k}: {v:.3f}")
                else:
                    print(f"      {k}: {v}")
        if exp.get("error_preview"):
            print(f"   Error: {exp['error_preview']}...")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Ablation results
    ablation = next(
        (e for e in report["experiments"] if e["name"] == "ablation_categorization"), None
    )
    if ablation and ablation["metrics"]:
        print("\n1. Categorization Ablation:")
        for k, v in ablation["metrics"].items():
            if "judge_avg" in k:
                print(f"   {k}: {v:.2f}/10")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Review individual experiment outputs in results/ directory
2. Compare judge scores across variants
3. If universal prompt wins: Simplify production code
4. If per-category wins: Deploy optimized category prompts
5. Update PROMPT_VERSION in jarvis/prompts/constants.py
6. Run regression tests: make test
""")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all prompt optimization experiments")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: use faster optimizers, smaller subsets",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=["ablation", "dspy_global", "dspy_per_category", "format"],
        help="Skip specific experiments",
        default=[],
    )
    args = parser.parse_args()

    print("=" * 70)
    print("JARVIS PROMPT OPTIMIZATION EXPERIMENT SUITE")
    print("=" * 70)
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}")
    print(f"Skip: {args.skip if args.skip else 'None'}")
    print(f"Results will be saved to: {PROJECT_ROOT}/results/")
    print("=" * 70)

    # Check API key
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        print("\n❌ ERROR: .env file not found!")
        print("   DSPy optimization requires CEREBRAS_API_KEY in .env")
        return 1

    env_content = env_path.read_text()
    if "CEREBRAS_API_KEY" not in env_content or "csk-" not in env_content:
        print("\n❌ ERROR: CEREBRAS_API_KEY not found in .env!")
        return 1

    print("\n✅ API key found in .env")

    # Run experiments
    results = []
    experiments_to_run = [
        ("ablation", run_ablation_study),
        ("format", run_format_comparison),
        ("dspy_global", run_dspy_global),
        ("dspy_per_category", run_dspy_per_category),
    ]

    for name, func in experiments_to_run:
        if name in args.skip:
            print(f"\n⏭️  Skipping {name}")
            continue

        try:
            if name.startswith("dspy"):
                result = func(quick=args.quick)
            else:
                result = func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Exception in {name}: {e}")
            results.append(
                ExperimentResult(
                    name=name,
                    status="failed",
                    duration_s=0,
                    error=str(e),
                )
            )

    # Compile and save report
    report = compile_report(results)
    report_path = PROJECT_ROOT / "results" / "prompt_experiments_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    # Print to console
    print_report(report)

    print(f"\n📊 Full report saved to: {report_path}")

    return 0 if all(r.status == "success" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
