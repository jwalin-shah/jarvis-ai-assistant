#!/usr/bin/env python3  # noqa: E501
"""Master script to run all prompt optimization experiments.  # noqa: E501
  # noqa: E501
Runs in sequence:  # noqa: E501
1. Categorization ablation study (categorized vs universal vs hint)  # noqa: E501
2. DSPy global optimization  # noqa: E501
3. DSPy per-category optimization  # noqa: E501
4. Format comparison (ChatML vs XML vs minimal)  # noqa: E501
5. Compiles results into comparison report  # noqa: E501
  # noqa: E501
Usage:  # noqa: E501
    uv run python evals/run_all_prompt_experiments.py --quick  # Fast mode (subset)  # noqa: E501
    uv run python evals/run_all_prompt_experiments.py --full   # Complete (slow)  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from __future__ import annotations  # noqa: E402  # noqa: E501

# noqa: E501
import argparse  # noqa: E501
import json  # noqa: E501
import subprocess  # noqa: E501
import sys  # noqa: E501
import time  # noqa: E501
from dataclasses import dataclass, field  # noqa: E402  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501
from typing import Any  # noqa: E402  # noqa: E501

  # noqa: E501
PROJECT_ROOT = Path(__file__).parent.parent  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class ExperimentResult:  # noqa: E501
    name: str  # noqa: E501
    status: str  # "success", "failed", "skipped"  # noqa: E501
    duration_s: float  # noqa: E501
    output_path: Path | None = None  # noqa: E501
    metrics: dict[str, Any] = field(default_factory=dict)  # noqa: E501
    error: str = ""  # noqa: E501
  # noqa: E501
  # noqa: E501
def run_command(  # noqa: E501
    cmd: list[str],  # noqa: E501
    description: str,  # noqa: E501
    timeout: int = 1800,  # noqa: E501
) -> tuple[bool, str, float]:  # noqa: E501
    """Run a command and return success, output, duration."""  # noqa: E501
    print(f"\n{'=' * 70}")  # noqa: E501
    print(f"Running: {description}")  # noqa: E501
    print(f"Command: {' '.join(cmd)}")  # noqa: E501
    print(f"{'=' * 70}\n")  # noqa: E501
  # noqa: E501
    start = time.perf_counter()  # noqa: E501
    try:  # noqa: E501
        result = subprocess.run(  # noqa: E501
            cmd,  # noqa: E501
            capture_output=True,  # noqa: E501
            text=True,  # noqa: E501
            timeout=timeout,  # noqa: E501
            cwd=PROJECT_ROOT,  # noqa: E501
        )  # noqa: E501
        duration = time.perf_counter() - start  # noqa: E501
  # noqa: E501
        # Print output  # noqa: E501
        if result.stdout:  # noqa: E501
            print(result.stdout)  # noqa: E501
        if result.stderr:  # noqa: E501
            print(result.stderr, file=sys.stderr)  # noqa: E501
  # noqa: E501
        success = result.returncode == 0  # noqa: E501
        return success, result.stdout + result.stderr, duration  # noqa: E501
  # noqa: E501
    except subprocess.TimeoutExpired:  # noqa: E501
        duration = time.perf_counter() - start  # noqa: E501
        return False, f"Timeout after {timeout}s", duration  # noqa: E501
    except Exception as e:  # noqa: E501
        duration = time.perf_counter() - start  # noqa: E501
        return False, str(e), duration  # noqa: E501
  # noqa: E501
  # noqa: E501
def run_ablation_study(quick: bool = False) -> ExperimentResult:  # noqa: E501
    """Run categorization ablation study."""  # noqa: E501
    name = "ablation_categorization"  # noqa: E501
    cmd = [  # noqa: E501
        sys.executable,  # noqa: E501
        "evals/ablation_categorization.py",  # noqa: E501
        "--variant",  # noqa: E501
        "all",  # noqa: E501
        "--judge",  # noqa: E501
    ]  # noqa: E501
  # noqa: E501
    if quick:  # noqa: E501
        # Will need to modify script to support --limit  # noqa: E501
        pass  # noqa: E501
  # noqa: E501
    success, output, duration = run_command(  # noqa: E501
        cmd,  # noqa: E501
        "Categorization Ablation Study (categorized vs universal vs hint)",  # noqa: E501
        timeout=1800,  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    output_path = PROJECT_ROOT / "results" / "ablation_categorization.json"  # noqa: E501
    metrics = {}  # noqa: E501
  # noqa: E501
    if success and output_path.exists():  # noqa: E501
        try:  # noqa: E501
            data = json.loads(output_path.read_text())  # noqa: E501
            analysis = data.get("analysis", {})  # noqa: E501
            for variant, stats in analysis.items():  # noqa: E501
                metrics[f"{variant}_judge_avg"] = stats.get("judge_avg", 0)  # noqa: E501
                metrics[f"{variant}_anti_ai_rate"] = stats.get("anti_ai_rate", 0)  # noqa: E501
        except Exception as e:  # noqa: E501
            print(f"Warning: Could not parse ablation results: {e}")  # noqa: E501
  # noqa: E501
    return ExperimentResult(  # noqa: E501
        name=name,  # noqa: E501
        status="success" if success else "failed",  # noqa: E501
        duration_s=duration,  # noqa: E501
        output_path=output_path if output_path.exists() else None,  # noqa: E501
        metrics=metrics,  # noqa: E501
        error="" if success else output[-500:],  # Last 500 chars  # noqa: E501
    )  # noqa: E501
  # noqa: E501
  # noqa: E501
def run_dspy_global(quick: bool = False) -> ExperimentResult:  # noqa: E501
    """Run DSPy global optimization."""  # noqa: E501
    name = "dspy_global"  # noqa: E501
  # noqa: E501
    # First run optimization  # noqa: E501
    opt_cmd = [  # noqa: E501
        sys.executable,  # noqa: E501
        "evals/dspy_optimize.py",  # noqa: E501
        "--optimizer",  # noqa: E501
        "bootstrap" if quick else "mipro",  # noqa: E501
    ]  # noqa: E501
  # noqa: E501
    success, output, duration = run_command(  # noqa: E501
        opt_cmd,  # noqa: E501
        "DSPy Global Optimization",  # noqa: E501
        timeout=3600 if not quick else 600,  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    if not success:  # noqa: E501
        return ExperimentResult(  # noqa: E501
            name=name,  # noqa: E501
            status="failed",  # noqa: E501
            duration_s=duration,  # noqa: E501
            error=output[-500:],  # noqa: E501
        )  # noqa: E501
  # noqa: E501
    # Then evaluate  # noqa: E501
    eval_cmd = [  # noqa: E501
        sys.executable,  # noqa: E501
        "evals/batch_eval.py",  # noqa: E501
        "--judge",  # noqa: E501
        "--optimized",  # noqa: E501
    ]  # noqa: E501
  # noqa: E501
    success, output, duration_eval = run_command(  # noqa: E501
        eval_cmd,  # noqa: E501
        "DSPy Global Evaluation",  # noqa: E501
        timeout=600,  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    return ExperimentResult(  # noqa: E501
        name=name,  # noqa: E501
        status="success" if success else "failed",  # noqa: E501
        duration_s=duration + duration_eval,  # noqa: E501
        output_path=PROJECT_ROOT / "results" / "batch_eval_latest.json",  # noqa: E501
        error="" if success else output[-500:],  # noqa: E501
    )  # noqa: E501
  # noqa: E501
  # noqa: E501
def run_dspy_per_category(quick: bool = False) -> ExperimentResult:  # noqa: E501
    """Run DSPy per-category optimization."""  # noqa: E501
    name = "dspy_per_category"  # noqa: E501
  # noqa: E501
    # Run optimization  # noqa: E501
    opt_cmd = [  # noqa: E501
        sys.executable,  # noqa: E501
        "evals/dspy_optimize.py",  # noqa: E501
        "--per-category",  # noqa: E501
        "--optimizer",  # noqa: E501
        "bootstrap" if quick else "mipro",  # noqa: E501
    ]  # noqa: E501
  # noqa: E501
    success, output, duration = run_command(  # noqa: E501
        opt_cmd,  # noqa: E501
        "DSPy Per-Category Optimization",  # noqa: E501
        timeout=3600 if not quick else 900,  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    if not success:  # noqa: E501
        return ExperimentResult(  # noqa: E501
            name=name,  # noqa: E501
            status="failed",  # noqa: E501
            duration_s=duration,  # noqa: E501
            error=output[-500:],  # noqa: E501
        )  # noqa: E501
  # noqa: E501
    # Evaluate  # noqa: E501
    eval_cmd = [  # noqa: E501
        sys.executable,  # noqa: E501
        "evals/dspy_optimize.py",  # noqa: E501
        "--eval-only",  # noqa: E501
        "--per-category",  # noqa: E501
    ]  # noqa: E501
  # noqa: E501
    success, output, duration_eval = run_command(  # noqa: E501
        eval_cmd,  # noqa: E501
        "DSPy Per-Category Evaluation",  # noqa: E501
        timeout=600,  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    return ExperimentResult(  # noqa: E501
        name=name,  # noqa: E501
        status="success" if success else "failed",  # noqa: E501
        duration_s=duration + duration_eval,  # noqa: E501
        output_path=PROJECT_ROOT / "evals" / "optimized_categories" / "summary.json",  # noqa: E501
        error="" if success else output[-500:],  # noqa: E501
    )  # noqa: E501
  # noqa: E501
  # noqa: E501
def run_format_comparison() -> ExperimentResult:  # noqa: E501
    """Compare prompt formats (ChatML vs XML vs minimal)."""  # noqa: E501
    name = "format_comparison"  # noqa: E501
  # noqa: E501
    # This would need a dedicated script - for now, placeholder  # noqa: E501
    print(f"\n{'=' * 70}")  # noqa: E501
    print("Format Comparison (ChatML vs XML vs Minimal)")  # noqa: E501
    print("NOTE: Using eval_pipeline as baseline for format comparison")  # noqa: E501
    print(f"{'=' * 70}\n")  # noqa: E501
  # noqa: E501
    cmd = [  # noqa: E501
        sys.executable,  # noqa: E501
        "evals/eval_pipeline.py",  # noqa: E501
        "--judge",  # noqa: E501
    ]  # noqa: E501
  # noqa: E501
    success, output, duration = run_command(  # noqa: E501
        cmd,  # noqa: E501
        "Baseline Eval Pipeline",  # noqa: E501
        timeout=600,  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    return ExperimentResult(  # noqa: E501
        name=name,  # noqa: E501
        status="success" if success else "failed",  # noqa: E501
        duration_s=duration,  # noqa: E501
        output_path=PROJECT_ROOT / "results" / "eval_pipeline_baseline.json",  # noqa: E501
        error="" if success else output[-500:],  # noqa: E501
    )  # noqa: E501
  # noqa: E501
  # noqa: E501
def compile_report(results: list[ExperimentResult]) -> dict:  # noqa: E501
    """Compile all results into a summary report."""  # noqa: E501
    report = {  # noqa: E501
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),  # noqa: E501
        "experiments": [],  # noqa: E501
        "summary": {  # noqa: E501
            "total": len(results),  # noqa: E501
            "passed": sum(1 for r in results if r.status == "success"),  # noqa: E501
            "failed": sum(1 for r in results if r.status == "failed"),  # noqa: E501
            "total_duration_s": sum(r.duration_s for r in results),  # noqa: E501
        },  # noqa: E501
    }  # noqa: E501
  # noqa: E501
    for r in results:  # noqa: E501
        exp_data = {  # noqa: E501
            "name": r.name,  # noqa: E501
            "status": r.status,  # noqa: E501
            "duration_s": round(r.duration_s, 1),  # noqa: E501
            "output_path": str(r.output_path) if r.output_path else None,  # noqa: E501
            "metrics": r.metrics,  # noqa: E501
        }  # noqa: E501
        if r.error:  # noqa: E501
            exp_data["error_preview"] = r.error[:200]  # noqa: E501
        report["experiments"].append(exp_data)  # noqa: E501
  # noqa: E501
    return report  # noqa: E501
  # noqa: E501
  # noqa: E501
def print_report(report: dict) -> None:  # noqa: E501
    """Print formatted report to console."""  # noqa: E501
    print("\n" + "=" * 70)  # noqa: E501
    print("PROMPT EXPERIMENT RESULTS SUMMARY")  # noqa: E501
    print("=" * 70)  # noqa: E501
  # noqa: E501
    summary = report["summary"]  # noqa: E501
    print(f"\nTotal Experiments: {summary['total']}")  # noqa: E501
    print(f"Passed: {summary['passed']} | Failed: {summary['failed']}")  # noqa: E501
    print(  # noqa: E501
        f"Total Duration: {summary['total_duration_s']:.1f}s ({summary['total_duration_s'] / 60:.1f} min)"  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    print("\n" + "-" * 70)  # noqa: E501
    print("INDIVIDUAL RESULTS")  # noqa: E501
    print("-" * 70)  # noqa: E501
  # noqa: E501
    for exp in report["experiments"]:  # noqa: E501
        status_icon = "✅" if exp["status"] == "success" else "❌"  # noqa: E501
        print(f"\n{status_icon} {exp['name']}")  # noqa: E501
        print(f"   Status: {exp['status']}")  # noqa: E501
        print(f"   Duration: {exp['duration_s']:.1f}s")  # noqa: E501
        if exp["output_path"]:  # noqa: E501
            print(f"   Output: {exp['output_path']}")  # noqa: E501
        if exp["metrics"]:  # noqa: E501
            print("   Metrics:")  # noqa: E501
            for k, v in exp["metrics"].items():  # noqa: E501
                if isinstance(v, float):  # noqa: E501
                    print(f"      {k}: {v:.3f}")  # noqa: E501
                else:  # noqa: E501
                    print(f"      {k}: {v}")  # noqa: E501
        if exp.get("error_preview"):  # noqa: E501
            print(f"   Error: {exp['error_preview']}...")  # noqa: E501
  # noqa: E501
    # Key findings  # noqa: E501
    print("\n" + "=" * 70)  # noqa: E501
    print("KEY FINDINGS")  # noqa: E501
    print("=" * 70)  # noqa: E501
  # noqa: E501
    # Ablation results  # noqa: E501
    ablation = next(  # noqa: E501
        (e for e in report["experiments"] if e["name"] == "ablation_categorization"), None  # noqa: E501
    )  # noqa: E501
    if ablation and ablation["metrics"]:  # noqa: E501
        print("\n1. Categorization Ablation:")  # noqa: E501
        for k, v in ablation["metrics"].items():  # noqa: E501
            if "judge_avg" in k:  # noqa: E501
                print(f"   {k}: {v:.2f}/10")  # noqa: E501
  # noqa: E501
    print("\n" + "=" * 70)  # noqa: E501
    print("NEXT STEPS")  # noqa: E501
    print("=" * 70)  # noqa: E501
    print("""  # noqa: E501
1. Review individual experiment outputs in results/ directory  # noqa: E501
2. Compare judge scores across variants  # noqa: E501
3. If universal prompt wins: Simplify production code  # noqa: E501
4. If per-category wins: Deploy optimized category prompts  # noqa: E501
5. Update PROMPT_VERSION in jarvis/prompts/constants.py  # noqa: E501
6. Run regression tests: make test  # noqa: E501
""")  # noqa: E501
  # noqa: E501
  # noqa: E501
def main() -> int:  # noqa: E501
    parser = argparse.ArgumentParser(description="Run all prompt optimization experiments")  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--quick",  # noqa: E501
        action="store_true",  # noqa: E501
        help="Quick mode: use faster optimizers, smaller subsets",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--skip",  # noqa: E501
        nargs="+",  # noqa: E501
        choices=["ablation", "dspy_global", "dspy_per_category", "format"],  # noqa: E501
        help="Skip specific experiments",  # noqa: E501
        default=[],  # noqa: E501
    )  # noqa: E501
    args = parser.parse_args()  # noqa: E501
  # noqa: E501
    print("=" * 70)  # noqa: E501
    print("JARVIS PROMPT OPTIMIZATION EXPERIMENT SUITE")  # noqa: E501
    print("=" * 70)  # noqa: E501
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}")  # noqa: E501
    print(f"Skip: {args.skip if args.skip else 'None'}")  # noqa: E501
    print(f"Results will be saved to: {PROJECT_ROOT}/results/")  # noqa: E501
    print("=" * 70)  # noqa: E501
  # noqa: E501
    # Check API key  # noqa: E501
    env_path = PROJECT_ROOT / ".env"  # noqa: E501
    if not env_path.exists():  # noqa: E501
        print("\n❌ ERROR: .env file not found!")  # noqa: E501
        print("   DSPy optimization requires CEREBRAS_API_KEY in .env")  # noqa: E501
        return 1  # noqa: E501
  # noqa: E501
    env_content = env_path.read_text()  # noqa: E501
    if "CEREBRAS_API_KEY" not in env_content or "csk-" not in env_content:  # noqa: E501
        print("\n❌ ERROR: CEREBRAS_API_KEY not found in .env!")  # noqa: E501
        return 1  # noqa: E501
  # noqa: E501
    print("\n✅ API key found in .env")  # noqa: E501
  # noqa: E501
    # Run experiments  # noqa: E501
    results = []  # noqa: E501
    experiments_to_run = [  # noqa: E501
        ("ablation", run_ablation_study),  # noqa: E501
        ("format", run_format_comparison),  # noqa: E501
        ("dspy_global", run_dspy_global),  # noqa: E501
        ("dspy_per_category", run_dspy_per_category),  # noqa: E501
    ]  # noqa: E501
  # noqa: E501
    for name, func in experiments_to_run:  # noqa: E501
        if name in args.skip:  # noqa: E501
            print(f"\n⏭️  Skipping {name}")  # noqa: E501
            continue  # noqa: E501
  # noqa: E501
        try:  # noqa: E501
            if name.startswith("dspy"):  # noqa: E501
                result = func(quick=args.quick)  # noqa: E501
            else:  # noqa: E501
                result = func()  # noqa: E501
            results.append(result)  # noqa: E501
        except Exception as e:  # noqa: E501
            print(f"\n❌ Exception in {name}: {e}")  # noqa: E501
            results.append(  # noqa: E501
                ExperimentResult(  # noqa: E501
                    name=name,  # noqa: E501
                    status="failed",  # noqa: E501
                    duration_s=0,  # noqa: E501
                    error=str(e),  # noqa: E501
                )  # noqa: E501
            )  # noqa: E501
  # noqa: E501
    # Compile and save report  # noqa: E501
    report = compile_report(results)  # noqa: E501
    report_path = PROJECT_ROOT / "results" / "prompt_experiments_report.json"  # noqa: E501
    report_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E501
    report_path.write_text(json.dumps(report, indent=2))  # noqa: E501
  # noqa: E501
    # Print to console  # noqa: E501
    print_report(report)  # noqa: E501
  # noqa: E501
    print(f"\n📊 Full report saved to: {report_path}")  # noqa: E501
  # noqa: E501
    return 0 if all(r.status == "success" for r in results) else 1  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    sys.exit(main())  # noqa: E501
