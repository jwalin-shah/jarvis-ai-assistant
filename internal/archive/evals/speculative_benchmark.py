#!/usr/bin/env python3
"""A/B benchmark: baseline vs speculative decoding.

Runs the same test prompts in two modes:
1. Baseline: target model only (no draft)
2. Speculative: target + draft model

Compares latency, tokens/sec, acceptance rate, and output quality.

Usage:
    uv run python evals/speculative_benchmark.py
    uv run python evals/speculative_benchmark.py --draft-model lfm-0.3b
    uv run python evals/speculative_benchmark.py --num-draft-tokens 5
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
  # noqa: E402
from evals.batch_eval import TEST_CASES, build_prompt, check_result  # noqa: E402


  # noqa: E402
  # noqa: E402
@dataclass  # noqa: E402
class BenchmarkResult:  # noqa: E402
    name: str  # noqa: E402
    variant: str  # "baseline" or "speculative"  # noqa: E402
    output: str  # noqa: E402
    latency_ms: float  # noqa: E402
    tokens_per_second: float  # noqa: E402
    tokens_generated: int  # noqa: E402
    acceptance_rate: float  # noqa: E402
    checks_passed: list[str]  # noqa: E402
    checks_failed: list[str]  # noqa: E402
    passed: bool  # noqa: E402
  # noqa: E402
  # noqa: E402
def run_variant(  # noqa: E402
    loader,  # noqa: E402
    test_cases: list[dict],  # noqa: E402
    variant: str,  # noqa: E402
) -> list[BenchmarkResult]:  # noqa: E402
    """Run all test cases with the given loader configuration."""  # noqa: E402
    results = []  # noqa: E402
    for tc in test_cases:  # noqa: E402
        prompt = build_prompt(tc)  # noqa: E402
        gen_start = time.perf_counter()  # noqa: E402
        try:  # noqa: E402
            result = loader.generate_sync(  # noqa: E402
                prompt=prompt,  # noqa: E402
                temperature=0.7,  # noqa: E402
                max_tokens=50,  # noqa: E402
                top_p=0.1,  # noqa: E402
                top_k=50,  # noqa: E402
                repetition_penalty=1.05,  # noqa: E402
            )  # noqa: E402
            output = result.text.strip()  # noqa: E402
            latency_ms = (time.perf_counter() - gen_start) * 1000  # noqa: E402
            tps = result.tokens_per_second  # noqa: E402
            tokens = result.tokens_generated  # noqa: E402
            acceptance = result.acceptance_rate  # noqa: E402
        except Exception as e:  # noqa: E402
            output = f"[ERROR: {e}]"  # noqa: E402
            latency_ms = (time.perf_counter() - gen_start) * 1000  # noqa: E402
            tps = 0.0  # noqa: E402
            tokens = 0  # noqa: E402
            acceptance = 0.0  # noqa: E402
  # noqa: E402
        passed_checks, failed_checks = check_result(tc, output)  # noqa: E402
        results.append(  # noqa: E402
            BenchmarkResult(  # noqa: E402
                name=tc["name"],  # noqa: E402
                variant=variant,  # noqa: E402
                output=output,  # noqa: E402
                latency_ms=latency_ms,  # noqa: E402
                tokens_per_second=tps,  # noqa: E402
                tokens_generated=tokens,  # noqa: E402
                acceptance_rate=acceptance,  # noqa: E402
                checks_passed=passed_checks,  # noqa: E402
                checks_failed=failed_checks,  # noqa: E402
                passed=len(failed_checks) == 0,  # noqa: E402
            )  # noqa: E402
        )  # noqa: E402
    return results  # noqa: E402
  # noqa: E402
  # noqa: E402
def print_comparison(baseline: list[BenchmarkResult], speculative: list[BenchmarkResult]) -> None:  # noqa: E402
    """Print side-by-side comparison table."""  # noqa: E402
    print()  # noqa: E402
    print("=" * 80)  # noqa: E402
    print("COMPARISON: Baseline vs Speculative Decoding")  # noqa: E402
    print("=" * 80)  # noqa: E402
  # noqa: E402
    # Per-case comparison  # noqa: E402
    print(f"\n{'Test Case':<30} {'Baseline ms':>12} {'Spec ms':>12} {'Speedup':>8} {'Accept%':>8}")  # noqa: E402
    print("-" * 80)  # noqa: E402
  # noqa: E402
    for b, s in zip(baseline, speculative):  # noqa: E402
        speedup = b.latency_ms / s.latency_ms if s.latency_ms > 0 else 0  # noqa: E402
        accept_pct = f"{s.acceptance_rate * 100:.0f}%" if s.acceptance_rate > 0 else "-"  # noqa: E402
        print(  # noqa: E402
            f"{b.name:<30} {b.latency_ms:>10.0f}ms {s.latency_ms:>10.0f}ms "  # noqa: E402
            f"{speedup:>7.1f}x {accept_pct:>8}"  # noqa: E402
        )  # noqa: E402
  # noqa: E402
    # Summary  # noqa: E402
    def summarize(results: list[BenchmarkResult]) -> dict:  # noqa: E402
        latencies = [r.latency_ms for r in results]  # noqa: E402
        sorted_lat = sorted(latencies)  # noqa: E402
        n = len(sorted_lat)  # noqa: E402
        tps_values = [r.tokens_per_second for r in results if r.tokens_per_second > 0]  # noqa: E402
        pass_count = sum(1 for r in results if r.passed)  # noqa: E402
        acceptance_values = [r.acceptance_rate for r in results if r.acceptance_rate > 0]  # noqa: E402
        return {  # noqa: E402
            "avg_latency": sum(latencies) / n if n else 0,  # noqa: E402
            "p50_latency": sorted_lat[n // 2] if n else 0,  # noqa: E402
            "p95_latency": sorted_lat[min(int(n * 0.95), n - 1)] if n else 0,  # noqa: E402
            "avg_tps": sum(tps_values) / len(tps_values) if tps_values else 0,  # noqa: E402
            "pass_rate": pass_count / n if n else 0,  # noqa: E402
            "avg_acceptance": (  # noqa: E402
                sum(acceptance_values) / len(acceptance_values) if acceptance_values else 0  # noqa: E402
            ),  # noqa: E402
        }  # noqa: E402
  # noqa: E402
    b_sum = summarize(baseline)  # noqa: E402
    s_sum = summarize(speculative)  # noqa: E402
  # noqa: E402
    speedup_avg = b_sum["avg_latency"] / s_sum["avg_latency"] if s_sum["avg_latency"] > 0 else 0  # noqa: E402
    speedup_p95 = b_sum["p95_latency"] / s_sum["p95_latency"] if s_sum["p95_latency"] > 0 else 0  # noqa: E402
  # noqa: E402
    print()  # noqa: E402
    print(f"{'Metric':<25} {'Baseline':>15} {'Speculative':>15} {'Delta':>10}")  # noqa: E402
    print("-" * 65)  # noqa: E402
    print(  # noqa: E402
        f"{'Avg Latency':<25} {b_sum['avg_latency']:>13.0f}ms {s_sum['avg_latency']:>13.0f}ms "  # noqa: E402
        f"{speedup_avg:>8.1f}x"  # noqa: E402
    )  # noqa: E402
    print(f"{'P50 Latency':<25} {b_sum['p50_latency']:>13.0f}ms {s_sum['p50_latency']:>13.0f}ms")  # noqa: E402
    print(  # noqa: E402
        f"{'P95 Latency':<25} {b_sum['p95_latency']:>13.0f}ms {s_sum['p95_latency']:>13.0f}ms "  # noqa: E402
        f"{speedup_p95:>8.1f}x"  # noqa: E402
    )  # noqa: E402
    print(f"{'Avg tok/s':<25} {b_sum['avg_tps']:>14.1f} {s_sum['avg_tps']:>14.1f}")  # noqa: E402
    print(f"{'Quality (local pass)':<25} {b_sum['pass_rate']:>14.0%} {s_sum['pass_rate']:>14.0%}")  # noqa: E402
    print(f"{'Acceptance Rate':<25} {'N/A':>15} {s_sum['avg_acceptance']:>14.0%}")  # noqa: E402
    print("=" * 80)  # noqa: E402
  # noqa: E402
  # noqa: E402
def main() -> int:  # noqa: E402
    import argparse  # noqa: E402
  # noqa: E402
    parser = argparse.ArgumentParser(description="Speculative Decoding Benchmark")  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--draft-model", default="models/lfm2-350m-extract-mlx-4bit", help="Draft model ID"  # noqa: E402
    )  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--num-draft-tokens", type=int, default=3, help="Draft tokens per step (default: 3)"  # noqa: E402
    )  # noqa: E402
    args = parser.parse_args()  # noqa: E402
  # noqa: E402
    print("=" * 80)  # noqa: E402
    print("SPECULATIVE DECODING BENCHMARK")  # noqa: E402
    print("=" * 80)  # noqa: E402
    print(f"Test cases:        {len(TEST_CASES)}")  # noqa: E402
    print(f"Draft model:       {args.draft_model}")  # noqa: E402
    print(f"Num draft tokens:  {args.num_draft_tokens}")  # noqa: E402
    print()  # noqa: E402
  # noqa: E402
    # Load target model  # noqa: E402
    from models.loader import MLXModelLoader  # noqa: E402
  # noqa: E402
    loader = MLXModelLoader()  # noqa: E402
    print("Loading target model...")  # noqa: E402
    load_start = time.perf_counter()  # noqa: E402
    loader.load()  # noqa: E402
    print(f"Target model loaded in {(time.perf_counter() - load_start) * 1000:.0f}ms")  # noqa: E402
  # noqa: E402
    # === Baseline run ===  # noqa: E402
    print("\n--- Baseline (target only) ---")  # noqa: E402
    baseline_start = time.perf_counter()  # noqa: E402
    baseline_results = run_variant(loader, TEST_CASES, "baseline")  # noqa: E402
    baseline_total = (time.perf_counter() - baseline_start) * 1000  # noqa: E402
    n_pass_b = sum(1 for r in baseline_results if r.passed)  # noqa: E402
    print(f"Baseline done: {n_pass_b}/{len(baseline_results)} passed, {baseline_total:.0f}ms total")  # noqa: E402
  # noqa: E402
    # === Load draft model ===  # noqa: E402
    print(f"\nLoading draft model ({args.draft_model})...")  # noqa: E402
    draft_start = time.perf_counter()  # noqa: E402
    if not loader.load_draft_model(args.draft_model):  # noqa: E402
        print("FATAL: Failed to load draft model. Tokenizer mismatch?")  # noqa: E402
        return 1  # noqa: E402
    print(f"Draft model loaded in {(time.perf_counter() - draft_start) * 1000:.0f}ms")  # noqa: E402
  # noqa: E402
    # === Speculative run ===  # noqa: E402
    print("\n--- Speculative (target + draft) ---")  # noqa: E402
    spec_start = time.perf_counter()  # noqa: E402
    speculative_results = run_variant(loader, TEST_CASES, "speculative")  # noqa: E402
    spec_total = (time.perf_counter() - spec_start) * 1000  # noqa: E402
    n_pass_s = sum(1 for r in speculative_results if r.passed)  # noqa: E402
    print(  # noqa: E402
        f"Speculative done: {n_pass_s}/{len(speculative_results)} passed, {spec_total:.0f}ms total"  # noqa: E402
    )  # noqa: E402
  # noqa: E402
    # === Comparison ===  # noqa: E402
    print_comparison(baseline_results, speculative_results)  # noqa: E402
  # noqa: E402
    # === Save results ===  # noqa: E402
    output_path = PROJECT_ROOT / "results" / "speculative_benchmark_latest.json"  # noqa: E402
    output_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E402
  # noqa: E402
    def serialize(results: list[BenchmarkResult]) -> list[dict]:  # noqa: E402
        return [  # noqa: E402
            {  # noqa: E402
                "name": r.name,  # noqa: E402
                "variant": r.variant,  # noqa: E402
                "output": r.output,  # noqa: E402
                "latency_ms": round(r.latency_ms, 1),  # noqa: E402
                "tokens_per_second": round(r.tokens_per_second, 1),  # noqa: E402
                "tokens_generated": r.tokens_generated,  # noqa: E402
                "acceptance_rate": round(r.acceptance_rate, 4),  # noqa: E402
                "passed": r.passed,  # noqa: E402
                "failed_checks": r.checks_failed,  # noqa: E402
            }  # noqa: E402
            for r in results  # noqa: E402
        ]  # noqa: E402
  # noqa: E402
    output_data = {  # noqa: E402
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),  # noqa: E402
        "draft_model": args.draft_model,  # noqa: E402
        "num_draft_tokens": args.num_draft_tokens,  # noqa: E402
        "test_cases": len(TEST_CASES),  # noqa: E402
        "baseline": {  # noqa: E402
            "total_ms": round(baseline_total, 1),  # noqa: E402
            "pass_rate": round(n_pass_b / len(TEST_CASES), 4),  # noqa: E402
            "results": serialize(baseline_results),  # noqa: E402
        },  # noqa: E402
        "speculative": {  # noqa: E402
            "total_ms": round(spec_total, 1),  # noqa: E402
            "pass_rate": round(n_pass_s / len(TEST_CASES), 4),  # noqa: E402
            "results": serialize(speculative_results),  # noqa: E402
        },  # noqa: E402
    }  # noqa: E402
    output_path.write_text(json.dumps(output_data, indent=2))  # noqa: E402
    print(f"\nResults saved to: {output_path}")  # noqa: E402
  # noqa: E402
    # Unload draft model  # noqa: E402
    loader.unload_draft_model()  # noqa: E402
  # noqa: E402
    return 0  # noqa: E402
  # noqa: E402
  # noqa: E402
if __name__ == "__main__":  # noqa: E402
    sys.exit(main())  # noqa: E402
