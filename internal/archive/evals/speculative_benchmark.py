#!/usr/bin/env python3  # noqa: E501
"""A/B benchmark: baseline vs speculative decoding.  # noqa: E501
  # noqa: E501
Runs the same test prompts in two modes:  # noqa: E501
1. Baseline: target model only (no draft)  # noqa: E501
2. Speculative: target + draft model  # noqa: E501
  # noqa: E501
Compares latency, tokens/sec, acceptance rate, and output quality.  # noqa: E501
  # noqa: E501
Usage:  # noqa: E501
    uv run python evals/speculative_benchmark.py  # noqa: E501
    uv run python evals/speculative_benchmark.py --draft-model lfm-0.3b  # noqa: E501
    uv run python evals/speculative_benchmark.py --num-draft-tokens 5  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from __future__ import annotations  # noqa: E402  # noqa: E501

# noqa: E501
import json  # noqa: E501
import sys  # noqa: E501
import time  # noqa: E501
from dataclasses import dataclass  # noqa: E402  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501

  # noqa: E501
PROJECT_ROOT = Path(__file__).parent.parent  # noqa: E501
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E501
  # noqa: E501
from evals.batch_eval import TEST_CASES, build_prompt, check_result  # noqa: E402  # noqa: E501


  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class BenchmarkResult:  # noqa: E501
    name: str  # noqa: E501
    variant: str  # "baseline" or "speculative"  # noqa: E501
    output: str  # noqa: E501
    latency_ms: float  # noqa: E501
    tokens_per_second: float  # noqa: E501
    tokens_generated: int  # noqa: E501
    acceptance_rate: float  # noqa: E501
    checks_passed: list[str]  # noqa: E501
    checks_failed: list[str]  # noqa: E501
    passed: bool  # noqa: E501
  # noqa: E501
  # noqa: E501
def run_variant(  # noqa: E501
    loader,  # noqa: E501
    test_cases: list[dict],  # noqa: E501
    variant: str,  # noqa: E501
) -> list[BenchmarkResult]:  # noqa: E501
    """Run all test cases with the given loader configuration."""  # noqa: E501
    results = []  # noqa: E501
    for tc in test_cases:  # noqa: E501
        prompt = build_prompt(tc)  # noqa: E501
        gen_start = time.perf_counter()  # noqa: E501
        try:  # noqa: E501
            result = loader.generate_sync(  # noqa: E501
                prompt=prompt,  # noqa: E501
                temperature=0.7,  # noqa: E501
                max_tokens=50,  # noqa: E501
                top_p=0.1,  # noqa: E501
                top_k=50,  # noqa: E501
                repetition_penalty=1.05,  # noqa: E501
            )  # noqa: E501
            output = result.text.strip()  # noqa: E501
            latency_ms = (time.perf_counter() - gen_start) * 1000  # noqa: E501
            tps = result.tokens_per_second  # noqa: E501
            tokens = result.tokens_generated  # noqa: E501
            acceptance = result.acceptance_rate  # noqa: E501
        except Exception as e:  # noqa: E501
            output = f"[ERROR: {e}]"  # noqa: E501
            latency_ms = (time.perf_counter() - gen_start) * 1000  # noqa: E501
            tps = 0.0  # noqa: E501
            tokens = 0  # noqa: E501
            acceptance = 0.0  # noqa: E501
  # noqa: E501
        passed_checks, failed_checks = check_result(tc, output)  # noqa: E501
        results.append(  # noqa: E501
            BenchmarkResult(  # noqa: E501
                name=tc["name"],  # noqa: E501
                variant=variant,  # noqa: E501
                output=output,  # noqa: E501
                latency_ms=latency_ms,  # noqa: E501
                tokens_per_second=tps,  # noqa: E501
                tokens_generated=tokens,  # noqa: E501
                acceptance_rate=acceptance,  # noqa: E501
                checks_passed=passed_checks,  # noqa: E501
                checks_failed=failed_checks,  # noqa: E501
                passed=len(failed_checks) == 0,  # noqa: E501
            )  # noqa: E501
        )  # noqa: E501
    return results  # noqa: E501
  # noqa: E501
  # noqa: E501
def print_comparison(baseline: list[BenchmarkResult], speculative: list[BenchmarkResult]) -> None:  # noqa: E501
    """Print side-by-side comparison table."""  # noqa: E501
    print()  # noqa: E501
    print("=" * 80)  # noqa: E501
    print("COMPARISON: Baseline vs Speculative Decoding")  # noqa: E501
    print("=" * 80)  # noqa: E501
  # noqa: E501
    # Per-case comparison  # noqa: E501
    print(f"\n{'Test Case':<30} {'Baseline ms':>12} {'Spec ms':>12} {'Speedup':>8} {'Accept%':>8}")  # noqa: E501
    print("-" * 80)  # noqa: E501
  # noqa: E501
    for b, s in zip(baseline, speculative):  # noqa: E501
        speedup = b.latency_ms / s.latency_ms if s.latency_ms > 0 else 0  # noqa: E501
        accept_pct = f"{s.acceptance_rate * 100:.0f}%" if s.acceptance_rate > 0 else "-"  # noqa: E501
        print(  # noqa: E501
            f"{b.name:<30} {b.latency_ms:>10.0f}ms {s.latency_ms:>10.0f}ms "  # noqa: E501
            f"{speedup:>7.1f}x {accept_pct:>8}"  # noqa: E501
        )  # noqa: E501
  # noqa: E501
    # Summary  # noqa: E501
    def summarize(results: list[BenchmarkResult]) -> dict:  # noqa: E501
        latencies = [r.latency_ms for r in results]  # noqa: E501
        sorted_lat = sorted(latencies)  # noqa: E501
        n = len(sorted_lat)  # noqa: E501
        tps_values = [r.tokens_per_second for r in results if r.tokens_per_second > 0]  # noqa: E501
        pass_count = sum(1 for r in results if r.passed)  # noqa: E501
        acceptance_values = [r.acceptance_rate for r in results if r.acceptance_rate > 0]  # noqa: E501
        return {  # noqa: E501
            "avg_latency": sum(latencies) / n if n else 0,  # noqa: E501
            "p50_latency": sorted_lat[n // 2] if n else 0,  # noqa: E501
            "p95_latency": sorted_lat[min(int(n * 0.95), n - 1)] if n else 0,  # noqa: E501
            "avg_tps": sum(tps_values) / len(tps_values) if tps_values else 0,  # noqa: E501
            "pass_rate": pass_count / n if n else 0,  # noqa: E501
            "avg_acceptance": (  # noqa: E501
                sum(acceptance_values) / len(acceptance_values) if acceptance_values else 0  # noqa: E501
            ),  # noqa: E501
        }  # noqa: E501
  # noqa: E501
    b_sum = summarize(baseline)  # noqa: E501
    s_sum = summarize(speculative)  # noqa: E501
  # noqa: E501
    speedup_avg = b_sum["avg_latency"] / s_sum["avg_latency"] if s_sum["avg_latency"] > 0 else 0  # noqa: E501
    speedup_p95 = b_sum["p95_latency"] / s_sum["p95_latency"] if s_sum["p95_latency"] > 0 else 0  # noqa: E501
  # noqa: E501
    print()  # noqa: E501
    print(f"{'Metric':<25} {'Baseline':>15} {'Speculative':>15} {'Delta':>10}")  # noqa: E501
    print("-" * 65)  # noqa: E501
    print(  # noqa: E501
        f"{'Avg Latency':<25} {b_sum['avg_latency']:>13.0f}ms {s_sum['avg_latency']:>13.0f}ms "  # noqa: E501
        f"{speedup_avg:>8.1f}x"  # noqa: E501
    )  # noqa: E501
    print(f"{'P50 Latency':<25} {b_sum['p50_latency']:>13.0f}ms {s_sum['p50_latency']:>13.0f}ms")  # noqa: E501
    print(  # noqa: E501
        f"{'P95 Latency':<25} {b_sum['p95_latency']:>13.0f}ms {s_sum['p95_latency']:>13.0f}ms "  # noqa: E501
        f"{speedup_p95:>8.1f}x"  # noqa: E501
    )  # noqa: E501
    print(f"{'Avg tok/s':<25} {b_sum['avg_tps']:>14.1f} {s_sum['avg_tps']:>14.1f}")  # noqa: E501
    print(f"{'Quality (local pass)':<25} {b_sum['pass_rate']:>14.0%} {s_sum['pass_rate']:>14.0%}")  # noqa: E501
    print(f"{'Acceptance Rate':<25} {'N/A':>15} {s_sum['avg_acceptance']:>14.0%}")  # noqa: E501
    print("=" * 80)  # noqa: E501
  # noqa: E501
  # noqa: E501
def main() -> int:  # noqa: E501
    import argparse  # noqa: E501
  # noqa: E501
    parser = argparse.ArgumentParser(description="Speculative Decoding Benchmark")  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--draft-model", default="models/lfm2-350m-extract-mlx-4bit", help="Draft model ID"  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--num-draft-tokens", type=int, default=3, help="Draft tokens per step (default: 3)"  # noqa: E501
    )  # noqa: E501
    args = parser.parse_args()  # noqa: E501
  # noqa: E501
    print("=" * 80)  # noqa: E501
    print("SPECULATIVE DECODING BENCHMARK")  # noqa: E501
    print("=" * 80)  # noqa: E501
    print(f"Test cases:        {len(TEST_CASES)}")  # noqa: E501
    print(f"Draft model:       {args.draft_model}")  # noqa: E501
    print(f"Num draft tokens:  {args.num_draft_tokens}")  # noqa: E501
    print()  # noqa: E501
  # noqa: E501
    # Load target model  # noqa: E501
    from models.loader import MLXModelLoader  # noqa: E501
  # noqa: E501
    loader = MLXModelLoader()  # noqa: E501
    print("Loading target model...")  # noqa: E501
    load_start = time.perf_counter()  # noqa: E501
    loader.load()  # noqa: E501
    print(f"Target model loaded in {(time.perf_counter() - load_start) * 1000:.0f}ms")  # noqa: E501
  # noqa: E501
    # === Baseline run ===  # noqa: E501
    print("\n--- Baseline (target only) ---")  # noqa: E501
    baseline_start = time.perf_counter()  # noqa: E501
    baseline_results = run_variant(loader, TEST_CASES, "baseline")  # noqa: E501
    baseline_total = (time.perf_counter() - baseline_start) * 1000  # noqa: E501
    n_pass_b = sum(1 for r in baseline_results if r.passed)  # noqa: E501
    print(f"Baseline done: {n_pass_b}/{len(baseline_results)} passed, {baseline_total:.0f}ms total")  # noqa: E501
  # noqa: E501
    # === Load draft model ===  # noqa: E501
    print(f"\nLoading draft model ({args.draft_model})...")  # noqa: E501
    draft_start = time.perf_counter()  # noqa: E501
    if not loader.load_draft_model(args.draft_model):  # noqa: E501
        print("FATAL: Failed to load draft model. Tokenizer mismatch?")  # noqa: E501
        return 1  # noqa: E501
    print(f"Draft model loaded in {(time.perf_counter() - draft_start) * 1000:.0f}ms")  # noqa: E501
  # noqa: E501
    # === Speculative run ===  # noqa: E501
    print("\n--- Speculative (target + draft) ---")  # noqa: E501
    spec_start = time.perf_counter()  # noqa: E501
    speculative_results = run_variant(loader, TEST_CASES, "speculative")  # noqa: E501
    spec_total = (time.perf_counter() - spec_start) * 1000  # noqa: E501
    n_pass_s = sum(1 for r in speculative_results if r.passed)  # noqa: E501
    print(  # noqa: E501
        f"Speculative done: {n_pass_s}/{len(speculative_results)} passed, {spec_total:.0f}ms total"  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    # === Comparison ===  # noqa: E501
    print_comparison(baseline_results, speculative_results)  # noqa: E501
  # noqa: E501
    # === Save results ===  # noqa: E501
    output_path = PROJECT_ROOT / "results" / "speculative_benchmark_latest.json"  # noqa: E501
    output_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E501
  # noqa: E501
    def serialize(results: list[BenchmarkResult]) -> list[dict]:  # noqa: E501
        return [  # noqa: E501
            {  # noqa: E501
                "name": r.name,  # noqa: E501
                "variant": r.variant,  # noqa: E501
                "output": r.output,  # noqa: E501
                "latency_ms": round(r.latency_ms, 1),  # noqa: E501
                "tokens_per_second": round(r.tokens_per_second, 1),  # noqa: E501
                "tokens_generated": r.tokens_generated,  # noqa: E501
                "acceptance_rate": round(r.acceptance_rate, 4),  # noqa: E501
                "passed": r.passed,  # noqa: E501
                "failed_checks": r.checks_failed,  # noqa: E501
            }  # noqa: E501
            for r in results  # noqa: E501
        ]  # noqa: E501
  # noqa: E501
    output_data = {  # noqa: E501
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),  # noqa: E501
        "draft_model": args.draft_model,  # noqa: E501
        "num_draft_tokens": args.num_draft_tokens,  # noqa: E501
        "test_cases": len(TEST_CASES),  # noqa: E501
        "baseline": {  # noqa: E501
            "total_ms": round(baseline_total, 1),  # noqa: E501
            "pass_rate": round(n_pass_b / len(TEST_CASES), 4),  # noqa: E501
            "results": serialize(baseline_results),  # noqa: E501
        },  # noqa: E501
        "speculative": {  # noqa: E501
            "total_ms": round(spec_total, 1),  # noqa: E501
            "pass_rate": round(n_pass_s / len(TEST_CASES), 4),  # noqa: E501
            "results": serialize(speculative_results),  # noqa: E501
        },  # noqa: E501
    }  # noqa: E501
    output_path.write_text(json.dumps(output_data, indent=2))  # noqa: E501
    print(f"\nResults saved to: {output_path}")  # noqa: E501
  # noqa: E501
    # Unload draft model  # noqa: E501
    loader.unload_draft_model()  # noqa: E501
  # noqa: E501
    return 0  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    sys.exit(main())  # noqa: E501
