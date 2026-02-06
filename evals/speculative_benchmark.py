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

from evals.batch_eval import TEST_CASES, build_prompt, check_result  # noqa: E402


@dataclass
class BenchmarkResult:
    name: str
    variant: str  # "baseline" or "speculative"
    output: str
    latency_ms: float
    tokens_per_second: float
    tokens_generated: int
    acceptance_rate: float
    checks_passed: list[str]
    checks_failed: list[str]
    passed: bool


def run_variant(
    loader,
    test_cases: list[dict],
    variant: str,
) -> list[BenchmarkResult]:
    """Run all test cases with the given loader configuration."""
    results = []
    for tc in test_cases:
        prompt = build_prompt(tc)
        gen_start = time.perf_counter()
        try:
            result = loader.generate_sync(
                prompt=prompt,
                temperature=0.7,
                max_tokens=50,
                top_p=0.1,
                top_k=50,
                repetition_penalty=1.05,
            )
            output = result.text.strip()
            latency_ms = (time.perf_counter() - gen_start) * 1000
            tps = result.tokens_per_second
            tokens = result.tokens_generated
            acceptance = result.acceptance_rate
        except Exception as e:
            output = f"[ERROR: {e}]"
            latency_ms = (time.perf_counter() - gen_start) * 1000
            tps = 0.0
            tokens = 0
            acceptance = 0.0

        passed_checks, failed_checks = check_result(tc, output)
        results.append(
            BenchmarkResult(
                name=tc["name"],
                variant=variant,
                output=output,
                latency_ms=latency_ms,
                tokens_per_second=tps,
                tokens_generated=tokens,
                acceptance_rate=acceptance,
                checks_passed=passed_checks,
                checks_failed=failed_checks,
                passed=len(failed_checks) == 0,
            )
        )
    return results


def print_comparison(baseline: list[BenchmarkResult], speculative: list[BenchmarkResult]) -> None:
    """Print side-by-side comparison table."""
    print()
    print("=" * 80)
    print("COMPARISON: Baseline vs Speculative Decoding")
    print("=" * 80)

    # Per-case comparison
    print(f"\n{'Test Case':<30} {'Baseline ms':>12} {'Spec ms':>12} {'Speedup':>8} {'Accept%':>8}")
    print("-" * 80)

    for b, s in zip(baseline, speculative):
        speedup = b.latency_ms / s.latency_ms if s.latency_ms > 0 else 0
        accept_pct = f"{s.acceptance_rate * 100:.0f}%" if s.acceptance_rate > 0 else "-"
        print(
            f"{b.name:<30} {b.latency_ms:>10.0f}ms {s.latency_ms:>10.0f}ms "
            f"{speedup:>7.1f}x {accept_pct:>8}"
        )

    # Summary
    def summarize(results: list[BenchmarkResult]) -> dict:
        latencies = [r.latency_ms for r in results]
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        tps_values = [r.tokens_per_second for r in results if r.tokens_per_second > 0]
        pass_count = sum(1 for r in results if r.passed)
        acceptance_values = [r.acceptance_rate for r in results if r.acceptance_rate > 0]
        return {
            "avg_latency": sum(latencies) / n if n else 0,
            "p50_latency": sorted_lat[n // 2] if n else 0,
            "p95_latency": sorted_lat[min(int(n * 0.95), n - 1)] if n else 0,
            "avg_tps": sum(tps_values) / len(tps_values) if tps_values else 0,
            "pass_rate": pass_count / n if n else 0,
            "avg_acceptance": (
                sum(acceptance_values) / len(acceptance_values) if acceptance_values else 0
            ),
        }

    b_sum = summarize(baseline)
    s_sum = summarize(speculative)

    speedup_avg = b_sum["avg_latency"] / s_sum["avg_latency"] if s_sum["avg_latency"] > 0 else 0
    speedup_p95 = b_sum["p95_latency"] / s_sum["p95_latency"] if s_sum["p95_latency"] > 0 else 0

    print()
    print(f"{'Metric':<25} {'Baseline':>15} {'Speculative':>15} {'Delta':>10}")
    print("-" * 65)
    print(
        f"{'Avg Latency':<25} {b_sum['avg_latency']:>13.0f}ms {s_sum['avg_latency']:>13.0f}ms "
        f"{speedup_avg:>8.1f}x"
    )
    print(f"{'P50 Latency':<25} {b_sum['p50_latency']:>13.0f}ms {s_sum['p50_latency']:>13.0f}ms")
    print(
        f"{'P95 Latency':<25} {b_sum['p95_latency']:>13.0f}ms {s_sum['p95_latency']:>13.0f}ms "
        f"{speedup_p95:>8.1f}x"
    )
    print(f"{'Avg tok/s':<25} {b_sum['avg_tps']:>14.1f} {s_sum['avg_tps']:>14.1f}")
    print(f"{'Quality (local pass)':<25} {b_sum['pass_rate']:>14.0%} {s_sum['pass_rate']:>14.0%}")
    print(f"{'Acceptance Rate':<25} {'N/A':>15} {s_sum['avg_acceptance']:>14.0%}")
    print("=" * 80)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Speculative Decoding Benchmark")
    parser.add_argument(
        "--draft-model", default="lfm-0.3b", help="Draft model ID (default: lfm-0.3b)"
    )
    parser.add_argument(
        "--num-draft-tokens", type=int, default=3, help="Draft tokens per step (default: 3)"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("SPECULATIVE DECODING BENCHMARK")
    print("=" * 80)
    print(f"Test cases:        {len(TEST_CASES)}")
    print(f"Draft model:       {args.draft_model}")
    print(f"Num draft tokens:  {args.num_draft_tokens}")
    print()

    # Load target model
    from models.loader import MLXModelLoader

    loader = MLXModelLoader()
    print("Loading target model...")
    load_start = time.perf_counter()
    loader.load()
    print(f"Target model loaded in {(time.perf_counter() - load_start) * 1000:.0f}ms")

    # === Baseline run ===
    print("\n--- Baseline (target only) ---")
    baseline_start = time.perf_counter()
    baseline_results = run_variant(loader, TEST_CASES, "baseline")
    baseline_total = (time.perf_counter() - baseline_start) * 1000
    n_pass_b = sum(1 for r in baseline_results if r.passed)
    print(f"Baseline done: {n_pass_b}/{len(baseline_results)} passed, {baseline_total:.0f}ms total")

    # === Load draft model ===
    print(f"\nLoading draft model ({args.draft_model})...")
    draft_start = time.perf_counter()
    if not loader.load_draft_model(args.draft_model):
        print("FATAL: Failed to load draft model. Tokenizer mismatch?")
        return 1
    print(f"Draft model loaded in {(time.perf_counter() - draft_start) * 1000:.0f}ms")

    # === Speculative run ===
    print("\n--- Speculative (target + draft) ---")
    spec_start = time.perf_counter()
    speculative_results = run_variant(loader, TEST_CASES, "speculative")
    spec_total = (time.perf_counter() - spec_start) * 1000
    n_pass_s = sum(1 for r in speculative_results if r.passed)
    print(
        f"Speculative done: {n_pass_s}/{len(speculative_results)} passed, {spec_total:.0f}ms total"
    )

    # === Comparison ===
    print_comparison(baseline_results, speculative_results)

    # === Save results ===
    output_path = PROJECT_ROOT / "results" / "speculative_benchmark_latest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def serialize(results: list[BenchmarkResult]) -> list[dict]:
        return [
            {
                "name": r.name,
                "variant": r.variant,
                "output": r.output,
                "latency_ms": round(r.latency_ms, 1),
                "tokens_per_second": round(r.tokens_per_second, 1),
                "tokens_generated": r.tokens_generated,
                "acceptance_rate": round(r.acceptance_rate, 4),
                "passed": r.passed,
                "failed_checks": r.checks_failed,
            }
            for r in results
        ]

    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "draft_model": args.draft_model,
        "num_draft_tokens": args.num_draft_tokens,
        "test_cases": len(TEST_CASES),
        "baseline": {
            "total_ms": round(baseline_total, 1),
            "pass_rate": round(n_pass_b / len(TEST_CASES), 4),
            "results": serialize(baseline_results),
        },
        "speculative": {
            "total_ms": round(spec_total, 1),
            "pass_rate": round(n_pass_s / len(TEST_CASES), 4),
            "results": serialize(speculative_results),
        },
    }
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved to: {output_path}")

    # Unload draft model
    loader.unload_draft_model()

    return 0


if __name__ == "__main__":
    sys.exit(main())
