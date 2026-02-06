"""Latency benchmark implementation and CLI entrypoint.

Workstream 4: Latency Benchmark

Usage: python -m benchmarks.latency.run --output results/latency.json

Implements the LatencyBenchmarker protocol from contracts/latency.py.
"""

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from evals.benchmarks.latency.scenarios import (
    Scenario,
    get_scenario_by_type,
)
from evals.benchmarks.latency.timer import (
    HighPrecisionTimer,
    force_model_unload,
    warmup_timer,
)
from contracts.latency import LatencyBenchmarkResult, LatencyResult

# Conditional MLX imports for environments without Apple Silicon
try:
    from models.loader import MLXModelLoader, ModelConfig

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    MLXModelLoader = None  # type: ignore[misc, assignment]

    # Stub ModelConfig for type checking when MLX not available
    class ModelConfig:  # type: ignore[no-redef]
        """Stub ModelConfig for environments without MLX."""

        def __init__(
            self,
            model_path: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            estimated_memory_mb: float = 800,
            memory_buffer_multiplier: float = 1.5,
            default_max_tokens: int = 100,
            default_temperature: float = 0.7,
        ) -> None:
            self.model_path = model_path
            self.estimated_memory_mb = estimated_memory_mb
            self.memory_buffer_multiplier = memory_buffer_multiplier
            self.default_max_tokens = default_max_tokens
            self.default_temperature = default_temperature


logger = logging.getLogger(__name__)

# Default number of benchmark iterations
DEFAULT_NUM_RUNS = 10


class MLXLatencyBenchmarker:
    """Latency benchmarker for MLX models.

    Implements LatencyBenchmarker protocol from contracts/latency.py.

    Measures latency for three scenarios:
    - Cold: Model not loaded, must load from disk
    - Warm: Model loaded, new prompt
    - Hot: Model loaded, same prompt prefix (tests caching)
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        """Initialize the benchmarker.

        Args:
            config: Model configuration. Uses defaults if not provided.
        """
        self.config = config or ModelConfig()
        self._loader = None  # Type: MLXModelLoader | None when MLX available

    def _check_mlx_available(self) -> None:
        """Check if MLX is available and raise if not."""
        if not HAS_MLX:
            msg = (
                "MLX is not available. This benchmark requires Apple Silicon "
                "with MLX installed. Install with: pip install mlx mlx-lm"
            )
            raise RuntimeError(msg)

    def _get_loader(self) -> "MLXModelLoader":
        """Get or create the model loader."""
        self._check_mlx_available()
        if self._loader is None:
            self._loader = MLXModelLoader(self.config)  # type: ignore[assignment]
        return self._loader  # type: ignore[return-value]

    def _ensure_unloaded(self) -> None:
        """Ensure model is completely unloaded."""
        if self._loader is not None:
            self._loader.unload()
        force_model_unload()

    def _ensure_loaded(self) -> bool:
        """Ensure model is loaded.

        Returns:
            True if model is loaded successfully.
        """
        loader = self._get_loader()
        if not loader.is_loaded():
            return bool(loader.load())
        return True

    def measure_single(
        self,
        model_path: str,
        scenario: Scenario,
        prompt: str,
        max_tokens: int,
    ) -> LatencyResult:
        """Measure latency for a single generation.

        Args:
            model_path: Path or HuggingFace model identifier.
            scenario: One of 'cold', 'warm', or 'hot'.
            prompt: Input prompt for generation.
            max_tokens: Maximum tokens to generate.

        Returns:
            LatencyResult with timing breakdown.
        """
        # Update config if model path differs
        if model_path != self.config.model_path:
            self.config = ModelConfig(model_path=model_path)
            self._loader = None

        load_time_ms = 0.0
        prefill_time_ms = 0.0
        generation_time_ms = 0.0
        tokens_generated = 0

        timer = HighPrecisionTimer()

        # Handle scenario-specific setup
        if scenario == "cold":
            # Cold start: ensure model is unloaded
            self._ensure_unloaded()

            # Time model loading
            timer.start()
            load_success = self._ensure_loaded()
            load_result = timer.stop()
            load_time_ms = load_result.elapsed_ms

            if not load_success:
                msg = f"Failed to load model: {model_path}"
                raise RuntimeError(msg)

        elif scenario == "warm":
            # Warm start: ensure model is loaded
            if not self._ensure_loaded():
                msg = f"Failed to load model: {model_path}"
                raise RuntimeError(msg)

        elif scenario == "hot":
            # Hot start: model should already be loaded from previous warm run
            if not self._ensure_loaded():
                msg = f"Failed to load model: {model_path}"
                raise RuntimeError(msg)

        # Time generation (includes prefill + token generation)
        timer.start()
        result = self._get_loader().generate_sync(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        gen_result = timer.stop()

        # Generation time includes both prefill and actual generation
        # We approximate prefill as ~10% of total for small prompts
        total_gen_ms = gen_result.elapsed_ms
        prefill_time_ms = total_gen_ms * 0.1  # Approximate
        generation_time_ms = total_gen_ms * 0.9  # Approximate
        tokens_generated = result.tokens_generated

        # Calculate total time
        total_time_ms = load_time_ms + total_gen_ms

        # Calculate tokens per second
        tokens_per_second = 0.0
        if generation_time_ms > 0:
            tokens_per_second = (tokens_generated / generation_time_ms) * 1000

        return LatencyResult(
            scenario=scenario,
            model_name=model_path,
            context_length=len(prompt),
            output_tokens=tokens_generated,
            load_time_ms=load_time_ms,
            prefill_time_ms=prefill_time_ms,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms,
            tokens_per_second=tokens_per_second,
            timestamp=datetime.now(UTC).isoformat(),
        )

    def run_benchmark(
        self,
        model_path: str,
        scenario: Scenario,
        num_runs: int = DEFAULT_NUM_RUNS,
    ) -> LatencyBenchmarkResult:
        """Run full benchmark with statistical analysis.

        Args:
            model_path: Path or HuggingFace model identifier.
            scenario: One of 'cold', 'warm', or 'hot'.
            num_runs: Number of iterations (default 10).

        Returns:
            LatencyBenchmarkResult with percentiles and statistics.
        """
        test_scenario = get_scenario_by_type(scenario)
        results: list[LatencyResult] = []

        logger.info(
            "Running %s benchmark: %d iterations for %s",
            scenario,
            num_runs,
            model_path,
        )

        # Warmup timer to minimize JIT effects
        warmup_timer()

        # For warm/hot scenarios, pre-load the model once
        if scenario in ("warm", "hot"):
            if not self._ensure_loaded():
                msg = f"Failed to load model: {model_path}"
                raise RuntimeError(msg)

        # Run benchmark iterations
        for i in range(num_runs):
            logger.debug("Iteration %d/%d", i + 1, num_runs)

            try:
                result = self.measure_single(
                    model_path=model_path,
                    scenario=scenario,
                    prompt=test_scenario.prompt,
                    max_tokens=test_scenario.max_tokens,
                )
                results.append(result)
            except Exception as e:
                logger.warning("Iteration %d failed: %s", i + 1, e)
                continue

        if not results:
            msg = f"All {num_runs} iterations failed"
            raise RuntimeError(msg)

        # Calculate statistics (exclude first run as JIT outlier if we have enough runs)
        analysis_results = results
        if len(results) > 3:
            analysis_results = results[1:]  # Exclude first run (JIT outlier)

        total_times = [r.total_time_ms for r in analysis_results]

        # Calculate percentiles
        p50_ms = float(np.percentile(total_times, 50))
        p95_ms = float(np.percentile(total_times, 95))
        p99_ms = float(np.percentile(total_times, 99))
        mean_ms = mean(total_times)
        std_ms = stdev(total_times) if len(total_times) > 1 else 0.0

        return LatencyBenchmarkResult(
            scenario=scenario,
            model_name=model_path,
            num_runs=len(results),
            p50_ms=p50_ms,
            p95_ms=p95_ms,
            p99_ms=p99_ms,
            mean_ms=mean_ms,
            std_ms=std_ms,
            results=results,
            timestamp=datetime.now(UTC).isoformat(),
        )

    def run_all_scenarios(
        self,
        model_path: str | None = None,
        num_runs: int = DEFAULT_NUM_RUNS,
    ) -> dict[Scenario, LatencyBenchmarkResult]:
        """Run benchmarks for all scenario types.

        Args:
            model_path: Path or HuggingFace model identifier.
                       Uses default from config if not provided.
            num_runs: Number of iterations per scenario.

        Returns:
            Dictionary mapping scenario type to benchmark results.
        """
        model_path = model_path or self.config.model_path
        results: dict[Scenario, LatencyBenchmarkResult] = {}

        # Run scenarios in order: cold, warm, hot
        # This order is important because warm/hot benefit from cold's loading
        for scenario in ("cold", "warm", "hot"):
            logger.info("Starting %s scenario benchmark", scenario)
            results[scenario] = self.run_benchmark(
                model_path=model_path,
                scenario=scenario,
                num_runs=num_runs,
            )

        return results

    def cleanup(self) -> None:
        """Clean up resources and unload model."""
        self._ensure_unloaded()


def format_results_for_output(
    results: dict[Scenario, LatencyBenchmarkResult],
) -> dict[str, object]:
    """Format benchmark results for JSON output.

    Args:
        results: Dictionary of benchmark results by scenario.

    Returns:
        Dictionary suitable for JSON serialization.
    """
    output: dict[str, object] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "scenarios": {},
    }

    scenarios_dict: dict[str, object] = {}
    for scenario, result in results.items():
        scenarios_dict[scenario] = {
            "model_name": result.model_name,
            "num_runs": result.num_runs,
            "p50_ms": round(result.p50_ms, 2),
            "p95_ms": round(result.p95_ms, 2),
            "p99_ms": round(result.p99_ms, 2),
            "mean_ms": round(result.mean_ms, 2),
            "std_ms": round(result.std_ms, 2),
            "individual_runs": [
                {
                    "load_time_ms": round(r.load_time_ms, 2),
                    "prefill_time_ms": round(r.prefill_time_ms, 2),
                    "generation_time_ms": round(r.generation_time_ms, 2),
                    "total_time_ms": round(r.total_time_ms, 2),
                    "tokens_per_second": round(r.tokens_per_second, 2),
                    "output_tokens": r.output_tokens,
                }
                for r in result.results
            ],
        }

    output["scenarios"] = scenarios_dict
    return output


def print_summary(results: dict[Scenario, LatencyBenchmarkResult]) -> None:
    """Print a summary of benchmark results.

    Args:
        results: Dictionary of benchmark results by scenario.
    """
    print("\n" + "=" * 60)
    print("LATENCY BENCHMARK RESULTS")
    print("=" * 60)

    for scenario, result in results.items():
        print(f"\n{scenario.upper()} START:")
        print(f"  Runs: {result.num_runs}")
        print(f"  P50:  {result.p50_ms:,.1f}ms")
        print(f"  P95:  {result.p95_ms:,.1f}ms")
        print(f"  P99:  {result.p99_ms:,.1f}ms")
        print(f"  Mean: {result.mean_ms:,.1f}ms (±{result.std_ms:.1f}ms)")

        # Show pass/fail for gates
        if scenario == "warm":
            status = (
                "PASS"
                if result.mean_ms < 3000
                else "CONDITIONAL"
                if result.mean_ms < 5000
                else "FAIL"
            )
            print(f"  Gate G4 (warm <3s): {status}")
        elif scenario == "cold":
            status = (
                "PASS"
                if result.mean_ms < 15000
                else "CONDITIONAL"
                if result.mean_ms < 20000
                else "FAIL"
            )
            print(f"  Gate G5 (cold <15s): {status}")

    print("\n" + "=" * 60)


def main() -> int:
    """Run latency benchmark CLI.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(description="Run latency benchmark for MLX models")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model path or HuggingFace ID (default: Qwen2.5-0.5B-Instruct-4bit)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["cold", "warm", "hot", "all"],
        default="all",
        help="Scenario to benchmark (default: all)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_NUM_RUNS,
        help=f"Number of iterations per scenario (default: {DEFAULT_NUM_RUNS})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.verbose:
        print("Initializing benchmarker...")

    try:
        # Create benchmarker with optional custom model
        config = ModelConfig(model_path=args.model) if args.model else None
        benchmarker = MLXLatencyBenchmarker(config=config)

        if args.scenario == "all":
            # Run all scenarios
            results = benchmarker.run_all_scenarios(
                model_path=args.model,
                num_runs=args.runs,
            )
        else:
            # Run single scenario
            result = benchmarker.run_benchmark(
                model_path=args.model or ModelConfig().model_path,
                scenario=args.scenario,
                num_runs=args.runs,
            )
            results = {args.scenario: result}

        # Format and save results
        output_data = format_results_for_output(results)

        # Ensure output directory exists
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output_data, indent=2))

        # Print summary
        print_summary(results)
        print(f"\nResults saved to: {args.output}")

        # Cleanup
        benchmarker.cleanup()

        return 0

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        logger.exception("Benchmark failed")
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
