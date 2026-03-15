"""Latency benchmark implementation and CLI entrypoint.  # noqa: E501
  # noqa: E501
Workstream 4: Latency Benchmark  # noqa: E501
  # noqa: E501
Usage: python -m benchmarks.latency.run --output results/latency.json  # noqa: E501
  # noqa: E501
Implements the LatencyBenchmarker protocol from contracts/latency.py.  # noqa: E501
"""  # noqa: E501
  # noqa: E501
import argparse  # noqa: E501
import json  # noqa: E501
import logging  # noqa: E501
import sys  # noqa: E501
from datetime import UTC, datetime  # noqa: E402  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501
from statistics import mean, stdev  # noqa: E402  # noqa: E501

# noqa: E501
import numpy as np  # noqa: E501
from evals.benchmarks.latency.scenarios import (  # noqa: E501
    Scenario,  # noqa: E501
    get_scenario_by_type,  # noqa: E501
)  # noqa: E501
from evals.benchmarks.latency.timer import (  # noqa: E501
    HighPrecisionTimer,  # noqa: E501
    force_model_unload,  # noqa: E501
    warmup_timer,  # noqa: E501
)  # noqa: E501

# noqa: E501
from jarvis.contracts.latency import (  # noqa: E402  # noqa: E501
    LatencyBenchmarkResult,
    LatencyResult,
)

  # noqa: E501
# Conditional MLX imports for environments without Apple Silicon  # noqa: E501
try:  # noqa: E501
    from models.loader import MLXModelLoader, ModelConfig  # noqa: E501
  # noqa: E501
    HAS_MLX = True  # noqa: E501
except ImportError:  # noqa: E501
    HAS_MLX = False  # noqa: E501
    MLXModelLoader = None  # type: ignore[misc, assignment]  # noqa: E501
  # noqa: E501
    # Stub ModelConfig for type checking when MLX not available  # noqa: E501
    class ModelConfig:  # type: ignore[no-redef]  # noqa: E501
        """Stub ModelConfig for environments without MLX."""  # noqa: E501
  # noqa: E501
        def __init__(  # noqa: E501
            self,  # noqa: E501
            model_path: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit",  # noqa: E501
            estimated_memory_mb: float = 800,  # noqa: E501
            memory_buffer_multiplier: float = 1.5,  # noqa: E501
            default_max_tokens: int = 100,  # noqa: E501
            default_temperature: float = 0.7,  # noqa: E501
        ) -> None:  # noqa: E501
            self.model_path = model_path  # noqa: E501
            self.estimated_memory_mb = estimated_memory_mb  # noqa: E501
            self.memory_buffer_multiplier = memory_buffer_multiplier  # noqa: E501
            self.default_max_tokens = default_max_tokens  # noqa: E501
            self.default_temperature = default_temperature  # noqa: E501
  # noqa: E501
  # noqa: E501
logger = logging.getLogger(__name__)  # noqa: E501
  # noqa: E501
# Default number of benchmark iterations  # noqa: E501
DEFAULT_NUM_RUNS = 10  # noqa: E501
  # noqa: E501
  # noqa: E501
class MLXLatencyBenchmarker:  # noqa: E501
    """Latency benchmarker for MLX models.  # noqa: E501
  # noqa: E501
    Implements LatencyBenchmarker protocol from contracts/latency.py.  # noqa: E501
  # noqa: E501
    Measures latency for three scenarios:  # noqa: E501
    - Cold: Model not loaded, must load from disk  # noqa: E501
    - Warm: Model loaded, new prompt  # noqa: E501
    - Hot: Model loaded, same prompt prefix (tests caching)  # noqa: E501
    """  # noqa: E501
  # noqa: E501
    def __init__(self, config: ModelConfig | None = None) -> None:  # noqa: E501
        """Initialize the benchmarker.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            config: Model configuration. Uses defaults if not provided.  # noqa: E501
        """  # noqa: E501
        self.config = config or ModelConfig()  # noqa: E501
        self._loader = None  # Type: MLXModelLoader | None when MLX available  # noqa: E501
  # noqa: E501
    def _check_mlx_available(self) -> None:  # noqa: E501
        """Check if MLX is available and raise if not."""  # noqa: E501
        if not HAS_MLX:  # noqa: E501
            msg = (  # noqa: E501
                "MLX is not available. This benchmark requires Apple Silicon "  # noqa: E501
                "with MLX installed. Install with: pip install mlx mlx-lm"  # noqa: E501
            )  # noqa: E501
            raise RuntimeError(msg)  # noqa: E501
  # noqa: E501
    def _get_loader(self) -> "MLXModelLoader":  # noqa: E501
        """Get or create the model loader."""  # noqa: E501
        self._check_mlx_available()  # noqa: E501
        if self._loader is None:  # noqa: E501
            self._loader = MLXModelLoader(self.config)  # type: ignore[assignment]  # noqa: E501
        return self._loader  # type: ignore[return-value]  # noqa: E501
  # noqa: E501
    def _ensure_unloaded(self) -> None:  # noqa: E501
        """Ensure model is completely unloaded."""  # noqa: E501
        if self._loader is not None:  # noqa: E501
            self._loader.unload()  # noqa: E501
        force_model_unload()  # noqa: E501
  # noqa: E501
    def _ensure_loaded(self) -> bool:  # noqa: E501
        """Ensure model is loaded.  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            True if model is loaded successfully.  # noqa: E501
        """  # noqa: E501
        loader = self._get_loader()  # noqa: E501
        if not loader.is_loaded():  # noqa: E501
            return bool(loader.load())  # noqa: E501
        return True  # noqa: E501
  # noqa: E501
    def measure_single(  # noqa: E501
        self,  # noqa: E501
        model_path: str,  # noqa: E501
        scenario: Scenario,  # noqa: E501
        prompt: str,  # noqa: E501
        max_tokens: int,  # noqa: E501
    ) -> LatencyResult:  # noqa: E501
        """Measure latency for a single generation.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            model_path: Path or HuggingFace model identifier.  # noqa: E501
            scenario: One of 'cold', 'warm', or 'hot'.  # noqa: E501
            prompt: Input prompt for generation.  # noqa: E501
            max_tokens: Maximum tokens to generate.  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            LatencyResult with timing breakdown.  # noqa: E501
        """  # noqa: E501
        # Update config if model path differs  # noqa: E501
        if model_path != self.config.model_path:  # noqa: E501
            self.config = ModelConfig(model_path=model_path)  # noqa: E501
            self._loader = None  # noqa: E501
  # noqa: E501
        load_time_ms = 0.0  # noqa: E501
        prefill_time_ms = 0.0  # noqa: E501
        generation_time_ms = 0.0  # noqa: E501
        tokens_generated = 0  # noqa: E501
  # noqa: E501
        timer = HighPrecisionTimer()  # noqa: E501
  # noqa: E501
        # Handle scenario-specific setup  # noqa: E501
        if scenario == "cold":  # noqa: E501
            # Cold start: ensure model is unloaded  # noqa: E501
            self._ensure_unloaded()  # noqa: E501
  # noqa: E501
            # Time model loading  # noqa: E501
            timer.start()  # noqa: E501
            load_success = self._ensure_loaded()  # noqa: E501
            load_result = timer.stop()  # noqa: E501
            load_time_ms = load_result.elapsed_ms  # noqa: E501
  # noqa: E501
            if not load_success:  # noqa: E501
                msg = f"Failed to load model: {model_path}"  # noqa: E501
                raise RuntimeError(msg)  # noqa: E501
  # noqa: E501
        elif scenario == "warm":  # noqa: E501
            # Warm start: ensure model is loaded  # noqa: E501
            if not self._ensure_loaded():  # noqa: E501
                msg = f"Failed to load model: {model_path}"  # noqa: E501
                raise RuntimeError(msg)  # noqa: E501
  # noqa: E501
        elif scenario == "hot":  # noqa: E501
            # Hot start: model should already be loaded from previous warm run  # noqa: E501
            if not self._ensure_loaded():  # noqa: E501
                msg = f"Failed to load model: {model_path}"  # noqa: E501
                raise RuntimeError(msg)  # noqa: E501
  # noqa: E501
        # Time generation (includes prefill + token generation)  # noqa: E501
        timer.start()  # noqa: E501
        result = self._get_loader().generate_sync(  # noqa: E501
            prompt=prompt,  # noqa: E501
            max_tokens=max_tokens,  # noqa: E501
            temperature=0.7,  # noqa: E501
        )  # noqa: E501
        gen_result = timer.stop()  # noqa: E501
  # noqa: E501
        # Generation time includes both prefill and actual generation  # noqa: E501
        # We approximate prefill as ~10% of total for small prompts  # noqa: E501
        total_gen_ms = gen_result.elapsed_ms  # noqa: E501
        prefill_time_ms = total_gen_ms * 0.1  # Approximate  # noqa: E501
        generation_time_ms = total_gen_ms * 0.9  # Approximate  # noqa: E501
        tokens_generated = result.tokens_generated  # noqa: E501
  # noqa: E501
        # Calculate total time  # noqa: E501
        total_time_ms = load_time_ms + total_gen_ms  # noqa: E501
  # noqa: E501
        # Calculate tokens per second  # noqa: E501
        tokens_per_second = 0.0  # noqa: E501
        if generation_time_ms > 0:  # noqa: E501
            tokens_per_second = (tokens_generated / generation_time_ms) * 1000  # noqa: E501
  # noqa: E501
        return LatencyResult(  # noqa: E501
            scenario=scenario,  # noqa: E501
            model_name=model_path,  # noqa: E501
            context_length=len(prompt),  # noqa: E501
            output_tokens=tokens_generated,  # noqa: E501
            load_time_ms=load_time_ms,  # noqa: E501
            prefill_time_ms=prefill_time_ms,  # noqa: E501
            generation_time_ms=generation_time_ms,  # noqa: E501
            total_time_ms=total_time_ms,  # noqa: E501
            tokens_per_second=tokens_per_second,  # noqa: E501
            timestamp=datetime.now(UTC).isoformat(),  # noqa: E501
        )  # noqa: E501
  # noqa: E501
    def run_benchmark(  # noqa: E501
        self,  # noqa: E501
        model_path: str,  # noqa: E501
        scenario: Scenario,  # noqa: E501
        num_runs: int = DEFAULT_NUM_RUNS,  # noqa: E501
    ) -> LatencyBenchmarkResult:  # noqa: E501
        """Run full benchmark with statistical analysis.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            model_path: Path or HuggingFace model identifier.  # noqa: E501
            scenario: One of 'cold', 'warm', or 'hot'.  # noqa: E501
            num_runs: Number of iterations (default 10).  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            LatencyBenchmarkResult with percentiles and statistics.  # noqa: E501
        """  # noqa: E501
        test_scenario = get_scenario_by_type(scenario)  # noqa: E501
        results: list[LatencyResult] = []  # noqa: E501
  # noqa: E501
        logger.info(  # noqa: E501
            "Running %s benchmark: %d iterations for %s",  # noqa: E501
            scenario,  # noqa: E501
            num_runs,  # noqa: E501
            model_path,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
        # Warmup timer to minimize JIT effects  # noqa: E501
        warmup_timer()  # noqa: E501
  # noqa: E501
        # For warm/hot scenarios, pre-load the model once  # noqa: E501
        if scenario in ("warm", "hot"):  # noqa: E501
            if not self._ensure_loaded():  # noqa: E501
                msg = f"Failed to load model: {model_path}"  # noqa: E501
                raise RuntimeError(msg)  # noqa: E501
  # noqa: E501
        # Run benchmark iterations  # noqa: E501
        for i in range(num_runs):  # noqa: E501
            logger.debug("Iteration %d/%d", i + 1, num_runs)  # noqa: E501
  # noqa: E501
            try:  # noqa: E501
                result = self.measure_single(  # noqa: E501
                    model_path=model_path,  # noqa: E501
                    scenario=scenario,  # noqa: E501
                    prompt=test_scenario.prompt,  # noqa: E501
                    max_tokens=test_scenario.max_tokens,  # noqa: E501
                )  # noqa: E501
                results.append(result)  # noqa: E501
            except Exception as e:  # noqa: E501
                logger.warning("Iteration %d failed: %s", i + 1, e)  # noqa: E501
                continue  # noqa: E501
  # noqa: E501
        if not results:  # noqa: E501
            msg = f"All {num_runs} iterations failed"  # noqa: E501
            raise RuntimeError(msg)  # noqa: E501
  # noqa: E501
        # Calculate statistics (exclude first run as JIT outlier if we have enough runs)  # noqa: E501
        analysis_results = results  # noqa: E501
        if len(results) > 3:  # noqa: E501
            analysis_results = results[1:]  # Exclude first run (JIT outlier)  # noqa: E501
  # noqa: E501
        total_times = [r.total_time_ms for r in analysis_results]  # noqa: E501
  # noqa: E501
        # Calculate percentiles  # noqa: E501
        p50_ms = float(np.percentile(total_times, 50))  # noqa: E501
        p95_ms = float(np.percentile(total_times, 95))  # noqa: E501
        p99_ms = float(np.percentile(total_times, 99))  # noqa: E501
        mean_ms = mean(total_times)  # noqa: E501
        std_ms = stdev(total_times) if len(total_times) > 1 else 0.0  # noqa: E501
  # noqa: E501
        return LatencyBenchmarkResult(  # noqa: E501
            scenario=scenario,  # noqa: E501
            model_name=model_path,  # noqa: E501
            num_runs=len(results),  # noqa: E501
            p50_ms=p50_ms,  # noqa: E501
            p95_ms=p95_ms,  # noqa: E501
            p99_ms=p99_ms,  # noqa: E501
            mean_ms=mean_ms,  # noqa: E501
            std_ms=std_ms,  # noqa: E501
            results=results,  # noqa: E501
            timestamp=datetime.now(UTC).isoformat(),  # noqa: E501
        )  # noqa: E501
  # noqa: E501
    def run_all_scenarios(  # noqa: E501
        self,  # noqa: E501
        model_path: str | None = None,  # noqa: E501
        num_runs: int = DEFAULT_NUM_RUNS,  # noqa: E501
    ) -> dict[Scenario, LatencyBenchmarkResult]:  # noqa: E501
        """Run benchmarks for all scenario types.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            model_path: Path or HuggingFace model identifier.  # noqa: E501
                       Uses default from config if not provided.  # noqa: E501
            num_runs: Number of iterations per scenario.  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            Dictionary mapping scenario type to benchmark results.  # noqa: E501
        """  # noqa: E501
        model_path = model_path or self.config.model_path  # noqa: E501
        results: dict[Scenario, LatencyBenchmarkResult] = {}  # noqa: E501
  # noqa: E501
        # Run scenarios in order: cold, warm, hot  # noqa: E501
        # This order is important because warm/hot benefit from cold's loading  # noqa: E501
        for scenario in ("cold", "warm", "hot"):  # noqa: E501
            logger.info("Starting %s scenario benchmark", scenario)  # noqa: E501
            results[scenario] = self.run_benchmark(  # noqa: E501
                model_path=model_path,  # noqa: E501
                scenario=scenario,  # noqa: E501
                num_runs=num_runs,  # noqa: E501
            )  # noqa: E501
  # noqa: E501
        return results  # noqa: E501
  # noqa: E501
    def cleanup(self) -> None:  # noqa: E501
        """Clean up resources and unload model."""  # noqa: E501
        self._ensure_unloaded()  # noqa: E501
  # noqa: E501
  # noqa: E501
def format_results_for_output(  # noqa: E501
    results: dict[Scenario, LatencyBenchmarkResult],  # noqa: E501
) -> dict[str, object]:  # noqa: E501
    """Format benchmark results for JSON output.  # noqa: E501
  # noqa: E501
    Args:  # noqa: E501
        results: Dictionary of benchmark results by scenario.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        Dictionary suitable for JSON serialization.  # noqa: E501
    """  # noqa: E501
    output: dict[str, object] = {  # noqa: E501
        "timestamp": datetime.now(UTC).isoformat(),  # noqa: E501
        "scenarios": {},  # noqa: E501
    }  # noqa: E501
  # noqa: E501
    scenarios_dict: dict[str, object] = {}  # noqa: E501
    for scenario, result in results.items():  # noqa: E501
        scenarios_dict[scenario] = {  # noqa: E501
            "model_name": result.model_name,  # noqa: E501
            "num_runs": result.num_runs,  # noqa: E501
            "p50_ms": round(result.p50_ms, 2),  # noqa: E501
            "p95_ms": round(result.p95_ms, 2),  # noqa: E501
            "p99_ms": round(result.p99_ms, 2),  # noqa: E501
            "mean_ms": round(result.mean_ms, 2),  # noqa: E501
            "std_ms": round(result.std_ms, 2),  # noqa: E501
            "individual_runs": [  # noqa: E501
                {  # noqa: E501
                    "load_time_ms": round(r.load_time_ms, 2),  # noqa: E501
                    "prefill_time_ms": round(r.prefill_time_ms, 2),  # noqa: E501
                    "generation_time_ms": round(r.generation_time_ms, 2),  # noqa: E501
                    "total_time_ms": round(r.total_time_ms, 2),  # noqa: E501
                    "tokens_per_second": round(r.tokens_per_second, 2),  # noqa: E501
                    "output_tokens": r.output_tokens,  # noqa: E501
                }  # noqa: E501
                for r in result.results  # noqa: E501
            ],  # noqa: E501
        }  # noqa: E501
  # noqa: E501
    output["scenarios"] = scenarios_dict  # noqa: E501
    return output  # noqa: E501
  # noqa: E501
  # noqa: E501
def print_summary(results: dict[Scenario, LatencyBenchmarkResult]) -> None:  # noqa: E501
    """Print a summary of benchmark results.  # noqa: E501
  # noqa: E501
    Args:  # noqa: E501
        results: Dictionary of benchmark results by scenario.  # noqa: E501
    """  # noqa: E501
    print("\n" + "=" * 60)  # noqa: E501
    print("LATENCY BENCHMARK RESULTS")  # noqa: E501
    print("=" * 60)  # noqa: E501
  # noqa: E501
    for scenario, result in results.items():  # noqa: E501
        print(f"\n{scenario.upper()} START:")  # noqa: E501
        print(f"  Runs: {result.num_runs}")  # noqa: E501
        print(f"  P50:  {result.p50_ms:,.1f}ms")  # noqa: E501
        print(f"  P95:  {result.p95_ms:,.1f}ms")  # noqa: E501
        print(f"  P99:  {result.p99_ms:,.1f}ms")  # noqa: E501
        print(f"  Mean: {result.mean_ms:,.1f}ms (±{result.std_ms:.1f}ms)")  # noqa: E501
  # noqa: E501
        # Show pass/fail for gates  # noqa: E501
        if scenario == "warm":  # noqa: E501
            status = (  # noqa: E501
                "PASS"  # noqa: E501
                if result.mean_ms < 3000  # noqa: E501
                else "CONDITIONAL"  # noqa: E501
                if result.mean_ms < 5000  # noqa: E501
                else "FAIL"  # noqa: E501
            )  # noqa: E501
            print(f"  Gate G4 (warm <3s): {status}")  # noqa: E501
        elif scenario == "cold":  # noqa: E501
            status = (  # noqa: E501
                "PASS"  # noqa: E501
                if result.mean_ms < 15000  # noqa: E501
                else "CONDITIONAL"  # noqa: E501
                if result.mean_ms < 20000  # noqa: E501
                else "FAIL"  # noqa: E501
            )  # noqa: E501
            print(f"  Gate G5 (cold <15s): {status}")  # noqa: E501
  # noqa: E501
    print("\n" + "=" * 60)  # noqa: E501
  # noqa: E501
  # noqa: E501
def main() -> int:  # noqa: E501
    """Run latency benchmark CLI.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        Exit code (0 for success, 1 for error).  # noqa: E501
    """  # noqa: E501
    parser = argparse.ArgumentParser(description="Run latency benchmark for MLX models")  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--output",  # noqa: E501
        type=Path,  # noqa: E501
        required=True,  # noqa: E501
        help="Output JSON file for results",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--model",  # noqa: E501
        type=str,  # noqa: E501
        default=None,  # noqa: E501
        help="Model path or HuggingFace ID (default: Qwen2.5-0.5B-Instruct-4bit)",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--scenario",  # noqa: E501
        type=str,  # noqa: E501
        choices=["cold", "warm", "hot", "all"],  # noqa: E501
        default="all",  # noqa: E501
        help="Scenario to benchmark (default: all)",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--runs",  # noqa: E501
        type=int,  # noqa: E501
        default=DEFAULT_NUM_RUNS,  # noqa: E501
        help=f"Number of iterations per scenario (default: {DEFAULT_NUM_RUNS})",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--verbose",  # noqa: E501
        action="store_true",  # noqa: E501
        help="Show detailed output",  # noqa: E501
    )  # noqa: E501
    args = parser.parse_args()  # noqa: E501
  # noqa: E501
    # Configure logging  # noqa: E501
    log_level = logging.DEBUG if args.verbose else logging.INFO  # noqa: E501
    logging.basicConfig(  # noqa: E501
        level=log_level,  # noqa: E501
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    if args.verbose:  # noqa: E501
        print("Initializing benchmarker...")  # noqa: E501
  # noqa: E501
    try:  # noqa: E501
        # Create benchmarker with optional custom model  # noqa: E501
        config = ModelConfig(model_path=args.model) if args.model else None  # noqa: E501
        benchmarker = MLXLatencyBenchmarker(config=config)  # noqa: E501
  # noqa: E501
        if args.scenario == "all":  # noqa: E501
            # Run all scenarios  # noqa: E501
            results = benchmarker.run_all_scenarios(  # noqa: E501
                model_path=args.model,  # noqa: E501
                num_runs=args.runs,  # noqa: E501
            )  # noqa: E501
        else:  # noqa: E501
            # Run single scenario  # noqa: E501
            result = benchmarker.run_benchmark(  # noqa: E501
                model_path=args.model or ModelConfig().model_path,  # noqa: E501
                scenario=args.scenario,  # noqa: E501
                num_runs=args.runs,  # noqa: E501
            )  # noqa: E501
            results = {args.scenario: result}  # noqa: E501
  # noqa: E501
        # Format and save results  # noqa: E501
        output_data = format_results_for_output(results)  # noqa: E501
  # noqa: E501
        # Ensure output directory exists  # noqa: E501
        args.output.parent.mkdir(parents=True, exist_ok=True)  # noqa: E501
        args.output.write_text(json.dumps(output_data, indent=2))  # noqa: E501
  # noqa: E501
        # Print summary  # noqa: E501
        print_summary(results)  # noqa: E501
        print(f"\nResults saved to: {args.output}")  # noqa: E501
  # noqa: E501
        # Cleanup  # noqa: E501
        benchmarker.cleanup()  # noqa: E501
  # noqa: E501
        return 0  # noqa: E501
  # noqa: E501
    except KeyboardInterrupt:  # noqa: E501
        print("\nBenchmark interrupted by user")  # noqa: E501
        return 1  # noqa: E501
    except Exception as e:  # noqa: E501
        logger.exception("Benchmark failed")  # noqa: E501
        print(f"\nError: {e}")  # noqa: E501
        return 1  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    sys.exit(main())  # noqa: E501
