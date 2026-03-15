"""CLI for running memory profiling benchmark.  # noqa: E501
  # noqa: E501
Usage: python -m benchmarks.memory.run --output results/memory.json  # noqa: E501
  # noqa: E501
Workstream 1: Memory Profiler  # noqa: E501
"""  # noqa: E501
  # noqa: E501
import argparse  # noqa: E501
import json  # noqa: E501
import logging  # noqa: E501
import sys  # noqa: E501
from dataclasses import asdict  # noqa: E402  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501

# noqa: E501
from evals.benchmarks.memory.models import (  # noqa: E501
    CONTEXT_LENGTHS,  # noqa: E501
    get_default_model,  # noqa: E501
    get_models_for_profiling,  # noqa: E501
)  # noqa: E501
from evals.benchmarks.memory.profiler import MLXMemoryProfiler  # noqa: E402  # noqa: E501

  # noqa: E501
logger = logging.getLogger(__name__)  # noqa: E501
  # noqa: E501
  # noqa: E501
def setup_logging(verbose: bool) -> None:  # noqa: E501
    """Configure logging based on verbosity."""  # noqa: E501
    level = logging.DEBUG if verbose else logging.INFO  # noqa: E501
    logging.basicConfig(  # noqa: E501
        level=level,  # noqa: E501
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # noqa: E501
    )  # noqa: E501
  # noqa: E501
  # noqa: E501
def main() -> int:  # noqa: E501
    """Run memory profiling benchmark.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        Exit code (0 for success, 1 for error)  # noqa: E501
    """  # noqa: E501
    parser = argparse.ArgumentParser(  # noqa: E501
        description="Profile MLX model memory usage",  # noqa: E501
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # noqa: E501
    )  # noqa: E501
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
        help="Specific model path to profile (default: all configured models)",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--context-lengths",  # noqa: E501
        type=int,  # noqa: E501
        nargs="+",  # noqa: E501
        default=CONTEXT_LENGTHS,  # noqa: E501
        help="Context lengths to test",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--quick",  # noqa: E501
        action="store_true",  # noqa: E501
        help="Quick mode: profile only default model at 512 context",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--with-generation",  # noqa: E501
        action="store_true",  # noqa: E501
        help="Include memory measurement during generation",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--verbose",  # noqa: E501
        action="store_true",  # noqa: E501
        help="Show detailed progress and debug output",  # noqa: E501
    )  # noqa: E501
    args = parser.parse_args()  # noqa: E501
  # noqa: E501
    setup_logging(args.verbose)  # noqa: E501
  # noqa: E501
    # Ensure output directory exists  # noqa: E501
    args.output.parent.mkdir(parents=True, exist_ok=True)  # noqa: E501
  # noqa: E501
    # Determine models to profile  # noqa: E501
    if args.model:  # noqa: E501
        from evals.benchmarks.memory.models import ModelSpec  # noqa: E501
  # noqa: E501
        models = [  # noqa: E501
            ModelSpec(  # noqa: E501
                path=args.model,  # noqa: E501
                name=args.model.split("/")[-1],  # noqa: E501
                estimated_memory_mb=0,  # noqa: E501
                description="User-specified model",  # noqa: E501
            )  # noqa: E501
        ]  # noqa: E501
    elif args.quick:  # noqa: E501
        models = [get_default_model()]  # noqa: E501
        args.context_lengths = [512]  # noqa: E501
    else:  # noqa: E501
        models = get_models_for_profiling()  # noqa: E501
  # noqa: E501
    # Initialize profiler  # noqa: E501
    profiler = MLXMemoryProfiler()  # noqa: E501
    results: list[dict[str, object]] = []  # noqa: E501
    errors: list[dict[str, object]] = []  # noqa: E501
  # noqa: E501
    total_profiles = len(models) * len(args.context_lengths)  # noqa: E501
    print(f"Profiling {len(models)} model(s) at {len(args.context_lengths)} context length(s)")  # noqa: E501
    print(f"Total profiles to run: {total_profiles}")  # noqa: E501
    print("-" * 60)  # noqa: E501
  # noqa: E501
    profile_count = 0  # noqa: E501
    for model in models:  # noqa: E501
        print(f"\nModel: {model.name}")  # noqa: E501
        print(f"  Path: {model.path}")  # noqa: E501
  # noqa: E501
        for ctx_len in args.context_lengths:  # noqa: E501
            profile_count += 1  # noqa: E501
            print(f"\n  [{profile_count}/{total_profiles}] Context length: {ctx_len}")  # noqa: E501
  # noqa: E501
            try:  # noqa: E501
                if args.with_generation:  # noqa: E501
                    profile = profiler.profile_with_generation(  # noqa: E501
                        model_path=model.path,  # noqa: E501
                        context_length=ctx_len,  # noqa: E501
                    )  # noqa: E501
                else:  # noqa: E501
                    profile = profiler.profile_model(  # noqa: E501
                        model_path=model.path,  # noqa: E501
                        context_length=ctx_len,  # noqa: E501
                    )  # noqa: E501
  # noqa: E501
                profile_dict = asdict(profile)  # noqa: E501
                results.append(profile_dict)  # noqa: E501
  # noqa: E501
                print(f"    RSS:   {profile.rss_mb:>8.1f} MB")  # noqa: E501
                print(f"    Metal: {profile.metal_mb:>8.1f} MB")  # noqa: E501
                print(f"    Load:  {profile.load_time_seconds:>8.2f} s")  # noqa: E501
  # noqa: E501
            except FileNotFoundError:  # noqa: E501
                error = {  # noqa: E501
                    "model": model.path,  # noqa: E501
                    "context_length": ctx_len,  # noqa: E501
                    "error": "Model not found",  # noqa: E501
                }  # noqa: E501
                errors.append(error)  # noqa: E501
                print(f"    ERROR: Model not found - {model.path}")  # noqa: E501
  # noqa: E501
            except Exception as e:  # noqa: E501
                error = {  # noqa: E501
                    "model": model.path,  # noqa: E501
                    "context_length": ctx_len,  # noqa: E501
                    "error": str(e),  # noqa: E501
                }  # noqa: E501
                errors.append(error)  # noqa: E501
                print(f"    ERROR: {e}")  # noqa: E501
                logger.exception("Error profiling %s", model.path)  # noqa: E501
  # noqa: E501
    # Prepare output data  # noqa: E501
    summary: dict[str, object] = {  # noqa: E501
        "total_profiles": len(results),  # noqa: E501
        "total_errors": len(errors),  # noqa: E501
        "models_tested": len(models),  # noqa: E501
        "context_lengths_tested": args.context_lengths,  # noqa: E501
    }  # noqa: E501
  # noqa: E501
    # Add aggregate stats if we have results  # noqa: E501
    if results:  # noqa: E501
        rss_values: list[float] = [r["rss_mb"] for r in results]  # type: ignore[misc]  # noqa: E501
        metal_values: list[float] = [r["metal_mb"] for r in results]  # type: ignore[misc]  # noqa: E501
        load_times: list[float] = [r["load_time_seconds"] for r in results]  # type: ignore[misc]  # noqa: E501
  # noqa: E501
        summary["max_rss_mb"] = max(rss_values)  # noqa: E501
        summary["avg_rss_mb"] = sum(rss_values) / len(rss_values)  # noqa: E501
        summary["max_metal_mb"] = max(metal_values)  # noqa: E501
        summary["avg_metal_mb"] = sum(metal_values) / len(metal_values)  # noqa: E501
        summary["avg_load_time_seconds"] = sum(load_times) / len(load_times)  # noqa: E501
  # noqa: E501
    output_data: dict[str, object] = {  # noqa: E501
        "profiles": results,  # noqa: E501
        "errors": errors,  # noqa: E501
        "summary": summary,  # noqa: E501
    }  # noqa: E501
  # noqa: E501
    # Save results  # noqa: E501
    args.output.write_text(json.dumps(output_data, indent=2))  # noqa: E501
  # noqa: E501
    # Print summary  # noqa: E501
    print("\n" + "=" * 60)  # noqa: E501
    print("SUMMARY")  # noqa: E501
    print("=" * 60)  # noqa: E501
    print(f"Profiles completed: {len(results)}")  # noqa: E501
    print(f"Errors: {len(errors)}")  # noqa: E501
  # noqa: E501
    if results:  # noqa: E501
        print(f"\nMax RSS Memory:   {summary['max_rss_mb']:.1f} MB")  # noqa: E501
        print(f"Max Metal Memory: {summary['max_metal_mb']:.1f} MB")  # noqa: E501
        print(f"Avg Load Time:    {summary['avg_load_time_seconds']:.2f} s")  # noqa: E501
  # noqa: E501
    print(f"\nResults saved to: {args.output}")  # noqa: E501
  # noqa: E501
    # Return error code if any profiles failed  # noqa: E501
    return 1 if errors else 0  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    sys.exit(main())  # noqa: E501
