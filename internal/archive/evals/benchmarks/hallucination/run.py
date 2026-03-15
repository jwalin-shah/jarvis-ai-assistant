"""CLI for running HHEM hallucination evaluation benchmark.  # noqa: E501
  # noqa: E501
Usage: python -m benchmarks.hallucination.run --output results/hhem.json  # noqa: E501
  # noqa: E501
Workstream 2: HHEM Hallucination Benchmark  # noqa: E501
"""  # noqa: E501
  # noqa: E501
import argparse  # noqa: E501
import json  # noqa: E501
import sys  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501

# noqa: E501
from evals.benchmarks.hallucination.datasets import (  # noqa: E501
    generate_grounded_pairs,  # noqa: E501
    generate_hallucinated_pairs,  # noqa: E501
    generate_mixed_dataset,  # noqa: E501
    get_dataset_metadata,  # noqa: E501
)  # noqa: E501
from evals.benchmarks.hallucination.hhem import get_evaluator  # noqa: E402  # noqa: E501


  # noqa: E501
  # noqa: E501
def main() -> int:  # noqa: E501
    """Run HHEM hallucination evaluation benchmark.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        Exit code (0 for success, 1 for error)  # noqa: E501
    """  # noqa: E501
    parser = argparse.ArgumentParser(description="Run HHEM hallucination evaluation benchmark")  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--output",  # noqa: E501
        type=Path,  # noqa: E501
        required=True,  # noqa: E501
        help="Output JSON file for results",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--model-name",  # noqa: E501
        type=str,  # noqa: E501
        default="default",  # noqa: E501
        help="Name of the model being evaluated (for metadata)",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--dataset",  # noqa: E501
        type=str,  # noqa: E501
        choices=["grounded", "hallucinated", "mixed"],  # noqa: E501
        default="mixed",  # noqa: E501
        help="Which dataset to evaluate: grounded (should score high), "  # noqa: E501
        "hallucinated (should score low), or mixed (both)",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--templates",  # noqa: E501
        type=str,  # noqa: E501
        nargs="*",  # noqa: E501
        default=[],  # noqa: E501
        help="Filter by prompt templates (e.g., basic rag few_shot)",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--verbose",  # noqa: E501
        action="store_true",  # noqa: E501
        help="Show progress and detailed output",  # noqa: E501
    )  # noqa: E501
    args = parser.parse_args()  # noqa: E501
  # noqa: E501
    if args.verbose:  # noqa: E501
        print("Initializing HHEM evaluator...")  # noqa: E501
        print(f"Model name: {args.model_name}")  # noqa: E501
        print(f"Dataset: {args.dataset}")  # noqa: E501
  # noqa: E501
    # Load appropriate dataset  # noqa: E501
    if args.dataset == "grounded":  # noqa: E501
        dataset = generate_grounded_pairs()  # noqa: E501
        if args.verbose:  # noqa: E501
            print(f"Loaded {len(dataset)} grounded pairs (should score high)")  # noqa: E501
    elif args.dataset == "hallucinated":  # noqa: E501
        dataset = generate_hallucinated_pairs()  # noqa: E501
        if args.verbose:  # noqa: E501
            print(f"Loaded {len(dataset)} hallucinated pairs (should score low)")  # noqa: E501
    else:  # mixed  # noqa: E501
        dataset = generate_mixed_dataset()  # noqa: E501
        if args.verbose:  # noqa: E501
            print(f"Loaded {len(dataset)} mixed pairs")  # noqa: E501
  # noqa: E501
    if args.verbose:  # noqa: E501
        metadata = get_dataset_metadata()  # noqa: E501
        print(f"Dataset metadata: {metadata}")  # noqa: E501
  # noqa: E501
    # Get evaluator  # noqa: E501
    evaluator = get_evaluator()  # noqa: E501
  # noqa: E501
    if args.verbose:  # noqa: E501
        print("Running HHEM evaluation...")  # noqa: E501
  # noqa: E501
    # Run benchmark  # noqa: E501
    result = evaluator.run_benchmark(  # noqa: E501
        model_name=args.model_name,  # noqa: E501
        dataset=dataset,  # noqa: E501
        prompt_templates=args.templates or [],  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    # Prepare output data (convert dataclass to dict, excluding full results for brevity)  # noqa: E501
    output_data = {  # noqa: E501
        "model_name": result.model_name,  # noqa: E501
        "num_samples": result.num_samples,  # noqa: E501
        "mean_score": round(result.mean_score, 4),  # noqa: E501
        "median_score": round(result.median_score, 4),  # noqa: E501
        "std_score": round(result.std_score, 4),  # noqa: E501
        "pass_rate_at_05": round(result.pass_rate_at_05, 4),  # noqa: E501
        "pass_rate_at_07": round(result.pass_rate_at_07, 4),  # noqa: E501
        "timestamp": result.timestamp,  # noqa: E501
        "dataset_type": args.dataset,  # noqa: E501
        "template_filter": args.templates or "all",  # noqa: E501
    }  # noqa: E501
  # noqa: E501
    # Optionally include detailed results  # noqa: E501
    if args.verbose:  # noqa: E501
        output_data["detailed_results"] = [  # noqa: E501
            {  # noqa: E501
                "prompt_template": r.prompt_template,  # noqa: E501
                "hhem_score": round(r.hhem_score, 4),  # noqa: E501
                "source_preview": r.source_text[:100] + "..."  # noqa: E501
                if len(r.source_text) > 100  # noqa: E501
                else r.source_text,  # noqa: E501
                "summary_preview": r.generated_summary[:100] + "..."  # noqa: E501
                if len(r.generated_summary) > 100  # noqa: E501
                else r.generated_summary,  # noqa: E501
            }  # noqa: E501
            for r in result.results  # noqa: E501
        ]  # noqa: E501
  # noqa: E501
    # Ensure output directory exists  # noqa: E501
    args.output.parent.mkdir(parents=True, exist_ok=True)  # noqa: E501
  # noqa: E501
    # Save results  # noqa: E501
    args.output.write_text(json.dumps(output_data, indent=2))  # noqa: E501
  # noqa: E501
    # Print summary  # noqa: E501
    print("\n" + "=" * 50)  # noqa: E501
    print("HHEM BENCHMARK RESULTS")  # noqa: E501
    print("=" * 50)  # noqa: E501
    print(f"Model: {result.model_name}")  # noqa: E501
    print(f"Samples: {result.num_samples}")  # noqa: E501
    print(f"Mean Score: {result.mean_score:.4f}")  # noqa: E501
    print(f"Median Score: {result.median_score:.4f}")  # noqa: E501
    print(f"Std Dev: {result.std_score:.4f}")  # noqa: E501
    print(f"Pass Rate @0.5: {result.pass_rate_at_05:.1%}")  # noqa: E501
    print(f"Pass Rate @0.7: {result.pass_rate_at_07:.1%}")  # noqa: E501
    print("=" * 50)  # noqa: E501
  # noqa: E501
    # Gate check (G3: mean >= 0.5)  # noqa: E501
    if result.mean_score >= 0.5:  # noqa: E501
        print("Gate G3: PASS (mean HHEM score >= 0.5)")  # noqa: E501
    elif result.mean_score >= 0.4:  # noqa: E501
        print("Gate G3: CONDITIONAL (mean HHEM score 0.4-0.5)")  # noqa: E501
    else:  # noqa: E501
        print("Gate G3: FAIL (mean HHEM score < 0.4)")  # noqa: E501
  # noqa: E501
    print(f"\nResults saved to: {args.output}")  # noqa: E501
  # noqa: E501
    return 0  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    sys.exit(main())  # noqa: E501
