#!/usr/bin/env python3
"""Benchmark NLI models for speech act classification.

Tests different NLI models on sample iMessage-style texts to measure:
- Latency (single hypothesis vs multi-hypothesis)
- Memory usage
- Classification quality (qualitative)

Usage:
    uv run python scripts/benchmark_speech_acts.py
    uv run python scripts/benchmark_speech_acts.py --model base --multi
    uv run python scripts/benchmark_speech_acts.py --all
    uv run python scripts/benchmark_speech_acts.py --detailed
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


# Sample iMessage-style texts covering all speech acts
# Organized by expected category for evaluation
SAMPLE_TEXTS = {
    "directive": [
        # Requests
        "Can you pick me up at 5?",
        "Send me the file when you get a chance",
        "Could you check if the door is locked?",
        "Call me back when you're free",
        "Please grab milk on your way home",
        # Invitations
        "Wanna grab lunch tomorrow?",
        "Let's hang out this weekend",
        "You down to go to the movies?",
        "Want to come over later?",
        # Questions requiring commitment (not just info)
        "Are you coming to the party?",
        "Can you make it tonight?",
        "You free Saturday?",
    ],
    "assertive": [
        # Facts/reports
        "I went to the grocery store",
        "The meeting got moved to 3pm",
        "Weather looks nice today",
        "The new restaurant downtown is pretty good",
        "I finished the project yesterday",
        "Traffic was terrible this morning",
        # Opinions
        "I think that's a good idea",
        "That movie was overrated",
        "I don't think it's going to rain",
    ],
    "commissive": [
        # Promises/plans
        "I'll be there at 5",
        "I'm going to call you later",
        "omw",
        "I'll pick up dinner on the way home",
        "I promise I'll be on time",
        "I'm gonna send you the link",
        "I'll take care of it",
        # Offers
        "I can help you move this weekend",
        "Want me to grab you something?",
    ],
    "expressive": [
        # Positive emotions
        "That's amazing!",
        "I'm so excited!!",
        "Congrats on the new job!",
        "That's hilarious 😂",
        "Love you",
        # Negative emotions
        "So sorry to hear that",
        "This sucks",
        "I miss you",
        "Ugh that's frustrating",
        # Thanks/apologies
        "Thanks so much for your help",
        "My bad, I forgot",
        "Sorry I'm late",
    ],
    "phatic": [
        # Greetings
        "Hey!",
        "What's up",
        "Good morning",
        "Yo",
        # Acknowledgments
        "Ok",
        "Got it",
        "Sounds good",
        "Yeah",
        "Cool",
        # Backchannels
        "Lol",
        "Haha",
        "Nice",
        # Farewells
        "Bye!",
        "Talk later",
        "No worries",
    ],
}


def benchmark_model(
    model_name: str,
    multi_hypothesis: bool = False,
    n_runs: int = 3,
    verbose: bool = True,
) -> dict:
    """Benchmark a single model configuration."""
    from jarvis.classifiers.zeroshot_nli import SpeechActClassifier, NLI_MODELS

    # Clear any cached models
    gc.collect()

    mem_before = get_memory_mb()

    if verbose:
        model_id = NLI_MODELS.get(model_name, model_name)
        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({model_id})")
        print(f"Multi-hypothesis: {multi_hypothesis}")
        print(f"{'='*60}")

    # Load model
    load_start = time.perf_counter()
    classifier = SpeechActClassifier(
        model_name=model_name,
        multi_hypothesis=multi_hypothesis,
    )
    # Force model load
    _ = classifier.model
    load_time = (time.perf_counter() - load_start) * 1000

    mem_after = get_memory_mb()
    mem_delta = mem_after - mem_before

    if verbose:
        print(f"Load time: {load_time:.0f}ms")
        print(f"Memory delta: {mem_delta:.0f}MB (total: {mem_after:.0f}MB)")
        print()

    # Flatten samples for benchmarking
    all_texts = []
    for texts in SAMPLE_TEXTS.values():
        all_texts.extend(texts)

    # Benchmark
    stats = classifier.benchmark(all_texts, n_runs=n_runs)

    if verbose:
        print(f"Latency stats ({len(all_texts)} texts x {n_runs} runs):")
        print(f"  Mean: {stats['mean_latency_ms']:.1f}ms")
        print(f"  Std:  {stats['std_latency_ms']:.1f}ms")
        print(f"  P50:  {stats['p50_latency_ms']:.1f}ms")
        print(f"  P95:  {stats['p95_latency_ms']:.1f}ms")
        print(f"  Min:  {stats['min_latency_ms']:.1f}ms")
        print(f"  Max:  {stats['max_latency_ms']:.1f}ms")

    # Show sample classifications
    if verbose:
        print(f"\nSample classifications:")
        samples = [
            "Can you pick me up?",
            "I'll be there at 5",
            "That's amazing!",
            "Ok",
            "The meeting is at 3pm",
        ]
        for text in samples:
            result = classifier.classify(text)
            print(f"  '{text}' -> {result.act.value} ({result.confidence:.2f})")

    return {
        **stats,
        "load_time_ms": load_time,
        "memory_delta_mb": mem_delta,
        "memory_total_mb": mem_after,
    }


def benchmark_all_models(n_runs: int = 2) -> None:
    """Benchmark all available models."""
    from jarvis.classifiers.zeroshot_nli import reset_classifier

    results = []

    # Test small models with both single and multi hypothesis
    test_configs = [
        ("xsmall", False),
        ("xsmall", True),
        ("mini", False),
        ("distil", False),
    ]

    for model_name, multi in test_configs:
        try:
            stats = benchmark_model(model_name, multi_hypothesis=multi, n_runs=n_runs)
            results.append({
                "model": model_name,
                "multi": multi,
                **stats,
            })
        except Exception as e:
            print(f"Error with {model_name}: {e}")

        # Clean up between models
        reset_classifier()
        gc.collect()

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Model':<15} {'Multi':<6} {'Load(ms)':<10} {'Mean(ms)':<10} {'P95(ms)':<10} {'Mem(MB)':<10}")
    print("-"*80)
    for r in results:
        print(
            f"{r['model']:<15} "
            f"{str(r['multi']):<6} "
            f"{r['load_time_ms']:<10.0f} "
            f"{r['mean_latency_ms']:<10.1f} "
            f"{r['p95_latency_ms']:<10.1f} "
            f"{r['memory_delta_mb']:<10.0f}"
        )


def show_detailed_results(model_name: str, multi: bool = False) -> None:
    """Show detailed classification results for all sample texts."""
    from jarvis.classifiers.zeroshot_nli import SpeechActClassifier, SpeechAct

    classifier = SpeechActClassifier(model_name=model_name, multi_hypothesis=multi)

    print(f"\n{'='*80}")
    print(f"Detailed Results: {model_name} (multi={multi})")
    print(f"{'='*80}")

    correct = 0
    total = 0

    for expected_act_str, texts in SAMPLE_TEXTS.items():
        expected_act = SpeechAct(expected_act_str)
        print(f"\n--- Expected: {expected_act.value.upper()} ---")

        for text in texts:
            result = classifier.classify(text)
            match = "✓" if result.act == expected_act else "✗"
            if result.act == expected_act:
                correct += 1
            total += 1

            # Show top 2 scores
            sorted_scores = sorted(result.scores.items(), key=lambda x: x[1], reverse=True)
            top2 = ", ".join(f"{a.value}:{s:.2f}" for a, s in sorted_scores[:2])

            print(f"  {match} '{text[:40]:<40}' -> {result.act.value:<12} [{top2}]")

    print(f"\n{'='*80}")
    print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"{'='*80}")


def compare_hypotheses(model_name: str = "xsmall") -> None:
    """Compare single vs multi hypothesis performance."""
    from jarvis.classifiers.zeroshot_nli import SpeechActClassifier, SpeechAct

    print(f"\n{'='*60}")
    print(f"Comparing Single vs Multi Hypothesis ({model_name})")
    print(f"{'='*60}")

    single_clf = SpeechActClassifier(model_name=model_name, multi_hypothesis=False)
    multi_clf = SpeechActClassifier(model_name=model_name, multi_hypothesis=True)

    single_correct = 0
    multi_correct = 0
    total = 0
    disagreements = []

    for expected_act_str, texts in SAMPLE_TEXTS.items():
        expected_act = SpeechAct(expected_act_str)

        for text in texts:
            single_result = single_clf.classify(text)
            multi_result = multi_clf.classify(text)

            if single_result.act == expected_act:
                single_correct += 1
            if multi_result.act == expected_act:
                multi_correct += 1
            total += 1

            if single_result.act != multi_result.act:
                disagreements.append({
                    "text": text,
                    "expected": expected_act.value,
                    "single": single_result.act.value,
                    "multi": multi_result.act.value,
                })

    print(f"\nSingle-hypothesis accuracy: {single_correct}/{total} ({100*single_correct/total:.1f}%)")
    print(f"Multi-hypothesis accuracy:  {multi_correct}/{total} ({100*multi_correct/total:.1f}%)")
    print(f"\nDisagreements: {len(disagreements)}")

    if disagreements:
        print("\nWhere they differ:")
        for d in disagreements[:10]:
            exp_marker = lambda x: "✓" if x == d["expected"] else "✗"
            print(f"  '{d['text'][:35]:<35}' expected={d['expected']}")
            print(f"      single={d['single']} {exp_marker(d['single'])}, multi={d['multi']} {exp_marker(d['multi'])}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark NLI speech act classifiers")
    parser.add_argument(
        "--model", "-m",
        default="xsmall",
        help="Model to benchmark (xsmall, mini, distil, base, large, bart)"
    )
    parser.add_argument(
        "--multi", "-M",
        action="store_true",
        help="Use multi-hypothesis mode"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Benchmark all models"
    )
    parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="Show detailed per-text results"
    )
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Compare single vs multi hypothesis"
    )
    parser.add_argument(
        "--runs", "-n",
        type=int,
        default=3,
        help="Number of benchmark runs"
    )

    args = parser.parse_args()

    if args.all:
        benchmark_all_models(n_runs=args.runs)
    elif args.detailed:
        show_detailed_results(args.model, multi=args.multi)
    elif args.compare:
        compare_hypotheses(args.model)
    else:
        benchmark_model(args.model, multi_hypothesis=args.multi, n_runs=args.runs)


if __name__ == "__main__":
    main()
