#!/usr/bin/env python3
"""Benchmark different NLI models on actual casual chat texts.

Tests the core problem: casual musings being misclassified as questions.
"I wonder if bulls make another trade" -> should be ACKNOWLEDGEABLE (musing)
"No way Dallas could've gotten lottery picks" -> should be ACKNOWLEDGEABLE (opinion)

Usage:
    uv run python scripts/benchmark_nli_models.py
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass

import numpy as np

# Problematic casual chat texts with expected categories
# These are ACTUAL texts that NLI models tend to misclassify
ACTUAL_TEXTS = [
    # Musings/wonderings - NOT questions, just thinking out loud
    ("I wonder if bulls make another trade", "ACKNOWLEDGEABLE"),
    ("Kind of curious what they will offer", "ACKNOWLEDGEABLE"),
    ("I wonder what happened", "ACKNOWLEDGEABLE"),
    ("Curious how they'll handle it", "ACKNOWLEDGEABLE"),
    ("Wonder if anyone else noticed", "ACKNOWLEDGEABLE"),
    # Opinions disguised as statements
    ("No way Dallas could've gotten lottery picks", "ACKNOWLEDGEABLE"),
    ("No way that's real", "ACKNOWLEDGEABLE"),  # NOT expressive, just disbelief statement
    ("No chance they win tonight", "ACKNOWLEDGEABLE"),
    ("I don't think it's gonna happen", "ACKNOWLEDGEABLE"),
    ("Doubt they'll actually do it", "ACKNOWLEDGEABLE"),
    # Rhetorical / statement questions
    ("Why do dads text like that", "ACKNOWLEDGEABLE"),  # rhetorical
    ("How does that even work", "ACKNOWLEDGEABLE"),  # rhetorical amazement
    ("Who even says that anymore", "ACKNOWLEDGEABLE"),  # rhetorical
    # Actual questions (SHOULD be ANSWERABLE)
    ("What time is the game", "ANSWERABLE"),
    ("Where are you", "ANSWERABLE"),
    ("Did you get my text", "ANSWERABLE"),
    ("What happened at the meeting", "ANSWERABLE"),
    # Actual requests (SHOULD be ACTIONABLE)
    ("Can you pick me up", "ACTIONABLE"),
    ("Wanna grab lunch", "ACTIONABLE"),
    ("Let me know when you're free", "ACTIONABLE"),
    ("Text me when you get there", "ACTIONABLE"),
    # Actual reactive content (SHOULD be REACTIVE)
    ("Omg I got the job!!", "REACTIVE"),
    ("That's so sad", "REACTIVE"),
    ("This is amazing!!", "REACTIVE"),
    ("I can't believe it!!!", "REACTIVE"),
    # Statements that need simple ack
    ("I went to the store", "ACKNOWLEDGEABLE"),
    ("The game starts at 7", "ACKNOWLEDGEABLE"),
    ("Traffic was bad today", "ACKNOWLEDGEABLE"),
    ("I'm heading out now", "ACKNOWLEDGEABLE"),
]


# Response-oriented hypotheses (same as response_nli.py)
HYPOTHESES = {
    "ACTIONABLE": "This requires a yes or no decision about doing something.",
    "ANSWERABLE": "This is a question asking for specific information.",
    "REACTIVE": "This shares exciting or upsetting news that deserves a reaction.",
    "ACKNOWLEDGEABLE": "This is a statement or observation, not a question or request.",
}


# Models to test
MODELS_TO_TEST = [
    ("cross-encoder/nli-deberta-v3-xsmall", "xsmall (70M)"),
    ("cross-encoder/nli-MiniLM2-L6-H768", "MiniLM (82M)"),
    ("cross-encoder/nli-deberta-v3-small", "small (100M)"),
    ("cross-encoder/nli-distilroberta-base", "distilroberta (82M)"),
    ("cross-encoder/nli-deberta-v3-base", "base (184M)"),
]


@dataclass
class BenchmarkResult:
    model_name: str
    accuracy: float
    accuracy_by_category: dict[str, float]
    misclassifications: list[tuple[str, str, str, dict[str, float]]]  # text, expected, got, scores
    load_time_ms: float
    avg_latency_ms: float
    memory_mb: float


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def load_model(model_id: str):
    """Load NLI cross-encoder model."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_id, max_length=512)


def classify_nli(model, text: str, hypotheses: dict[str, str]) -> tuple[str, dict[str, float]]:
    """Classify text using NLI model.

    Returns (predicted_category, scores_dict)
    """
    pairs = [(text, h) for h in hypotheses.values()]
    logits = model.predict(pairs)

    scores = {}
    for i, cat in enumerate(hypotheses.keys()):
        exp = np.exp(logits[i] - np.max(logits[i]))
        probs = exp / exp.sum()
        scores[cat] = float(probs[1])  # entailment index

    best = max(scores, key=scores.get)
    return best, scores


def benchmark_model(model_id: str, model_label: str) -> BenchmarkResult:
    """Benchmark a single model on all test texts."""
    gc.collect()
    mem_before = get_memory_mb()

    # Load model
    load_start = time.perf_counter()
    model = load_model(model_id)
    load_time = (time.perf_counter() - load_start) * 1000

    mem_after = get_memory_mb()

    # Run classification
    correct = 0
    correct_by_cat: dict[str, int] = {cat: 0 for cat in HYPOTHESES.keys()}
    total_by_cat: dict[str, int] = {cat: 0 for cat in HYPOTHESES.keys()}
    misclassifications = []
    latencies = []

    for text, expected in ACTUAL_TEXTS:
        total_by_cat[expected] += 1

        start = time.perf_counter()
        predicted, scores = classify_nli(model, text, HYPOTHESES)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        if predicted == expected:
            correct += 1
            correct_by_cat[expected] += 1
        else:
            misclassifications.append((text, expected, predicted, scores))

    accuracy = correct / len(ACTUAL_TEXTS)
    accuracy_by_cat = {
        cat: correct_by_cat[cat] / total_by_cat[cat] if total_by_cat[cat] > 0 else 0.0
        for cat in HYPOTHESES.keys()
    }

    return BenchmarkResult(
        model_name=model_label,
        accuracy=accuracy,
        accuracy_by_category=accuracy_by_cat,
        misclassifications=misclassifications,
        load_time_ms=load_time,
        avg_latency_ms=np.mean(latencies),
        memory_mb=mem_after - mem_before,
    )


def print_result(result: BenchmarkResult, verbose: bool = True):
    """Print benchmark result."""
    print(f"\n{'='*70}")
    print(f"Model: {result.model_name}")
    print(f"{'='*70}")
    print(f"Overall Accuracy: {result.accuracy:.1%}")
    print(
        f"Load Time: {result.load_time_ms:.0f}ms | Latency: {result.avg_latency_ms:.1f}ms | Memory: {result.memory_mb:.0f}MB"
    )
    print()
    print("Accuracy by category:")
    for cat, acc in result.accuracy_by_category.items():
        print(f"  {cat:15s}: {acc:.1%}")

    if verbose and result.misclassifications:
        print(f"\nMisclassifications ({len(result.misclassifications)}):")
        for text, expected, got, scores in result.misclassifications[:10]:
            score_str = ", ".join(
                f"{k[:3]}:{v:.2f}" for k, v in sorted(scores.items(), key=lambda x: -x[1])[:3]
            )
            print(f"  '{text[:45]:<45s}' expected={expected:14s} got={got:14s} [{score_str}]")


def main():
    print("NLI Model Comparison for Casual Chat Classification")
    print("=" * 70)
    print(f"Testing on {len(ACTUAL_TEXTS)} actual texts")
    print()

    results = []

    for model_id, model_label in MODELS_TO_TEST:
        try:
            result = benchmark_model(model_id, model_label)
            results.append(result)
            print_result(result, verbose=True)
        except Exception as e:
            print(f"Error with {model_label}: {e}")

        # Clean up
        gc.collect()

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"{'Model':<25s} {'Accuracy':>10s} {'ANSWER':>10s} {'ACK':>10s} {'Latency':>10s} {'Memory':>10s}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r.model_name:<25s} "
            f"{r.accuracy:>9.1%} "
            f"{r.accuracy_by_category.get('ANSWERABLE', 0):>9.1%} "
            f"{r.accuracy_by_category.get('ACKNOWLEDGEABLE', 0):>9.1%} "
            f"{r.avg_latency_ms:>9.1f}ms "
            f"{r.memory_mb:>9.0f}MB"
        )

    # Find best model
    best = max(results, key=lambda x: x.accuracy)
    print(f"\nBest model: {best.model_name} ({best.accuracy:.1%})")


if __name__ == "__main__":
    main()
