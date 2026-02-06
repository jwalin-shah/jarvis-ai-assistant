"""Response Classifier Benchmarking Suite.

Measures and compares performance of V1 vs V2 response classifiers:
- Throughput: messages per second
- Latency: p50, p95, p99 percentiles
- Memory usage
- Accuracy vs speed tradeoffs

Usage:
    # Run full benchmark suite
    python -m benchmarks.classifier.classifier_benchmark

    # Run specific benchmark
    python -m benchmarks.classifier.classifier_benchmark --benchmark throughput

    # Compare V1 vs V2
    python -m benchmarks.classifier.classifier_benchmark --compare
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import statistics
import sys
import tracemalloc
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from evals.benchmarks.latency.timer import HighPrecisionTimer, force_model_unload, warmup_timer

logger = logging.getLogger(__name__)


# =============================================================================
# Benchmark Data
# =============================================================================

# Representative test messages for each category
BENCHMARK_MESSAGES: dict[str, list[str]] = {
    "agree": [
        "Yes!",
        "Yeah definitely",
        "I'm down",
        "Sounds good",
        "Sure thing",
        "Let's do it",
        "Count me in",
        "For sure",
        "Works for me",
        "That works",
    ],
    "decline": [
        "No",
        "Nope",
        "Can't make it",
        "Sorry I'm busy",
        "I'll pass",
        "Not tonight",
        "Won't be able to",
        "Rain check",
        "I can't",
        "Unfortunately no",
    ],
    "defer": [
        "Maybe",
        "Let me check",
        "I'll see",
        "Not sure yet",
        "Depends",
        "We'll see",
        "Might be able to",
        "TBD",
        "Let me think about it",
        "I'll let you know",
    ],
    "question": [
        "What time?",
        "Where is it?",
        "Who's going?",
        "How do I get there?",
        "When does it start?",
        "Can you send me the address?",
        "What should I bring?",
        "Is parking available?",
        "Do I need to RSVP?",
        "What's the dress code?",
    ],
    "acknowledge": [
        "Ok",
        "Got it",
        "Alright",
        "Cool",
        "Noted",
        "Will do",
        "No problem",
        "Makes sense",
        "I see",
        "Word",
    ],
    "react_positive": [
        "Congrats!",
        "That's awesome!",
        "So happy for you!",
        "OMG",
        "Yay!",
        "Nice!",
        "Let's gooo",
        "Well done",
        "lol",
        "haha that's hilarious",
    ],
    "react_sympathy": [
        "I'm sorry to hear that",
        "That sucks",
        "Damn",
        "Here for you",
        "Hang in there",
        "Let me know if you need anything",
        "Oh no",
        "That's rough",
        "Sending hugs",
        "Take care of yourself",
    ],
    "greeting": [
        "Hey!",
        "Hi",
        "Hello",
        "Yo",
        "Sup",
        "What's up",
        "Good morning",
        "Evening",
        "Hiya",
        "Hey there",
    ],
    "statement": [
        "I went to the store today",
        "The weather is nice outside",
        "I finished the project",
        "Traffic was bad this morning",
        "The movie was really good",
        "I'm working from home",
        "My flight got delayed",
        "The restaurant was packed",
        "I started a new book",
        "My cat is sleeping",
    ],
    "mixed": [
        "Yeah I think so but let me check first",
        "That sounds great, when should we meet?",
        "I can't tomorrow but maybe Thursday?",
        "lol no way that happened!",
        "Thanks for letting me know, I'll be there",
        "Sorry to hear that, let me know if I can help",
        "Good morning! How's it going?",
        "Nice! Where did you find it?",
        "Hmm not sure, maybe ask John?",
        "Ok cool, see you then!",
    ],
}


def generate_benchmark_data(n_messages: int = 1000) -> list[str]:
    """Generate a diverse set of benchmark messages.

    Args:
        n_messages: Total number of messages to generate.

    Returns:
        List of benchmark messages.
    """
    all_messages = []
    for category, messages in BENCHMARK_MESSAGES.items():
        all_messages.extend(messages)

    # Repeat and shuffle to get desired count
    result: list[str] = []
    while len(result) < n_messages:
        result.extend(all_messages)

    np.random.shuffle(result)
    return result[:n_messages]


# =============================================================================
# Benchmark Results
# =============================================================================


@dataclass
class LatencyStats:
    """Latency statistics in milliseconds."""

    p50: float
    p95: float
    p99: float
    mean: float
    min: float
    max: float
    std: float


@dataclass
class ThroughputStats:
    """Throughput statistics."""

    messages_per_second: float
    total_messages: int
    total_time_ms: float
    batch_size: int | None = None


@dataclass
class MemoryStats:
    """Memory usage statistics in MB."""

    peak_mb: float
    current_mb: float
    allocated_mb: float


@dataclass
class AccuracyStats:
    """Classification accuracy statistics."""

    total: int
    correct: int
    accuracy: float
    per_class_accuracy: dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a classifier version."""

    version: str
    latency: LatencyStats | None = None
    throughput: ThroughputStats | None = None
    memory: MemoryStats | None = None
    accuracy: AccuracyStats | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {"version": self.version, "metadata": self.metadata}
        if self.latency:
            result["latency"] = asdict(self.latency)
        if self.throughput:
            result["throughput"] = asdict(self.throughput)
        if self.memory:
            result["memory"] = asdict(self.memory)
        if self.accuracy:
            result["accuracy"] = asdict(self.accuracy)
        return result


# =============================================================================
# Benchmark Functions
# =============================================================================


def measure_latency(
    classify_fn: Callable[[str], Any],
    messages: list[str],
    warmup_iterations: int = 10,
) -> LatencyStats:
    """Measure single-message classification latency.

    Args:
        classify_fn: Function that classifies a single message.
        messages: List of messages to classify.
        warmup_iterations: Number of warmup iterations.

    Returns:
        LatencyStats with percentile measurements.
    """
    # Warmup
    for msg in messages[:warmup_iterations]:
        classify_fn(msg)

    # Measure
    latencies_ms: list[float] = []
    timer = HighPrecisionTimer()

    for msg in messages:
        timer.start()
        classify_fn(msg)
        result = timer.stop()
        latencies_ms.append(result.elapsed_ms)
        timer.reset()

    return LatencyStats(
        p50=float(np.percentile(latencies_ms, 50)),
        p95=float(np.percentile(latencies_ms, 95)),
        p99=float(np.percentile(latencies_ms, 99)),
        mean=statistics.mean(latencies_ms),
        min=min(latencies_ms),
        max=max(latencies_ms),
        std=statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
    )


def measure_batch_latency(
    classify_batch_fn: Callable[[list[str]], list[Any]],
    messages: list[str],
    batch_size: int = 64,
    warmup_batches: int = 2,
) -> LatencyStats:
    """Measure batch classification latency (per-message).

    Args:
        classify_batch_fn: Function that classifies a batch of messages.
        messages: List of messages to classify.
        batch_size: Batch size for processing.
        warmup_batches: Number of warmup batches.

    Returns:
        LatencyStats with per-message latency percentiles.
    """
    # Split into batches
    batches = [messages[i : i + batch_size] for i in range(0, len(messages), batch_size)]

    # Warmup
    for batch in batches[:warmup_batches]:
        classify_batch_fn(batch)

    # Measure
    per_message_latencies: list[float] = []
    timer = HighPrecisionTimer()

    for batch in batches:
        timer.start()
        classify_batch_fn(batch)
        result = timer.stop()
        per_msg_latency = result.elapsed_ms / len(batch)
        per_message_latencies.extend([per_msg_latency] * len(batch))
        timer.reset()

    return LatencyStats(
        p50=float(np.percentile(per_message_latencies, 50)),
        p95=float(np.percentile(per_message_latencies, 95)),
        p99=float(np.percentile(per_message_latencies, 99)),
        mean=statistics.mean(per_message_latencies),
        min=min(per_message_latencies),
        max=max(per_message_latencies),
        std=statistics.stdev(per_message_latencies) if len(per_message_latencies) > 1 else 0.0,
    )


def measure_throughput(
    classify_batch_fn: Callable[[list[str]], list[Any]],
    messages: list[str],
    batch_size: int = 64,
    warmup_batches: int = 2,
) -> ThroughputStats:
    """Measure classification throughput.

    Args:
        classify_batch_fn: Function that classifies a batch of messages.
        messages: List of messages to classify.
        batch_size: Batch size for processing.
        warmup_batches: Number of warmup batches.

    Returns:
        ThroughputStats with messages per second.
    """
    # Split into batches
    batches = [messages[i : i + batch_size] for i in range(0, len(messages), batch_size)]

    # Warmup
    for batch in batches[:warmup_batches]:
        classify_batch_fn(batch)

    # Measure total time
    timer = HighPrecisionTimer()
    timer.start()

    for batch in batches:
        classify_batch_fn(batch)

    result = timer.stop()

    messages_per_second = len(messages) / (result.elapsed_ms / 1000)

    return ThroughputStats(
        messages_per_second=messages_per_second,
        total_messages=len(messages),
        total_time_ms=result.elapsed_ms,
        batch_size=batch_size,
    )


def measure_memory(
    classify_batch_fn: Callable[[list[str]], list[Any]],
    messages: list[str],
    batch_size: int = 64,
) -> MemoryStats:
    """Measure memory usage during classification.

    Args:
        classify_batch_fn: Function that classifies a batch of messages.
        messages: List of messages to classify.
        batch_size: Batch size for processing.

    Returns:
        MemoryStats with peak and current memory usage.
    """
    # Force cleanup before measurement
    gc.collect()
    force_model_unload()

    # Start memory tracking
    tracemalloc.start()

    # Process all messages
    batches = [messages[i : i + batch_size] for i in range(0, len(messages), batch_size)]
    for batch in batches:
        classify_batch_fn(batch)

    # Get memory stats
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return MemoryStats(
        peak_mb=peak / (1024 * 1024),
        current_mb=current / (1024 * 1024),
        allocated_mb=peak / (1024 * 1024),
    )


def measure_accuracy(
    classify_fn: Callable[[str], Any],
    messages: list[str],
    expected_labels: list[str],
    label_extractor: Callable[[Any], str] | None = None,
) -> AccuracyStats:
    """Measure classification accuracy against expected labels.

    Args:
        classify_fn: Function that classifies a single message.
        messages: List of messages to classify.
        expected_labels: Expected classification labels.
        label_extractor: Function to extract label string from result.

    Returns:
        AccuracyStats with overall and per-class accuracy.
    """

    def default_label_extractor(r: Any) -> str:
        return r.label.value if hasattr(r.label, "value") else str(r.label)

    if label_extractor is None:
        label_extractor = default_label_extractor

    correct = 0
    per_class_correct: dict[str, int] = {}
    per_class_total: dict[str, int] = {}

    for msg, expected in zip(messages, expected_labels):
        result = classify_fn(msg)
        predicted = label_extractor(result)

        # Track per-class stats
        per_class_total[expected] = per_class_total.get(expected, 0) + 1

        if predicted.upper() == expected.upper():
            correct += 1
            per_class_correct[expected] = per_class_correct.get(expected, 0) + 1

    # Calculate per-class accuracy
    per_class_accuracy = {}
    for label in per_class_total:
        per_class_accuracy[label] = per_class_correct.get(label, 0) / per_class_total[label]

    return AccuracyStats(
        total=len(messages),
        correct=correct,
        accuracy=correct / len(messages) if messages else 0.0,
        per_class_accuracy=per_class_accuracy,
    )


# =============================================================================
# Classifier Benchmarks
# =============================================================================


def benchmark_v1_classifier(messages: list[str], batch_size: int = 64) -> BenchmarkResult:
    """Benchmark V1 response classifier.

    Args:
        messages: Test messages.
        batch_size: Batch size for throughput measurement.

    Returns:
        BenchmarkResult with all measurements.
    """
    from jarvis.classifiers.response_classifier import (
        get_response_classifier,
        reset_response_classifier,
    )

    # Reset to ensure clean state
    reset_response_classifier()
    classifier = get_response_classifier()

    # Warmup
    classifier.classify("Hello")

    result = BenchmarkResult(version="v1")

    # Single-message latency
    print("  Measuring V1 single-message latency...")
    result.latency = measure_latency(classifier.classify, messages[:500])

    # Batch throughput
    print("  Measuring V1 throughput...")
    result.throughput = measure_throughput(classifier.classify_batch, messages, batch_size)

    # Memory usage
    print("  Measuring V1 memory usage...")
    result.memory = measure_memory(classifier.classify_batch, messages, batch_size)

    result.metadata = {
        "has_svm": getattr(classifier, "_svm", None) is not None,
        "has_centroids": bool(getattr(classifier, "centroids", None)),
    }

    return result


def benchmark_v2_classifier(
    messages: list[str],
    batch_size: int = 64,
    enable_caching: bool = True,
    enable_ensemble: bool = True,
) -> BenchmarkResult:
    """Benchmark V2 response classifier.

    Args:
        messages: Test messages.
        batch_size: Batch size for throughput measurement.
        enable_caching: Enable embedding/feature caching.
        enable_ensemble: Enable ensemble voting.

    Returns:
        BenchmarkResult with all measurements.
    """
    from jarvis.classifiers.response_classifier_v2 import (
        BatchResponseClassifier,
        reset_batch_response_classifier,
    )

    # Reset to ensure clean state
    reset_batch_response_classifier()
    classifier = BatchResponseClassifier(
        enable_caching=enable_caching,
        enable_ensemble=enable_ensemble,
        use_v2_api=True,
    )

    # Warmup
    classifier.warmup()

    result = BenchmarkResult(version="v2")

    # Single-message latency
    print("  Measuring V2 single-message latency...")
    result.latency = measure_latency(classifier.classify, messages[:500])

    # Batch latency (per-message)
    print("  Measuring V2 batch latency...")
    batch_latency = measure_batch_latency(classifier.classify_batch, messages, batch_size)
    result.metadata["batch_latency"] = asdict(batch_latency)

    # Batch throughput
    print("  Measuring V2 throughput...")
    result.throughput = measure_throughput(classifier.classify_batch, messages, batch_size)

    # Memory usage
    print("  Measuring V2 memory usage...")
    result.memory = measure_memory(classifier.classify_batch, messages, batch_size)

    result.metadata.update(
        {
            "enable_caching": enable_caching,
            "enable_ensemble": enable_ensemble,
            "cache_stats": classifier.get_cache_stats(),
            "has_svm": getattr(classifier, "_svm", None) is not None,
        }
    )

    classifier.shutdown()
    return result


def compare_classifiers(messages: list[str], batch_size: int = 64) -> dict[str, Any]:
    """Compare V1 vs V2 classifiers.

    Args:
        messages: Test messages.
        batch_size: Batch size for throughput measurement.

    Returns:
        Dictionary with comparison results.
    """
    print("\n" + "=" * 60)
    print("RESPONSE CLASSIFIER BENCHMARK COMPARISON")
    print("=" * 60)
    print(f"\nTest set: {len(messages)} messages")
    print(f"Batch size: {batch_size}")

    results: dict[str, Any] = {}

    # Benchmark V1
    print("\n--- V1 Classifier ---")
    try:
        v1_result = benchmark_v1_classifier(messages, batch_size)
        results["v1"] = v1_result.to_dict()
    except Exception as e:
        print(f"  V1 benchmark failed: {e}")
        results["v1"] = {"error": str(e)}

    # Benchmark V2 with caching
    print("\n--- V2 Classifier (with caching) ---")
    try:
        v2_result = benchmark_v2_classifier(messages, batch_size, enable_caching=True)
        results["v2_cached"] = v2_result.to_dict()
    except Exception as e:
        print(f"  V2 benchmark failed: {e}")
        results["v2_cached"] = {"error": str(e)}

    # Benchmark V2 without caching
    print("\n--- V2 Classifier (no caching) ---")
    try:
        v2_nocache_result = benchmark_v2_classifier(messages, batch_size, enable_caching=False)
        results["v2_no_cache"] = v2_nocache_result.to_dict()
    except Exception as e:
        print(f"  V2 no-cache benchmark failed: {e}")
        results["v2_no_cache"] = {"error": str(e)}

    # Calculate improvements
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    if "v1" in results and "v2_cached" in results:
        if "error" not in results["v1"] and "error" not in results["v2_cached"]:
            v1 = results["v1"]
            v2 = results["v2_cached"]

            # Latency comparison
            if v1.get("latency") and v2.get("latency"):
                v1_p95 = v1["latency"]["p95"]
                v2_p95 = v2["latency"]["p95"]
                latency_improvement = v1_p95 / v2_p95 if v2_p95 > 0 else 0

                print("\nLatency (p95):")
                print(f"  V1: {v1_p95:.2f}ms")
                print(f"  V2: {v2_p95:.2f}ms")
                print(f"  Improvement: {latency_improvement:.1f}x")

                results["comparison"] = {"latency_improvement": latency_improvement}

            # Throughput comparison
            if v1.get("throughput") and v2.get("throughput"):
                v1_tput = v1["throughput"]["messages_per_second"]
                v2_tput = v2["throughput"]["messages_per_second"]
                throughput_improvement = v2_tput / v1_tput if v1_tput > 0 else 0

                print("\nThroughput:")
                print(f"  V1: {v1_tput:.0f} msgs/sec")
                print(f"  V2: {v2_tput:.0f} msgs/sec")
                print(f"  Improvement: {throughput_improvement:.1f}x")

                results["comparison"]["throughput_improvement"] = throughput_improvement

            # Memory comparison
            if v1.get("memory") and v2.get("memory"):
                v1_mem = v1["memory"]["peak_mb"]
                v2_mem = v2["memory"]["peak_mb"]
                memory_ratio = v2_mem / v1_mem if v1_mem > 0 else 0

                print("\nPeak Memory:")
                print(f"  V1: {v1_mem:.1f}MB")
                print(f"  V2: {v2_mem:.1f}MB")
                print(f"  Ratio: {memory_ratio:.2f}x")

                results["comparison"]["memory_ratio"] = memory_ratio

    # Check if targets met
    print("\n" + "=" * 60)
    print("TARGET VERIFICATION")
    print("=" * 60)

    targets_met = True

    if "v2_cached" in results and "error" not in results["v2_cached"]:
        v2 = results["v2_cached"]

        # Target: <5ms p95 latency
        if v2.get("latency"):
            p95 = v2["latency"]["p95"]
            target_met = p95 < 5.0
            status = "PASS" if target_met else "FAIL"
            print(f"\nLatency p95 < 5ms: {status} ({p95:.2f}ms)")
            targets_met = targets_met and target_met

        # Target: 10x throughput improvement
        if "comparison" in results and "throughput_improvement" in results["comparison"]:
            improvement = results["comparison"]["throughput_improvement"]
            target_met = improvement >= 10.0
            status = "PASS" if target_met else "FAIL"
            print(f"Throughput 10x improvement: {status} ({improvement:.1f}x)")
            targets_met = targets_met and target_met

    results["targets_met"] = targets_met

    return results


# =============================================================================
# CLI Interface
# =============================================================================


def main() -> int:
    """Run the benchmark suite."""
    parser = argparse.ArgumentParser(description="Response Classifier Benchmark Suite")
    parser.add_argument(
        "--benchmark",
        choices=["latency", "throughput", "memory", "all"],
        default="all",
        help="Specific benchmark to run",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare V1 vs V2 classifiers",
    )
    parser.add_argument(
        "--messages",
        type=int,
        default=1000,
        help="Number of test messages",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for throughput tests",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Warmup timer
    warmup_timer()

    # Generate test data
    messages = generate_benchmark_data(args.messages)

    if args.compare:
        final_results = compare_classifiers(messages, args.batch_size)
    else:
        print(f"\nRunning {args.benchmark} benchmark...")
        bench_result = benchmark_v2_classifier(messages, args.batch_size)
        final_results = bench_result.to_dict()

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(final_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
