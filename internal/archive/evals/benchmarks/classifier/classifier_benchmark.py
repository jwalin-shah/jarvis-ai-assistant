"""Response Classifier Benchmarking Suite.

Measures performance of the response classifier:
- Throughput: messages per second
- Latency: p50, p95, p99 percentiles
- Memory usage
- Accuracy vs speed tradeoffs

Usage:
    # Run full benchmark suite
    python -m benchmarks.classifier.classifier_benchmark

    # Run specific benchmark
    python -m benchmarks.classifier.classifier_benchmark --benchmark throughput
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
  # noqa: E402
from evals.benchmarks.latency.timer import (  # noqa: E402
    HighPrecisionTimer,  # noqa: E402
    force_model_unload,  # noqa: E402
    warmup_timer,  # noqa: E402
)  # noqa: E402

  # noqa: E402
logger = logging.getLogger(__name__)  # noqa: E402
  # noqa: E402
  # noqa: E402
# =============================================================================  # noqa: E402
# Benchmark Data  # noqa: E402
# =============================================================================  # noqa: E402
  # noqa: E402
# Representative test messages for each category  # noqa: E402
BENCHMARK_MESSAGES: dict[str, list[str]] = {  # noqa: E402
    "agree": [  # noqa: E402
        "Yes!",  # noqa: E402
        "Yeah definitely",  # noqa: E402
        "I'm down",  # noqa: E402
        "Sounds good",  # noqa: E402
        "Sure thing",  # noqa: E402
        "Let's do it",  # noqa: E402
        "Count me in",  # noqa: E402
        "For sure",  # noqa: E402
        "Works for me",  # noqa: E402
        "That works",  # noqa: E402
    ],  # noqa: E402
    "decline": [  # noqa: E402
        "No",  # noqa: E402
        "Nope",  # noqa: E402
        "Can't make it",  # noqa: E402
        "Sorry I'm busy",  # noqa: E402
        "I'll pass",  # noqa: E402
        "Not tonight",  # noqa: E402
        "Won't be able to",  # noqa: E402
        "Rain check",  # noqa: E402
        "I can't",  # noqa: E402
        "Unfortunately no",  # noqa: E402
    ],  # noqa: E402
    "defer": [  # noqa: E402
        "Maybe",  # noqa: E402
        "Let me check",  # noqa: E402
        "I'll see",  # noqa: E402
        "Not sure yet",  # noqa: E402
        "Depends",  # noqa: E402
        "We'll see",  # noqa: E402
        "Might be able to",  # noqa: E402
        "TBD",  # noqa: E402
        "Let me think about it",  # noqa: E402
        "I'll let you know",  # noqa: E402
    ],  # noqa: E402
    "question": [  # noqa: E402
        "What time?",  # noqa: E402
        "Where is it?",  # noqa: E402
        "Who's going?",  # noqa: E402
        "How do I get there?",  # noqa: E402
        "When does it start?",  # noqa: E402
        "Can you send me the address?",  # noqa: E402
        "What should I bring?",  # noqa: E402
        "Is parking available?",  # noqa: E402
        "Do I need to RSVP?",  # noqa: E402
        "What's the dress code?",  # noqa: E402
    ],  # noqa: E402
    "acknowledge": [  # noqa: E402
        "Ok",  # noqa: E402
        "Got it",  # noqa: E402
        "Alright",  # noqa: E402
        "Cool",  # noqa: E402
        "Noted",  # noqa: E402
        "Will do",  # noqa: E402
        "No problem",  # noqa: E402
        "Makes sense",  # noqa: E402
        "I see",  # noqa: E402
        "Word",  # noqa: E402
    ],  # noqa: E402
    "react_positive": [  # noqa: E402
        "Congrats!",  # noqa: E402
        "That's awesome!",  # noqa: E402
        "So happy for you!",  # noqa: E402
        "OMG",  # noqa: E402
        "Yay!",  # noqa: E402
        "Nice!",  # noqa: E402
        "Let's gooo",  # noqa: E402
        "Well done",  # noqa: E402
        "lol",  # noqa: E402
        "haha that's hilarious",  # noqa: E402
    ],  # noqa: E402
    "react_sympathy": [  # noqa: E402
        "I'm sorry to hear that",  # noqa: E402
        "That sucks",  # noqa: E402
        "Damn",  # noqa: E402
        "Here for you",  # noqa: E402
        "Hang in there",  # noqa: E402
        "Let me know if you need anything",  # noqa: E402
        "Oh no",  # noqa: E402
        "That's rough",  # noqa: E402
        "Sending hugs",  # noqa: E402
        "Take care of yourself",  # noqa: E402
    ],  # noqa: E402
    "greeting": [  # noqa: E402
        "Hey!",  # noqa: E402
        "Hi",  # noqa: E402
        "Hello",  # noqa: E402
        "Yo",  # noqa: E402
        "Sup",  # noqa: E402
        "What's up",  # noqa: E402
        "Good morning",  # noqa: E402
        "Evening",  # noqa: E402
        "Hiya",  # noqa: E402
        "Hey there",  # noqa: E402
    ],  # noqa: E402
    "statement": [  # noqa: E402
        "I went to the store today",  # noqa: E402
        "The weather is nice outside",  # noqa: E402
        "I finished the project",  # noqa: E402
        "Traffic was bad this morning",  # noqa: E402
        "The movie was really good",  # noqa: E402
        "I'm working from home",  # noqa: E402
        "My flight got delayed",  # noqa: E402
        "The restaurant was packed",  # noqa: E402
        "I started a new book",  # noqa: E402
        "My cat is sleeping",  # noqa: E402
    ],  # noqa: E402
    "mixed": [  # noqa: E402
        "Yeah I think so but let me check first",  # noqa: E402
        "That sounds great, when should we meet?",  # noqa: E402
        "I can't tomorrow but maybe Thursday?",  # noqa: E402
        "lol no way that happened!",  # noqa: E402
        "Thanks for letting me know, I'll be there",  # noqa: E402
        "Sorry to hear that, let me know if I can help",  # noqa: E402
        "Good morning! How's it going?",  # noqa: E402
        "Nice! Where did you find it?",  # noqa: E402
        "Hmm not sure, maybe ask John?",  # noqa: E402
        "Ok cool, see you then!",  # noqa: E402
    ],  # noqa: E402
}  # noqa: E402
  # noqa: E402
  # noqa: E402
def generate_benchmark_data(n_messages: int = 1000) -> list[str]:  # noqa: E402
    """Generate a diverse set of benchmark messages.  # noqa: E402
  # noqa: E402
    Args:  # noqa: E402
        n_messages: Total number of messages to generate.  # noqa: E402
  # noqa: E402
    Returns:  # noqa: E402
        List of benchmark messages.  # noqa: E402
    """  # noqa: E402
    all_messages = []  # noqa: E402
    for category, messages in BENCHMARK_MESSAGES.items():  # noqa: E402
        all_messages.extend(messages)  # noqa: E402
  # noqa: E402
    # Repeat and shuffle to get desired count  # noqa: E402
    result: list[str] = []  # noqa: E402
    while len(result) < n_messages:  # noqa: E402
        result.extend(all_messages)  # noqa: E402
  # noqa: E402
    np.random.shuffle(result)  # noqa: E402
    return result[:n_messages]  # noqa: E402
  # noqa: E402
  # noqa: E402
# =============================================================================  # noqa: E402
# Benchmark Results  # noqa: E402
# =============================================================================  # noqa: E402
  # noqa: E402
  # noqa: E402
@dataclass  # noqa: E402
class LatencyStats:  # noqa: E402
    """Latency statistics in milliseconds."""  # noqa: E402
  # noqa: E402
    p50: float  # noqa: E402
    p95: float  # noqa: E402
    p99: float  # noqa: E402
    mean: float  # noqa: E402
    min: float  # noqa: E402
    max: float  # noqa: E402
    std: float  # noqa: E402
  # noqa: E402
  # noqa: E402
@dataclass  # noqa: E402
class ThroughputStats:  # noqa: E402
    """Throughput statistics."""  # noqa: E402
  # noqa: E402
    messages_per_second: float  # noqa: E402
    total_messages: int  # noqa: E402
    total_time_ms: float  # noqa: E402
    batch_size: int | None = None  # noqa: E402
  # noqa: E402
  # noqa: E402
@dataclass  # noqa: E402
class MemoryStats:  # noqa: E402
    """Memory usage statistics in MB."""  # noqa: E402
  # noqa: E402
    peak_mb: float  # noqa: E402
    current_mb: float  # noqa: E402
    allocated_mb: float  # noqa: E402
  # noqa: E402
  # noqa: E402
@dataclass  # noqa: E402
class AccuracyStats:  # noqa: E402
    """Classification accuracy statistics."""  # noqa: E402
  # noqa: E402
    total: int  # noqa: E402
    correct: int  # noqa: E402
    accuracy: float  # noqa: E402
    per_class_accuracy: dict[str, float] = field(default_factory=dict)  # noqa: E402
  # noqa: E402
  # noqa: E402
@dataclass  # noqa: E402
class BenchmarkResult:  # noqa: E402
    """Complete benchmark result for a classifier version."""  # noqa: E402
  # noqa: E402
    version: str  # noqa: E402
    latency: LatencyStats | None = None  # noqa: E402
    throughput: ThroughputStats | None = None  # noqa: E402
    memory: MemoryStats | None = None  # noqa: E402
    accuracy: AccuracyStats | None = None  # noqa: E402
    metadata: dict[str, Any] = field(default_factory=dict)  # noqa: E402
  # noqa: E402
    def to_dict(self) -> dict[str, Any]:  # noqa: E402
        """Convert to dictionary for JSON serialization."""  # noqa: E402
        result = {"version": self.version, "metadata": self.metadata}  # noqa: E402
        if self.latency:  # noqa: E402
            result["latency"] = asdict(self.latency)  # noqa: E402
        if self.throughput:  # noqa: E402
            result["throughput"] = asdict(self.throughput)  # noqa: E402
        if self.memory:  # noqa: E402
            result["memory"] = asdict(self.memory)  # noqa: E402
        if self.accuracy:  # noqa: E402
            result["accuracy"] = asdict(self.accuracy)  # noqa: E402
        return result  # noqa: E402
  # noqa: E402
  # noqa: E402
# =============================================================================  # noqa: E402
# Benchmark Functions  # noqa: E402
# =============================================================================  # noqa: E402
  # noqa: E402
  # noqa: E402
def measure_latency(  # noqa: E402
    classify_fn: Callable[[str], Any],  # noqa: E402
    messages: list[str],  # noqa: E402
    warmup_iterations: int = 10,  # noqa: E402
) -> LatencyStats:  # noqa: E402
    """Measure single-message classification latency.  # noqa: E402
  # noqa: E402
    Args:  # noqa: E402
        classify_fn: Function that classifies a single message.  # noqa: E402
        messages: List of messages to classify.  # noqa: E402
        warmup_iterations: Number of warmup iterations.  # noqa: E402
  # noqa: E402
    Returns:  # noqa: E402
        LatencyStats with percentile measurements.  # noqa: E402
    """  # noqa: E402
    # Warmup  # noqa: E402
    for msg in messages[:warmup_iterations]:  # noqa: E402
        classify_fn(msg)  # noqa: E402
  # noqa: E402
    # Measure  # noqa: E402
    latencies_ms: list[float] = []  # noqa: E402
    timer = HighPrecisionTimer()  # noqa: E402
  # noqa: E402
    for msg in messages:  # noqa: E402
        timer.start()  # noqa: E402
        classify_fn(msg)  # noqa: E402
        result = timer.stop()  # noqa: E402
        latencies_ms.append(result.elapsed_ms)  # noqa: E402
        timer.reset()  # noqa: E402
  # noqa: E402
    return LatencyStats(  # noqa: E402
        p50=float(np.percentile(latencies_ms, 50)),  # noqa: E402
        p95=float(np.percentile(latencies_ms, 95)),  # noqa: E402
        p99=float(np.percentile(latencies_ms, 99)),  # noqa: E402
        mean=statistics.mean(latencies_ms),  # noqa: E402
        min=min(latencies_ms),  # noqa: E402
        max=max(latencies_ms),  # noqa: E402
        std=statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,  # noqa: E402
    )  # noqa: E402
  # noqa: E402
  # noqa: E402
def measure_batch_latency(  # noqa: E402
    classify_batch_fn: Callable[[list[str]], list[Any]],  # noqa: E402
    messages: list[str],  # noqa: E402
    batch_size: int = 64,  # noqa: E402
    warmup_batches: int = 2,  # noqa: E402
) -> LatencyStats:  # noqa: E402
    """Measure batch classification latency (per-message).  # noqa: E402
  # noqa: E402
    Args:  # noqa: E402
        classify_batch_fn: Function that classifies a batch of messages.  # noqa: E402
        messages: List of messages to classify.  # noqa: E402
        batch_size: Batch size for processing.  # noqa: E402
        warmup_batches: Number of warmup batches.  # noqa: E402
  # noqa: E402
    Returns:  # noqa: E402
        LatencyStats with per-message latency percentiles.  # noqa: E402
    """  # noqa: E402
    # Split into batches  # noqa: E402
    batches = [messages[i : i + batch_size] for i in range(0, len(messages), batch_size)]  # noqa: E402
  # noqa: E402
    # Warmup  # noqa: E402
    for batch in batches[:warmup_batches]:  # noqa: E402
        classify_batch_fn(batch)  # noqa: E402
  # noqa: E402
    # Measure  # noqa: E402
    per_message_latencies: list[float] = []  # noqa: E402
    timer = HighPrecisionTimer()  # noqa: E402
  # noqa: E402
    for batch in batches:  # noqa: E402
        timer.start()  # noqa: E402
        classify_batch_fn(batch)  # noqa: E402
        result = timer.stop()  # noqa: E402
        per_msg_latency = result.elapsed_ms / len(batch)  # noqa: E402
        per_message_latencies.extend([per_msg_latency] * len(batch))  # noqa: E402
        timer.reset()  # noqa: E402
  # noqa: E402
    return LatencyStats(  # noqa: E402
        p50=float(np.percentile(per_message_latencies, 50)),  # noqa: E402
        p95=float(np.percentile(per_message_latencies, 95)),  # noqa: E402
        p99=float(np.percentile(per_message_latencies, 99)),  # noqa: E402
        mean=statistics.mean(per_message_latencies),  # noqa: E402
        min=min(per_message_latencies),  # noqa: E402
        max=max(per_message_latencies),  # noqa: E402
        std=statistics.stdev(per_message_latencies) if len(per_message_latencies) > 1 else 0.0,  # noqa: E402
    )  # noqa: E402
  # noqa: E402
  # noqa: E402
def measure_throughput(  # noqa: E402
    classify_batch_fn: Callable[[list[str]], list[Any]],  # noqa: E402
    messages: list[str],  # noqa: E402
    batch_size: int = 64,  # noqa: E402
    warmup_batches: int = 2,  # noqa: E402
) -> ThroughputStats:  # noqa: E402
    """Measure classification throughput.  # noqa: E402
  # noqa: E402
    Args:  # noqa: E402
        classify_batch_fn: Function that classifies a batch of messages.  # noqa: E402
        messages: List of messages to classify.  # noqa: E402
        batch_size: Batch size for processing.  # noqa: E402
        warmup_batches: Number of warmup batches.  # noqa: E402
  # noqa: E402
    Returns:  # noqa: E402
        ThroughputStats with messages per second.  # noqa: E402
    """  # noqa: E402
    # Split into batches  # noqa: E402
    batches = [messages[i : i + batch_size] for i in range(0, len(messages), batch_size)]  # noqa: E402
  # noqa: E402
    # Warmup  # noqa: E402
    for batch in batches[:warmup_batches]:  # noqa: E402
        classify_batch_fn(batch)  # noqa: E402
  # noqa: E402
    # Measure total time  # noqa: E402
    timer = HighPrecisionTimer()  # noqa: E402
    timer.start()  # noqa: E402
  # noqa: E402
    for batch in batches:  # noqa: E402
        classify_batch_fn(batch)  # noqa: E402
  # noqa: E402
    result = timer.stop()  # noqa: E402
  # noqa: E402
    messages_per_second = len(messages) / (result.elapsed_ms / 1000)  # noqa: E402
  # noqa: E402
    return ThroughputStats(  # noqa: E402
        messages_per_second=messages_per_second,  # noqa: E402
        total_messages=len(messages),  # noqa: E402
        total_time_ms=result.elapsed_ms,  # noqa: E402
        batch_size=batch_size,  # noqa: E402
    )  # noqa: E402
  # noqa: E402
  # noqa: E402
def measure_memory(  # noqa: E402
    classify_batch_fn: Callable[[list[str]], list[Any]],  # noqa: E402
    messages: list[str],  # noqa: E402
    batch_size: int = 64,  # noqa: E402
) -> MemoryStats:  # noqa: E402
    """Measure memory usage during classification.  # noqa: E402
  # noqa: E402
    Args:  # noqa: E402
        classify_batch_fn: Function that classifies a batch of messages.  # noqa: E402
        messages: List of messages to classify.  # noqa: E402
        batch_size: Batch size for processing.  # noqa: E402
  # noqa: E402
    Returns:  # noqa: E402
        MemoryStats with peak and current memory usage.  # noqa: E402
    """  # noqa: E402
    # Force cleanup before measurement  # noqa: E402
    gc.collect()  # noqa: E402
    force_model_unload()  # noqa: E402
  # noqa: E402
    # Start memory tracking  # noqa: E402
    tracemalloc.start()  # noqa: E402
  # noqa: E402
    # Process all messages  # noqa: E402
    batches = [messages[i : i + batch_size] for i in range(0, len(messages), batch_size)]  # noqa: E402
    for batch in batches:  # noqa: E402
        classify_batch_fn(batch)  # noqa: E402
  # noqa: E402
    # Get memory stats  # noqa: E402
    current, peak = tracemalloc.get_traced_memory()  # noqa: E402
    tracemalloc.stop()  # noqa: E402
  # noqa: E402
    return MemoryStats(  # noqa: E402
        peak_mb=peak / (1024 * 1024),  # noqa: E402
        current_mb=current / (1024 * 1024),  # noqa: E402
        allocated_mb=peak / (1024 * 1024),  # noqa: E402
    )  # noqa: E402
  # noqa: E402
  # noqa: E402
def measure_accuracy(  # noqa: E402
    classify_fn: Callable[[str], Any],  # noqa: E402
    messages: list[str],  # noqa: E402
    expected_labels: list[str],  # noqa: E402
    label_extractor: Callable[[Any], str] | None = None,  # noqa: E402
) -> AccuracyStats:  # noqa: E402
    """Measure classification accuracy against expected labels.  # noqa: E402
  # noqa: E402
    Args:  # noqa: E402
        classify_fn: Function that classifies a single message.  # noqa: E402
        messages: List of messages to classify.  # noqa: E402
        expected_labels: Expected classification labels.  # noqa: E402
        label_extractor: Function to extract label string from result.  # noqa: E402
  # noqa: E402
    Returns:  # noqa: E402
        AccuracyStats with overall and per-class accuracy.  # noqa: E402
    """  # noqa: E402
  # noqa: E402
    def default_label_extractor(r: Any) -> str:  # noqa: E402
        return r.label.value if hasattr(r.label, "value") else str(r.label)  # noqa: E402
  # noqa: E402
    if label_extractor is None:  # noqa: E402
        label_extractor = default_label_extractor  # noqa: E402
  # noqa: E402
    correct = 0  # noqa: E402
    per_class_correct: dict[str, int] = {}  # noqa: E402
    per_class_total: dict[str, int] = {}  # noqa: E402
  # noqa: E402
    for msg, expected in zip(messages, expected_labels):  # noqa: E402
        result = classify_fn(msg)  # noqa: E402
        predicted = label_extractor(result)  # noqa: E402
  # noqa: E402
        # Track per-class stats  # noqa: E402
        per_class_total[expected] = per_class_total.get(expected, 0) + 1  # noqa: E402
  # noqa: E402
        if predicted.upper() == expected.upper():  # noqa: E402
            correct += 1  # noqa: E402
            per_class_correct[expected] = per_class_correct.get(expected, 0) + 1  # noqa: E402
  # noqa: E402
    # Calculate per-class accuracy  # noqa: E402
    per_class_accuracy = {}  # noqa: E402
    for label in per_class_total:  # noqa: E402
        per_class_accuracy[label] = per_class_correct.get(label, 0) / per_class_total[label]  # noqa: E402
  # noqa: E402
    return AccuracyStats(  # noqa: E402
        total=len(messages),  # noqa: E402
        correct=correct,  # noqa: E402
        accuracy=correct / len(messages) if messages else 0.0,  # noqa: E402
        per_class_accuracy=per_class_accuracy,  # noqa: E402
    )  # noqa: E402
  # noqa: E402
  # noqa: E402
# =============================================================================  # noqa: E402
# Classifier Benchmarks  # noqa: E402
# =============================================================================  # noqa: E402
  # noqa: E402
  # noqa: E402
def benchmark_v1_classifier(messages: list[str], batch_size: int = 64) -> BenchmarkResult:  # noqa: E402
    """Benchmark V1 response classifier.  # noqa: E402
  # noqa: E402
    Args:  # noqa: E402
        messages: Test messages.  # noqa: E402
        batch_size: Batch size for throughput measurement.  # noqa: E402
  # noqa: E402
    Returns:  # noqa: E402
        BenchmarkResult with all measurements.  # noqa: E402
    """  # noqa: E402
    from jarvis.classifiers.response_classifier import (  # noqa: E402
        get_response_classifier,  # noqa: E402
        reset_response_classifier,  # noqa: E402
    )  # noqa: E402
  # noqa: E402
    # Reset to ensure clean state  # noqa: E402
    reset_response_classifier()  # noqa: E402
    classifier = get_response_classifier()  # noqa: E402
  # noqa: E402
    # Warmup  # noqa: E402
    classifier.classify("Hello")  # noqa: E402
  # noqa: E402
    result = BenchmarkResult(version="v1")  # noqa: E402
  # noqa: E402
    # Single-message latency  # noqa: E402
    print("  Measuring V1 single-message latency...")  # noqa: E402
    result.latency = measure_latency(classifier.classify, messages[:500])  # noqa: E402
  # noqa: E402
    # Batch throughput  # noqa: E402
    print("  Measuring V1 throughput...")  # noqa: E402
    result.throughput = measure_throughput(classifier.classify_batch, messages, batch_size)  # noqa: E402
  # noqa: E402
    # Memory usage  # noqa: E402
    print("  Measuring V1 memory usage...")  # noqa: E402
    result.memory = measure_memory(classifier.classify_batch, messages, batch_size)  # noqa: E402
  # noqa: E402
    result.metadata = {  # noqa: E402
        "has_svm": getattr(classifier, "_svm", None) is not None,  # noqa: E402
        "has_centroids": bool(getattr(classifier, "centroids", None)),  # noqa: E402
    }  # noqa: E402
  # noqa: E402
    return result  # noqa: E402
  # noqa: E402
  # noqa: E402
def compare_classifiers(messages: list[str], batch_size: int = 64) -> dict[str, Any]:  # noqa: E402
    """Benchmark the response classifier.  # noqa: E402
  # noqa: E402
    Args:  # noqa: E402
        messages: Test messages.  # noqa: E402
        batch_size: Batch size for throughput measurement.  # noqa: E402
  # noqa: E402
    Returns:  # noqa: E402
        Dictionary with benchmark results.  # noqa: E402
    """  # noqa: E402
    print("\n" + "=" * 60)  # noqa: E402
    print("RESPONSE CLASSIFIER BENCHMARK")  # noqa: E402
    print("=" * 60)  # noqa: E402
    print(f"\nTest set: {len(messages)} messages")  # noqa: E402
    print(f"Batch size: {batch_size}")  # noqa: E402
  # noqa: E402
    results: dict[str, Any] = {}  # noqa: E402
  # noqa: E402
    # Benchmark V1  # noqa: E402
    print("\n--- V1 Classifier ---")  # noqa: E402
    try:  # noqa: E402
        v1_result = benchmark_v1_classifier(messages, batch_size)  # noqa: E402
        results["v1"] = v1_result.to_dict()  # noqa: E402
    except Exception as e:  # noqa: E402
        print(f"  V1 benchmark failed: {e}")  # noqa: E402
        results["v1"] = {"error": str(e)}  # noqa: E402
  # noqa: E402
    # Calculate improvements  # noqa: E402
    print("\n" + "=" * 60)  # noqa: E402
    print("COMPARISON RESULTS")  # noqa: E402
    print("=" * 60)  # noqa: E402
  # noqa: E402
    # Print results  # noqa: E402
    print("\n" + "=" * 60)  # noqa: E402
    print("RESULTS")  # noqa: E402
    print("=" * 60)  # noqa: E402
  # noqa: E402
    if "v1" in results and "error" not in results["v1"]:  # noqa: E402
        v1 = results["v1"]  # noqa: E402
        if v1.get("latency"):  # noqa: E402
            print(f"\nLatency p95: {v1['latency']['p95']:.2f}ms")  # noqa: E402
        if v1.get("throughput"):  # noqa: E402
            print(f"Throughput: {v1['throughput']['messages_per_second']:.0f} msgs/sec")  # noqa: E402
        if v1.get("memory"):  # noqa: E402
            print(f"Peak Memory: {v1['memory']['peak_mb']:.1f}MB")  # noqa: E402
  # noqa: E402
    return results  # noqa: E402
  # noqa: E402
  # noqa: E402
# =============================================================================  # noqa: E402
# CLI Interface  # noqa: E402
# =============================================================================  # noqa: E402
  # noqa: E402
  # noqa: E402
def main() -> int:  # noqa: E402
    """Run the benchmark suite."""  # noqa: E402
    parser = argparse.ArgumentParser(description="Response Classifier Benchmark Suite")  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--benchmark",  # noqa: E402
        choices=["latency", "throughput", "memory", "all"],  # noqa: E402
        default="all",  # noqa: E402
        help="Specific benchmark to run",  # noqa: E402
    )  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--compare",  # noqa: E402
        action="store_true",  # noqa: E402
        help="Run comparison benchmark",  # noqa: E402
    )  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--messages",  # noqa: E402
        type=int,  # noqa: E402
        default=1000,  # noqa: E402
        help="Number of test messages",  # noqa: E402
    )  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--batch-size",  # noqa: E402
        type=int,  # noqa: E402
        default=64,  # noqa: E402
        help="Batch size for throughput tests",  # noqa: E402
    )  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--output",  # noqa: E402
        type=str,  # noqa: E402
        help="Output JSON file for results",  # noqa: E402
    )  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--quiet",  # noqa: E402
        action="store_true",  # noqa: E402
        help="Suppress detailed output",  # noqa: E402
    )  # noqa: E402
  # noqa: E402
    args = parser.parse_args()  # noqa: E402
  # noqa: E402
    # Configure logging  # noqa: E402
    logging.basicConfig(  # noqa: E402
        level=logging.WARNING if args.quiet else logging.INFO,  # noqa: E402
        format="%(levelname)s: %(message)s",  # noqa: E402
    )  # noqa: E402
  # noqa: E402
    # Warmup timer  # noqa: E402
    warmup_timer()  # noqa: E402
  # noqa: E402
    # Generate test data  # noqa: E402
    messages = generate_benchmark_data(args.messages)  # noqa: E402
  # noqa: E402
    if args.compare:  # noqa: E402
        final_results = compare_classifiers(messages, args.batch_size)  # noqa: E402
    else:  # noqa: E402
        print(f"\nRunning {args.benchmark} benchmark...")  # noqa: E402
        bench_result = benchmark_v1_classifier(messages, args.batch_size)  # noqa: E402
        final_results = bench_result.to_dict()  # noqa: E402
  # noqa: E402
    # Save results if requested  # noqa: E402
    if args.output:  # noqa: E402
        output_path = Path(args.output)  # noqa: E402
        output_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E402
        with open(output_path, "w") as f:  # noqa: E402
            json.dump(final_results, f, indent=2)  # noqa: E402
        print(f"\nResults saved to: {output_path}")  # noqa: E402
  # noqa: E402
    return 0  # noqa: E402
  # noqa: E402
  # noqa: E402
if __name__ == "__main__":  # noqa: E402
    sys.exit(main())  # noqa: E402
