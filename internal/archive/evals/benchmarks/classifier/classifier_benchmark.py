"""Response Classifier Benchmarking Suite.  # noqa: E501
  # noqa: E501
Measures performance of the response classifier:  # noqa: E501
- Throughput: messages per second  # noqa: E501
- Latency: p50, p95, p99 percentiles  # noqa: E501
- Memory usage  # noqa: E501
- Accuracy vs speed tradeoffs  # noqa: E501
  # noqa: E501
Usage:  # noqa: E501
    # Run full benchmark suite  # noqa: E501
    python -m benchmarks.classifier.classifier_benchmark  # noqa: E501
  # noqa: E501
    # Run specific benchmark  # noqa: E501
    python -m benchmarks.classifier.classifier_benchmark --benchmark throughput  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from __future__ import annotations  # noqa: E402  # noqa: E501

# noqa: E501
import argparse  # noqa: E501
import gc  # noqa: E501
import json  # noqa: E501
import logging  # noqa: E501
import statistics  # noqa: E501
import sys  # noqa: E501
import tracemalloc  # noqa: E501
from collections.abc import Callable  # noqa: E402  # noqa: E501
from dataclasses import asdict, dataclass, field  # noqa: E402  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501
from typing import Any  # noqa: E402  # noqa: E501

# noqa: E501
import numpy as np  # noqa: E501

  # noqa: E501
# Add project root to path  # noqa: E501
sys.path.insert(0, str(Path(__file__).parents[2]))  # noqa: E501
  # noqa: E501
from evals.benchmarks.latency.timer import (  # noqa: E402  # noqa: E501
    HighPrecisionTimer,
    force_model_unload,
    warmup_timer,
)

  # noqa: E501
logger = logging.getLogger(__name__)  # noqa: E501
  # noqa: E501
  # noqa: E501
# =============================================================================  # noqa: E501
# Benchmark Data  # noqa: E501
# =============================================================================  # noqa: E501
  # noqa: E501
# Representative test messages for each category  # noqa: E501
BENCHMARK_MESSAGES: dict[str, list[str]] = {  # noqa: E501
    "agree": [  # noqa: E501
        "Yes!",  # noqa: E501
        "Yeah definitely",  # noqa: E501
        "I'm down",  # noqa: E501
        "Sounds good",  # noqa: E501
        "Sure thing",  # noqa: E501
        "Let's do it",  # noqa: E501
        "Count me in",  # noqa: E501
        "For sure",  # noqa: E501
        "Works for me",  # noqa: E501
        "That works",  # noqa: E501
    ],  # noqa: E501
    "decline": [  # noqa: E501
        "No",  # noqa: E501
        "Nope",  # noqa: E501
        "Can't make it",  # noqa: E501
        "Sorry I'm busy",  # noqa: E501
        "I'll pass",  # noqa: E501
        "Not tonight",  # noqa: E501
        "Won't be able to",  # noqa: E501
        "Rain check",  # noqa: E501
        "I can't",  # noqa: E501
        "Unfortunately no",  # noqa: E501
    ],  # noqa: E501
    "defer": [  # noqa: E501
        "Maybe",  # noqa: E501
        "Let me check",  # noqa: E501
        "I'll see",  # noqa: E501
        "Not sure yet",  # noqa: E501
        "Depends",  # noqa: E501
        "We'll see",  # noqa: E501
        "Might be able to",  # noqa: E501
        "TBD",  # noqa: E501
        "Let me think about it",  # noqa: E501
        "I'll let you know",  # noqa: E501
    ],  # noqa: E501
    "question": [  # noqa: E501
        "What time?",  # noqa: E501
        "Where is it?",  # noqa: E501
        "Who's going?",  # noqa: E501
        "How do I get there?",  # noqa: E501
        "When does it start?",  # noqa: E501
        "Can you send me the address?",  # noqa: E501
        "What should I bring?",  # noqa: E501
        "Is parking available?",  # noqa: E501
        "Do I need to RSVP?",  # noqa: E501
        "What's the dress code?",  # noqa: E501
    ],  # noqa: E501
    "acknowledge": [  # noqa: E501
        "Ok",  # noqa: E501
        "Got it",  # noqa: E501
        "Alright",  # noqa: E501
        "Cool",  # noqa: E501
        "Noted",  # noqa: E501
        "Will do",  # noqa: E501
        "No problem",  # noqa: E501
        "Makes sense",  # noqa: E501
        "I see",  # noqa: E501
        "Word",  # noqa: E501
    ],  # noqa: E501
    "react_positive": [  # noqa: E501
        "Congrats!",  # noqa: E501
        "That's awesome!",  # noqa: E501
        "So happy for you!",  # noqa: E501
        "OMG",  # noqa: E501
        "Yay!",  # noqa: E501
        "Nice!",  # noqa: E501
        "Let's gooo",  # noqa: E501
        "Well done",  # noqa: E501
        "lol",  # noqa: E501
        "haha that's hilarious",  # noqa: E501
    ],  # noqa: E501
    "react_sympathy": [  # noqa: E501
        "I'm sorry to hear that",  # noqa: E501
        "That sucks",  # noqa: E501
        "Damn",  # noqa: E501
        "Here for you",  # noqa: E501
        "Hang in there",  # noqa: E501
        "Let me know if you need anything",  # noqa: E501
        "Oh no",  # noqa: E501
        "That's rough",  # noqa: E501
        "Sending hugs",  # noqa: E501
        "Take care of yourself",  # noqa: E501
    ],  # noqa: E501
    "greeting": [  # noqa: E501
        "Hey!",  # noqa: E501
        "Hi",  # noqa: E501
        "Hello",  # noqa: E501
        "Yo",  # noqa: E501
        "Sup",  # noqa: E501
        "What's up",  # noqa: E501
        "Good morning",  # noqa: E501
        "Evening",  # noqa: E501
        "Hiya",  # noqa: E501
        "Hey there",  # noqa: E501
    ],  # noqa: E501
    "statement": [  # noqa: E501
        "I went to the store today",  # noqa: E501
        "The weather is nice outside",  # noqa: E501
        "I finished the project",  # noqa: E501
        "Traffic was bad this morning",  # noqa: E501
        "The movie was really good",  # noqa: E501
        "I'm working from home",  # noqa: E501
        "My flight got delayed",  # noqa: E501
        "The restaurant was packed",  # noqa: E501
        "I started a new book",  # noqa: E501
        "My cat is sleeping",  # noqa: E501
    ],  # noqa: E501
    "mixed": [  # noqa: E501
        "Yeah I think so but let me check first",  # noqa: E501
        "That sounds great, when should we meet?",  # noqa: E501
        "I can't tomorrow but maybe Thursday?",  # noqa: E501
        "lol no way that happened!",  # noqa: E501
        "Thanks for letting me know, I'll be there",  # noqa: E501
        "Sorry to hear that, let me know if I can help",  # noqa: E501
        "Good morning! How's it going?",  # noqa: E501
        "Nice! Where did you find it?",  # noqa: E501
        "Hmm not sure, maybe ask John?",  # noqa: E501
        "Ok cool, see you then!",  # noqa: E501
    ],  # noqa: E501
}  # noqa: E501
  # noqa: E501
  # noqa: E501
def generate_benchmark_data(n_messages: int = 1000) -> list[str]:  # noqa: E501
    """Generate a diverse set of benchmark messages.  # noqa: E501
  # noqa: E501
    Args:  # noqa: E501
        n_messages: Total number of messages to generate.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        List of benchmark messages.  # noqa: E501
    """  # noqa: E501
    all_messages = []  # noqa: E501
    for category, messages in BENCHMARK_MESSAGES.items():  # noqa: E501
        all_messages.extend(messages)  # noqa: E501
  # noqa: E501
    # Repeat and shuffle to get desired count  # noqa: E501
    result: list[str] = []  # noqa: E501
    while len(result) < n_messages:  # noqa: E501
        result.extend(all_messages)  # noqa: E501
  # noqa: E501
    np.random.shuffle(result)  # noqa: E501
    return result[:n_messages]  # noqa: E501
  # noqa: E501
  # noqa: E501
# =============================================================================  # noqa: E501
# Benchmark Results  # noqa: E501
# =============================================================================  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class LatencyStats:  # noqa: E501
    """Latency statistics in milliseconds."""  # noqa: E501
  # noqa: E501
    p50: float  # noqa: E501
    p95: float  # noqa: E501
    p99: float  # noqa: E501
    mean: float  # noqa: E501
    min: float  # noqa: E501
    max: float  # noqa: E501
    std: float  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class ThroughputStats:  # noqa: E501
    """Throughput statistics."""  # noqa: E501
  # noqa: E501
    messages_per_second: float  # noqa: E501
    total_messages: int  # noqa: E501
    total_time_ms: float  # noqa: E501
    batch_size: int | None = None  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class MemoryStats:  # noqa: E501
    """Memory usage statistics in MB."""  # noqa: E501
  # noqa: E501
    peak_mb: float  # noqa: E501
    current_mb: float  # noqa: E501
    allocated_mb: float  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class AccuracyStats:  # noqa: E501
    """Classification accuracy statistics."""  # noqa: E501
  # noqa: E501
    total: int  # noqa: E501
    correct: int  # noqa: E501
    accuracy: float  # noqa: E501
    per_class_accuracy: dict[str, float] = field(default_factory=dict)  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class BenchmarkResult:  # noqa: E501
    """Complete benchmark result for a classifier version."""  # noqa: E501
  # noqa: E501
    version: str  # noqa: E501
    latency: LatencyStats | None = None  # noqa: E501
    throughput: ThroughputStats | None = None  # noqa: E501
    memory: MemoryStats | None = None  # noqa: E501
    accuracy: AccuracyStats | None = None  # noqa: E501
    metadata: dict[str, Any] = field(default_factory=dict)  # noqa: E501
  # noqa: E501
    def to_dict(self) -> dict[str, Any]:  # noqa: E501
        """Convert to dictionary for JSON serialization."""  # noqa: E501
        result = {"version": self.version, "metadata": self.metadata}  # noqa: E501
        if self.latency:  # noqa: E501
            result["latency"] = asdict(self.latency)  # noqa: E501
        if self.throughput:  # noqa: E501
            result["throughput"] = asdict(self.throughput)  # noqa: E501
        if self.memory:  # noqa: E501
            result["memory"] = asdict(self.memory)  # noqa: E501
        if self.accuracy:  # noqa: E501
            result["accuracy"] = asdict(self.accuracy)  # noqa: E501
        return result  # noqa: E501
  # noqa: E501
  # noqa: E501
# =============================================================================  # noqa: E501
# Benchmark Functions  # noqa: E501
# =============================================================================  # noqa: E501
  # noqa: E501
  # noqa: E501
def measure_latency(  # noqa: E501
    classify_fn: Callable[[str], Any],  # noqa: E501
    messages: list[str],  # noqa: E501
    warmup_iterations: int = 10,  # noqa: E501
) -> LatencyStats:  # noqa: E501
    """Measure single-message classification latency.  # noqa: E501
  # noqa: E501
    Args:  # noqa: E501
        classify_fn: Function that classifies a single message.  # noqa: E501
        messages: List of messages to classify.  # noqa: E501
        warmup_iterations: Number of warmup iterations.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        LatencyStats with percentile measurements.  # noqa: E501
    """  # noqa: E501
    # Warmup  # noqa: E501
    for msg in messages[:warmup_iterations]:  # noqa: E501
        classify_fn(msg)  # noqa: E501
  # noqa: E501
    # Measure  # noqa: E501
    latencies_ms: list[float] = []  # noqa: E501
    timer = HighPrecisionTimer()  # noqa: E501
  # noqa: E501
    for msg in messages:  # noqa: E501
        timer.start()  # noqa: E501
        classify_fn(msg)  # noqa: E501
        result = timer.stop()  # noqa: E501
        latencies_ms.append(result.elapsed_ms)  # noqa: E501
        timer.reset()  # noqa: E501
  # noqa: E501
    return LatencyStats(  # noqa: E501
        p50=float(np.percentile(latencies_ms, 50)),  # noqa: E501
        p95=float(np.percentile(latencies_ms, 95)),  # noqa: E501
        p99=float(np.percentile(latencies_ms, 99)),  # noqa: E501
        mean=statistics.mean(latencies_ms),  # noqa: E501
        min=min(latencies_ms),  # noqa: E501
        max=max(latencies_ms),  # noqa: E501
        std=statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,  # noqa: E501
    )  # noqa: E501
  # noqa: E501
  # noqa: E501
def measure_batch_latency(  # noqa: E501
    classify_batch_fn: Callable[[list[str]], list[Any]],  # noqa: E501
    messages: list[str],  # noqa: E501
    batch_size: int = 64,  # noqa: E501
    warmup_batches: int = 2,  # noqa: E501
) -> LatencyStats:  # noqa: E501
    """Measure batch classification latency (per-message).  # noqa: E501
  # noqa: E501
    Args:  # noqa: E501
        classify_batch_fn: Function that classifies a batch of messages.  # noqa: E501
        messages: List of messages to classify.  # noqa: E501
        batch_size: Batch size for processing.  # noqa: E501
        warmup_batches: Number of warmup batches.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        LatencyStats with per-message latency percentiles.  # noqa: E501
    """  # noqa: E501
    # Split into batches  # noqa: E501
    batches = [messages[i : i + batch_size] for i in range(0, len(messages), batch_size)]  # noqa: E501
  # noqa: E501
    # Warmup  # noqa: E501
    for batch in batches[:warmup_batches]:  # noqa: E501
        classify_batch_fn(batch)  # noqa: E501
  # noqa: E501
    # Measure  # noqa: E501
    per_message_latencies: list[float] = []  # noqa: E501
    timer = HighPrecisionTimer()  # noqa: E501
  # noqa: E501
    for batch in batches:  # noqa: E501
        timer.start()  # noqa: E501
        classify_batch_fn(batch)  # noqa: E501
        result = timer.stop()  # noqa: E501
        per_msg_latency = result.elapsed_ms / len(batch)  # noqa: E501
        per_message_latencies.extend([per_msg_latency] * len(batch))  # noqa: E501
        timer.reset()  # noqa: E501
  # noqa: E501
    return LatencyStats(  # noqa: E501
        p50=float(np.percentile(per_message_latencies, 50)),  # noqa: E501
        p95=float(np.percentile(per_message_latencies, 95)),  # noqa: E501
        p99=float(np.percentile(per_message_latencies, 99)),  # noqa: E501
        mean=statistics.mean(per_message_latencies),  # noqa: E501
        min=min(per_message_latencies),  # noqa: E501
        max=max(per_message_latencies),  # noqa: E501
        std=statistics.stdev(per_message_latencies) if len(per_message_latencies) > 1 else 0.0,  # noqa: E501
    )  # noqa: E501
  # noqa: E501
  # noqa: E501
def measure_throughput(  # noqa: E501
    classify_batch_fn: Callable[[list[str]], list[Any]],  # noqa: E501
    messages: list[str],  # noqa: E501
    batch_size: int = 64,  # noqa: E501
    warmup_batches: int = 2,  # noqa: E501
) -> ThroughputStats:  # noqa: E501
    """Measure classification throughput.  # noqa: E501
  # noqa: E501
    Args:  # noqa: E501
        classify_batch_fn: Function that classifies a batch of messages.  # noqa: E501
        messages: List of messages to classify.  # noqa: E501
        batch_size: Batch size for processing.  # noqa: E501
        warmup_batches: Number of warmup batches.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        ThroughputStats with messages per second.  # noqa: E501
    """  # noqa: E501
    # Split into batches  # noqa: E501
    batches = [messages[i : i + batch_size] for i in range(0, len(messages), batch_size)]  # noqa: E501
  # noqa: E501
    # Warmup  # noqa: E501
    for batch in batches[:warmup_batches]:  # noqa: E501
        classify_batch_fn(batch)  # noqa: E501
  # noqa: E501
    # Measure total time  # noqa: E501
    timer = HighPrecisionTimer()  # noqa: E501
    timer.start()  # noqa: E501
  # noqa: E501
    for batch in batches:  # noqa: E501
        classify_batch_fn(batch)  # noqa: E501
  # noqa: E501
    result = timer.stop()  # noqa: E501
  # noqa: E501
    messages_per_second = len(messages) / (result.elapsed_ms / 1000)  # noqa: E501
  # noqa: E501
    return ThroughputStats(  # noqa: E501
        messages_per_second=messages_per_second,  # noqa: E501
        total_messages=len(messages),  # noqa: E501
        total_time_ms=result.elapsed_ms,  # noqa: E501
        batch_size=batch_size,  # noqa: E501
    )  # noqa: E501
  # noqa: E501
  # noqa: E501
def measure_memory(  # noqa: E501
    classify_batch_fn: Callable[[list[str]], list[Any]],  # noqa: E501
    messages: list[str],  # noqa: E501
    batch_size: int = 64,  # noqa: E501
) -> MemoryStats:  # noqa: E501
    """Measure memory usage during classification.  # noqa: E501
  # noqa: E501
    Args:  # noqa: E501
        classify_batch_fn: Function that classifies a batch of messages.  # noqa: E501
        messages: List of messages to classify.  # noqa: E501
        batch_size: Batch size for processing.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        MemoryStats with peak and current memory usage.  # noqa: E501
    """  # noqa: E501
    # Force cleanup before measurement  # noqa: E501
    gc.collect()  # noqa: E501
    force_model_unload()  # noqa: E501
  # noqa: E501
    # Start memory tracking  # noqa: E501
    tracemalloc.start()  # noqa: E501
  # noqa: E501
    # Process all messages  # noqa: E501
    batches = [messages[i : i + batch_size] for i in range(0, len(messages), batch_size)]  # noqa: E501
    for batch in batches:  # noqa: E501
        classify_batch_fn(batch)  # noqa: E501
  # noqa: E501
    # Get memory stats  # noqa: E501
    current, peak = tracemalloc.get_traced_memory()  # noqa: E501
    tracemalloc.stop()  # noqa: E501
  # noqa: E501
    return MemoryStats(  # noqa: E501
        peak_mb=peak / (1024 * 1024),  # noqa: E501
        current_mb=current / (1024 * 1024),  # noqa: E501
        allocated_mb=peak / (1024 * 1024),  # noqa: E501
    )  # noqa: E501
  # noqa: E501
  # noqa: E501
def measure_accuracy(  # noqa: E501
    classify_fn: Callable[[str], Any],  # noqa: E501
    messages: list[str],  # noqa: E501
    expected_labels: list[str],  # noqa: E501
    label_extractor: Callable[[Any], str] | None = None,  # noqa: E501
) -> AccuracyStats:  # noqa: E501
    """Measure classification accuracy against expected labels.  # noqa: E501
  # noqa: E501
    Args:  # noqa: E501
        classify_fn: Function that classifies a single message.  # noqa: E501
        messages: List of messages to classify.  # noqa: E501
        expected_labels: Expected classification labels.  # noqa: E501
        label_extractor: Function to extract label string from result.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        AccuracyStats with overall and per-class accuracy.  # noqa: E501
    """  # noqa: E501
  # noqa: E501
    def default_label_extractor(r: Any) -> str:  # noqa: E501
        return r.label.value if hasattr(r.label, "value") else str(r.label)  # noqa: E501
  # noqa: E501
    if label_extractor is None:  # noqa: E501
        label_extractor = default_label_extractor  # noqa: E501
  # noqa: E501
    correct = 0  # noqa: E501
    per_class_correct: dict[str, int] = {}  # noqa: E501
    per_class_total: dict[str, int] = {}  # noqa: E501
  # noqa: E501
    for msg, expected in zip(messages, expected_labels):  # noqa: E501
        result = classify_fn(msg)  # noqa: E501
        predicted = label_extractor(result)  # noqa: E501
  # noqa: E501
        # Track per-class stats  # noqa: E501
        per_class_total[expected] = per_class_total.get(expected, 0) + 1  # noqa: E501
  # noqa: E501
        if predicted.upper() == expected.upper():  # noqa: E501
            correct += 1  # noqa: E501
            per_class_correct[expected] = per_class_correct.get(expected, 0) + 1  # noqa: E501
  # noqa: E501
    # Calculate per-class accuracy  # noqa: E501
    per_class_accuracy = {}  # noqa: E501
    for label in per_class_total:  # noqa: E501
        per_class_accuracy[label] = per_class_correct.get(label, 0) / per_class_total[label]  # noqa: E501
  # noqa: E501
    return AccuracyStats(  # noqa: E501
        total=len(messages),  # noqa: E501
        correct=correct,  # noqa: E501
        accuracy=correct / len(messages) if messages else 0.0,  # noqa: E501
        per_class_accuracy=per_class_accuracy,  # noqa: E501
    )  # noqa: E501
  # noqa: E501
  # noqa: E501
# =============================================================================  # noqa: E501
# Classifier Benchmarks  # noqa: E501
# =============================================================================  # noqa: E501
  # noqa: E501
  # noqa: E501
def benchmark_v1_classifier(messages: list[str], batch_size: int = 64) -> BenchmarkResult:  # noqa: E501
    """Benchmark V1 response classifier.  # noqa: E501
  # noqa: E501
    Args:  # noqa: E501
        messages: Test messages.  # noqa: E501
        batch_size: Batch size for throughput measurement.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        BenchmarkResult with all measurements.  # noqa: E501
    """  # noqa: E501
    from jarvis.classifiers.response_classifier import (  # noqa: E501
        get_response_classifier,  # noqa: E501
        reset_response_classifier,  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    # Reset to ensure clean state  # noqa: E501
    reset_response_classifier()  # noqa: E501
    classifier = get_response_classifier()  # noqa: E501
  # noqa: E501
    # Warmup  # noqa: E501
    classifier.classify("Hello")  # noqa: E501
  # noqa: E501
    result = BenchmarkResult(version="v1")  # noqa: E501
  # noqa: E501
    # Single-message latency  # noqa: E501
    print("  Measuring V1 single-message latency...")  # noqa: E501
    result.latency = measure_latency(classifier.classify, messages[:500])  # noqa: E501
  # noqa: E501
    # Batch throughput  # noqa: E501
    print("  Measuring V1 throughput...")  # noqa: E501
    result.throughput = measure_throughput(classifier.classify_batch, messages, batch_size)  # noqa: E501
  # noqa: E501
    # Memory usage  # noqa: E501
    print("  Measuring V1 memory usage...")  # noqa: E501
    result.memory = measure_memory(classifier.classify_batch, messages, batch_size)  # noqa: E501
  # noqa: E501
    result.metadata = {  # noqa: E501
        "has_svm": getattr(classifier, "_svm", None) is not None,  # noqa: E501
        "has_centroids": bool(getattr(classifier, "centroids", None)),  # noqa: E501
    }  # noqa: E501
  # noqa: E501
    return result  # noqa: E501
  # noqa: E501
  # noqa: E501
def compare_classifiers(messages: list[str], batch_size: int = 64) -> dict[str, Any]:  # noqa: E501
    """Benchmark the response classifier.  # noqa: E501
  # noqa: E501
    Args:  # noqa: E501
        messages: Test messages.  # noqa: E501
        batch_size: Batch size for throughput measurement.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        Dictionary with benchmark results.  # noqa: E501
    """  # noqa: E501
    print("\n" + "=" * 60)  # noqa: E501
    print("RESPONSE CLASSIFIER BENCHMARK")  # noqa: E501
    print("=" * 60)  # noqa: E501
    print(f"\nTest set: {len(messages)} messages")  # noqa: E501
    print(f"Batch size: {batch_size}")  # noqa: E501
  # noqa: E501
    results: dict[str, Any] = {}  # noqa: E501
  # noqa: E501
    # Benchmark V1  # noqa: E501
    print("\n--- V1 Classifier ---")  # noqa: E501
    try:  # noqa: E501
        v1_result = benchmark_v1_classifier(messages, batch_size)  # noqa: E501
        results["v1"] = v1_result.to_dict()  # noqa: E501
    except Exception as e:  # noqa: E501
        print(f"  V1 benchmark failed: {e}")  # noqa: E501
        results["v1"] = {"error": str(e)}  # noqa: E501
  # noqa: E501
    # Calculate improvements  # noqa: E501
    print("\n" + "=" * 60)  # noqa: E501
    print("COMPARISON RESULTS")  # noqa: E501
    print("=" * 60)  # noqa: E501
  # noqa: E501
    # Print results  # noqa: E501
    print("\n" + "=" * 60)  # noqa: E501
    print("RESULTS")  # noqa: E501
    print("=" * 60)  # noqa: E501
  # noqa: E501
    if "v1" in results and "error" not in results["v1"]:  # noqa: E501
        v1 = results["v1"]  # noqa: E501
        if v1.get("latency"):  # noqa: E501
            print(f"\nLatency p95: {v1['latency']['p95']:.2f}ms")  # noqa: E501
        if v1.get("throughput"):  # noqa: E501
            print(f"Throughput: {v1['throughput']['messages_per_second']:.0f} msgs/sec")  # noqa: E501
        if v1.get("memory"):  # noqa: E501
            print(f"Peak Memory: {v1['memory']['peak_mb']:.1f}MB")  # noqa: E501
  # noqa: E501
    return results  # noqa: E501
  # noqa: E501
  # noqa: E501
# =============================================================================  # noqa: E501
# CLI Interface  # noqa: E501
# =============================================================================  # noqa: E501
  # noqa: E501
  # noqa: E501
def main() -> int:  # noqa: E501
    """Run the benchmark suite."""  # noqa: E501
    parser = argparse.ArgumentParser(description="Response Classifier Benchmark Suite")  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--benchmark",  # noqa: E501
        choices=["latency", "throughput", "memory", "all"],  # noqa: E501
        default="all",  # noqa: E501
        help="Specific benchmark to run",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--compare",  # noqa: E501
        action="store_true",  # noqa: E501
        help="Run comparison benchmark",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--messages",  # noqa: E501
        type=int,  # noqa: E501
        default=1000,  # noqa: E501
        help="Number of test messages",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--batch-size",  # noqa: E501
        type=int,  # noqa: E501
        default=64,  # noqa: E501
        help="Batch size for throughput tests",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--output",  # noqa: E501
        type=str,  # noqa: E501
        help="Output JSON file for results",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--quiet",  # noqa: E501
        action="store_true",  # noqa: E501
        help="Suppress detailed output",  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    args = parser.parse_args()  # noqa: E501
  # noqa: E501
    # Configure logging  # noqa: E501
    logging.basicConfig(  # noqa: E501
        level=logging.WARNING if args.quiet else logging.INFO,  # noqa: E501
        format="%(levelname)s: %(message)s",  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    # Warmup timer  # noqa: E501
    warmup_timer()  # noqa: E501
  # noqa: E501
    # Generate test data  # noqa: E501
    messages = generate_benchmark_data(args.messages)  # noqa: E501
  # noqa: E501
    if args.compare:  # noqa: E501
        final_results = compare_classifiers(messages, args.batch_size)  # noqa: E501
    else:  # noqa: E501
        print(f"\nRunning {args.benchmark} benchmark...")  # noqa: E501
        bench_result = benchmark_v1_classifier(messages, args.batch_size)  # noqa: E501
        final_results = bench_result.to_dict()  # noqa: E501
  # noqa: E501
    # Save results if requested  # noqa: E501
    if args.output:  # noqa: E501
        output_path = Path(args.output)  # noqa: E501
        output_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E501
        with open(output_path, "w") as f:  # noqa: E501
            json.dump(final_results, f, indent=2)  # noqa: E501
        print(f"\nResults saved to: {output_path}")  # noqa: E501
  # noqa: E501
    return 0  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    sys.exit(main())  # noqa: E501
