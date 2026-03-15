"""Quality benchmarking suite for JARVIS.  # noqa: E501
  # noqa: E501
Provides:  # noqa: E501
- Standard evaluation datasets  # noqa: E501
- A/B testing framework  # noqa: E501
- Automated regression testing  # noqa: E501
- Human evaluation integration  # noqa: E501
  # noqa: E501
Usage:  # noqa: E501
    uv run python -m evals.benchmarks.quality --help  # noqa: E501
    uv run python -m evals.benchmarks.quality run --dataset standard  # noqa: E501
    uv run python -m evals.benchmarks.quality compare --model-a default --model-b new  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from __future__ import annotations  # noqa: E402  # noqa: E501

# noqa: E501
import json  # noqa: E501
import logging  # noqa: E501
import time  # noqa: E501
from dataclasses import dataclass, field  # noqa: E402  # noqa: E501
from datetime import UTC, datetime  # noqa: E402  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501
from statistics import mean, median, stdev  # noqa: E402  # noqa: E501
from typing import Any  # noqa: E402  # noqa: E501

  # noqa: E501
logger = logging.getLogger(__name__)  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class BenchmarkSample:  # noqa: E501
    """A single benchmark sample."""  # noqa: E501
  # noqa: E501
    sample_id: str  # noqa: E501
    source_text: str  # noqa: E501
    response_text: str  # noqa: E501
    expected_quality: dict[str, float] | None = None  # Human-labeled  # noqa: E501
    metadata: dict[str, Any] = field(default_factory=dict)  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class BenchmarkResult:  # noqa: E501
    """Result for a single benchmark sample."""  # noqa: E501
  # noqa: E501
    sample_id: str  # noqa: E501
    quality_scores: dict[str, float]  # noqa: E501
    gate_result: dict[str, Any]  # noqa: E501
    latency_ms: float  # noqa: E501
    passed: bool  # noqa: E501
    human_score: float | None = None  # For calibration  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class BenchmarkReport:  # noqa: E501
    """Full benchmark report."""  # noqa: E501
  # noqa: E501
    benchmark_name: str  # noqa: E501
    model_name: str  # noqa: E501
    timestamp: str  # noqa: E501
    total_samples: int  # noqa: E501
    passed_samples: int  # noqa: E501
    pass_rate: float  # noqa: E501
    # Aggregate scores  # noqa: E501
    mean_scores: dict[str, float]  # noqa: E501
    median_scores: dict[str, float]  # noqa: E501
    std_scores: dict[str, float]  # noqa: E501
    # Latency stats  # noqa: E501
    mean_latency_ms: float  # noqa: E501
    p50_latency_ms: float  # noqa: E501
    p95_latency_ms: float  # noqa: E501
    p99_latency_ms: float  # noqa: E501
    # Per-sample results  # noqa: E501
    results: list[BenchmarkResult] = field(default_factory=list)  # noqa: E501
    # Comparison with baseline (if available)  # noqa: E501
    baseline_comparison: dict[str, float] | None = None  # noqa: E501
  # noqa: E501
    def to_dict(self) -> dict[str, Any]:  # noqa: E501
        """Convert to dictionary."""  # noqa: E501
        return {  # noqa: E501
            "benchmark_name": self.benchmark_name,  # noqa: E501
            "model_name": self.model_name,  # noqa: E501
            "timestamp": self.timestamp,  # noqa: E501
            "total_samples": self.total_samples,  # noqa: E501
            "passed_samples": self.passed_samples,  # noqa: E501
            "pass_rate": round(self.pass_rate, 4),  # noqa: E501
            "mean_scores": {k: round(v, 4) for k, v in self.mean_scores.items()},  # noqa: E501
            "median_scores": {k: round(v, 4) for k, v in self.median_scores.items()},  # noqa: E501
            "std_scores": {k: round(v, 4) for k, v in self.std_scores.items()},  # noqa: E501
            "latency": {  # noqa: E501
                "mean_ms": round(self.mean_latency_ms, 2),  # noqa: E501
                "p50_ms": round(self.p50_latency_ms, 2),  # noqa: E501
                "p95_ms": round(self.p95_latency_ms, 2),  # noqa: E501
                "p99_ms": round(self.p99_latency_ms, 2),  # noqa: E501
            },  # noqa: E501
            "baseline_comparison": self.baseline_comparison,  # noqa: E501
        }  # noqa: E501
  # noqa: E501
    def save(self, path: Path) -> None:  # noqa: E501
        """Save report to JSON file."""  # noqa: E501
        path.write_text(json.dumps(self.to_dict(), indent=2))  # noqa: E501
        logger.info("Saved benchmark report to %s", path)  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class ABTestResult:  # noqa: E501
    """Result of A/B test comparison."""  # noqa: E501
  # noqa: E501
    model_a: str  # noqa: E501
    model_b: str  # noqa: E501
    winner: str | None  # None if tie  # noqa: E501
    confidence: float  # noqa: E501
    dimension_comparison: dict[str, dict[str, float]]  # noqa: E501
    sample_size: int  # noqa: E501
    p_value: float | None = None  # noqa: E501
  # noqa: E501
    def to_dict(self) -> dict[str, Any]:  # noqa: E501
        """Convert to dictionary."""  # noqa: E501
        return {  # noqa: E501
            "model_a": self.model_a,  # noqa: E501
            "model_b": self.model_b,  # noqa: E501
            "winner": self.winner,  # noqa: E501
            "confidence": round(self.confidence, 4),  # noqa: E501
            "dimension_comparison": {  # noqa: E501
                dim: {k: round(v, 4) for k, v in scores.items()}  # noqa: E501
                for dim, scores in self.dimension_comparison.items()  # noqa: E501
            },  # noqa: E501
            "sample_size": self.sample_size,  # noqa: E501
            "p_value": round(self.p_value, 4) if self.p_value else None,  # noqa: E501
        }  # noqa: E501
  # noqa: E501
  # noqa: E501
class StandardDataset:  # noqa: E501
    """Standard evaluation dataset for quality benchmarking."""  # noqa: E501
  # noqa: E501
    SAMPLES = [  # noqa: E501
        # Grounded responses (should pass)  # noqa: E501
        BenchmarkSample(  # noqa: E501
            sample_id="grounded_001",  # noqa: E501
            source_text="Hey, want to grab lunch tomorrow at noon?",  # noqa: E501
            response_text="Sure, noon works for me! Where should we meet?",  # noqa: E501
            expected_quality={"hallucination": 0.1, "relevance": 0.9, "coherence": 0.9},  # noqa: E501
        ),  # noqa: E501
        BenchmarkSample(  # noqa: E501
            sample_id="grounded_002",  # noqa: E501
            source_text="Can you send me the report by end of day?",  # noqa: E501
            response_text="Yes, I'll have the report to you before 5pm.",  # noqa: E501
            expected_quality={"hallucination": 0.1, "relevance": 0.9, "coherence": 0.9},  # noqa: E501
        ),  # noqa: E501
        BenchmarkSample(  # noqa: E501
            sample_id="grounded_003",  # noqa: E501
            source_text="Happy birthday! Hope you have an amazing day!",  # noqa: E501
            response_text="Thank you so much! Really appreciate the birthday wishes!",  # noqa: E501
            expected_quality={"hallucination": 0.1, "relevance": 0.9, "tone": 0.9},  # noqa: E501
        ),  # noqa: E501
        BenchmarkSample(  # noqa: E501
            sample_id="grounded_004",  # noqa: E501
            source_text="Did you see the game last night?",  # noqa: E501
            response_text="Yes! What a finish, I couldn't believe that last play.",  # noqa: E501
            expected_quality={"hallucination": 0.2, "relevance": 0.8, "coherence": 0.9},  # noqa: E501
        ),  # noqa: E501
        BenchmarkSample(  # noqa: E501
            sample_id="grounded_005",  # noqa: E501
            source_text="Running late, be there in 20 minutes.",  # noqa: E501
            response_text="No problem, take your time. I'll grab us a table.",  # noqa: E501
            expected_quality={"hallucination": 0.1, "relevance": 0.9, "tone": 0.9},  # noqa: E501
        ),  # noqa: E501
        # Potentially hallucinated (should fail or warn)  # noqa: E501
        BenchmarkSample(  # noqa: E501
            sample_id="hallucination_001",  # noqa: E501
            source_text="What time is the meeting?",  # noqa: E501
            response_text=(  # noqa: E501
                "The meeting with John is at 3pm in the conference room on the 5th floor."  # noqa: E501
            ),  # noqa: E501
            expected_quality={"hallucination": 0.8, "factuality": 0.2},  # noqa: E501
            metadata={"note": "Adds specific details not in source"},  # noqa: E501
        ),  # noqa: E501
        BenchmarkSample(  # noqa: E501
            sample_id="hallucination_002",  # noqa: E501
            source_text="How was your weekend?",  # noqa: E501
            response_text="You mentioned going to the beach with Sarah, that sounds fun!",  # noqa: E501
            expected_quality={"hallucination": 0.9, "factuality": 0.1},  # noqa: E501
            metadata={"note": "Fabricates details about what user said"},  # noqa: E501
        ),  # noqa: E501
        BenchmarkSample(  # noqa: E501
            sample_id="hallucination_003",  # noqa: E501
            source_text="Thanks for the help yesterday!",  # noqa: E501
            response_text=(  # noqa: E501
                "You're welcome! I'm glad the presentation went well "  # noqa: E501
                "and that your boss loved the charts we made."  # noqa: E501
            ),  # noqa: E501
            expected_quality={"hallucination": 0.8, "factuality": 0.2},  # noqa: E501
            metadata={"note": "Assumes specific outcomes not mentioned"},  # noqa: E501
        ),  # noqa: E501
        # Coherence test cases  # noqa: E501
        BenchmarkSample(  # noqa: E501
            sample_id="coherence_001",  # noqa: E501
            source_text="Are you free tomorrow evening?",  # noqa: E501
            response_text="Tomorrow evening? I'll need to check my calendar.",  # noqa: E501
            expected_quality={"coherence": 0.9, "relevance": 0.9},  # noqa: E501
        ),  # noqa: E501
        BenchmarkSample(  # noqa: E501
            sample_id="incoherent_001",  # noqa: E501
            source_text="How's the weather there?",  # noqa: E501
            response_text=(  # noqa: E501
                "The weather is nice. By the way, did you finish the project? "  # noqa: E501
                "Also, I had pizza for lunch."  # noqa: E501
            ),  # noqa: E501
            expected_quality={"coherence": 0.4, "relevance": 0.5},  # noqa: E501
            metadata={"note": "Jumps between unrelated topics"},  # noqa: E501
        ),  # noqa: E501
        # Tone appropriateness  # noqa: E501
        BenchmarkSample(  # noqa: E501
            sample_id="tone_formal_001",  # noqa: E501
            source_text="Dear Sir/Madam, I am writing to inquire about the position.",  # noqa: E501
            response_text=(  # noqa: E501
                "Thank you for your interest in the position. "  # noqa: E501
                "We would be happy to provide more information."  # noqa: E501
            ),  # noqa: E501
            expected_quality={"tone": 0.9, "coherence": 0.9},  # noqa: E501
            metadata={"expected_tone": "formal"},  # noqa: E501
        ),  # noqa: E501
        BenchmarkSample(  # noqa: E501
            sample_id="tone_casual_001",  # noqa: E501
            source_text="yo whats up! you coming to the party?",  # noqa: E501
            response_text="Hey! Yeah definitely, what time does it start?",  # noqa: E501
            expected_quality={"tone": 0.9, "coherence": 0.9},  # noqa: E501
            metadata={"expected_tone": "casual"},  # noqa: E501
        ),  # noqa: E501
        # Length appropriateness  # noqa: E501
        BenchmarkSample(  # noqa: E501
            sample_id="length_short_001",  # noqa: E501
            source_text="Yes or no?",  # noqa: E501
            response_text="Yes.",  # noqa: E501
            expected_quality={"length": 0.9},  # noqa: E501
        ),  # noqa: E501
        BenchmarkSample(  # noqa: E501
            sample_id="length_verbose_001",  # noqa: E501
            source_text="Can you come?",  # noqa: E501
            response_text=(  # noqa: E501
                "That's a great question. Let me think about it. There are several "  # noqa: E501
                "factors to consider here, including my schedule, other commitments, "  # noqa: E501
                "transportation options, and whether I have anything else planned. "  # noqa: E501
                "After careful consideration of all these factors, I would say that "  # noqa: E501
                "yes, I believe I can attend."  # noqa: E501
            ),  # noqa: E501
            expected_quality={"length": 0.3},  # noqa: E501
            metadata={"note": "Overly verbose for simple question"},  # noqa: E501
        ),  # noqa: E501
    ]  # noqa: E501
  # noqa: E501
    @classmethod  # noqa: E501
    def get_samples(cls) -> list[BenchmarkSample]:  # noqa: E501
        """Get all standard benchmark samples."""  # noqa: E501
        return cls.SAMPLES  # noqa: E501
  # noqa: E501
    @classmethod  # noqa: E501
    def get_grounded_samples(cls) -> list[BenchmarkSample]:  # noqa: E501
        """Get only grounded (should-pass) samples."""  # noqa: E501
        return [s for s in cls.SAMPLES if s.sample_id.startswith("grounded")]  # noqa: E501
  # noqa: E501
    @classmethod  # noqa: E501
    def get_hallucination_samples(cls) -> list[BenchmarkSample]:  # noqa: E501
        """Get hallucination test samples."""  # noqa: E501
        return [s for s in cls.SAMPLES if s.sample_id.startswith("hallucination")]  # noqa: E501
  # noqa: E501
  # noqa: E501
class QualityBenchmark:  # noqa: E501
    """Quality benchmark runner."""  # noqa: E501
  # noqa: E501
    def __init__(  # noqa: E501
        self,  # noqa: E501
        model_name: str = "default",  # noqa: E501
        gate_threshold: float = 0.5,  # noqa: E501
    ) -> None:  # noqa: E501
        """Initialize the benchmark runner.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            model_name: Name of the model being benchmarked  # noqa: E501
            gate_threshold: Quality gate threshold  # noqa: E501
        """  # noqa: E501
        self._model_name = model_name  # noqa: E501
        self._gate_threshold = gate_threshold  # noqa: E501
        self._quality_gate: Any = None  # noqa: E501
  # noqa: E501
    def _get_quality_gate(self) -> Any:  # noqa: E501
        """Lazy load quality gate."""  # noqa: E501
        if self._quality_gate is None:  # noqa: E501
            try:  # noqa: E501
                from jarvis.quality.gates import QualityGate, QualityGateConfig  # noqa: E501
  # noqa: E501
                config = QualityGateConfig(  # noqa: E501
                    hallucination_threshold=self._gate_threshold,  # noqa: E501
                    factuality_threshold=self._gate_threshold,  # noqa: E501
                )  # noqa: E501
                self._quality_gate = QualityGate(config)  # noqa: E501
            except Exception as e:  # noqa: E501
                logger.error("Failed to load quality gate: %s", e)  # noqa: E501
                raise  # noqa: E501
        return self._quality_gate  # noqa: E501
  # noqa: E501
    def run_benchmark(  # noqa: E501
        self,  # noqa: E501
        samples: list[BenchmarkSample],  # noqa: E501
        benchmark_name: str = "standard",  # noqa: E501
    ) -> BenchmarkReport:  # noqa: E501
        """Run benchmark on a set of samples.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            samples: Benchmark samples to evaluate  # noqa: E501
            benchmark_name: Name for the benchmark  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            BenchmarkReport with results  # noqa: E501
        """  # noqa: E501
        gate = self._get_quality_gate()  # noqa: E501
        results: list[BenchmarkResult] = []  # noqa: E501
        latencies: list[float] = []  # noqa: E501
        dimension_scores: dict[str, list[float]] = {}  # noqa: E501
  # noqa: E501
        logger.info("Running benchmark '%s' with %d samples", benchmark_name, len(samples))  # noqa: E501
  # noqa: E501
        for sample in samples:  # noqa: E501
            start_time = time.perf_counter()  # noqa: E501
  # noqa: E501
            # Run quality check  # noqa: E501
            gate_result = gate.check(  # noqa: E501
                response=sample.response_text,  # noqa: E501
                source=sample.source_text,  # noqa: E501
            )  # noqa: E501
  # noqa: E501
            latency_ms = (time.perf_counter() - start_time) * 1000  # noqa: E501
            latencies.append(latency_ms)  # noqa: E501
  # noqa: E501
            # Extract scores  # noqa: E501
            scores = {"overall": gate_result.quality_score}  # noqa: E501
            for gr in gate_result.gate_results:  # noqa: E501
                scores[gr.gate_name] = gr.score  # noqa: E501
  # noqa: E501
            # Track dimension scores  # noqa: E501
            for dim, score in scores.items():  # noqa: E501
                if dim not in dimension_scores:  # noqa: E501
                    dimension_scores[dim] = []  # noqa: E501
                dimension_scores[dim].append(score)  # noqa: E501
  # noqa: E501
            results.append(  # noqa: E501
                BenchmarkResult(  # noqa: E501
                    sample_id=sample.sample_id,  # noqa: E501
                    quality_scores=scores,  # noqa: E501
                    gate_result=gate_result.to_dict(),  # noqa: E501
                    latency_ms=latency_ms,  # noqa: E501
                    passed=gate_result.should_send,  # noqa: E501
                    human_score=(  # noqa: E501
                        sample.expected_quality.get("overall") if sample.expected_quality else None  # noqa: E501
                    ),  # noqa: E501
                )  # noqa: E501
            )  # noqa: E501
  # noqa: E501
        # Calculate aggregates  # noqa: E501
        passed_count = sum(1 for r in results if r.passed)  # noqa: E501
        pass_rate = passed_count / len(results) if results else 0.0  # noqa: E501
  # noqa: E501
        mean_scores = {dim: mean(scores) for dim, scores in dimension_scores.items()}  # noqa: E501
        median_scores = {dim: median(scores) for dim, scores in dimension_scores.items()}  # noqa: E501
        std_scores = {  # noqa: E501
            dim: stdev(scores) if len(scores) > 1 else 0.0  # noqa: E501
            for dim, scores in dimension_scores.items()  # noqa: E501
        }  # noqa: E501
  # noqa: E501
        # Latency percentiles  # noqa: E501
        sorted_latencies = sorted(latencies)  # noqa: E501
        p50_idx = int(len(sorted_latencies) * 0.50)  # noqa: E501
        p95_idx = int(len(sorted_latencies) * 0.95)  # noqa: E501
        p99_idx = int(len(sorted_latencies) * 0.99)  # noqa: E501
  # noqa: E501
        # Compute latency percentiles (handle empty case)  # noqa: E501
        p50_latency = sorted_latencies[p50_idx] if sorted_latencies else 0.0  # noqa: E501
        p95_latency = (  # noqa: E501
            sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)] if sorted_latencies else 0.0  # noqa: E501
        )  # noqa: E501
        p99_latency = (  # noqa: E501
            sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)] if sorted_latencies else 0.0  # noqa: E501
        )  # noqa: E501
  # noqa: E501
        return BenchmarkReport(  # noqa: E501
            benchmark_name=benchmark_name,  # noqa: E501
            model_name=self._model_name,  # noqa: E501
            timestamp=datetime.now(UTC).isoformat(),  # noqa: E501
            total_samples=len(samples),  # noqa: E501
            passed_samples=passed_count,  # noqa: E501
            pass_rate=pass_rate,  # noqa: E501
            mean_scores=mean_scores,  # noqa: E501
            median_scores=median_scores,  # noqa: E501
            std_scores=std_scores,  # noqa: E501
            mean_latency_ms=mean(latencies) if latencies else 0.0,  # noqa: E501
            p50_latency_ms=p50_latency,  # noqa: E501
            p95_latency_ms=p95_latency,  # noqa: E501
            p99_latency_ms=p99_latency,  # noqa: E501
            results=results,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
    def run_regression_test(  # noqa: E501
        self,  # noqa: E501
        baseline_report: BenchmarkReport,  # noqa: E501
        samples: list[BenchmarkSample] | None = None,  # noqa: E501
        regression_threshold: float = 0.05,  # noqa: E501
    ) -> tuple[bool, dict[str, float]]:  # noqa: E501
        """Run regression test against baseline.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            baseline_report: Baseline benchmark report  # noqa: E501
            samples: Samples to test (uses standard if None)  # noqa: E501
            regression_threshold: Max allowed decline (5% default)  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            Tuple of (passed, dimension_changes)  # noqa: E501
        """  # noqa: E501
        if samples is None:  # noqa: E501
            samples = StandardDataset.get_samples()  # noqa: E501
  # noqa: E501
        # Run current benchmark  # noqa: E501
        current_report = self.run_benchmark(samples, benchmark_name="regression_test")  # noqa: E501
  # noqa: E501
        # Compare against baseline  # noqa: E501
        dimension_changes: dict[str, float] = {}  # noqa: E501
        regressions: list[str] = []  # noqa: E501
  # noqa: E501
        for dim in current_report.mean_scores:  # noqa: E501
            if dim in baseline_report.mean_scores:  # noqa: E501
                baseline_score = baseline_report.mean_scores[dim]  # noqa: E501
                current_score = current_report.mean_scores[dim]  # noqa: E501
  # noqa: E501
                if baseline_score > 0:  # noqa: E501
                    change = (current_score - baseline_score) / baseline_score  # noqa: E501
                    dimension_changes[dim] = change  # noqa: E501
  # noqa: E501
                    if change < -regression_threshold:  # noqa: E501
                        regressions.append(dim)  # noqa: E501
                        logger.warning(  # noqa: E501
                            "Regression in %s: %.2f%% decline",  # noqa: E501
                            dim,  # noqa: E501
                            change * 100,  # noqa: E501
                        )  # noqa: E501
  # noqa: E501
        passed = len(regressions) == 0  # noqa: E501
  # noqa: E501
        if passed:  # noqa: E501
            logger.info("Regression test PASSED - no significant quality decline")  # noqa: E501
        else:  # noqa: E501
            logger.error(  # noqa: E501
                "Regression test FAILED - regressions in: %s",  # noqa: E501
                ", ".join(regressions),  # noqa: E501
            )  # noqa: E501
  # noqa: E501
        return passed, dimension_changes  # noqa: E501
  # noqa: E501
  # noqa: E501
class ABTestFramework:  # noqa: E501
    """A/B testing framework for model comparison."""  # noqa: E501
  # noqa: E501
    def __init__(self) -> None:  # noqa: E501
        """Initialize the A/B test framework."""  # noqa: E501
        pass  # noqa: E501
  # noqa: E501
    def compare_models(  # noqa: E501
        self,  # noqa: E501
        model_a_name: str,  # noqa: E501
        model_b_name: str,  # noqa: E501
        samples: list[BenchmarkSample] | None = None,  # noqa: E501
        significance_threshold: float = 0.05,  # noqa: E501
    ) -> ABTestResult:  # noqa: E501
        """Compare two models with A/B testing.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            model_a_name: Name of model A  # noqa: E501
            model_b_name: Name of model B  # noqa: E501
            samples: Samples to use (standard if None)  # noqa: E501
            significance_threshold: P-value threshold for significance  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            ABTestResult with comparison  # noqa: E501
        """  # noqa: E501
        if samples is None:  # noqa: E501
            samples = StandardDataset.get_samples()  # noqa: E501
  # noqa: E501
        # Run benchmarks for both models  # noqa: E501
        benchmark_a = QualityBenchmark(model_name=model_a_name)  # noqa: E501
        benchmark_b = QualityBenchmark(model_name=model_b_name)  # noqa: E501
  # noqa: E501
        report_a = benchmark_a.run_benchmark(samples, f"ab_test_{model_a_name}")  # noqa: E501
        report_b = benchmark_b.run_benchmark(samples, f"ab_test_{model_b_name}")  # noqa: E501
  # noqa: E501
        # Compare dimensions  # noqa: E501
        dimension_comparison: dict[str, dict[str, float]] = {}  # noqa: E501
        wins_a = 0  # noqa: E501
        wins_b = 0  # noqa: E501
  # noqa: E501
        for dim in report_a.mean_scores:  # noqa: E501
            if dim in report_b.mean_scores:  # noqa: E501
                score_a = report_a.mean_scores[dim]  # noqa: E501
                score_b = report_b.mean_scores[dim]  # noqa: E501
                diff = score_b - score_a  # noqa: E501
  # noqa: E501
                dimension_comparison[dim] = {  # noqa: E501
                    "model_a": score_a,  # noqa: E501
                    "model_b": score_b,  # noqa: E501
                    "difference": diff,  # noqa: E501
                }  # noqa: E501
  # noqa: E501
                # Count wins (>2% difference)  # noqa: E501
                if diff > 0.02:  # noqa: E501
                    wins_b += 1  # noqa: E501
                elif diff < -0.02:  # noqa: E501
                    wins_a += 1  # noqa: E501
  # noqa: E501
        # Determine winner  # noqa: E501
        total_dims = wins_a + wins_b  # noqa: E501
        if total_dims == 0:  # noqa: E501
            winner = None  # noqa: E501
            confidence = 0.5  # noqa: E501
        elif wins_a > wins_b:  # noqa: E501
            winner = model_a_name  # noqa: E501
            confidence = wins_a / total_dims  # noqa: E501
        elif wins_b > wins_a:  # noqa: E501
            winner = model_b_name  # noqa: E501
            confidence = wins_b / total_dims  # noqa: E501
        else:  # noqa: E501
            winner = None  # noqa: E501
            confidence = 0.5  # noqa: E501
  # noqa: E501
        # Simple statistical test (paired t-test approximation)  # noqa: E501
        p_value = self._compute_p_value(  # noqa: E501
            list(report_a.mean_scores.values()),  # noqa: E501
            list(report_b.mean_scores.values()),  # noqa: E501
        )  # noqa: E501
  # noqa: E501
        return ABTestResult(  # noqa: E501
            model_a=model_a_name,  # noqa: E501
            model_b=model_b_name,  # noqa: E501
            winner=winner if p_value and p_value < significance_threshold else None,  # noqa: E501
            confidence=confidence,  # noqa: E501
            dimension_comparison=dimension_comparison,  # noqa: E501
            sample_size=len(samples),  # noqa: E501
            p_value=p_value,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
    def _compute_p_value(self, scores_a: list[float], scores_b: list[float]) -> float | None:  # noqa: E501
        """Compute p-value for paired comparison."""  # noqa: E501
        if len(scores_a) != len(scores_b) or len(scores_a) < 3:  # noqa: E501
            return None  # noqa: E501
  # noqa: E501
        try:  # noqa: E501
            # Simple paired difference test  # noqa: E501
            differences = [b - a for a, b in zip(scores_a, scores_b)]  # noqa: E501
            mean_diff = mean(differences)  # noqa: E501
            std_diff = stdev(differences) if len(differences) > 1 else 1.0  # noqa: E501
  # noqa: E501
            if std_diff == 0:  # noqa: E501
                return 0.0 if mean_diff == 0 else 1.0  # noqa: E501
  # noqa: E501
            # T-statistic  # noqa: E501
            n = len(differences)  # noqa: E501
            t_stat = mean_diff / (std_diff / (n**0.5))  # noqa: E501
  # noqa: E501
            # Approximate p-value (two-tailed)  # noqa: E501
            # Using normal approximation for simplicity  # noqa: E501
            import math  # noqa: E501
  # noqa: E501
            p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))  # noqa: E501
            return p_value  # noqa: E501
        except Exception as e:  # noqa: E501
            logger.warning("P-value computation failed: %s", e)  # noqa: E501
            return None  # noqa: E501
  # noqa: E501
  # noqa: E501
class HumanEvaluationIntegration:  # noqa: E501
    """Integration for human evaluation studies."""  # noqa: E501
  # noqa: E501
    def __init__(self, output_dir: Path | None = None) -> None:  # noqa: E501
        """Initialize human evaluation integration.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            output_dir: Directory for evaluation files  # noqa: E501
        """  # noqa: E501
        self._output_dir = output_dir or Path("evaluations")  # noqa: E501
        self._output_dir.mkdir(parents=True, exist_ok=True)  # noqa: E501
  # noqa: E501
    def generate_evaluation_batch(  # noqa: E501
        self,  # noqa: E501
        samples: list[BenchmarkSample],  # noqa: E501
        batch_name: str,  # noqa: E501
        include_predictions: bool = True,  # noqa: E501
    ) -> Path:  # noqa: E501
        """Generate a batch file for human evaluation.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            samples: Samples to evaluate  # noqa: E501
            batch_name: Name for the batch  # noqa: E501
            include_predictions: Include model predictions for comparison  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            Path to the generated evaluation file  # noqa: E501
        """  # noqa: E501
        evaluation_items = []  # noqa: E501
  # noqa: E501
        for sample in samples:  # noqa: E501
            item = {  # noqa: E501
                "sample_id": sample.sample_id,  # noqa: E501
                "source": sample.source_text,  # noqa: E501
                "response": sample.response_text,  # noqa: E501
                "evaluation": {  # noqa: E501
                    "hallucination": None,  # 1-5 scale  # noqa: E501
                    "relevance": None,  # noqa: E501
                    "coherence": None,  # noqa: E501
                    "tone": None,  # noqa: E501
                    "overall": None,  # noqa: E501
                    "comments": "",  # noqa: E501
                },  # noqa: E501
            }  # noqa: E501
  # noqa: E501
            if include_predictions and sample.expected_quality:  # noqa: E501
                item["predicted_quality"] = sample.expected_quality  # noqa: E501
  # noqa: E501
            evaluation_items.append(item)  # noqa: E501
  # noqa: E501
        output_path = self._output_dir / f"{batch_name}_evaluation.json"  # noqa: E501
        output_path.write_text(json.dumps(evaluation_items, indent=2))  # noqa: E501
  # noqa: E501
        logger.info("Generated evaluation batch with %d items: %s", len(samples), output_path)  # noqa: E501
  # noqa: E501
        return output_path  # noqa: E501
  # noqa: E501
    def import_human_evaluations(self, evaluation_file: Path) -> list[tuple[str, dict[str, float]]]:  # noqa: E501
        """Import human evaluations from a completed file.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            evaluation_file: Path to the completed evaluation file  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            List of (sample_id, scores) tuples  # noqa: E501
        """  # noqa: E501
        data = json.loads(evaluation_file.read_text())  # noqa: E501
        results = []  # noqa: E501
  # noqa: E501
        for item in data:  # noqa: E501
            sample_id = item["sample_id"]  # noqa: E501
            evaluation = item.get("evaluation", {})  # noqa: E501
  # noqa: E501
            # Convert 1-5 scale to 0-1  # noqa: E501
            scores = {}  # noqa: E501
            for dim in ["hallucination", "relevance", "coherence", "tone", "overall"]:  # noqa: E501
                if evaluation.get(dim) is not None:  # noqa: E501
                    # Invert hallucination (higher = worse)  # noqa: E501
                    if dim == "hallucination":  # noqa: E501
                        scores[dim] = 1 - (evaluation[dim] - 1) / 4  # noqa: E501
                    else:  # noqa: E501
                        scores[dim] = (evaluation[dim] - 1) / 4  # noqa: E501
  # noqa: E501
            if scores:  # noqa: E501
                results.append((sample_id, scores))  # noqa: E501
  # noqa: E501
        logger.info("Imported %d human evaluations from %s", len(results), evaluation_file)  # noqa: E501
  # noqa: E501
        return results  # noqa: E501
  # noqa: E501
    def compute_inter_annotator_agreement(  # noqa: E501
        self, evaluations: list[dict[str, Any]]  # noqa: E501
    ) -> dict[str, float]:  # noqa: E501
        """Compute inter-annotator agreement metrics.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            evaluations: List of evaluation dicts from multiple annotators  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            Agreement metrics per dimension  # noqa: E501
        """  # noqa: E501
        # This is a simplified version - would need multiple annotator data  # noqa: E501
        # Returns placeholder values  # noqa: E501
        return {  # noqa: E501
            "hallucination": 0.0,  # noqa: E501
            "relevance": 0.0,  # noqa: E501
            "coherence": 0.0,  # noqa: E501
            "tone": 0.0,  # noqa: E501
            "overall": 0.0,  # noqa: E501
        }  # noqa: E501
  # noqa: E501
  # noqa: E501
def main() -> None:  # noqa: E501
    """CLI entry point for quality benchmark."""  # noqa: E501
    import argparse  # noqa: E501
  # noqa: E501
    parser = argparse.ArgumentParser(description="JARVIS Quality Benchmark")  # noqa: E501
    subparsers = parser.add_subparsers(dest="command", help="Commands")  # noqa: E501
  # noqa: E501
    # Run benchmark  # noqa: E501
    run_parser = subparsers.add_parser("run", help="Run benchmark")  # noqa: E501
    run_parser.add_argument(  # noqa: E501
        "--dataset",  # noqa: E501
        choices=["standard", "grounded", "hallucination"],  # noqa: E501
        default="standard",  # noqa: E501
        help="Dataset to use",  # noqa: E501
    )  # noqa: E501
    run_parser.add_argument("--model", default="default", help="Model name")  # noqa: E501
    run_parser.add_argument("--output", type=Path, help="Output file for report")  # noqa: E501
  # noqa: E501
    # Compare models  # noqa: E501
    compare_parser = subparsers.add_parser("compare", help="Compare two models")  # noqa: E501
    compare_parser.add_argument("--model-a", required=True, help="First model")  # noqa: E501
    compare_parser.add_argument("--model-b", required=True, help="Second model")  # noqa: E501
  # noqa: E501
    # Generate human evaluation batch  # noqa: E501
    eval_parser = subparsers.add_parser("generate-eval", help="Generate human evaluation batch")  # noqa: E501
    eval_parser.add_argument("--name", required=True, help="Batch name")  # noqa: E501
    eval_parser.add_argument(  # noqa: E501
        "--output-dir", type=Path, default=Path("evaluations"), help="Output directory"  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    args = parser.parse_args()  # noqa: E501
  # noqa: E501
    logging.basicConfig(level=logging.INFO)  # noqa: E501
  # noqa: E501
    if args.command == "run":  # noqa: E501
        # Get samples  # noqa: E501
        if args.dataset == "standard":  # noqa: E501
            samples = StandardDataset.get_samples()  # noqa: E501
        elif args.dataset == "grounded":  # noqa: E501
            samples = StandardDataset.get_grounded_samples()  # noqa: E501
        else:  # noqa: E501
            samples = StandardDataset.get_hallucination_samples()  # noqa: E501
  # noqa: E501
        # Run benchmark  # noqa: E501
        benchmark = QualityBenchmark(model_name=args.model)  # noqa: E501
        report = benchmark.run_benchmark(samples, benchmark_name=args.dataset)  # noqa: E501
  # noqa: E501
        # Print summary  # noqa: E501
        print(f"\n=== Benchmark Report: {report.benchmark_name} ===")  # noqa: E501
        print(f"Model: {report.model_name}")  # noqa: E501
        print(f"Samples: {report.total_samples}")  # noqa: E501
        print(f"Passed: {report.passed_samples} ({report.pass_rate:.1%})")  # noqa: E501
        print("\nMean Scores:")  # noqa: E501
        for dim, score in report.mean_scores.items():  # noqa: E501
            print(f"  {dim}: {score:.3f}")  # noqa: E501
        print("\nLatency:")  # noqa: E501
        print(f"  Mean: {report.mean_latency_ms:.1f}ms")  # noqa: E501
        print(f"  P95: {report.p95_latency_ms:.1f}ms")  # noqa: E501
  # noqa: E501
        # Save if output specified  # noqa: E501
        if args.output:  # noqa: E501
            report.save(args.output)  # noqa: E501
  # noqa: E501
    elif args.command == "compare":  # noqa: E501
        framework = ABTestFramework()  # noqa: E501
        result = framework.compare_models(args.model_a, args.model_b)  # noqa: E501
  # noqa: E501
        print(f"\n=== A/B Test: {args.model_a} vs {args.model_b} ===")  # noqa: E501
        print(f"Winner: {result.winner or 'Tie'}")  # noqa: E501
        print(f"Confidence: {result.confidence:.1%}")  # noqa: E501
        if result.p_value:  # noqa: E501
            print(f"P-value: {result.p_value:.4f}")  # noqa: E501
        print("\nDimension Comparison:")  # noqa: E501
        for dim, scores in result.dimension_comparison.items():  # noqa: E501
            a_score = scores["model_a"]  # noqa: E501
            b_score = scores["model_b"]  # noqa: E501
            diff = scores["difference"]  # noqa: E501
            print(f"  {dim}: A={a_score:.3f}, B={b_score:.3f}, diff={diff:+.3f}")  # noqa: E501
  # noqa: E501
    elif args.command == "generate-eval":  # noqa: E501
        integration = HumanEvaluationIntegration(output_dir=args.output_dir)  # noqa: E501
        samples = StandardDataset.get_samples()  # noqa: E501
        output_path = integration.generate_evaluation_batch(samples, args.name)  # noqa: E501
        print(f"Generated evaluation batch: {output_path}")  # noqa: E501
  # noqa: E501
    else:  # noqa: E501
        parser.print_help()  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    main()  # noqa: E501
