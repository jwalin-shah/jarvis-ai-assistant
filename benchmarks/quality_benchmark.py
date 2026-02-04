"""Quality benchmarking suite for JARVIS.

Provides:
- Standard evaluation datasets
- A/B testing framework
- Automated regression testing
- Human evaluation integration

Usage:
    uv run python -m benchmarks.quality_benchmark --help
    uv run python -m benchmarks.quality_benchmark run --dataset standard
    uv run python -m benchmarks.quality_benchmark compare --model-a default --model-b new
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkSample:
    """A single benchmark sample."""

    sample_id: str
    source_text: str
    response_text: str
    expected_quality: dict[str, float] | None = None  # Human-labeled
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result for a single benchmark sample."""

    sample_id: str
    quality_scores: dict[str, float]
    gate_result: dict[str, Any]
    latency_ms: float
    passed: bool
    human_score: float | None = None  # For calibration


@dataclass
class BenchmarkReport:
    """Full benchmark report."""

    benchmark_name: str
    model_name: str
    timestamp: str
    total_samples: int
    passed_samples: int
    pass_rate: float
    # Aggregate scores
    mean_scores: dict[str, float]
    median_scores: dict[str, float]
    std_scores: dict[str, float]
    # Latency stats
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    # Per-sample results
    results: list[BenchmarkResult] = field(default_factory=list)
    # Comparison with baseline (if available)
    baseline_comparison: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "total_samples": self.total_samples,
            "passed_samples": self.passed_samples,
            "pass_rate": round(self.pass_rate, 4),
            "mean_scores": {k: round(v, 4) for k, v in self.mean_scores.items()},
            "median_scores": {k: round(v, 4) for k, v in self.median_scores.items()},
            "std_scores": {k: round(v, 4) for k, v in self.std_scores.items()},
            "latency": {
                "mean_ms": round(self.mean_latency_ms, 2),
                "p50_ms": round(self.p50_latency_ms, 2),
                "p95_ms": round(self.p95_latency_ms, 2),
                "p99_ms": round(self.p99_latency_ms, 2),
            },
            "baseline_comparison": self.baseline_comparison,
        }

    def save(self, path: Path) -> None:
        """Save report to JSON file."""
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("Saved benchmark report to %s", path)


@dataclass
class ABTestResult:
    """Result of A/B test comparison."""

    model_a: str
    model_b: str
    winner: str | None  # None if tie
    confidence: float
    dimension_comparison: dict[str, dict[str, float]]
    sample_size: int
    p_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "winner": self.winner,
            "confidence": round(self.confidence, 4),
            "dimension_comparison": {
                dim: {k: round(v, 4) for k, v in scores.items()}
                for dim, scores in self.dimension_comparison.items()
            },
            "sample_size": self.sample_size,
            "p_value": round(self.p_value, 4) if self.p_value else None,
        }


class StandardDataset:
    """Standard evaluation dataset for quality benchmarking."""

    SAMPLES = [
        # Grounded responses (should pass)
        BenchmarkSample(
            sample_id="grounded_001",
            source_text="Hey, want to grab lunch tomorrow at noon?",
            response_text="Sure, noon works for me! Where should we meet?",
            expected_quality={"hallucination": 0.1, "relevance": 0.9, "coherence": 0.9},
        ),
        BenchmarkSample(
            sample_id="grounded_002",
            source_text="Can you send me the report by end of day?",
            response_text="Yes, I'll have the report to you before 5pm.",
            expected_quality={"hallucination": 0.1, "relevance": 0.9, "coherence": 0.9},
        ),
        BenchmarkSample(
            sample_id="grounded_003",
            source_text="Happy birthday! Hope you have an amazing day!",
            response_text="Thank you so much! Really appreciate the birthday wishes!",
            expected_quality={"hallucination": 0.1, "relevance": 0.9, "tone": 0.9},
        ),
        BenchmarkSample(
            sample_id="grounded_004",
            source_text="Did you see the game last night?",
            response_text="Yes! What a finish, I couldn't believe that last play.",
            expected_quality={"hallucination": 0.2, "relevance": 0.8, "coherence": 0.9},
        ),
        BenchmarkSample(
            sample_id="grounded_005",
            source_text="Running late, be there in 20 minutes.",
            response_text="No problem, take your time. I'll grab us a table.",
            expected_quality={"hallucination": 0.1, "relevance": 0.9, "tone": 0.9},
        ),
        # Potentially hallucinated (should fail or warn)
        BenchmarkSample(
            sample_id="hallucination_001",
            source_text="What time is the meeting?",
            response_text=(
                "The meeting with John is at 3pm in the conference room on the 5th floor."
            ),
            expected_quality={"hallucination": 0.8, "factuality": 0.2},
            metadata={"note": "Adds specific details not in source"},
        ),
        BenchmarkSample(
            sample_id="hallucination_002",
            source_text="How was your weekend?",
            response_text="You mentioned going to the beach with Sarah, that sounds fun!",
            expected_quality={"hallucination": 0.9, "factuality": 0.1},
            metadata={"note": "Fabricates details about what user said"},
        ),
        BenchmarkSample(
            sample_id="hallucination_003",
            source_text="Thanks for the help yesterday!",
            response_text=(
                "You're welcome! I'm glad the presentation went well "
                "and that your boss loved the charts we made."
            ),
            expected_quality={"hallucination": 0.8, "factuality": 0.2},
            metadata={"note": "Assumes specific outcomes not mentioned"},
        ),
        # Coherence test cases
        BenchmarkSample(
            sample_id="coherence_001",
            source_text="Are you free tomorrow evening?",
            response_text="Tomorrow evening? I'll need to check my calendar.",
            expected_quality={"coherence": 0.9, "relevance": 0.9},
        ),
        BenchmarkSample(
            sample_id="incoherent_001",
            source_text="How's the weather there?",
            response_text=(
                "The weather is nice. By the way, did you finish the project? "
                "Also, I had pizza for lunch."
            ),
            expected_quality={"coherence": 0.4, "relevance": 0.5},
            metadata={"note": "Jumps between unrelated topics"},
        ),
        # Tone appropriateness
        BenchmarkSample(
            sample_id="tone_formal_001",
            source_text="Dear Sir/Madam, I am writing to inquire about the position.",
            response_text=(
                "Thank you for your interest in the position. "
                "We would be happy to provide more information."
            ),
            expected_quality={"tone": 0.9, "coherence": 0.9},
            metadata={"expected_tone": "formal"},
        ),
        BenchmarkSample(
            sample_id="tone_casual_001",
            source_text="yo whats up! you coming to the party?",
            response_text="Hey! Yeah definitely, what time does it start?",
            expected_quality={"tone": 0.9, "coherence": 0.9},
            metadata={"expected_tone": "casual"},
        ),
        # Length appropriateness
        BenchmarkSample(
            sample_id="length_short_001",
            source_text="Yes or no?",
            response_text="Yes.",
            expected_quality={"length": 0.9},
        ),
        BenchmarkSample(
            sample_id="length_verbose_001",
            source_text="Can you come?",
            response_text=(
                "That's a great question. Let me think about it. There are several "
                "factors to consider here, including my schedule, other commitments, "
                "transportation options, and whether I have anything else planned. "
                "After careful consideration of all these factors, I would say that "
                "yes, I believe I can attend."
            ),
            expected_quality={"length": 0.3},
            metadata={"note": "Overly verbose for simple question"},
        ),
    ]

    @classmethod
    def get_samples(cls) -> list[BenchmarkSample]:
        """Get all standard benchmark samples."""
        return cls.SAMPLES

    @classmethod
    def get_grounded_samples(cls) -> list[BenchmarkSample]:
        """Get only grounded (should-pass) samples."""
        return [s for s in cls.SAMPLES if s.sample_id.startswith("grounded")]

    @classmethod
    def get_hallucination_samples(cls) -> list[BenchmarkSample]:
        """Get hallucination test samples."""
        return [s for s in cls.SAMPLES if s.sample_id.startswith("hallucination")]


class QualityBenchmark:
    """Quality benchmark runner."""

    def __init__(
        self,
        model_name: str = "default",
        gate_threshold: float = 0.5,
    ) -> None:
        """Initialize the benchmark runner.

        Args:
            model_name: Name of the model being benchmarked
            gate_threshold: Quality gate threshold
        """
        self._model_name = model_name
        self._gate_threshold = gate_threshold
        self._quality_gate: Any = None

    def _get_quality_gate(self) -> Any:
        """Lazy load quality gate."""
        if self._quality_gate is None:
            try:
                from jarvis.quality.gates import QualityGate, QualityGateConfig

                config = QualityGateConfig(
                    hallucination_threshold=self._gate_threshold,
                    factuality_threshold=self._gate_threshold,
                )
                self._quality_gate = QualityGate(config)
            except Exception as e:
                logger.error("Failed to load quality gate: %s", e)
                raise
        return self._quality_gate

    def run_benchmark(
        self,
        samples: list[BenchmarkSample],
        benchmark_name: str = "standard",
    ) -> BenchmarkReport:
        """Run benchmark on a set of samples.

        Args:
            samples: Benchmark samples to evaluate
            benchmark_name: Name for the benchmark

        Returns:
            BenchmarkReport with results
        """
        gate = self._get_quality_gate()
        results: list[BenchmarkResult] = []
        latencies: list[float] = []
        dimension_scores: dict[str, list[float]] = {}

        logger.info("Running benchmark '%s' with %d samples", benchmark_name, len(samples))

        for sample in samples:
            start_time = time.perf_counter()

            # Run quality check
            gate_result = gate.check(
                response=sample.response_text,
                source=sample.source_text,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(latency_ms)

            # Extract scores
            scores = {"overall": gate_result.quality_score}
            for gr in gate_result.gate_results:
                scores[gr.gate_name] = gr.score

            # Track dimension scores
            for dim, score in scores.items():
                if dim not in dimension_scores:
                    dimension_scores[dim] = []
                dimension_scores[dim].append(score)

            results.append(
                BenchmarkResult(
                    sample_id=sample.sample_id,
                    quality_scores=scores,
                    gate_result=gate_result.to_dict(),
                    latency_ms=latency_ms,
                    passed=gate_result.should_send,
                    human_score=(
                        sample.expected_quality.get("overall") if sample.expected_quality else None
                    ),
                )
            )

        # Calculate aggregates
        passed_count = sum(1 for r in results if r.passed)
        pass_rate = passed_count / len(results) if results else 0.0

        mean_scores = {dim: mean(scores) for dim, scores in dimension_scores.items()}
        median_scores = {dim: median(scores) for dim, scores in dimension_scores.items()}
        std_scores = {
            dim: stdev(scores) if len(scores) > 1 else 0.0
            for dim, scores in dimension_scores.items()
        }

        # Latency percentiles
        sorted_latencies = sorted(latencies)
        p50_idx = int(len(sorted_latencies) * 0.50)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p99_idx = int(len(sorted_latencies) * 0.99)

        # Compute latency percentiles (handle empty case)
        p50_latency = sorted_latencies[p50_idx] if sorted_latencies else 0.0
        p95_latency = (
            sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)] if sorted_latencies else 0.0
        )
        p99_latency = (
            sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)] if sorted_latencies else 0.0
        )

        return BenchmarkReport(
            benchmark_name=benchmark_name,
            model_name=self._model_name,
            timestamp=datetime.now(UTC).isoformat(),
            total_samples=len(samples),
            passed_samples=passed_count,
            pass_rate=pass_rate,
            mean_scores=mean_scores,
            median_scores=median_scores,
            std_scores=std_scores,
            mean_latency_ms=mean(latencies) if latencies else 0.0,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            results=results,
        )

    def run_regression_test(
        self,
        baseline_report: BenchmarkReport,
        samples: list[BenchmarkSample] | None = None,
        regression_threshold: float = 0.05,
    ) -> tuple[bool, dict[str, float]]:
        """Run regression test against baseline.

        Args:
            baseline_report: Baseline benchmark report
            samples: Samples to test (uses standard if None)
            regression_threshold: Max allowed decline (5% default)

        Returns:
            Tuple of (passed, dimension_changes)
        """
        if samples is None:
            samples = StandardDataset.get_samples()

        # Run current benchmark
        current_report = self.run_benchmark(samples, benchmark_name="regression_test")

        # Compare against baseline
        dimension_changes: dict[str, float] = {}
        regressions: list[str] = []

        for dim in current_report.mean_scores:
            if dim in baseline_report.mean_scores:
                baseline_score = baseline_report.mean_scores[dim]
                current_score = current_report.mean_scores[dim]

                if baseline_score > 0:
                    change = (current_score - baseline_score) / baseline_score
                    dimension_changes[dim] = change

                    if change < -regression_threshold:
                        regressions.append(dim)
                        logger.warning(
                            "Regression in %s: %.2f%% decline",
                            dim,
                            change * 100,
                        )

        passed = len(regressions) == 0

        if passed:
            logger.info("Regression test PASSED - no significant quality decline")
        else:
            logger.error(
                "Regression test FAILED - regressions in: %s",
                ", ".join(regressions),
            )

        return passed, dimension_changes


class ABTestFramework:
    """A/B testing framework for model comparison."""

    def __init__(self) -> None:
        """Initialize the A/B test framework."""
        pass

    def compare_models(
        self,
        model_a_name: str,
        model_b_name: str,
        samples: list[BenchmarkSample] | None = None,
        significance_threshold: float = 0.05,
    ) -> ABTestResult:
        """Compare two models with A/B testing.

        Args:
            model_a_name: Name of model A
            model_b_name: Name of model B
            samples: Samples to use (standard if None)
            significance_threshold: P-value threshold for significance

        Returns:
            ABTestResult with comparison
        """
        if samples is None:
            samples = StandardDataset.get_samples()

        # Run benchmarks for both models
        benchmark_a = QualityBenchmark(model_name=model_a_name)
        benchmark_b = QualityBenchmark(model_name=model_b_name)

        report_a = benchmark_a.run_benchmark(samples, f"ab_test_{model_a_name}")
        report_b = benchmark_b.run_benchmark(samples, f"ab_test_{model_b_name}")

        # Compare dimensions
        dimension_comparison: dict[str, dict[str, float]] = {}
        wins_a = 0
        wins_b = 0

        for dim in report_a.mean_scores:
            if dim in report_b.mean_scores:
                score_a = report_a.mean_scores[dim]
                score_b = report_b.mean_scores[dim]
                diff = score_b - score_a

                dimension_comparison[dim] = {
                    "model_a": score_a,
                    "model_b": score_b,
                    "difference": diff,
                }

                # Count wins (>2% difference)
                if diff > 0.02:
                    wins_b += 1
                elif diff < -0.02:
                    wins_a += 1

        # Determine winner
        total_dims = wins_a + wins_b
        if total_dims == 0:
            winner = None
            confidence = 0.5
        elif wins_a > wins_b:
            winner = model_a_name
            confidence = wins_a / total_dims
        elif wins_b > wins_a:
            winner = model_b_name
            confidence = wins_b / total_dims
        else:
            winner = None
            confidence = 0.5

        # Simple statistical test (paired t-test approximation)
        p_value = self._compute_p_value(
            list(report_a.mean_scores.values()),
            list(report_b.mean_scores.values()),
        )

        return ABTestResult(
            model_a=model_a_name,
            model_b=model_b_name,
            winner=winner if p_value and p_value < significance_threshold else None,
            confidence=confidence,
            dimension_comparison=dimension_comparison,
            sample_size=len(samples),
            p_value=p_value,
        )

    def _compute_p_value(self, scores_a: list[float], scores_b: list[float]) -> float | None:
        """Compute p-value for paired comparison."""
        if len(scores_a) != len(scores_b) or len(scores_a) < 3:
            return None

        try:
            # Simple paired difference test
            differences = [b - a for a, b in zip(scores_a, scores_b)]
            mean_diff = mean(differences)
            std_diff = stdev(differences) if len(differences) > 1 else 1.0

            if std_diff == 0:
                return 0.0 if mean_diff == 0 else 1.0

            # T-statistic
            n = len(differences)
            t_stat = mean_diff / (std_diff / (n**0.5))

            # Approximate p-value (two-tailed)
            # Using normal approximation for simplicity
            import math

            p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))
            return p_value
        except Exception as e:
            logger.warning("P-value computation failed: %s", e)
            return None


class HumanEvaluationIntegration:
    """Integration for human evaluation studies."""

    def __init__(self, output_dir: Path | None = None) -> None:
        """Initialize human evaluation integration.

        Args:
            output_dir: Directory for evaluation files
        """
        self._output_dir = output_dir or Path("evaluations")
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate_evaluation_batch(
        self,
        samples: list[BenchmarkSample],
        batch_name: str,
        include_predictions: bool = True,
    ) -> Path:
        """Generate a batch file for human evaluation.

        Args:
            samples: Samples to evaluate
            batch_name: Name for the batch
            include_predictions: Include model predictions for comparison

        Returns:
            Path to the generated evaluation file
        """
        evaluation_items = []

        for sample in samples:
            item = {
                "sample_id": sample.sample_id,
                "source": sample.source_text,
                "response": sample.response_text,
                "evaluation": {
                    "hallucination": None,  # 1-5 scale
                    "relevance": None,
                    "coherence": None,
                    "tone": None,
                    "overall": None,
                    "comments": "",
                },
            }

            if include_predictions and sample.expected_quality:
                item["predicted_quality"] = sample.expected_quality

            evaluation_items.append(item)

        output_path = self._output_dir / f"{batch_name}_evaluation.json"
        output_path.write_text(json.dumps(evaluation_items, indent=2))

        logger.info("Generated evaluation batch with %d items: %s", len(samples), output_path)

        return output_path

    def import_human_evaluations(self, evaluation_file: Path) -> list[tuple[str, dict[str, float]]]:
        """Import human evaluations from a completed file.

        Args:
            evaluation_file: Path to the completed evaluation file

        Returns:
            List of (sample_id, scores) tuples
        """
        data = json.loads(evaluation_file.read_text())
        results = []

        for item in data:
            sample_id = item["sample_id"]
            evaluation = item.get("evaluation", {})

            # Convert 1-5 scale to 0-1
            scores = {}
            for dim in ["hallucination", "relevance", "coherence", "tone", "overall"]:
                if evaluation.get(dim) is not None:
                    # Invert hallucination (higher = worse)
                    if dim == "hallucination":
                        scores[dim] = 1 - (evaluation[dim] - 1) / 4
                    else:
                        scores[dim] = (evaluation[dim] - 1) / 4

            if scores:
                results.append((sample_id, scores))

        logger.info("Imported %d human evaluations from %s", len(results), evaluation_file)

        return results

    def compute_inter_annotator_agreement(
        self, evaluations: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Compute inter-annotator agreement metrics.

        Args:
            evaluations: List of evaluation dicts from multiple annotators

        Returns:
            Agreement metrics per dimension
        """
        # This is a simplified version - would need multiple annotator data
        # Returns placeholder values
        return {
            "hallucination": 0.0,
            "relevance": 0.0,
            "coherence": 0.0,
            "tone": 0.0,
            "overall": 0.0,
        }


def main() -> None:
    """CLI entry point for quality benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="JARVIS Quality Benchmark")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run benchmark
    run_parser = subparsers.add_parser("run", help="Run benchmark")
    run_parser.add_argument(
        "--dataset",
        choices=["standard", "grounded", "hallucination"],
        default="standard",
        help="Dataset to use",
    )
    run_parser.add_argument("--model", default="default", help="Model name")
    run_parser.add_argument("--output", type=Path, help="Output file for report")

    # Compare models
    compare_parser = subparsers.add_parser("compare", help="Compare two models")
    compare_parser.add_argument("--model-a", required=True, help="First model")
    compare_parser.add_argument("--model-b", required=True, help="Second model")

    # Generate human evaluation batch
    eval_parser = subparsers.add_parser("generate-eval", help="Generate human evaluation batch")
    eval_parser.add_argument("--name", required=True, help="Batch name")
    eval_parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluations"), help="Output directory"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.command == "run":
        # Get samples
        if args.dataset == "standard":
            samples = StandardDataset.get_samples()
        elif args.dataset == "grounded":
            samples = StandardDataset.get_grounded_samples()
        else:
            samples = StandardDataset.get_hallucination_samples()

        # Run benchmark
        benchmark = QualityBenchmark(model_name=args.model)
        report = benchmark.run_benchmark(samples, benchmark_name=args.dataset)

        # Print summary
        print(f"\n=== Benchmark Report: {report.benchmark_name} ===")
        print(f"Model: {report.model_name}")
        print(f"Samples: {report.total_samples}")
        print(f"Passed: {report.passed_samples} ({report.pass_rate:.1%})")
        print("\nMean Scores:")
        for dim, score in report.mean_scores.items():
            print(f"  {dim}: {score:.3f}")
        print("\nLatency:")
        print(f"  Mean: {report.mean_latency_ms:.1f}ms")
        print(f"  P95: {report.p95_latency_ms:.1f}ms")

        # Save if output specified
        if args.output:
            report.save(args.output)

    elif args.command == "compare":
        framework = ABTestFramework()
        result = framework.compare_models(args.model_a, args.model_b)

        print(f"\n=== A/B Test: {args.model_a} vs {args.model_b} ===")
        print(f"Winner: {result.winner or 'Tie'}")
        print(f"Confidence: {result.confidence:.1%}")
        if result.p_value:
            print(f"P-value: {result.p_value:.4f}")
        print("\nDimension Comparison:")
        for dim, scores in result.dimension_comparison.items():
            a_score = scores["model_a"]
            b_score = scores["model_b"]
            diff = scores["difference"]
            print(f"  {dim}: A={a_score:.3f}, B={b_score:.3f}, diff={diff:+.3f}")

    elif args.command == "generate-eval":
        integration = HumanEvaluationIntegration(output_dir=args.output_dir)
        samples = StandardDataset.get_samples()
        output_path = integration.generate_evaluation_batch(samples, args.name)
        print(f"Generated evaluation batch: {output_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
