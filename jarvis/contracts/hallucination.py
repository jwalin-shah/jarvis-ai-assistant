"""Hallucination evaluation interfaces.

Workstream 2 implements against these contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class HHEMResult:
    """Result of evaluating a single source/summary pair.

    Attributes:
        model_name: Name/identifier of the model being evaluated.
        prompt_template: Prompt template used for generation.
        source_text: Original source text.
        generated_summary: Generated summary to evaluate.
        hhem_score: HHEM score (0.0 = hallucinated, 1.0 = grounded).
        timestamp: ISO format timestamp of evaluation.
    """

    model_name: str
    prompt_template: str
    source_text: str
    generated_summary: str
    hhem_score: float
    timestamp: str

    def __post_init__(self) -> None:
        """Validate field constraints."""
        if not 0.0 <= self.hhem_score <= 1.0:
            msg = f"HHEM score must be between 0.0 and 1.0, got {self.hhem_score}"
            raise ValueError(msg)


@dataclass
class HHEMBenchmarkResult:
    """Aggregate results from a benchmark run.

    Attributes:
        model_name: Name/identifier of the model being evaluated.
        num_samples: Number of samples evaluated.
        mean_score: Mean HHEM score across all samples.
        median_score: Median HHEM score across all samples.
        std_score: Standard deviation of HHEM scores.
        pass_rate_at_05: Percentage of samples with score >= 0.5.
        pass_rate_at_07: Percentage of samples with score >= 0.7.
        results: Individual evaluation results.
        timestamp: ISO format timestamp of benchmark run.
    """

    model_name: str
    num_samples: int
    mean_score: float
    median_score: float
    std_score: float
    pass_rate_at_05: float
    pass_rate_at_07: float
    results: list[HHEMResult]
    timestamp: str

    def __post_init__(self) -> None:
        """Validate field constraints."""
        if self.num_samples != len(self.results):
            msg = f"num_samples ({self.num_samples}) != len(results) ({len(self.results)})"
            raise ValueError(msg)
        if self.num_samples < 1:
            msg = f"num_samples must be >= 1, got {self.num_samples}"
            raise ValueError(msg)
        if not 0.0 <= self.pass_rate_at_05 <= 100.0:
            msg = f"pass_rate_at_05 must be 0-100, got {self.pass_rate_at_05}"
            raise ValueError(msg)
        if not 0.0 <= self.pass_rate_at_07 <= 100.0:
            msg = f"pass_rate_at_07 must be 0-100, got {self.pass_rate_at_07}"
            raise ValueError(msg)


class HallucinationEvaluator(Protocol):
    """Interface for hallucination evaluation (Workstream 2)."""

    def evaluate_single(self, source: str, summary: str) -> float:
        """Return HHEM score for a source/summary pair.

        Args:
            source: Original source text.
            summary: Generated summary to evaluate.

        Returns:
            HHEM score between 0.0 (hallucinated) and 1.0 (grounded).
        """
        ...

    def evaluate_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Batch evaluate multiple pairs. More efficient than single calls.

        Args:
            pairs: List of (source, summary) tuples.

        Returns:
            List of HHEM scores, one per input pair.
        """
        ...

    def run_benchmark(
        self, model_name: str, dataset_path: str, prompt_templates: list[str]
    ) -> HHEMBenchmarkResult:
        """Run full benchmark and return aggregate results.

        Args:
            model_name: Name/identifier of the model being benchmarked.
            dataset_path: Path to benchmark dataset file.
            prompt_templates: List of prompt templates to test.

        Returns:
            Aggregate benchmark results with statistics.
        """
        ...
