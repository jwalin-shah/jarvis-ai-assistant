"""Latency benchmarking interfaces.

Workstream 4 implements against these contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    pass

Scenario = Literal["cold", "warm", "hot"]


@dataclass
class LatencyResult:
    """Result of a single latency measurement.

    Attributes:
        scenario: Test scenario (cold/warm/hot start).
        model_name: Name/identifier of the model.
        context_length: Input context length in tokens.
        output_tokens: Number of tokens generated.
        load_time_ms: Model loading time in milliseconds.
        prefill_time_ms: Context prefill time in milliseconds.
        generation_time_ms: Token generation time in milliseconds.
        total_time_ms: Total time from start to finish in milliseconds.
        tokens_per_second: Generation throughput in tokens/second.
        timestamp: ISO format timestamp of measurement.
    """

    scenario: Scenario
    model_name: str
    context_length: int
    output_tokens: int
    load_time_ms: float
    prefill_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    tokens_per_second: float
    timestamp: str

    def __post_init__(self) -> None:
        """Validate field constraints."""
        if self.context_length < 0:
            msg = f"context_length must be >= 0, got {self.context_length}"
            raise ValueError(msg)
        if self.output_tokens < 0:
            msg = f"output_tokens must be >= 0, got {self.output_tokens}"
            raise ValueError(msg)
        if self.total_time_ms < 0:
            msg = f"total_time_ms must be >= 0, got {self.total_time_ms}"
            raise ValueError(msg)


@dataclass
class LatencyBenchmarkResult:
    """Aggregate latency benchmark results.

    Attributes:
        scenario: Test scenario (cold/warm/hot start).
        model_name: Name/identifier of the model.
        num_runs: Number of benchmark runs performed.
        p50_ms: 50th percentile latency in milliseconds.
        p95_ms: 95th percentile latency in milliseconds.
        p99_ms: 99th percentile latency in milliseconds.
        mean_ms: Mean latency in milliseconds.
        std_ms: Standard deviation of latency in milliseconds.
        results: Individual measurement results.
        timestamp: ISO format timestamp of benchmark run.
    """

    scenario: Scenario
    model_name: str
    num_runs: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    results: list[LatencyResult]
    timestamp: str

    def __post_init__(self) -> None:
        """Validate field constraints."""
        if self.num_runs != len(self.results):
            msg = f"num_runs ({self.num_runs}) != len(results) ({len(self.results)})"
            raise ValueError(msg)
        if self.num_runs < 1:
            msg = f"num_runs must be >= 1, got {self.num_runs}"
            raise ValueError(msg)


class LatencyBenchmarker(Protocol):
    """Interface for latency benchmarking (Workstream 4)."""

    def measure_single(
        self, model_path: str, scenario: Scenario, prompt: str, max_tokens: int
    ) -> LatencyResult:
        """Measure latency for a single generation.

        Args:
            model_path: Path to the model to benchmark.
            scenario: Test scenario (cold/warm/hot start).
            prompt: Input prompt for generation.
            max_tokens: Maximum tokens to generate.

        Returns:
            Single latency measurement result.
        """
        ...

    def run_benchmark(
        self, model_path: str, scenario: Scenario, num_runs: int = 10
    ) -> LatencyBenchmarkResult:
        """Run full benchmark with statistical analysis.

        Args:
            model_path: Path to the model to benchmark.
            scenario: Test scenario (cold/warm/hot start).
            num_runs: Number of benchmark runs to perform.

        Returns:
            Aggregate benchmark results with percentile statistics.
        """
        ...
