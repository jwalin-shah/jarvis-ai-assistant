"""Latency benchmarking (Workstream 4).

This module provides latency measurement tools for MLX models.

Key components:
- MLXLatencyBenchmarker: Main benchmarker implementing LatencyBenchmarker protocol
- HighPrecisionTimer: Nanosecond-precision timer for accurate measurements
- LatencyScenario: Definitions for cold, warm, and hot start scenarios

Usage:
    python -m benchmarks.latency.run --output results/latency.json

Note: The MLXLatencyBenchmarker requires Apple Silicon with MLX installed.
Timer utilities and scenario definitions work on all platforms.
"""

from benchmarks.latency.scenarios import (
    DEFAULT_MAX_TOKENS,
    LatencyScenario,
    Scenario,
    get_benchmark_prompts,
    get_default_scenarios,
    get_scenario_by_type,
)
from benchmarks.latency.timer import (
    HighPrecisionTimer,
    TimingResult,
    force_model_unload,
    measure_operation,
    timed_operation,
    warmup_timer,
)

# Conditional import of MLXLatencyBenchmarker (requires Apple Silicon)
try:
    from benchmarks.latency.run import MLXLatencyBenchmarker

    HAS_MLX_BENCHMARKER = True
except ImportError:
    MLXLatencyBenchmarker = None  # type: ignore[misc, assignment]
    HAS_MLX_BENCHMARKER = False

__all__ = [
    # Main benchmarker (may be None if MLX unavailable)
    "MLXLatencyBenchmarker",
    "HAS_MLX_BENCHMARKER",
    # Timer utilities
    "HighPrecisionTimer",
    "TimingResult",
    "timed_operation",
    "measure_operation",
    "force_model_unload",
    "warmup_timer",
    # Scenario definitions
    "Scenario",
    "LatencyScenario",
    "DEFAULT_MAX_TOKENS",
    "get_default_scenarios",
    "get_scenario_by_type",
    "get_benchmark_prompts",
]
