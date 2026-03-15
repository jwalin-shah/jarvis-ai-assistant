"""Latency benchmarking (Workstream 4).  # noqa: E501
  # noqa: E501
This module provides latency measurement tools for MLX models.  # noqa: E501
  # noqa: E501
Key components:  # noqa: E501
- MLXLatencyBenchmarker: Main benchmarker implementing LatencyBenchmarker protocol  # noqa: E501
- HighPrecisionTimer: Nanosecond-precision timer for accurate measurements  # noqa: E501
- LatencyScenario: Definitions for cold, warm, and hot start scenarios  # noqa: E501
  # noqa: E501
Usage:  # noqa: E501
    python -m benchmarks.latency.run --output results/latency.json  # noqa: E501
  # noqa: E501
Note: The MLXLatencyBenchmarker requires Apple Silicon with MLX installed.  # noqa: E501
Timer utilities and scenario definitions work on all platforms.  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from evals.benchmarks.latency.scenarios import (  # noqa: E501
    DEFAULT_MAX_TOKENS,  # noqa: E501
    LatencyScenario,  # noqa: E501
    Scenario,  # noqa: E501
    get_benchmark_prompts,  # noqa: E501
    get_default_scenarios,  # noqa: E501
    get_scenario_by_type,  # noqa: E501
)  # noqa: E501
from evals.benchmarks.latency.timer import (  # noqa: E501
    HighPrecisionTimer,  # noqa: E501
    TimingResult,  # noqa: E501
    force_model_unload,  # noqa: E501
    measure_operation,  # noqa: E501
    timed_operation,  # noqa: E501
    warmup_timer,  # noqa: E501
)  # noqa: E501

  # noqa: E501
# Conditional import of MLXLatencyBenchmarker (requires Apple Silicon)  # noqa: E501
try:  # noqa: E501
    from evals.benchmarks.latency.run import MLXLatencyBenchmarker  # noqa: E501
  # noqa: E501
    HAS_MLX_BENCHMARKER = True  # noqa: E501
except ImportError:  # noqa: E501
    MLXLatencyBenchmarker = None  # type: ignore[misc, assignment]  # noqa: E501
    HAS_MLX_BENCHMARKER = False  # noqa: E501
  # noqa: E501
__all__ = [  # noqa: E501
    # Main benchmarker (may be None if MLX unavailable)  # noqa: E501
    "MLXLatencyBenchmarker",  # noqa: E501
    "HAS_MLX_BENCHMARKER",  # noqa: E501
    # Timer utilities  # noqa: E501
    "HighPrecisionTimer",  # noqa: E501
    "TimingResult",  # noqa: E501
    "timed_operation",  # noqa: E501
    "measure_operation",  # noqa: E501
    "force_model_unload",  # noqa: E501
    "warmup_timer",  # noqa: E501
    # Scenario definitions  # noqa: E501
    "Scenario",  # noqa: E501
    "LatencyScenario",  # noqa: E501
    "DEFAULT_MAX_TOKENS",  # noqa: E501
    "get_default_scenarios",  # noqa: E501
    "get_scenario_by_type",  # noqa: E501
    "get_benchmark_prompts",  # noqa: E501
]  # noqa: E501
