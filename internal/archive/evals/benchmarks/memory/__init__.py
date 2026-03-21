"""Memory profiling benchmark (Workstream 1).

This module provides memory profiling for MLX models to validate
that the model stack fits within memory constraints.

Example usage:
    from evals.benchmarks.memory import MLXMemoryProfiler, get_default_model

    profiler = MLXMemoryProfiler()
    model = get_default_model()
    profile = profiler.profile_model(model.path, context_length=512)
    print(f"RSS: {profile.rss_mb} MB")

CLI usage:
    python -m benchmarks.memory.run --output results/memory.json
"""

from evals.benchmarks.memory.models import (  # noqa: E402
    CONTEXT_LENGTHS,  # noqa: E402
    ModelSpec,
    get_context_lengths,
    get_default_model,
    get_models_for_profiling,
)
from evals.benchmarks.memory.profiler import MLXMemoryProfiler  # noqa: E402

__all__ = [
    "MLXMemoryProfiler",
    "ModelSpec",
    "CONTEXT_LENGTHS",
    "get_default_model",
    "get_models_for_profiling",
    "get_context_lengths",
]
