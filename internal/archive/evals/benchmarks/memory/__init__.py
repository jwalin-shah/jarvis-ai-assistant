"""Memory profiling benchmark (Workstream 1).  # noqa: E501
  # noqa: E501
This module provides memory profiling for MLX models to validate  # noqa: E501
that the model stack fits within memory constraints.  # noqa: E501
  # noqa: E501
Example usage:  # noqa: E501
    from evals.benchmarks.memory import MLXMemoryProfiler, get_default_model  # noqa: E501
  # noqa: E501
    profiler = MLXMemoryProfiler()  # noqa: E501
    model = get_default_model()  # noqa: E501
    profile = profiler.profile_model(model.path, context_length=512)  # noqa: E501
    print(f"RSS: {profile.rss_mb} MB")  # noqa: E501
  # noqa: E501
CLI usage:  # noqa: E501
    python -m benchmarks.memory.run --output results/memory.json  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from evals.benchmarks.memory.models import (  # noqa: E501
    CONTEXT_LENGTHS,  # noqa: E501
    ModelSpec,  # noqa: E501
    get_context_lengths,  # noqa: E501
    get_default_model,  # noqa: E501
    get_models_for_profiling,  # noqa: E501
)  # noqa: E501
from evals.benchmarks.memory.profiler import MLXMemoryProfiler  # noqa: E402  # noqa: E501

  # noqa: E501
__all__ = [  # noqa: E501
    "MLXMemoryProfiler",  # noqa: E501
    "ModelSpec",  # noqa: E501
    "CONTEXT_LENGTHS",  # noqa: E501
    "get_default_model",  # noqa: E501
    "get_models_for_profiling",  # noqa: E501
    "get_context_lengths",  # noqa: E501
]  # noqa: E501
