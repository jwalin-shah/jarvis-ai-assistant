"""Model configurations for memory profiling.  # noqa: E501
  # noqa: E501
Workstream 1: Memory Profiler  # noqa: E501
  # noqa: E501
Defines the models and context lengths to profile for memory benchmarking.  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from dataclasses import dataclass  # noqa: E402  # noqa: E501


  # noqa: E501
  # noqa: E501
@dataclass(frozen=True)  # noqa: E501
class ModelSpec:  # noqa: E501
    """Specification for a model to profile.  # noqa: E501
  # noqa: E501
    Attributes:  # noqa: E501
        path: HuggingFace model path or local path  # noqa: E501
        name: Human-readable name  # noqa: E501
        estimated_memory_mb: Estimated memory usage for planning  # noqa: E501
        description: Brief description of the model  # noqa: E501
    """  # noqa: E501
  # noqa: E501
    path: str  # noqa: E501
    name: str  # noqa: E501
    estimated_memory_mb: float  # noqa: E501
    description: str  # noqa: E501
  # noqa: E501
  # noqa: E501
# Context lengths to test (as specified in requirements)  # noqa: E501
CONTEXT_LENGTHS: list[int] = [512, 1024, 2048, 4096]  # noqa: E501
  # noqa: E501
  # noqa: E501
# Default model (from project configuration)  # noqa: E501
DEFAULT_MODEL = ModelSpec(  # noqa: E501
    path="mlx-community/Qwen2.5-0.5B-Instruct-4bit",  # noqa: E501
    name="Qwen2.5-0.5B-4bit",  # noqa: E501
    estimated_memory_mb=800,  # noqa: E501
    description="Default JARVIS model - Qwen 0.5B with 4-bit quantization",  # noqa: E501
)  # noqa: E501
  # noqa: E501
  # noqa: E501
# Models to profile for benchmarking  # noqa: E501
# Start with smaller models to validate profiler, then expand  # noqa: E501
BENCHMARK_MODELS: list[ModelSpec] = [  # noqa: E501
    DEFAULT_MODEL,  # noqa: E501
    # Additional models can be added here as needed  # noqa: E501
    # ModelSpec(  # noqa: E501
    #     path="mlx-community/Qwen2.5-1.5B-Instruct-4bit",  # noqa: E501
    #     name="Qwen2.5-1.5B-4bit",  # noqa: E501
    #     estimated_memory_mb=1500,  # noqa: E501
    #     description="Qwen 1.5B with 4-bit quantization",  # noqa: E501
    # ),  # noqa: E501
    # ModelSpec(  # noqa: E501
    #     path="mlx-community/Qwen2.5-3B-Instruct-4bit",  # noqa: E501
    #     name="Qwen2.5-3B-4bit",  # noqa: E501
    #     estimated_memory_mb=2500,  # noqa: E501
    #     description="Qwen 3B with 4-bit quantization",  # noqa: E501
    # ),  # noqa: E501
]  # noqa: E501
  # noqa: E501
  # noqa: E501
def get_models_for_profiling() -> list[ModelSpec]:  # noqa: E501
    """Return list of models to profile.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        List of ModelSpec instances  # noqa: E501
    """  # noqa: E501
    return list(BENCHMARK_MODELS)  # noqa: E501
  # noqa: E501
  # noqa: E501
def get_context_lengths() -> list[int]:  # noqa: E501
    """Return list of context lengths to test.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        List of context length values  # noqa: E501
    """  # noqa: E501
    return list(CONTEXT_LENGTHS)  # noqa: E501
  # noqa: E501
  # noqa: E501
def get_default_model() -> ModelSpec:  # noqa: E501
    """Return the default model for quick profiling.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        Default ModelSpec  # noqa: E501
    """  # noqa: E501
    return DEFAULT_MODEL  # noqa: E501
