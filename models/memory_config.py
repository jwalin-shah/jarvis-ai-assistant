"""Centralized MLX memory configuration for Apple Silicon systems.

All MLX memory limits should be set through this module to avoid
inconsistent settings across different model loaders.

Limits scale adaptively based on available system RAM.
"""

import psutil

_total_ram = psutil.virtual_memory().total

# Memory limits for different model types (in bytes)
# Scale with available RAM, capped at reasonable maximums
LLM_MEMORY_LIMIT = min(2 * 1024 * 1024 * 1024, int(_total_ram * 0.25))
LLM_CACHE_LIMIT = min(1 * 1024 * 1024 * 1024, int(_total_ram * 0.125))

# Embedders, cross-encoders, and utility models are smaller
EMBEDDER_MEMORY_LIMIT = min(1 * 1024 * 1024 * 1024, int(_total_ram * 0.125))
EMBEDDER_CACHE_LIMIT = min(512 * 1024 * 1024, int(_total_ram * 0.0625))


def apply_llm_limits() -> None:
    """Apply MLX memory limits for LLM model loading."""
    import mlx.core as mx

    mx.set_memory_limit(LLM_MEMORY_LIMIT)
    mx.set_cache_limit(LLM_CACHE_LIMIT)


def apply_embedder_limits() -> None:
    """Apply MLX memory limits for embedder/utility model loading."""
    import mlx.core as mx

    mx.set_memory_limit(EMBEDDER_MEMORY_LIMIT)
    mx.set_cache_limit(EMBEDDER_CACHE_LIMIT)
