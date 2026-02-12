"""Centralized MLX memory configuration for 8GB Apple Silicon systems.

All MLX memory limits should be set through this module to avoid
inconsistent settings across different model loaders.
"""

# Memory limits for different model types (in bytes)
# LLM models are larger and need more headroom
LLM_MEMORY_LIMIT = 2 * 1024 * 1024 * 1024  # 2 GB
LLM_CACHE_LIMIT = 1 * 1024 * 1024 * 1024  # 1 GB

# Embedders, cross-encoders, and utility models are smaller
EMBEDDER_MEMORY_LIMIT = 1 * 1024 * 1024 * 1024  # 1 GB
EMBEDDER_CACHE_LIMIT = 512 * 1024 * 1024  # 512 MB


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
