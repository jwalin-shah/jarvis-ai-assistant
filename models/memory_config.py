"""Centralized MLX memory configuration for Apple Silicon systems.

All MLX memory limits should be set through this module to avoid
inconsistent settings across different model loaders.

Limits scale adaptively based on available system RAM.
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from contextlib import contextmanager

import psutil

_total_ram = psutil.virtual_memory().total

# Memory limits for different model types (in bytes)
# Scale with available RAM, capped at reasonable maximums
LLM_MEMORY_LIMIT = min(2 * 1024 * 1024 * 1024, int(_total_ram * 0.25))
LLM_CACHE_LIMIT = min(1 * 1024 * 1024 * 1024, int(_total_ram * 0.125))

# Embedders, cross-encoders, and utility models are smaller
EMBEDDER_MEMORY_LIMIT = min(1 * 1024 * 1024 * 1024, int(_total_ram * 0.125))
EMBEDDER_CACHE_LIMIT = min(512 * 1024 * 1024, int(_total_ram * 0.0625))


def _get_gpu_lock() -> threading.Lock:
    """Get the shared MLX GPU lock from MLXModelLoader.

    Lazy import to avoid circular dependency (loader imports memory_config).
    """
    from models.loader import MLXModelLoader

    return MLXModelLoader._mlx_load_lock


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


@contextmanager
def gpu_context(*, embedder: bool = True) -> Generator[None, None, None]:
    """Acquire the shared GPU lock and apply appropriate memory limits.

    Combines two operations that are always done together:
    1. Acquire MLXModelLoader._mlx_load_lock to serialize Metal GPU access
    2. Apply memory limits (embedder or LLM) before any GPU operations

    Args:
        embedder: If True (default), apply embedder memory limits.
                  If False, apply LLM memory limits.

    Usage:
        from models.memory_config import gpu_context

        with gpu_context():
            # Embedder/cross-encoder GPU operations here
            ...

        with gpu_context(embedder=False):
            # LLM GPU operations here
            ...
    """
    with _get_gpu_lock():
        if embedder:
            apply_embedder_limits()
        else:
            apply_llm_limits()
        yield
