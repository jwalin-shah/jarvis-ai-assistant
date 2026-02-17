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
# Defaults are used if config load fails.
DEFAULT_LLM_MEMORY_LIMIT = 1024 * 1024 * 1024  # 1 GB for weights/working set
DEFAULT_LLM_CACHE_LIMIT = 512 * 1024 * 1024  # 512 MB for KV/cache

# Embedders, cross-encoders, and utility models are smaller.
DEFAULT_EMBEDDER_MEMORY_LIMIT = min(256 * 1024 * 1024, int(_total_ram * 0.05))
DEFAULT_EMBEDDER_CACHE_LIMIT = min(128 * 1024 * 1024, int(_total_ram * 0.025))


def _mb_to_bytes(value_mb: int) -> int:
    return max(1, value_mb) * 1024 * 1024


def _resolve_limits() -> tuple[int, int, int, int]:
    try:
        from jarvis.config import get_config

        model_cfg = get_config().model
        llm_memory = _mb_to_bytes(model_cfg.llm_memory_limit_mb)
        llm_cache = _mb_to_bytes(model_cfg.llm_cache_limit_mb)
        embedder_memory = _mb_to_bytes(model_cfg.embedder_memory_limit_mb)
        embedder_cache = _mb_to_bytes(model_cfg.embedder_cache_limit_mb)
        return llm_memory, llm_cache, embedder_memory, embedder_cache
    except Exception:
        return (
            DEFAULT_LLM_MEMORY_LIMIT,
            DEFAULT_LLM_CACHE_LIMIT,
            DEFAULT_EMBEDDER_MEMORY_LIMIT,
            DEFAULT_EMBEDDER_CACHE_LIMIT,
        )


def _get_gpu_lock() -> threading.Lock:
    """Get the shared MLX GPU lock from MLXModelLoader.

    Lazy import to avoid circular dependency (loader imports memory_config).
    """
    from models.loader import MLXModelLoader

    return MLXModelLoader._mlx_load_lock


def apply_llm_limits() -> None:
    """Apply MLX memory limits for LLM model loading."""
    import mlx.core as mx

    llm_memory, llm_cache, _, _ = _resolve_limits()
    mx.set_memory_limit(llm_memory)
    mx.set_cache_limit(llm_cache)


def apply_embedder_limits() -> None:
    """Apply MLX memory limits for embedder/utility model loading."""
    import mlx.core as mx

    _, _, embedder_memory, embedder_cache = _resolve_limits()
    mx.set_memory_limit(embedder_memory)
    mx.set_cache_limit(embedder_cache)


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
