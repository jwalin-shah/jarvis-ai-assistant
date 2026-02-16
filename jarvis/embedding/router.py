"""Embedding backend router - selects MLX (GPU) or CPU based on context.

Routing strategies:
- "auto": Dynamic - use MLX if GPU free, else CPU
- "mlx": Always use MLX (GPU)
- "cpu": Always use CPU
- "backfill": Use MLX (GPU only doing LLM, embeddings benefit from speed)
- "parallel": Use CPU (to parallelize with GPU LLM)

Usage:
    from jarvis.embedding.router import get_embedder_for_context

    # During backfill (GPU only doing LLM)
    embedder = get_embedder_for_context("backfill")

    # During full pipeline (segmentation parallel to LLM)
    embedder = get_embedder_for_context("parallel")
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from jarvis.embedding.cpu_embedder import get_cpu_embedder

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class EmbedderRouter:
    """Routes embedding requests to appropriate backend."""

    def __init__(self):
        self._mlx_embedder = None
        self._cpu_embedder = None
        self._lock = threading.Lock()

    def _get_mlx_embedder(self):
        """Lazy load MLX embedder."""
        if self._mlx_embedder is None:
            from jarvis.embedding_adapter import get_embedder as get_mlx

            self._mlx_embedder = get_mlx
        return self._mlx_embedder

    def _get_cpu_embedder(self):
        """Lazy load CPU embedder."""
        if self._cpu_embedder is None:
            self._cpu_embedder = get_cpu_embedder()
        return self._cpu_embedder

    def get_embedder(self, context: str = "auto"):
        """Get embedder based on context.

        Args:
            context: Routing strategy
                - "auto": Dynamic selection
                - "mlx": Force MLX/GPU
                - "cpu": Force CPU
                - "backfill": Use MLX (fastest when GPU available)
                - "parallel": Use CPU (for parallel processing)

        Returns:
            Embedder with encode() method
        """
        if context == "cpu":
            cpu = self._get_cpu_embedder()
            if cpu:
                return cpu
            logger.warning("CPU embedder not available, falling back to MLX")
            return self._get_mlx_embedder()

        elif context == "mlx" or context == "backfill":
            # Backfill: GPU is doing LLM, but embeddings are fast with MLX
            # Still use MLX because fact embeddings happen after LLM
            return self._get_mlx_embedder()

        elif context == "parallel":
            # Explicitly want parallel processing with GPU LLM
            cpu = self._get_cpu_embedder()
            if cpu:
                return cpu
            logger.warning("CPU embedder not available, falling back to MLX")
            return self._get_mlx_embedder()

        elif context == "auto":
            # Dynamic: Check if GPU is busy
            if self._is_gpu_free():
                return self._get_mlx_embedder()
            else:
                cpu = self._get_cpu_embedder()
                if cpu:
                    return cpu
                return self._get_mlx_embedder()

        else:
            raise ValueError(f"Unknown context: {context}")

    def _is_gpu_free(self) -> bool:
        """Check if GPU is currently available.

        Tries to acquire the MLX lock without blocking.
        """
        try:
            from models.loader import MLXModelLoader

            acquired = MLXModelLoader._mlx_load_lock.acquire(blocking=False)
            if acquired:
                MLXModelLoader._mlx_load_lock.release()
                return True
            return False
        except Exception:
            return True  # Assume free if can't check


# Global router instance
_router = EmbedderRouter()


def get_embedder_for_context(context: str = "auto"):
    """Get embedder for specific context.

    Args:
        context: "auto", "mlx", "cpu", "backfill", or "parallel"

    Returns:
        Embedder instance
    """
    return _router.get_embedder(context)


def get_parallel_embedder():
    """Get CPU embedder for parallel processing with GPU.

    Use this when you want embeddings to run in parallel
    with GPU LLM generation.
    """
    return _router.get_embedder("parallel")
