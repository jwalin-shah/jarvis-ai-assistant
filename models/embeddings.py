"""MLX Embedding constants and exceptions.

IMPORTANT: Most code should NOT import from this module directly.
Instead, use the unified interface in jarvis/embedding_adapter.py:

    from jarvis.embedding_adapter import get_embedder
    embedder = get_embedder()

This module provides:
- Exception classes for embedding errors
- Constants (embedding dimension, default model)
- is_mlx_available() check
"""

from __future__ import annotations

import logging

from jarvis.errors import ErrorCode, JarvisError

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Embedding dimension is 384 for all supported models
MLX_EMBEDDING_DIM = 384

# Legacy constant for backwards compatibility
DEFAULT_MLX_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# =============================================================================
# Exceptions
# =============================================================================


class MLXEmbeddingError(JarvisError):
    """Raised when MLX embedding operations fail."""

    default_message = "MLX embedding operation failed"
    default_code = ErrorCode.MDL_LOAD_FAILED


class MLXServiceNotAvailableError(MLXEmbeddingError):
    """Raised when MLX embedding is not available."""

    default_message = "MLX embedding is not available"


class MLXModelLoadError(MLXEmbeddingError):
    """Raised when MLX embedding model fails to load."""

    default_message = "Failed to load MLX embedding model"


# For backwards compatibility
MLXModelNotAvailableError = MLXServiceNotAvailableError


# =============================================================================
# Availability Check
# =============================================================================


def is_mlx_available() -> bool:
    """Check if in-process MLX embeddings are available.

    Returns:
        True if MLX can be imported (always True on Apple Silicon).
    """
    try:
        import mlx.core  # noqa: F401

        return True
    except ImportError:
        return False


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    "DEFAULT_MLX_EMBEDDING_MODEL",
    "MLX_EMBEDDING_DIM",
    # Exceptions
    "MLXEmbeddingError",
    "MLXServiceNotAvailableError",
    "MLXModelNotAvailableError",  # backwards compat
    "MLXModelLoadError",
    # Functions
    "is_mlx_available",
]
