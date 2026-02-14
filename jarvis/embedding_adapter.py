"""Unified Embedding Adapter - Minimal proxy layer.

Moved canonical implementation to models/bert_embedder.py to break circular
imports with models/templates.py.
"""

from __future__ import annotations

from models.bert_embedder import (
    EMBEDDING_MODEL_REGISTRY,
    MLX_EMBEDDING_DIM,
    DEFAULT_MLX_EMBEDDING_MODEL,
    CachedEmbedder,
    MLXEmbedder,
    get_embedder,
    get_model_info,
    get_configured_model_name,
    is_embedder_available,
    reset_embedder,
)

# Legacy aliases for backward compatibility
EMBEDDING_DIM = MLX_EMBEDDING_DIM
EMBEDDING_MODEL = DEFAULT_MLX_EMBEDDING_MODEL
UnifiedEmbedder = MLXEmbedder

__all__ = [
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL_REGISTRY",
    "get_model_info",
    "get_configured_model_name",
    "MLXEmbedder",
    "UnifiedEmbedder",
    "CachedEmbedder",
    "get_embedder",
    "reset_embedder",
    "is_embedder_available",
]
