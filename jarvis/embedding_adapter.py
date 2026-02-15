"""Unified Embedding Adapter - Minimal proxy layer.

Moved canonical implementation to models/bert_embedder.py to break circular
imports with models/templates.py.
"""

from __future__ import annotations

from models.bert_embedder import (
    DEFAULT_MLX_EMBEDDING_MODEL,
    EMBEDDING_MODEL_REGISTRY,
    MLX_EMBEDDING_DIM,
    CachedEmbedder,
    MLXEmbedder,
    get_configured_model_name,
    get_embedder,
    get_model_info,
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
