"""Embedding cache for JARVIS v2.

Provides fast, cached embeddings for semantic search and template matching.
"""

from .cache import EmbeddingCache, get_embedding_cache, content_hash
from .model import EmbeddingModel, get_embedding_model
from .similarity import (
    cosine_similarity,
    find_most_similar,
    semantic_match,
    batch_cosine_similarity,
)

__all__ = [
    # Cache
    "EmbeddingCache",
    "get_embedding_cache",
    "content_hash",
    # Model
    "EmbeddingModel",
    "get_embedding_model",
    # Similarity
    "cosine_similarity",
    "find_most_similar",
    "semantic_match",
    "batch_cosine_similarity",
]
