"""Embedding system for JARVIS v2.

Provides:
- Message indexing and similarity search
- Style learning from your past messages
- Cached embeddings for performance
"""

from .cache import EmbeddingCache, get_embedding_cache, content_hash
from .model import EmbeddingModel, get_embedding_model
from .similarity import (
    cosine_similarity,
    find_most_similar,
    semantic_match,
    batch_cosine_similarity,
)
from .store import (
    EmbeddingStore,
    get_embedding_store,
    SimilarMessage,
    ConversationContext,
    StyleProfile,
)
from .indexer import MessageIndexer, run_indexing, IndexingStats

__all__ = [
    # Store (main interface)
    "EmbeddingStore",
    "get_embedding_store",
    "SimilarMessage",
    "ConversationContext",
    "StyleProfile",
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
