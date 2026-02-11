"""Repository for search-related data access.

Wraps ``jarvis.search.vec_search.VecSearcher`` (sqlite-vec backed) and
``jarvis.search.semantic_search.EmbeddingCache`` (SQLite blob cache)
behind a unified interface.  Callers are NOT migrated yet -- this is the
extraction step only.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from jarvis.repositories.base import BaseRepository
from jarvis.search.vec_search import VecSearcher, VecSearchResult

if TYPE_CHECKING:
    from contracts.imessage import Message
    from jarvis.search.semantic_search import EmbeddingCache

logger = logging.getLogger(__name__)


class SearchRepository(BaseRepository):
    """Data access for vector search and embedding cache.

    Delegates to ``VecSearcher`` for sqlite-vec operations and exposes
    ``EmbeddingCache`` accessors for the legacy blob-based cache.
    """

    def __init__(
        self,
        vec_searcher: VecSearcher | None = None,
        embedding_cache: EmbeddingCache | None = None,
    ) -> None:
        super().__init__(db=vec_searcher.db if vec_searcher else None)
        self._vec_searcher = vec_searcher
        self._embedding_cache = embedding_cache

    # ------------------------------------------------------------------
    # Lazy accessors
    # ------------------------------------------------------------------

    @property
    def vec(self) -> VecSearcher:
        if self._vec_searcher is None:
            from jarvis.search.vec_search import get_vec_searcher

            self._vec_searcher = get_vec_searcher()
        return self._vec_searcher

    @property
    def cache(self) -> EmbeddingCache:
        if self._embedding_cache is None:
            from jarvis.search.semantic_search import EmbeddingCache

            self._embedding_cache = EmbeddingCache()
        return self._embedding_cache

    # ------------------------------------------------------------------
    # Vector search (wraps VecSearcher)
    # ------------------------------------------------------------------

    def search_messages(
        self, query: str, chat_id: str | None = None, limit: int = 10
    ) -> list[VecSearchResult]:
        """Semantic search over indexed messages."""
        return self.vec.search(query, chat_id=chat_id, limit=limit)

    def search_with_pairs(
        self,
        query: str,
        limit: int = 5,
        response_type: str | None = None,
        contact_id: int | None = None,
        embedder: Any | None = None,
    ) -> list[VecSearchResult]:
        """Search conversation pairs/chunks for RAG."""
        return self.vec.search_with_pairs(
            query,
            limit=limit,
            response_type=response_type,
            contact_id=contact_id,
            embedder=embedder,
        )

    def search_with_pairs_global(
        self,
        query: str,
        limit: int = 5,
        embedder: Any | None = None,
    ) -> list[VecSearchResult]:
        """Two-phase global search: hamming pre-filter then int8 re-rank."""
        return self.vec.search_with_pairs_global(query, limit=limit, embedder=embedder)

    def index_message(self, message: Message) -> bool:
        """Index a single message into vec_messages."""
        return self.vec.index_message(message)

    def index_messages(self, messages: list[Message]) -> int:
        """Batch-index messages."""
        return self.vec.index_messages(messages)

    def get_vec_stats(self) -> dict[str, Any]:
        """Get vector index statistics."""
        return self.vec.get_stats()

    # ------------------------------------------------------------------
    # Embedding cache (wraps EmbeddingCache)
    # ------------------------------------------------------------------

    def get_cached_embedding(self, message_id: int) -> Any:
        """Get a single cached embedding by message id."""
        return self.cache.get(message_id)

    def get_cached_embeddings_batch(self, message_ids: list[int]) -> dict[int, Any]:
        """Batch-fetch cached embeddings."""
        return self.cache.get_batch(message_ids)

    def cache_embedding(
        self, message_id: int, chat_id: str, text_hash: str, embedding: Any
    ) -> None:
        """Store an embedding in the cache."""
        self.cache.set(message_id, chat_id, text_hash, embedding)

    def cache_embeddings_batch(self, items: list[tuple[int, str, str, Any]]) -> None:
        """Batch-store embeddings."""
        self.cache.set_batch(items)

    def invalidate_embedding(self, message_id: int) -> None:
        """Remove a cached embedding."""
        self.cache.invalidate(message_id)

    def invalidate_chat_embeddings(self, chat_id: str) -> None:
        """Remove all cached embeddings for a chat."""
        self.cache.invalidate_chat(chat_id)
