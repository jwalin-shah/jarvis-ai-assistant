"""Hybrid Search - Combined semantic and keyword retrieval.

Combines VecSearcher (semantic) and BM25Searcher (keyword) using
Reciprocal Rank Fusion (RRF) to provide the best of both worlds:
- Deep semantic understanding from embeddings.
- Precise exact matching from BM25.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from jarvis.search.bm25_search import BM25Searcher
from jarvis.search.vec_search import VecSearchResult, get_vec_searcher

logger = logging.getLogger(__name__)


class HybridSearcher:
    """Combines semantic and keyword search with score fusion."""

    def __init__(self) -> None:
        self.vec_searcher = get_vec_searcher()
        self.bm25_searcher = BM25Searcher()
        self._lock = threading.RLock()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazily build BM25 index from SQLite chunks."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            try:
                # Fetch all chunks from vec_chunks for the BM25 index
                with self.vec_searcher.db.connection() as conn:
                    rows = conn.execute(
                        "SELECT rowid, context_text, reply_text FROM vec_chunks"
                    ).fetchall()

                chunks = []
                for row in rows:
                    text = f"{row['context_text'] or ''} {row['reply_text'] or ''}".strip()
                    if text:
                        chunks.append({"rowid": row["rowid"], "text": text})

                if chunks:
                    self.bm25_searcher.index_chunks(chunks)

                self._initialized = True
                logger.info("HybridSearcher initialized with %d BM25 documents", len(chunks))
            except Exception as e:
                logger.error("Failed to initialize HybridSearcher: %s", e)

    def search(
        self,
        query: str,
        limit: int = 5,
        rerank: bool = True,
    ) -> list[dict[str, Any]]:
        """Perform hybrid search using RRF.

        Args:
            query: Search query.
            limit: Max results.
            rerank: Whether to enable cross-encoder reranking in vec search.

        Returns:
            Ranked list of chunk dictionaries.
        """
        self._ensure_initialized()

        # 1. Get semantic results (top 20)
        vec_results = self.vec_searcher.search_with_chunks_global(
            query=query,
            limit=max(20, limit * 2),
            rerank=rerank
        )

        # 2. Get keyword results (top 20)
        bm25_results = self.bm25_searcher.search(query, limit=20)

        # 3. Reciprocal Rank Fusion (RRF)
        # score = sum(1 / (k + rank))
        k = 60
        rrf_scores: dict[int, float] = {}

        # Semantic ranks
        for rank, res in enumerate(vec_results, 1):
            rrf_scores[res.rowid] = rrf_scores.get(res.rowid, 0.0) + (1.0 / (k + rank))

        # Keyword ranks
        for rank, res in enumerate(bm25_results, 1):
            rrf_scores[res.rowid] = rrf_scores.get(res.rowid, 0.0) + (1.0 / (k + rank))

        # 4. Sort by fused score
        sorted_rowids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:limit]

        # 5. Fetch full metadata for top results
        if not sorted_rowids:
            return []

        # Use vec_searcher's logic to fetch enriched metadata
        # Create dummy VecSearchResult objects for the fetcher
        top_results = []
        for rid, score in sorted_rowids:
            top_results.append(VecSearchResult(rowid=rid, distance=0.0, score=score))

        # Enrich with segment data if possible
        # We'll use search_with_full_segments approach but with our fused rowids
        enriched = self._enrich_results([rid for rid, _ in sorted_rowids])

        # Add fused scores to output
        for item in enriched:
            item["fused_score"] = rrf_scores.get(item["rowid"], 0.0)

        return enriched

    def _enrich_results(self, rowids: list[int]) -> list[dict[str, Any]]:
        """Enrich rowids with full chunk metadata and segment info."""
        if not rowids:
            return []

        try:
            from jarvis.search.vec_search import _validate_placeholders
            placeholders = ",".join("?" * len(rowids))
            _validate_placeholders(placeholders)

            with self.vec_searcher.db.connection() as conn:
                rows = conn.execute(
                    f"""
                    SELECT rowid, chat_id, context_text, reply_text,
                           topic_label
                    FROM vec_chunks
                    WHERE rowid IN ({placeholders})
                    """,
                    rowids,
                ).fetchall()

            # Map results to original rowid order
            row_map = {r["rowid"]: dict(r) for r in rows}
            results = []
            for rid in rowids:
                if rid in row_map:
                    results.append(row_map[rid])

            return results
        except Exception as e:
            logger.error("Enrichment failed: %s", e)
            return [{"rowid": rid} for rid in rowids]


_hybrid_searcher: HybridSearcher | None = None
_hybrid_lock = threading.Lock()


def get_hybrid_searcher() -> HybridSearcher:
    """Get singleton HybridSearcher."""
    global _hybrid_searcher
    if _hybrid_searcher is None:
        with _hybrid_lock:
            if _hybrid_searcher is None:
                _hybrid_searcher = HybridSearcher()
    return _hybrid_searcher
