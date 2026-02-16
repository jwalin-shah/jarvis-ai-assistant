"""Hybrid Search - Combined semantic and keyword retrieval.

Combines VecSearcher (semantic) and BM25Searcher (keyword) using
Reciprocal Rank Fusion (RRF) to provide the best of both worlds:
- Deep semantic understanding from embeddings.
- Precise exact matching from BM25.
"""

from __future__ import annotations

import json
import logging
import pickle  # nosec B403
import sqlite3
import threading
from pathlib import Path
from typing import Any

from jarvis.search.bm25_search import BM25Searcher
from jarvis.search.vec_search import VecSearchResult, get_vec_searcher

logger = logging.getLogger(__name__)

# Cache directory for BM25 index
_CACHE_DIR = Path(".cache")
_BM25_CACHE_FILE = _CACHE_DIR / "bm25_index.pkl"


class HybridSearcher:
    """Combines semantic and keyword search with score fusion.

    BM25 index is cached to disk to avoid rebuilding on every instantiation.
    Cache is invalidated when chunk count or max timestamp changes.
    """

    def __init__(self) -> None:
        self.vec_searcher = get_vec_searcher()
        self.bm25_searcher = BM25Searcher()
        self._lock = threading.RLock()
        self._initialized = False
        self._cache_metadata: dict[str, Any] | None = None

    def _get_cache_metadata(self) -> dict[str, Any] | None:
        """Get metadata for cache staleness check.

        Returns:
            Dict with chunk_count and max_timestamp, or None if cache invalid.
        """
        if not _BM25_CACHE_FILE.exists():
            return None

        try:
            with open(_BM25_CACHE_FILE, "rb") as f:
                cached = pickle.load(f)  # nosec B301
                metadata = cached.get("metadata")
                if isinstance(metadata, dict):
                    return metadata
                return None
        except Exception:
            return None

    def _get_current_metadata(self) -> dict[str, Any] | None:
        """Get current DB state for cache comparison."""
        try:
            with self.vec_searcher.db.connection() as conn:
                # Get chunk count and latest timestamp
                row = conn.execute(
                    "SELECT COUNT(*) as count, MAX(source_timestamp) as max_ts FROM vec_chunks"
                ).fetchone()

                if row and row["count"] > 0:
                    return {
                        "chunk_count": row["count"],
                        "max_timestamp": row["max_ts"] or 0,
                    }
        except sqlite3.Error as e:
            logger.debug("Could not get metadata: %s", e)
        return None

    def _is_cache_stale(self, cached_meta: dict[str, Any], current_meta: dict[str, Any]) -> bool:
        """Check if cached index is stale."""
        return cached_meta.get("chunk_count") != current_meta.get("chunk_count") or cached_meta.get(
            "max_timestamp"
        ) != current_meta.get("max_timestamp")

    def _load_cached_index(self) -> bool:
        """Load BM25 index from cache if valid.

        Returns:
            True if loaded from cache, False otherwise.
        """
        try:
            cached_meta = self._get_cache_metadata()
            if cached_meta is None:
                return False

            current_meta = self._get_current_metadata()
            if current_meta is None:
                return False

            if self._is_cache_stale(cached_meta, current_meta):
                logger.debug("BM25 cache is stale, rebuilding")
                return False

            with open(_BM25_CACHE_FILE, "rb") as f:
                cached = pickle.load(f)  # nosec B301
                self.bm25_searcher = cached["searcher"]
                self._cache_metadata = cached_meta
                logger.info("Loaded BM25 index from cache (%d chunks)", cached_meta["chunk_count"])
                return True
        except Exception as e:
            logger.debug("Failed to load BM25 cache: %s", e)
            return False

    def _save_index_to_cache(self, metadata: dict[str, Any]) -> None:
        """Save BM25 index to cache."""
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(_BM25_CACHE_FILE, "wb") as f:
                pickle.dump(
                    {
                        "metadata": metadata,
                        "searcher": self.bm25_searcher,
                    },
                    f,
                )
        except Exception as e:
            logger.debug("Failed to save BM25 cache: %s", e)

    def _build_index_from_db(self) -> dict[str, Any]:
        """Build BM25 index from database."""
        with self.vec_searcher.db.connection() as conn:
            rows = conn.execute("SELECT rowid, context_text, reply_text FROM vec_chunks").fetchall()

        chunks = []
        for row in rows:
            text = f"{row['context_text'] or ''} {row['reply_text'] or ''}".strip()
            if text:
                chunks.append({"rowid": row["rowid"], "text": text})

        if chunks:
            self.bm25_searcher.index_chunks(chunks)

        return self._get_current_metadata() or {"chunk_count": 0, "max_timestamp": 0}

    def _ensure_initialized(self) -> None:
        """Lazily build BM25 index from SQLite chunks.

        Uses cached index if available and not stale.
        """
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            try:
                # Try to load from cache first
                if self._load_cached_index():
                    self._initialized = True
                    return

                # Build from DB
                metadata = self._build_index_from_db()
                self._cache_metadata = metadata

                # Save to cache for next time
                self._save_index_to_cache(metadata)

                self._initialized = True
                logger.info(
                    "HybridSearcher initialized with %d BM25 documents", metadata["chunk_count"]
                )
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
            query=query, limit=max(20, limit * 2), rerank=rerank
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
        for rank, bm25_res in enumerate(bm25_results, 1):
            rrf_scores[bm25_res.rowid] = rrf_scores.get(bm25_res.rowid, 0.0) + (1.0 / (k + rank))

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
            # Batch fetch in chunks to stay within SQLite parameter limits
            all_rows = []
            for i in range(0, len(rowids), 900):
                batch = rowids[i : i + 900]

                with self.vec_searcher.db.connection() as conn:
                    rows = conn.execute(
                        """
                        SELECT rowid, chat_id, context_text, reply_text,
                               topic_label
                        FROM vec_chunks
                        WHERE rowid IN (SELECT value FROM json_each(?))
                        """,
                        (json.dumps(batch),),
                    ).fetchall()
                    all_rows.extend(rows)

            # Map results to original rowid order
            row_map = {r["rowid"]: dict(r) for r in all_rows}
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
