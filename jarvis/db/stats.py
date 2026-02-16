"""Statistics operations mixin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class StatsMixin:
    """Mixin providing database statistics operations."""

    # Type hints for attributes provided by JarvisDBBase
    _stats_cache: Any

    def get_stats(self: Any, use_cache: bool = True) -> dict[str, Any]:
        """Get database statistics.

        Results are cached with 60-second TTL since stats don't change frequently.

        Args:
            use_cache: If True (default), use cached results if available.
        """
        cache_key = "db_stats"
        if use_cache:
            hit, cached = self._stats_cache.get(cache_key)
            if hit:
                return cached  # type: ignore[no-any-return]

        with self.connection() as conn:
            stats: dict[str, Any] = {}

            # Single query for all scalar counts
            row = conn.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM contacts) as contacts,
                    (SELECT COUNT(*) FROM conversation_segments) as chunks
                """
            ).fetchone()
            stats["contacts"] = row["contacts"]
            stats["chunks"] = row["chunks"]

            # Active index
            idx_row = conn.execute(
                "SELECT version_id FROM index_versions WHERE is_active = TRUE LIMIT 1"
            ).fetchone()
            stats["active_index"] = idx_row["version_id"] if idx_row else None

            # Chunks per contact
            cursor = conn.execute(
                """
                SELECT c.display_name, COUNT(cs.id) as chunk_count
                FROM contacts c
                LEFT JOIN conversation_segments cs ON c.chat_id = cs.chat_id
                GROUP BY c.id
                ORDER BY chunk_count DESC
                LIMIT 10
                """
            )
            stats["chunks_per_contact"] = [
                {"name": row["display_name"], "count": row["chunk_count"]} for row in cursor
            ]

            self._stats_cache.set(cache_key, stats)
            return stats

    def get_vector_stats(self: Any) -> dict[str, Any]:
        """Get statistics about vector embeddings.

        Returns stats about vec_chunks table (sqlite-vec based storage).
        """
        with self.connection() as conn:
            stats: dict[str, Any] = {}

            # Total chunks with embeddings
            row = conn.execute("SELECT COUNT(*) as total FROM vec_chunks").fetchone()
            stats["total_vectors"] = row["total"] if row else 0

            # Chunks per contact
            cursor = conn.execute(
                """
                SELECT c.display_name, COUNT(vc.chunk_id) as chunk_count
                FROM contacts c
                LEFT JOIN conversation_segments cs ON c.chat_id = cs.chat_id
                LEFT JOIN vec_chunks vc ON cs.id = vc.chunk_id
                GROUP BY c.id
                ORDER BY chunk_count DESC
                LIMIT 10
                """
            )
            stats["vectors_per_contact"] = [
                {"name": row["display_name"], "count": row["chunk_count"]} for row in cursor
            ]

            return stats
