"""Statistics operations mixin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jarvis.db.core import JarvisDBBase


class StatsMixin:
    """Mixin providing database statistics operations."""

    # Type hints for attributes provided by JarvisDBBase
    _stats_cache: Any

    def get_stats(self: JarvisDBBase, use_cache: bool = True) -> dict[str, Any]:
        """Get database statistics.

        Results are cached with 60-second TTL since stats don't change frequently.

        Args:
            use_cache: If True (default), use cached results if available.
        """
        cache_key = "db_stats"
        if use_cache:
            hit, cached = self._stats_cache.get(cache_key)
            if hit:
                return cached

        with self.connection() as conn:
            stats: dict[str, Any] = {}

            # Contact count
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM contacts")
            stats["contacts"] = cursor.fetchone()["cnt"]

            # Pair count (with quality breakdown)
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM pairs")
            stats["pairs"] = cursor.fetchone()["cnt"]

            cursor = conn.execute("SELECT COUNT(*) as cnt FROM pairs WHERE quality_score >= 0.5")
            stats["pairs_quality_gte_50"] = cursor.fetchone()["cnt"]

            # Cluster count
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM clusters")
            stats["clusters"] = cursor.fetchone()["cnt"]

            # Embedding count
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM pair_embeddings")
            stats["embeddings"] = cursor.fetchone()["cnt"]

            # Active index
            active_index = self.get_active_index()
            stats["active_index"] = active_index.version_id if active_index else None

            # Pairs per contact
            cursor = conn.execute(
                """
                SELECT c.display_name, COUNT(p.id) as pair_count
                FROM contacts c
                LEFT JOIN pairs p ON c.id = p.contact_id
                GROUP BY c.id
                ORDER BY pair_count DESC
                LIMIT 10
                """
            )
            stats["pairs_per_contact"] = [
                {"name": row["display_name"], "count": row["pair_count"]} for row in cursor
            ]

            self._stats_cache.set(cache_key, stats)
            return stats

    def get_gate_stats(self: JarvisDBBase) -> dict[str, Any]:
        """Get statistics about validity gate results."""
        with self.connection() as conn:
            stats: dict[str, Any] = {}

            # Total pairs with gate data
            cursor = conn.execute(
                "SELECT COUNT(*) as cnt FROM pairs WHERE validity_status IS NOT NULL"
            )
            stats["total_gated"] = cursor.fetchone()["cnt"]

            # By validity status
            for status in ["valid", "invalid", "uncertain"]:
                cursor = conn.execute(
                    "SELECT COUNT(*) as cnt FROM pairs WHERE validity_status = ?",
                    (status,),
                )
                stats[f"status_{status}"] = cursor.fetchone()["cnt"]

            # Gate A rejections
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM pairs WHERE gate_a_passed = FALSE")
            stats["gate_a_rejected"] = cursor.fetchone()["cnt"]

            # Gate A rejection reasons (from artifacts)
            cursor = conn.execute(
                """
                SELECT gate_a_reason, COUNT(*) as cnt
                FROM pair_artifacts
                WHERE gate_a_reason IS NOT NULL
                GROUP BY gate_a_reason
                ORDER BY cnt DESC
                """
            )
            stats["gate_a_reasons"] = {row["gate_a_reason"]: row["cnt"] for row in cursor}

            # Gate B score distribution
            cursor = conn.execute(
                """
                SELECT
                    CASE
                        WHEN gate_b_score >= 0.62 THEN 'accept'
                        WHEN gate_b_score >= 0.48 THEN 'borderline'
                        ELSE 'reject'
                    END as band,
                    COUNT(*) as cnt
                FROM pairs
                WHERE gate_b_score IS NOT NULL
                GROUP BY band
                """
            )
            stats["gate_b_bands"] = {row["band"]: row["cnt"] for row in cursor}

            # Gate C verdicts
            cursor = conn.execute(
                """
                SELECT gate_c_verdict, COUNT(*) as cnt
                FROM pairs
                WHERE gate_c_verdict IS NOT NULL
                GROUP BY gate_c_verdict
                """
            )
            stats["gate_c_verdicts"] = {row["gate_c_verdict"]: row["cnt"] for row in cursor}

            return stats
