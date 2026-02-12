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

            # Single query for all scalar counts (was 5 separate queries)
            row = conn.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM contacts) as contacts,
                    (SELECT COUNT(*) FROM pairs) as pairs,
                    (SELECT COUNT(*) FROM pairs WHERE quality_score >= 0.5) as pairs_quality,
                    (SELECT COUNT(*) FROM clusters) as clusters,
                    (SELECT COUNT(*) FROM pair_embeddings) as embeddings
                """
            ).fetchone()
            stats["contacts"] = row["contacts"]
            stats["pairs"] = row["pairs"]
            stats["pairs_quality_gte_50"] = row["pairs_quality"]
            stats["clusters"] = row["clusters"]
            stats["embeddings"] = row["embeddings"]

            # Active index (inlined to reuse same connection instead of opening a new one)
            idx_row = conn.execute(
                "SELECT version_id FROM index_versions WHERE is_active = TRUE LIMIT 1"
            ).fetchone()
            stats["active_index"] = idx_row["version_id"] if idx_row else None

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
        """Get statistics about validity gate results.

        Uses 3 queries instead of 7+: one for all pair-level counts (via GROUP BY
        and conditional aggregation), one for gate_a_reasons, one for gate_c_verdicts.
        """
        with self.connection() as conn:
            stats: dict[str, Any] = {}

            # Single query for all pair-level scalar stats + grouped counts
            # Replaces 5 separate queries (total_gated, 3x status loop, gate_a_rejected)
            # plus gate_b_bands (conditional aggregation)
            row = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN validity_status IS NOT NULL THEN 1 ELSE 0 END) as total_gated,
                    SUM(CASE WHEN validity_status = 'valid' THEN 1 ELSE 0 END) as status_valid,
                    SUM(CASE WHEN validity_status = 'invalid' THEN 1 ELSE 0 END) as status_invalid,
                    SUM(CASE WHEN validity_status = 'uncertain' THEN 1 ELSE 0 END)
                        as status_uncertain,
                    SUM(CASE WHEN gate_a_passed = FALSE THEN 1 ELSE 0 END) as gate_a_rejected,
                    SUM(CASE WHEN gate_b_score >= 0.62 THEN 1 ELSE 0 END) as gate_b_accept,
                    SUM(CASE WHEN gate_b_score >= 0.48 AND gate_b_score < 0.62 THEN 1 ELSE 0 END)
                        as gate_b_borderline,
                    SUM(CASE WHEN gate_b_score < 0.48 AND gate_b_score IS NOT NULL THEN 1 ELSE 0
                        END) as gate_b_reject
                FROM pairs
                """
            ).fetchone()

            stats["total_gated"] = row["total_gated"] or 0
            stats["status_valid"] = row["status_valid"] or 0
            stats["status_invalid"] = row["status_invalid"] or 0
            stats["status_uncertain"] = row["status_uncertain"] or 0
            stats["gate_a_rejected"] = row["gate_a_rejected"] or 0
            stats["gate_b_bands"] = {
                "accept": row["gate_b_accept"] or 0,
                "borderline": row["gate_b_borderline"] or 0,
                "reject": row["gate_b_reject"] or 0,
            }

            # Gate A rejection reasons (already uses GROUP BY - kept as-is)
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

            # Gate C verdicts (already uses GROUP BY - kept as-is)
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
