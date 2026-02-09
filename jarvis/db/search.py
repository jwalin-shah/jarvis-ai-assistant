"""Pair search, DA queries, pattern matching, and train/test split mixin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jarvis.db.models import Pair

if TYPE_CHECKING:
    from jarvis.db.core import JarvisDBBase


class PairSearchMixin:
    """Mixin providing pair search, DA filtering, and train/test split operations."""

    # Type hints for attributes provided by JarvisDBBase
    _trigger_pattern_cache: Any

    # Pre-computed acknowledgment triggers for pattern matching
    _ACK_TRIGGERS = (
        "ok",
        "okay",
        "sure",
        "yes",
        "yeah",
        "yep",
        "yup",
        "got it",
        "k",
        "kk",
        "alright",
        "sounds good",
    )

    def get_da_distribution(self: JarvisDBBase) -> dict[str, Any]:
        """Get distribution of DA types in the database.

        Returns:
            Dictionary with trigger and response DA type counts.
        """
        with self.connection() as conn:
            # Trigger DA distribution
            cursor = conn.execute(
                """SELECT trigger_da_type, COUNT(*) as cnt
                   FROM pairs WHERE trigger_da_type IS NOT NULL
                   GROUP BY trigger_da_type ORDER BY cnt DESC"""
            )
            trigger_dist = {row["trigger_da_type"]: row["cnt"] for row in cursor}

            # Response DA distribution
            cursor = conn.execute(
                """SELECT response_da_type, COUNT(*) as cnt
                   FROM pairs WHERE response_da_type IS NOT NULL
                   GROUP BY response_da_type ORDER BY cnt DESC"""
            )
            response_dist = {row["response_da_type"]: row["cnt"] for row in cursor}

            # Total classified
            cursor = conn.execute(
                "SELECT COUNT(*) as cnt FROM pairs WHERE trigger_da_type IS NOT NULL"
            )
            total_classified = cursor.fetchone()["cnt"]

            return {
                "trigger_da": trigger_dist,
                "response_da": response_dist,
                "total_classified": total_classified,
            }

    def get_pairs_by_trigger_pattern(
        self: JarvisDBBase,
        contact_id: int,
        pattern_type: str = "acknowledgment",
        limit: int = 10,
    ) -> list[Pair]:
        """Get pairs matching a trigger pattern type.

        Used to analyze user's typical response patterns to certain message types.
        For example, checking if user typically provides substantive info after
        acknowledgments like "ok" or "sure".

        Results are cached with 30-second TTL since this is called frequently
        during acknowledgment message routing.

        Args:
            contact_id: Contact to query.
            pattern_type: Pattern to match. Currently supported:
                - "acknowledgment": Short ack triggers like "ok", "sure", "yes"
            limit: Max pairs to return.

        Returns:
            List of Pair objects matching the pattern, ordered by recency.
        """
        # Check cache first
        cache_key = (contact_id, pattern_type, limit)
        hit, cached = self._trigger_pattern_cache.get(cache_key)
        if hit:
            return cached

        if pattern_type == "acknowledgment":
            # Build query with parameterized placeholders
            placeholders = ",".join("?" * len(self._ACK_TRIGGERS))
            query = (
                "SELECT id, contact_id, trigger_text, response_text, trigger_timestamp, "
                "response_timestamp, chat_id, trigger_msg_id, response_msg_id, "
                "trigger_msg_ids_json, response_msg_ids_json, context_text, "
                "quality_score, flags_json, is_group, is_holdout, "
                "gate_a_passed, gate_b_score, gate_c_verdict, validity_status, "
                "trigger_da_type, trigger_da_conf, response_da_type, "
                "response_da_conf, cluster_id "
                "FROM pairs "
                "WHERE contact_id = ? "
                f"AND LOWER(TRIM(trigger_text)) IN ({placeholders}) "
                "AND quality_score >= 0.5 "
                "ORDER BY trigger_timestamp DESC "
                "LIMIT ?"
            )
            with self.connection() as conn:
                cursor = conn.execute(query, (contact_id, *self._ACK_TRIGGERS, limit))
                result = [self._row_to_pair(row) for row in cursor]
                self._trigger_pattern_cache.set(cache_key, result)
                return result

        # Unknown pattern type - cache empty result
        self._trigger_pattern_cache.set(cache_key, [])
        return []

    def get_pairs_by_trigger_patterns_batch(
        self: JarvisDBBase,
        contact_ids: list[int],
        pattern_type: str = "acknowledgment",
        limit_per_contact: int = 10,
    ) -> dict[int, list[Pair]]:
        """Batch fetch pairs by trigger pattern for multiple contacts.

        More efficient than calling get_pairs_by_trigger_pattern in a loop
        when checking patterns for multiple contacts.

        Args:
            contact_ids: List of contact IDs to query.
            pattern_type: Pattern to match (currently "acknowledgment").
            limit_per_contact: Max pairs per contact.

        Returns:
            Dict mapping contact_id -> list of matching Pairs.
        """
        if not contact_ids or pattern_type != "acknowledgment":
            return {cid: [] for cid in contact_ids}

        # Check cache for already-cached contacts
        result: dict[int, list[Pair]] = {}
        uncached_ids: list[int] = []

        for cid in contact_ids:
            cache_key = (cid, pattern_type, limit_per_contact)
            hit, cached = self._trigger_pattern_cache.get(cache_key)
            if hit:
                result[cid] = cached
            else:
                uncached_ids.append(cid)

        if not uncached_ids:
            return result

        # Batch query for uncached contacts with chunking
        for j in range(0, len(uncached_ids), 900):
            chunk = uncached_ids[j : j + 900]
            contact_placeholders = ",".join("?" * len(chunk))
            trigger_placeholders = ",".join("?" * len(self._ACK_TRIGGERS))

            # Use window function to limit per contact
            query = (
                "SELECT id, contact_id, trigger_text, response_text, trigger_timestamp, "
                "response_timestamp, chat_id, trigger_msg_id, response_msg_id, "
                "trigger_msg_ids_json, response_msg_ids_json, context_text, "
                "quality_score, flags_json, is_group, is_holdout, "
                "gate_a_passed, gate_b_score, gate_c_verdict, validity_status, "
                "trigger_da_type, trigger_da_conf, response_da_type, "
                "response_da_conf, cluster_id "
                "FROM ( "
                "SELECT *, ROW_NUMBER() OVER ( "
                "PARTITION BY contact_id ORDER BY trigger_timestamp DESC "
                ") as rn "
                "FROM pairs "
                f"WHERE contact_id IN ({contact_placeholders}) "
                f"AND LOWER(TRIM(trigger_text)) IN ({trigger_placeholders}) "
                "AND quality_score >= 0.5 "
                ") WHERE rn <= ?"
            )

            with self.connection() as conn:
                cursor = conn.execute(query, (*chunk, *self._ACK_TRIGGERS, limit_per_contact))

                # Group results by contact_id
                for cid in chunk:
                    if cid not in result:
                        result[cid] = []

                for row in cursor:
                    pair = self._row_to_pair(row)
                    if pair.contact_id in result:
                        result[pair.contact_id].append(pair)

        # Cache all results
        for cid in uncached_ids:
            cache_key = (cid, pattern_type, limit_per_contact)
            self._trigger_pattern_cache.set(cache_key, result.get(cid, []))

        return result

    # -------------------------------------------------------------------------
    # Train/Test Split Operations
    # -------------------------------------------------------------------------

    def split_train_test(
        self: JarvisDBBase,
        holdout_ratio: float = 0.2,
        min_pairs_per_contact: int = 5,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Split pairs into training and holdout sets by contact.

        All pairs for a contact go to the same set to test generalization
        to new conversation styles, not just new messages from known contacts.

        Args:
            holdout_ratio: Fraction of contacts to hold out (default 0.2 = 20%).
            min_pairs_per_contact: Minimum pairs a contact must have to be
                considered for holdout (default 5).
            seed: Random seed for reproducibility.

        Returns:
            Statistics about the split.
        """
        import random

        if seed is not None:
            random.seed(seed)

        with self.connection() as conn:
            # Get contacts with their pair counts
            cursor = conn.execute(
                """
                SELECT contact_id, COUNT(*) as pair_count
                FROM pairs
                WHERE contact_id IS NOT NULL
                GROUP BY contact_id
                HAVING pair_count >= ?
                """,
                (min_pairs_per_contact,),
            )
            eligible_contacts = [(row["contact_id"], row["pair_count"]) for row in cursor]

            if not eligible_contacts:
                return {
                    "success": False,
                    "error": f"No contacts with >= {min_pairs_per_contact} pairs",
                    "contacts_total": 0,
                    "contacts_holdout": 0,
                }

            # Shuffle and select holdout contacts
            random.shuffle(eligible_contacts)
            num_holdout = max(1, int(len(eligible_contacts) * holdout_ratio))
            holdout_contacts = [c[0] for c in eligible_contacts[:num_holdout]]
            training_contacts = [c[0] for c in eligible_contacts[num_holdout:]]

            # Reset all pairs to training first
            conn.execute("UPDATE pairs SET is_holdout = FALSE")

            # Mark holdout contact pairs
            if holdout_contacts:
                # Chunk to avoid SQLite parameter limits
                for i in range(0, len(holdout_contacts), 900):
                    chunk = holdout_contacts[i : i + 900]
                    placeholders = ",".join("?" * len(chunk))
                    query = f"UPDATE pairs SET is_holdout = TRUE WHERE contact_id IN ({placeholders})"
                    conn.execute(query, chunk)

            # Get final counts
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM pairs WHERE is_holdout = FALSE")
            training_pairs = cursor.fetchone()["cnt"]

            cursor = conn.execute("SELECT COUNT(*) as cnt FROM pairs WHERE is_holdout = TRUE")
            holdout_pairs = cursor.fetchone()["cnt"]

            return {
                "success": True,
                "contacts_total": len(eligible_contacts),
                "contacts_holdout": len(holdout_contacts),
                "contacts_training": len(training_contacts),
                "pairs_training": training_pairs,
                "pairs_holdout": holdout_pairs,
                "holdout_ratio_actual": holdout_pairs / (training_pairs + holdout_pairs)
                if (training_pairs + holdout_pairs) > 0
                else 0,
                "holdout_contact_ids": holdout_contacts,
            }

    def get_training_pairs(
        self: JarvisDBBase, min_quality: float = 0.0, limit: int | None = None
    ) -> list[Pair]:
        """Get pairs designated for training (is_holdout=False)."""
        with self.connection() as conn:
            if limit is None:
                cursor = conn.execute(
                    """
                    SELECT * FROM pairs
                    WHERE is_holdout = FALSE AND quality_score >= ?
                    ORDER BY trigger_timestamp DESC
                    """,
                    (min_quality,),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM pairs
                    WHERE is_holdout = FALSE AND quality_score >= ?
                    ORDER BY trigger_timestamp DESC
                    LIMIT ?
                    """,
                    (min_quality, limit),
                )
            return [self._row_to_pair(row) for row in cursor]

    def get_holdout_pairs(
        self: JarvisDBBase, min_quality: float = 0.0, limit: int | None = None
    ) -> list[Pair]:
        """Get pairs designated for evaluation (is_holdout=True)."""
        with self.connection() as conn:
            if limit is None:
                cursor = conn.execute(
                    """
                    SELECT * FROM pairs
                    WHERE is_holdout = TRUE AND quality_score >= ?
                    ORDER BY trigger_timestamp DESC
                    """,
                    (min_quality,),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM pairs
                    WHERE is_holdout = TRUE AND quality_score >= ?
                    ORDER BY trigger_timestamp DESC
                    LIMIT ?
                    """,
                    (min_quality, limit),
                )
            return [self._row_to_pair(row) for row in cursor]

    def get_split_stats(self: JarvisDBBase) -> dict[str, Any]:
        """Get statistics about the current train/test split."""
        with self.connection() as conn:
            stats: dict[str, Any] = {}

            # Training pairs
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM pairs WHERE is_holdout = FALSE")
            stats["training_pairs"] = cursor.fetchone()["cnt"]

            # Holdout pairs
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM pairs WHERE is_holdout = TRUE")
            stats["holdout_pairs"] = cursor.fetchone()["cnt"]

            # Training contacts
            cursor = conn.execute(
                """
                SELECT COUNT(DISTINCT contact_id) as cnt
                FROM pairs WHERE is_holdout = FALSE AND contact_id IS NOT NULL
                """
            )
            stats["training_contacts"] = cursor.fetchone()["cnt"]

            # Holdout contacts
            cursor = conn.execute(
                """
                SELECT COUNT(DISTINCT contact_id) as cnt
                FROM pairs WHERE is_holdout = TRUE AND contact_id IS NOT NULL
                """
            )
            stats["holdout_contacts"] = cursor.fetchone()["cnt"]

            total = stats["training_pairs"] + stats["holdout_pairs"]
            stats["holdout_ratio"] = stats["holdout_pairs"] / total if total > 0 else 0

            return stats

    def get_valid_pairs(
        self: JarvisDBBase, min_quality: float = 0.0, limit: int = 100000
    ) -> list[Pair]:
        """Get pairs with validity_status='valid'."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM pairs
                WHERE validity_status = 'valid' AND quality_score >= ?
                ORDER BY trigger_timestamp DESC
                LIMIT ?
                """,
                (min_quality, limit),
            )
            return [self._row_to_pair(row) for row in cursor]

    # -------------------------------------------------------------------------
    # DA-Filtered Retrieval Operations
    # -------------------------------------------------------------------------

    def get_pairs_by_response_da(
        self: JarvisDBBase,
        response_da: str,
        min_conf: float = 0.5,
        min_quality: float = 0.0,
        limit: int = 100,
        exclude_holdout: bool = True,
    ) -> list[Pair]:
        """Get pairs filtered by response dialogue act type.

        Used for DA-filtered retrieval to find examples of specific response types.

        Args:
            response_da: Response dialogue act type (e.g., 'AGREE', 'DECLINE').
            min_conf: Minimum classifier confidence (default 0.5).
            min_quality: Minimum quality score (default 0.0).
            limit: Maximum pairs to return.
            exclude_holdout: If True, exclude holdout pairs (default True).

        Returns:
            List of Pair objects matching the criteria.
        """
        with self.connection() as conn:
            if exclude_holdout:
                cursor = conn.execute(
                    """
                    SELECT * FROM pairs
                    WHERE response_da_type = ?
                    AND response_da_conf >= ?
                    AND quality_score >= ?
                    AND is_holdout = FALSE
                    ORDER BY response_da_conf DESC, quality_score DESC
                    LIMIT ?
                    """,
                    (response_da, min_conf, min_quality, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM pairs
                    WHERE response_da_type = ?
                    AND response_da_conf >= ?
                    AND quality_score >= ?
                    ORDER BY response_da_conf DESC, quality_score DESC
                    LIMIT ?
                    """,
                    (response_da, min_conf, min_quality, limit),
                )
            return [self._row_to_pair(row) for row in cursor]

    def get_pairs_by_trigger_da(
        self: JarvisDBBase,
        trigger_da: str,
        min_conf: float = 0.5,
        min_quality: float = 0.0,
        limit: int = 100,
        exclude_holdout: bool = True,
    ) -> list[Pair]:
        """Get pairs filtered by trigger dialogue act type.

        Used for finding pairs where the trigger is a specific type
        (e.g., INVITATION triggers to find how user responds to invitations).

        Args:
            trigger_da: Trigger dialogue act type (e.g., 'INVITATION', 'YN_QUESTION').
            min_conf: Minimum classifier confidence (default 0.5).
            min_quality: Minimum quality score (default 0.0).
            limit: Maximum pairs to return.
            exclude_holdout: If True, exclude holdout pairs (default True).

        Returns:
            List of Pair objects matching the criteria.
        """
        with self.connection() as conn:
            if exclude_holdout:
                cursor = conn.execute(
                    """
                    SELECT * FROM pairs
                    WHERE trigger_da_type = ?
                    AND trigger_da_conf >= ?
                    AND quality_score >= ?
                    AND is_holdout = FALSE
                    ORDER BY trigger_da_conf DESC, quality_score DESC
                    LIMIT ?
                    """,
                    (trigger_da, min_conf, min_quality, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM pairs
                    WHERE trigger_da_type = ?
                    AND trigger_da_conf >= ?
                    AND quality_score >= ?
                    ORDER BY trigger_da_conf DESC, quality_score DESC
                    LIMIT ?
                    """,
                    (trigger_da, min_conf, min_quality, limit),
                )
            return [self._row_to_pair(row) for row in cursor]

    def get_pairs_by_trigger_and_response_da(
        self: JarvisDBBase,
        trigger_da: str,
        response_da: str,
        min_conf: float = 0.5,
        min_quality: float = 0.0,
        limit: int = 100,
        exclude_holdout: bool = True,
    ) -> list[Pair]:
        """Get pairs filtered by both trigger and response DA types.

        Used for finding examples of specific trigger->response patterns
        (e.g., INVITATION->AGREE pairs for affirmative response examples).

        Args:
            trigger_da: Trigger dialogue act type.
            response_da: Response dialogue act type.
            min_conf: Minimum classifier confidence for both (default 0.5).
            min_quality: Minimum quality score (default 0.0).
            limit: Maximum pairs to return.
            exclude_holdout: If True, exclude holdout pairs (default True).

        Returns:
            List of Pair objects matching the criteria.
        """
        with self.connection() as conn:
            if exclude_holdout:
                cursor = conn.execute(
                    """
                    SELECT * FROM pairs
                    WHERE trigger_da_type = ?
                    AND response_da_type = ?
                    AND trigger_da_conf >= ?
                    AND response_da_conf >= ?
                    AND quality_score >= ?
                    AND is_holdout = FALSE
                    ORDER BY quality_score DESC, response_da_conf DESC
                    LIMIT ?
                    """,
                    (trigger_da, response_da, min_conf, min_conf, min_quality, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM pairs
                    WHERE trigger_da_type = ?
                    AND response_da_type = ?
                    AND trigger_da_conf >= ?
                    AND response_da_conf >= ?
                    AND quality_score >= ?
                    ORDER BY quality_score DESC, response_da_conf DESC
                    LIMIT ?
                    """,
                    (trigger_da, response_da, min_conf, min_conf, min_quality, limit),
                )
            return [self._row_to_pair(row) for row in cursor]

    def get_high_quality_exemplars(
        self: JarvisDBBase,
        response_da: str,
        min_quality: float = 0.7,
        min_conf: float = 0.7,
        limit: int = 50,
    ) -> list[Pair]:
        """Get high-quality exemplar pairs for a response type.

        These are the best examples for few-shot learning - high quality scores
        AND high classifier confidence for the response DA type.

        Args:
            response_da: Response dialogue act type.
            min_quality: Minimum quality score (default 0.7 on 0-1 scale).
            min_conf: Minimum DA classifier confidence (default 0.7).
            limit: Maximum exemplars to return.

        Returns:
            List of high-quality Pair objects for this response type.
        """
        with self.connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM pairs
                WHERE response_da_type = ?
                AND response_da_conf >= ?
                AND quality_score >= ?
                AND is_holdout = FALSE
                ORDER BY quality_score DESC, response_da_conf DESC
                LIMIT ?
                """,
                (response_da, min_conf, min_quality, limit),
            )
            return [self._row_to_pair(row) for row in cursor]

    def get_da_cross_tabulation(self: JarvisDBBase) -> dict[str, dict[str, int]]:
        """Get cross-tabulation of trigger DA vs response DA types.

        Shows how different trigger types are responded to in the data.
        Useful for understanding response patterns and validating DA mappings.

        Returns:
            Nested dict of {trigger_da: {response_da: count}}.
        """
        with self.connection() as conn:
            cursor = conn.execute(
                """
                SELECT trigger_da_type, response_da_type, COUNT(*) as cnt
                FROM pairs
                WHERE trigger_da_type IS NOT NULL
                AND response_da_type IS NOT NULL
                GROUP BY trigger_da_type, response_da_type
                ORDER BY trigger_da_type, cnt DESC
                """
            )
            result: dict[str, dict[str, int]] = {}
            for row in cursor:
                trigger = row["trigger_da_type"]
                response = row["response_da_type"]
                if trigger not in result:
                    result[trigger] = {}
                result[trigger][response] = row["cnt"]
            return result
