"""Pair CRUD and bulk operations mixin."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING, Any

from jarvis.db.models import Pair

if TYPE_CHECKING:
    from jarvis.db.core import JarvisDBBase


class PairMixin:
    """Mixin providing pair CRUD and bulk operations."""

    def add_pair(
        self: JarvisDBBase,
        trigger_text: str,
        response_text: str,
        trigger_timestamp: datetime,
        response_timestamp: datetime,
        chat_id: str,
        contact_id: int | None = None,
        trigger_msg_id: int | None = None,
        response_msg_id: int | None = None,
        trigger_msg_ids: list[int] | None = None,
        response_msg_ids: list[int] | None = None,
        context_text: str | None = None,
        quality_score: float = 1.0,
        flags: dict[str, Any] | None = None,
    ) -> Pair | None:
        """Add a (trigger, response) pair.

        Ignores duplicates based on (trigger_msg_id, response_msg_id).

        Args:
            context_text: Previous messages before trigger for LLM context.

        Returns:
            The created Pair, or None if duplicate.
        """
        with self.connection() as conn:
            trigger_msg_ids_json = json.dumps(trigger_msg_ids) if trigger_msg_ids else None
            response_msg_ids_json = json.dumps(response_msg_ids) if response_msg_ids else None
            flags_json = json.dumps(flags) if flags else None

            try:
                cursor = conn.execute(
                    """
                    INSERT INTO pairs
                    (contact_id, trigger_text, response_text, trigger_timestamp,
                     response_timestamp, chat_id, trigger_msg_id, response_msg_id,
                     trigger_msg_ids_json, response_msg_ids_json, context_text,
                     quality_score, flags_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        contact_id,
                        trigger_text,
                        response_text,
                        trigger_timestamp,
                        response_timestamp,
                        chat_id,
                        trigger_msg_id,
                        response_msg_id,
                        trigger_msg_ids_json,
                        response_msg_ids_json,
                        context_text,
                        quality_score,
                        flags_json,
                    ),
                )
                return Pair(
                    id=cursor.lastrowid,
                    contact_id=contact_id,
                    trigger_text=trigger_text,
                    response_text=response_text,
                    trigger_timestamp=trigger_timestamp,
                    response_timestamp=response_timestamp,
                    chat_id=chat_id,
                    trigger_msg_id=trigger_msg_id,
                    response_msg_id=response_msg_id,
                    trigger_msg_ids_json=trigger_msg_ids_json,
                    response_msg_ids_json=response_msg_ids_json,
                    context_text=context_text,
                    quality_score=quality_score,
                    flags_json=flags_json,
                )
            except sqlite3.IntegrityError:
                # Duplicate pair
                return None

    def add_pairs_bulk(
        self: JarvisDBBase,
        pairs: list[dict[str, Any]],
        dedupe_by_content: bool = True,
    ) -> int:
        """Add multiple pairs in a single transaction.

        Args:
            pairs: List of pair dictionaries.
            dedupe_by_content: If True, skip pairs with duplicate content_hash.
                This prevents adding semantically identical pairs even if message IDs differ.

        Returns:
            Number of pairs successfully added.
        """
        import hashlib

        if not pairs:
            return 0

        # Precompute values and content hashes for all pairs
        processed = []
        for pair in pairs:
            trigger_msg_ids_json = (
                json.dumps(pair.get("trigger_msg_ids")) if pair.get("trigger_msg_ids") else None
            )
            response_msg_ids_json = (
                json.dumps(pair.get("response_msg_ids")) if pair.get("response_msg_ids") else None
            )
            flags_json = json.dumps(pair.get("flags")) if pair.get("flags") else None

            # Compute content hash for text-based deduplication
            trigger_normalized = pair["trigger_text"].lower().strip()
            response_normalized = pair["response_text"].lower().strip()
            content_str = f"{trigger_normalized}|{response_normalized}"
            content_hash = hashlib.md5(content_str.encode()).hexdigest()

            processed.append(
                {
                    "data": (
                        pair.get("contact_id"),
                        pair["trigger_text"],
                        pair["response_text"],
                        pair["trigger_timestamp"],
                        pair["response_timestamp"],
                        pair["chat_id"],
                        pair.get("trigger_msg_id"),
                        pair.get("response_msg_id"),
                        trigger_msg_ids_json,
                        response_msg_ids_json,
                        pair.get("context_text"),
                        pair.get("quality_score", 1.0),
                        flags_json,
                        pair.get("is_group", False),
                        pair.get("source_timestamp"),
                        content_hash,
                    ),
                    "content_hash": content_hash,
                }
            )

        added = 0
        with self.connection() as conn:
            # Efficient content deduplication
            if dedupe_by_content:
                # Get unique hashes from input batch
                unique_hashes = list({p["content_hash"] for p in processed})

                # Find which ones already exist in DB (chunked to avoid SQLite parameter limits)
                existing_hashes = set()
                for i in range(0, len(unique_hashes), 900):
                    chunk = unique_hashes[i : i + 900]
                    placeholders = ",".join("?" * len(chunk))
                    cursor = conn.execute(
                        f"SELECT content_hash FROM pairs WHERE content_hash IN ({placeholders})",
                        chunk,
                    )
                    existing_hashes.update(row["content_hash"] for row in cursor)

                # Filter batch: skip existing in DB and duplicates within the batch itself
                seen_in_batch = set()
                final_batch = []
                for p in processed:
                    h = p["content_hash"]
                    if h not in existing_hashes and h not in seen_in_batch:
                        final_batch.append(p["data"])
                        seen_in_batch.add(h)
            else:
                final_batch = [p["data"] for p in processed]

            if final_batch:
                # Use executemany with INSERT OR IGNORE for high-performance batch insertion
                # IGNORE handles UNIQUE(trigger_msg_id, response_msg_id) violations
                cursor = conn.executemany(
                    """
                    INSERT OR IGNORE INTO pairs
                    (contact_id, trigger_text, response_text, trigger_timestamp,
                     response_timestamp, chat_id, trigger_msg_id, response_msg_id,
                     trigger_msg_ids_json, response_msg_ids_json, context_text,
                     quality_score, flags_json, is_group, source_timestamp, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    final_batch,
                )
                added = cursor.rowcount

        return added

    def get_pairs(
        self: JarvisDBBase,
        contact_id: int | None = None,
        chat_id: str | None = None,
        min_quality: float = 0.0,
        limit: int = 10000,
    ) -> list[Pair]:
        """Get pairs with optional filtering."""
        with self.connection() as conn:
            conditions = ["quality_score >= ?"]
            params: list[Any] = [min_quality]

            if contact_id is not None:
                conditions.append("contact_id = ?")
                params.append(contact_id)
            if chat_id is not None:
                conditions.append("chat_id = ?")
                params.append(chat_id)

            where_clause = " AND ".join(conditions)
            params.append(limit)

            cursor = conn.execute(
                f"SELECT * FROM pairs WHERE {where_clause} ORDER BY trigger_timestamp DESC LIMIT ?",
                params,
            )
            return [self._row_to_pair(row) for row in cursor]

    def get_pair(self: JarvisDBBase, pair_id: int) -> Pair | None:
        """Get a single pair by ID.

        Args:
            pair_id: The pair ID to fetch.

        Returns:
            The Pair if found, None otherwise.
        """
        with self.connection() as conn:
            cursor = conn.execute("SELECT * FROM pairs WHERE id = ?", (pair_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_pair(row)
            return None

    def get_pairs_by_ids(self: JarvisDBBase, pair_ids: list[int]) -> dict[int, Pair]:
        """Batch fetch pairs by IDs.

        More efficient than calling get_pair in a loop.
        Uses chunking to avoid SQLite host parameter limits.

        Args:
            pair_ids: List of pair IDs to fetch.

        Returns:
            Dict mapping pair_id -> Pair for found pairs.
        """
        if not pair_ids:
            return {}

        result = {}
        with self.connection() as conn:
            # Chunk the IDs to avoid SQLite's limit on host parameters (often 999)
            for i in range(0, len(pair_ids), 900):
                chunk = pair_ids[i : i + 900]
                placeholders = ",".join("?" * len(chunk))
                cursor = conn.execute(
                    f"SELECT * FROM pairs WHERE id IN ({placeholders})",
                    chunk,
                )
                for row in cursor:
                    result[row["id"]] = self._row_to_pair(row)
        return result

    def get_all_pairs(self: JarvisDBBase, min_quality: float = 0.0) -> list[Pair]:
        """Get all pairs in the database."""
        return self.get_pairs(min_quality=min_quality, limit=1000000)

    def count_pairs(self: JarvisDBBase, min_quality: float = 0.0) -> int:
        """Count total pairs in database."""
        with self.connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) as cnt FROM pairs WHERE quality_score >= ?",
                (min_quality,),
            )
            row = cursor.fetchone()
            return row["cnt"] if row else 0

    def update_pair_quality(
        self: JarvisDBBase, pair_id: int, quality_score: float, flags: dict[str, Any] | None = None
    ) -> bool:
        """Update a pair's quality score and flags."""
        with self.connection() as conn:
            if flags is not None:
                cursor = conn.execute(
                    "UPDATE pairs SET quality_score = ?, flags_json = ? WHERE id = ?",
                    (quality_score, json.dumps(flags), pair_id),
                )
            else:
                cursor = conn.execute(
                    "UPDATE pairs SET quality_score = ? WHERE id = ?",
                    (quality_score, pair_id),
                )
            return cursor.rowcount > 0

    def clear_pairs(self: JarvisDBBase) -> int:
        """Delete all pairs from the database."""
        with self.connection() as conn:
            conn.execute("DELETE FROM pair_embeddings")
            cursor = conn.execute("DELETE FROM pairs")
            return cursor.rowcount

    def update_da_classifications(
        self: JarvisDBBase,
        updates: list[tuple[int, str, float, str, float]],
    ) -> int:
        """Bulk update DA classifications for pairs.

        Args:
            updates: List of (pair_id, trigger_da_type, trigger_da_conf,
                             response_da_type, response_da_conf) tuples.

        Returns:
            Number of pairs updated.
        """
        with self.connection() as conn:
            cursor = conn.executemany(
                """UPDATE pairs SET
                   trigger_da_type = ?,
                   trigger_da_conf = ?,
                   response_da_type = ?,
                   response_da_conf = ?
                   WHERE id = ?""",
                [
                    (trigger_da, trigger_conf, response_da, response_conf, pair_id)
                    for pair_id, trigger_da, trigger_conf, response_da, response_conf in updates
                ],
            )
            return cursor.rowcount

    def update_cluster_assignments(
        self: JarvisDBBase,
        assignments: list[tuple[int, int]],
    ) -> int:
        """Bulk update cluster assignments for pairs.

        Args:
            assignments: List of (pair_id, cluster_id) tuples.
                        Use cluster_id=-1 for noise/outliers.

        Returns:
            Number of pairs updated.
        """
        with self.connection() as conn:
            cursor = conn.executemany(
                "UPDATE pairs SET cluster_id = ? WHERE id = ?",
                [(cluster_id, pair_id) for pair_id, cluster_id in assignments],
            )
            return cursor.rowcount

    @staticmethod
    def _row_to_pair(row: sqlite3.Row) -> Pair:
        """Convert a database row to a Pair object."""
        keys = row.keys()
        return Pair(
            id=row["id"],
            contact_id=row["contact_id"],
            trigger_text=row["trigger_text"],
            response_text=row["response_text"],
            trigger_timestamp=row["trigger_timestamp"],
            response_timestamp=row["response_timestamp"],
            chat_id=row["chat_id"],
            trigger_msg_id=row["trigger_msg_id"] if "trigger_msg_id" in keys else None,
            response_msg_id=row["response_msg_id"] if "response_msg_id" in keys else None,
            trigger_msg_ids_json=row["trigger_msg_ids_json"]
            if "trigger_msg_ids_json" in keys
            else None,
            response_msg_ids_json=row["response_msg_ids_json"]
            if "response_msg_ids_json" in keys
            else None,
            context_text=row["context_text"] if "context_text" in keys else None,
            quality_score=row["quality_score"] if "quality_score" in keys else 1.0,
            flags_json=row["flags_json"] if "flags_json" in keys else None,
            is_group=bool(row["is_group"]) if "is_group" in keys else False,
            is_holdout=bool(row["is_holdout"]) if "is_holdout" in keys else False,
            gate_a_passed=(
                bool(row["gate_a_passed"])
                if "gate_a_passed" in keys and row["gate_a_passed"] is not None
                else None
            ),
            gate_b_score=(
                float(row["gate_b_score"])
                if "gate_b_score" in keys and row["gate_b_score"] is not None
                else None
            ),
            gate_c_verdict=row["gate_c_verdict"] if "gate_c_verdict" in keys else None,
            validity_status=row["validity_status"] if "validity_status" in keys else None,
            trigger_da_type=row["trigger_da_type"] if "trigger_da_type" in keys else None,
            trigger_da_conf=(
                float(row["trigger_da_conf"])
                if "trigger_da_conf" in keys and row["trigger_da_conf"] is not None
                else None
            ),
            response_da_type=row["response_da_type"] if "response_da_type" in keys else None,
            response_da_conf=(
                float(row["response_da_conf"])
                if "response_da_conf" in keys and row["response_da_conf"] is not None
                else None
            ),
            cluster_id=(
                int(row["cluster_id"])
                if "cluster_id" in keys and row["cluster_id"] is not None
                else None
            ),
        )
