"""Cluster CRUD operations mixin."""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

from jarvis.db.models import Cluster

if TYPE_CHECKING:
    from jarvis.db.core import JarvisDBBase


class ClusterMixin:
    """Mixin providing cluster CRUD operations."""

    def add_cluster(
        self: JarvisDBBase,
        name: str,
        description: str | None = None,
        example_triggers: list[str] | None = None,
        example_responses: list[str] | None = None,
    ) -> Cluster:
        """Add or update a cluster."""
        with self.connection() as conn:
            triggers_json = json.dumps(example_triggers or [])
            responses_json = json.dumps(example_responses or [])

            conn.execute(
                """
                INSERT INTO clusters (name, description, example_triggers, example_responses)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description = excluded.description,
                    example_triggers = excluded.example_triggers,
                    example_responses = excluded.example_responses
                """,
                (name, description, triggers_json, responses_json),
            )

            cursor = conn.execute("SELECT * FROM clusters WHERE name = ?", (name,))
            row = cursor.fetchone()

            return Cluster(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                example_triggers=json.loads(row["example_triggers"])
                if row["example_triggers"]
                else [],
                example_responses=json.loads(row["example_responses"])
                if row["example_responses"]
                else [],
                created_at=row["created_at"],
            )

    def get_cluster(self: JarvisDBBase, cluster_id: int) -> Cluster | None:
        """Get a cluster by ID."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT * FROM clusters WHERE id = ?", (cluster_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_cluster(row)
            return None

    def get_cluster_by_name(self: JarvisDBBase, name: str) -> Cluster | None:
        """Get a cluster by name."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT * FROM clusters WHERE name = ?", (name,))
            row = cursor.fetchone()
            if row:
                return self._row_to_cluster(row)
            return None

    def list_clusters(self: JarvisDBBase) -> list[Cluster]:
        """List all clusters."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT * FROM clusters ORDER BY name")
            return [self._row_to_cluster(row) for row in cursor]

    def update_cluster_label(
        self: JarvisDBBase, cluster_id: int, name: str, description: str | None = None
    ) -> bool:
        """Update a cluster's name and description."""
        with self.connection() as conn:
            if description is not None:
                cursor = conn.execute(
                    "UPDATE clusters SET name = ?, description = ? WHERE id = ?",
                    (name, description, cluster_id),
                )
            else:
                cursor = conn.execute(
                    "UPDATE clusters SET name = ? WHERE id = ?",
                    (name, cluster_id),
                )
            return cursor.rowcount > 0

    def clear_clusters(self: JarvisDBBase) -> int:
        """Delete all clusters."""
        with self.connection() as conn:
            conn.execute("UPDATE pair_embeddings SET cluster_id = NULL")
            cursor = conn.execute("DELETE FROM clusters")
            return cursor.rowcount

    def get_clusters_batch(self: JarvisDBBase, cluster_ids: list[int]) -> dict[int, Cluster]:
        """Batch fetch clusters by IDs.

        More efficient than calling get_cluster in a loop.
        Uses chunking to handle large input lists.

        Args:
            cluster_ids: List of cluster IDs to fetch.

        Returns:
            Dict mapping cluster_id -> Cluster for found clusters.
        """
        if not cluster_ids:
            return {}

        # Filter out None values and get unique IDs
        valid_ids = list({cid for cid in cluster_ids if cid is not None})
        if not valid_ids:
            return {}

        result = {}
        with self.connection() as conn:
            for i in range(0, len(valid_ids), 900):
                chunk = valid_ids[i : i + 900]
                placeholders = ",".join("?" * len(chunk))
                cursor = conn.execute(
                    f"SELECT * FROM clusters WHERE id IN ({placeholders})",
                    chunk,
                )
                for row in cursor:
                    result[row["id"]] = self._row_to_cluster(row)
        return result

    @staticmethod
    def _row_to_cluster(row: sqlite3.Row) -> Cluster:
        """Convert a database row to a Cluster object."""
        return Cluster(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            example_triggers=json.loads(row["example_triggers"]) if row["example_triggers"] else [],
            example_responses=json.loads(row["example_responses"])
            if row["example_responses"]
            else [],
            created_at=row["created_at"],
        )
