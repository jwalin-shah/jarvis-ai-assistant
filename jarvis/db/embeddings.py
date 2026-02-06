"""Embedding CRUD and FAISS lookup operations mixin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jarvis.db.models import Pair, PairEmbedding

if TYPE_CHECKING:
    from jarvis.db.core import JarvisDBBase


class EmbeddingMixin:
    """Mixin providing embedding CRUD and FAISS lookup operations."""

    def add_embedding(
        self: JarvisDBBase,
        pair_id: int,
        faiss_id: int,
        cluster_id: int | None = None,
        index_version: str | None = None,
    ) -> PairEmbedding:
        """Add or update a FAISS embedding reference."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pair_embeddings
                (pair_id, faiss_id, cluster_id, index_version)
                VALUES (?, ?, ?, ?)
                """,
                (pair_id, faiss_id, cluster_id, index_version),
            )
            return PairEmbedding(
                pair_id=pair_id,
                faiss_id=faiss_id,
                cluster_id=cluster_id,
                index_version=index_version,
            )

    def add_embeddings_bulk(self: JarvisDBBase, embeddings: list[dict[str, Any]]) -> int:
        """Add multiple embeddings in a single transaction."""
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO pair_embeddings
                (pair_id, faiss_id, cluster_id, index_version)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (e["pair_id"], e["faiss_id"], e.get("cluster_id"), e.get("index_version"))
                    for e in embeddings
                ],
            )
            return len(embeddings)

    def get_embedding_by_pair(self: JarvisDBBase, pair_id: int) -> PairEmbedding | None:
        """Get embedding by pair ID (stable key)."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT * FROM pair_embeddings WHERE pair_id = ?", (pair_id,))
            row = cursor.fetchone()
            if row:
                return PairEmbedding(
                    pair_id=row["pair_id"],
                    faiss_id=row["faiss_id"],
                    cluster_id=row["cluster_id"],
                    index_version=row["index_version"] if "index_version" in row.keys() else None,
                )
            return None

    def get_pair_by_faiss_id(
        self: JarvisDBBase, faiss_id: int, index_version: str | None = None
    ) -> Pair | None:
        """Get the pair associated with a FAISS ID."""
        with self.connection() as conn:
            if index_version:
                cursor = conn.execute(
                    """
                    SELECT p.* FROM pairs p
                    JOIN pair_embeddings e ON p.id = e.pair_id
                    WHERE e.faiss_id = ? AND e.index_version = ?
                    """,
                    (faiss_id, index_version),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT p.* FROM pairs p
                    JOIN pair_embeddings e ON p.id = e.pair_id
                    WHERE e.faiss_id = ?
                    """,
                    (faiss_id,),
                )
            row = cursor.fetchone()
            if row:
                return self._row_to_pair(row)
            return None

    def get_pairs_by_faiss_ids(
        self: JarvisDBBase, faiss_ids: list[int], index_version: str | None = None
    ) -> dict[int, Pair]:
        """Batch fetch pairs by FAISS IDs.

        More efficient than calling get_pair_by_faiss_id in a loop.
        Uses chunking to handle large input lists.

        Args:
            faiss_ids: List of FAISS IDs to fetch.
            index_version: Optional index version to filter by.

        Returns:
            Dict mapping faiss_id -> Pair for found pairs.
        """
        if not faiss_ids:
            return {}

        result = {}
        with self.connection() as conn:
            for i in range(0, len(faiss_ids), 900):
                chunk = faiss_ids[i : i + 900]
                placeholders = ",".join("?" * len(chunk))
                if index_version:
                    cursor = conn.execute(
                        f"""
                        SELECT p.*, e.faiss_id FROM pairs p
                        JOIN pair_embeddings e ON p.id = e.pair_id
                        WHERE e.faiss_id IN ({placeholders}) AND e.index_version = ?
                        """,
                        (*chunk, index_version),
                    )
                else:
                    cursor = conn.execute(
                        f"""
                        SELECT p.*, e.faiss_id FROM pairs p
                        JOIN pair_embeddings e ON p.id = e.pair_id
                        WHERE e.faiss_id IN ({placeholders})
                        """,
                        chunk,
                    )
                for row in cursor:
                    pair = self._row_to_pair(row)
                    result[row["faiss_id"]] = pair
        return result

    def get_embeddings_by_pair_ids(
        self: JarvisDBBase, pair_ids: list[int]
    ) -> dict[int, PairEmbedding]:
        """Batch fetch embeddings by pair IDs.

        More efficient than calling get_embedding_by_pair in a loop.
        Uses chunking to handle large input lists.

        Args:
            pair_ids: List of pair IDs to fetch embeddings for.

        Returns:
            Dict mapping pair_id -> PairEmbedding for found embeddings.
        """
        if not pair_ids:
            return {}

        result = {}
        with self.connection() as conn:
            for i in range(0, len(pair_ids), 900):
                chunk = pair_ids[i : i + 900]
                placeholders = ",".join("?" * len(chunk))
                cursor = conn.execute(
                    f"SELECT * FROM pair_embeddings WHERE pair_id IN ({placeholders})",
                    chunk,
                )
                for row in cursor:
                    result[row["pair_id"]] = PairEmbedding(
                        pair_id=row["pair_id"],
                        faiss_id=row["faiss_id"],
                        cluster_id=row["cluster_id"],
                        index_version=row["index_version"]
                        if "index_version" in row.keys()
                        else None,
                    )
        return result

    def get_pairs_with_clusters_by_faiss_ids(
        self: JarvisDBBase,
        faiss_ids: list[int],
        index_version: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Batch fetch pairs with embedding and cluster info in a single query.

        Consolidates 3 queries (pairs, embeddings, clusters) into 1 with JOINs.
        More efficient for search_with_pairs which needs all this data.
        Uses chunking to handle large input lists.

        Args:
            faiss_ids: List of FAISS IDs to fetch.
            index_version: Optional index version to filter by.
            limit: Optional limit on results (for safety on large result sets).

        Returns:
            List of dicts with pair, embedding, and cluster info.
        """
        if not faiss_ids:
            return []

        # Apply limit if specified
        ids_to_query = faiss_ids[:limit] if limit else faiss_ids
        results = []

        with self.connection() as conn:
            for i in range(0, len(ids_to_query), 900):
                chunk = ids_to_query[i : i + 900]
                placeholders = ",".join("?" * len(chunk))
                if index_version:
                    cursor = conn.execute(
                        f"""
                        SELECT p.*, e.faiss_id, e.cluster_id as embedding_cluster_id,
                               c.name as cluster_name, c.description as cluster_description
                        FROM pairs p
                        JOIN pair_embeddings e ON p.id = e.pair_id
                        LEFT JOIN clusters c ON e.cluster_id = c.id
                        WHERE e.faiss_id IN ({placeholders}) AND e.index_version = ?
                        """,
                        (*chunk, index_version),
                    )
                else:
                    cursor = conn.execute(
                        f"""
                        SELECT p.*, e.faiss_id, e.cluster_id as embedding_cluster_id,
                               c.name as cluster_name, c.description as cluster_description
                        FROM pairs p
                        JOIN pair_embeddings e ON p.id = e.pair_id
                        LEFT JOIN clusters c ON e.cluster_id = c.id
                        WHERE e.faiss_id IN ({placeholders})
                        """,
                        chunk,
                    )

                for row in cursor:
                    pair = self._row_to_pair(row)
                    results.append(
                        {
                            "pair": pair,
                            "faiss_id": row["faiss_id"],
                            "cluster_id": row["embedding_cluster_id"],
                            "cluster_name": row["cluster_name"],
                            "cluster_description": row["cluster_description"],
                        }
                    )
        return results

    def clear_embeddings(self: JarvisDBBase, index_version: str | None = None) -> int:
        """Delete embeddings, optionally for a specific index version."""
        with self.connection() as conn:
            if index_version:
                cursor = conn.execute(
                    "DELETE FROM pair_embeddings WHERE index_version = ?",
                    (index_version,),
                )
            else:
                cursor = conn.execute("DELETE FROM pair_embeddings")
            return cursor.rowcount

    def count_embeddings(self: JarvisDBBase, index_version: str | None = None) -> int:
        """Count embeddings."""
        with self.connection() as conn:
            if index_version:
                cursor = conn.execute(
                    "SELECT COUNT(*) as cnt FROM pair_embeddings WHERE index_version = ?",
                    (index_version,),
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) as cnt FROM pair_embeddings")
            row = cursor.fetchone()
            return row["cnt"] if row else 0
