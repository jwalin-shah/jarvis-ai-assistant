"""Index version management mixin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jarvis.db.models import IndexVersion

if TYPE_CHECKING:
    pass


class IndexVersionMixin:
    """Mixin providing FAISS index version management."""

    def add_index_version(
        self: Any,
        version_id: str,
        model_name: str,
        embedding_dim: int,
        num_vectors: int,
        index_path: str,
        is_active: bool = False,
    ) -> IndexVersion:
        """Add a new index version."""
        with self.connection() as conn:
            # If setting as active, deactivate others
            if is_active:
                conn.execute("UPDATE index_versions SET is_active = FALSE")

            cursor = conn.execute(
                """
                INSERT INTO index_versions
                (version_id, model_name, embedding_dim, num_vectors, index_path, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (version_id, model_name, embedding_dim, num_vectors, index_path, is_active),
            )

            return IndexVersion(
                id=cursor.lastrowid,
                version_id=version_id,
                model_name=model_name,
                embedding_dim=embedding_dim,
                num_vectors=num_vectors,
                index_path=index_path,
                is_active=is_active,
            )

    def get_active_index(self: Any) -> IndexVersion | None:
        """Get the currently active index version."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT * FROM index_versions WHERE is_active = TRUE LIMIT 1")
            row = cursor.fetchone()
            if row:
                return IndexVersion(
                    id=row["id"],
                    version_id=row["version_id"],
                    model_name=row["model_name"],
                    embedding_dim=row["embedding_dim"],
                    num_vectors=row["num_vectors"],
                    index_path=row["index_path"],
                    is_active=row["is_active"],
                    created_at=row["created_at"],
                )
            return None

    def set_active_index(self: Any, version_id: str) -> bool:
        """Set the active index version."""
        with self.connection() as conn:
            conn.execute("UPDATE index_versions SET is_active = FALSE")
            cursor = conn.execute(
                "UPDATE index_versions SET is_active = TRUE WHERE version_id = ?",
                (version_id,),
            )
            result: bool = cursor.rowcount > 0
            return result

    def list_index_versions(self: Any) -> list[IndexVersion]:
        """List all index versions."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT * FROM index_versions ORDER BY created_at DESC")
            return [
                IndexVersion(
                    id=row["id"],
                    version_id=row["version_id"],
                    model_name=row["model_name"],
                    embedding_dim=row["embedding_dim"],
                    num_vectors=row["num_vectors"],
                    index_path=row["index_path"],
                    is_active=row["is_active"],
                    created_at=row["created_at"],
                )
                for row in cursor
            ]
