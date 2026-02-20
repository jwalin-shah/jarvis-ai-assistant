from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import orjson

from jarvis.infrastructure.cache.base import CacheBackend

T = TypeVar("T")


class SQLiteBackend(CacheBackend):
    """Persistent SQLite-based cache backend with tag support."""

    def __init__(self, db_path: str | Path, default_ttl: float = 3600.0) -> None:
        self.db_path = Path(db_path)
        self.default_ttl = default_ttl
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    expires_at REAL,
                    created_at REAL,
                    tags TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at)")
            # In a real app we might want a separate tags table for better query performance
            # but for simplicity we'll use a JSON array in a TEXT column.

    def get(self, key: str) -> Any | None:
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM cache WHERE key = ? AND expires_at > ?",
                (key, now),
            )
            row = cursor.fetchone()
            if row:
                try:
                    return orjson.loads(row[0])
                except orjson.JSONDecodeError:
                    return None
            return None

    def set(
        self, key: str, value: Any, ttl: float | None = None, tags: list[str] | None = None
    ) -> None:
        ttl = ttl if ttl is not None else self.default_ttl
        expires_at = time.time() + ttl
        created_at = time.time()
        tags_json = orjson.dumps(tags or []).decode("utf-8")

        try:
            value_json = orjson.dumps(value).decode("utf-8")
        except (TypeError, ValueError):
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache (key, value, expires_at, created_at, tags)
                VALUES (?, ?, ?, ?, ?)
                """,
                (key, value_json, expires_at, created_at, tags_json),
            )

    def delete(self, key: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            return cursor.rowcount > 0

    def invalidate_by_tag(self, tag: str) -> int:
        count = 0
        with sqlite3.connect(self.db_path) as conn:
            # Simple but slow tag matching using LIKE
            # For better performance, we'd use a separate tags table.
            cursor = conn.execute("SELECT key, tags FROM cache")
            rows = cursor.fetchall()
            to_delete = []
            for key, tags_json in rows:
                try:
                    tags = orjson.loads(tags_json)
                    if tag in tags:
                        to_delete.append(key)
                except orjson.JSONDecodeError:
                    continue

            if to_delete:
                conn.executemany("DELETE FROM cache WHERE key = ?", [(k,) for k in to_delete])
                count = len(to_delete)
        return count

    def invalidate_by_pattern(self, pattern: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM cache WHERE key LIKE ?", (f"{pattern}%",))
            return cursor.rowcount

    def clear(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")

    def get_or_set(self, key: str, factory: Callable[[], T], ttl: float | None = None) -> T:
        """Get from cache or compute via factory and store."""
        cached = self.get(key)
        if cached is not None:
            return cast(T, cached)

        result = factory()
        self.set(key, result, ttl=ttl)
        return result

    def stats(self) -> dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            return {
                "entries": count,
                "db_path": str(self.db_path),
            }
