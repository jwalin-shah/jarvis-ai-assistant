"""Multi-tier caching system for speculative prefetching.

Provides a hierarchical cache with automatic tier promotion/demotion:
- L1: In-memory LRU (hot data, <1ms latency)
- L2: SQLite cache table (warm data, <10ms latency)
- L3: Disk cache (cold data, <50ms latency)

Usage:
    cache = MultiTierCache()
    cache.set("draft:chat123", draft_response, tier=CacheTier.L1)
    result = cache.get("draft:chat123")  # Checks L1 -> L2 -> L3
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from jarvis.errors import ErrorCode, JarvisError

logger = logging.getLogger(__name__)

# Type variable for cache values
T = TypeVar("T")

# Default cache directory
CACHE_DIR = Path.home() / ".jarvis" / "cache"

# Default SQLite cache path
CACHE_DB_PATH = CACHE_DIR / "prefetch_cache.db"


class CacheError(JarvisError):
    """Cache operation failed."""

    default_message = "Cache operation failed"
    default_code = ErrorCode.UNKNOWN


class CacheTier(IntEnum):
    """Cache tier levels."""

    L1 = 1  # In-memory LRU (<1ms)
    L2 = 2  # SQLite (<10ms)
    L3 = 3  # Disk file (<50ms)


@dataclass
class CacheEntry:
    """A cache entry with metadata."""

    key: str
    value: Any
    tier: CacheTier
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    ttl_seconds: float = 300.0  # 5 minute default TTL
    access_count: int = 0
    size_bytes: int = 0
    tags: list[str] = field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() - self.created_at > self.ttl_seconds

    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics for monitoring."""

    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    l3_hits: int = 0
    l3_misses: int = 0
    promotions: int = 0
    demotions: int = 0
    evictions: int = 0
    total_bytes: int = 0

    @property
    def l1_hit_rate(self) -> float:
        """L1 cache hit rate."""
        total = self.l1_hits + self.l1_misses
        return self.l1_hits / total if total > 0 else 0.0

    @property
    def l2_hit_rate(self) -> float:
        """L2 cache hit rate."""
        total = self.l2_hits + self.l2_misses
        return self.l2_hits / total if total > 0 else 0.0

    @property
    def overall_hit_rate(self) -> float:
        """Overall cache hit rate (any tier)."""
        hits = self.l1_hits + self.l2_hits + self.l3_hits
        total = hits + self.l3_misses  # L3 miss = complete miss
        return hits / total if total > 0 else 0.0


class L1Cache:
    """In-memory LRU cache with TTL support (<1ms access).

    Thread-safe implementation using OrderedDict for LRU tracking.
    """

    def __init__(
        self,
        maxsize: int = 1000,
        max_bytes: int = 100 * 1024 * 1024,  # 100MB default
    ) -> None:
        """Initialize L1 cache.

        Args:
            maxsize: Maximum number of entries.
            max_bytes: Maximum total size in bytes.
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._maxsize = maxsize
        self._max_bytes = max_bytes
        self._current_bytes = 0
        self._lock = threading.RLock()

    def get(self, key: str) -> CacheEntry | None:
        """Get entry from cache.

        Args:
            key: Cache key.

        Returns:
            CacheEntry if found and not expired, None otherwise.
        """
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]
            if entry.is_expired:
                self._remove(key)
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            return entry

    def set(self, key: str, entry: CacheEntry) -> None:
        """Set entry in cache.

        Args:
            key: Cache key.
            entry: Cache entry.
        """
        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                self._remove(key)

            # Evict entries if needed
            while (
                len(self._cache) >= self._maxsize
                or self._current_bytes + entry.size_bytes > self._max_bytes
            ) and self._cache:
                # Remove oldest entry
                oldest_key = next(iter(self._cache))
                self._remove(oldest_key)

            # Add new entry
            entry.tier = CacheTier.L1
            self._cache[key] = entry
            self._current_bytes += entry.size_bytes

    def remove(self, key: str) -> bool:
        """Remove entry from cache.

        Args:
            key: Cache key.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            return self._remove(key)

    def _remove(self, key: str) -> bool:
        """Internal remove without lock."""
        if key not in self._cache:
            return False
        entry = self._cache.pop(key)
        self._current_bytes -= entry.size_bytes
        return True

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._current_bytes = 0

    def keys(self) -> list[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self._cache.keys())

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "entries": len(self._cache),
                "maxsize": self._maxsize,
                "bytes": self._current_bytes,
                "max_bytes": self._max_bytes,
            }


class L2Cache:
    """SQLite-backed cache with TTL support (<10ms access).

    Provides persistent caching with automatic expiration cleanup.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize L2 cache.

        Args:
            db_path: Path to SQLite database file.
        """
        self._db_path = db_path or CACHE_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create thread-local connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self._db_path),
                timeout=10.0,
                check_same_thread=False,
            )
            self._local.connection.row_factory = sqlite3.Row
            # Optimize for read-heavy workloads
            self._local.connection.execute("PRAGMA journal_mode = WAL")
            self._local.connection.execute("PRAGMA synchronous = NORMAL")
            self._local.connection.execute("PRAGMA cache_size = -4000")  # 4MB
            self._local.connection.execute("PRAGMA temp_store = MEMORY")
        return self._local.connection

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                value BLOB NOT NULL,
                value_type TEXT NOT NULL,
                created_at REAL NOT NULL,
                accessed_at REAL NOT NULL,
                ttl_seconds REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                size_bytes INTEGER DEFAULT 0,
                tags_json TEXT DEFAULT '[]'
            );

            CREATE INDEX IF NOT EXISTS idx_cache_created_at ON cache_entries(created_at);
            CREATE INDEX IF NOT EXISTS idx_cache_accessed_at ON cache_entries(accessed_at);
            CREATE INDEX IF NOT EXISTS idx_cache_ttl ON cache_entries(created_at, ttl_seconds);
            """
        )
        conn.commit()

    def get(self, key: str) -> CacheEntry | None:
        """Get entry from cache.

        Args:
            key: Cache key.

        Returns:
            CacheEntry if found and not expired, None otherwise.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT key, value, value_type, created_at, accessed_at,
                   ttl_seconds, access_count, size_bytes, tags_json
            FROM cache_entries
            WHERE key = ?
            """,
            (key,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        # Check expiration
        created_at = row["created_at"]
        ttl_seconds = row["ttl_seconds"]
        if time.time() - created_at > ttl_seconds:
            self.remove(key)
            return None

        # Deserialize value
        value = self._deserialize(row["value"], row["value_type"])

        # Parse tags
        try:
            tags = json.loads(row["tags_json"]) if row["tags_json"] else []
        except json.JSONDecodeError:
            tags = []

        entry = CacheEntry(
            key=row["key"],
            value=value,
            tier=CacheTier.L2,
            created_at=row["created_at"],
            accessed_at=row["accessed_at"],
            ttl_seconds=row["ttl_seconds"],
            access_count=row["access_count"],
            size_bytes=row["size_bytes"],
            tags=tags,
        )

        # Update access time
        conn.execute(
            """
            UPDATE cache_entries
            SET accessed_at = ?, access_count = access_count + 1
            WHERE key = ?
            """,
            (time.time(), key),
        )
        conn.commit()

        entry.touch()
        return entry

    def set(self, key: str, entry: CacheEntry) -> None:
        """Set entry in cache.

        Args:
            key: Cache key.
            entry: Cache entry.
        """
        conn = self._get_connection()
        value_blob, value_type = self._serialize(entry.value)
        tags_json = json.dumps(entry.tags) if entry.tags else "[]"

        conn.execute(
            """
            INSERT OR REPLACE INTO cache_entries
            (key, value, value_type, created_at, accessed_at, ttl_seconds,
             access_count, size_bytes, tags_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                value_blob,
                value_type,
                entry.created_at,
                entry.accessed_at,
                entry.ttl_seconds,
                entry.access_count,
                entry.size_bytes,
                tags_json,
            ),
        )
        conn.commit()

    def remove(self, key: str) -> bool:
        """Remove entry from cache.

        Args:
            key: Cache key.

        Returns:
            True if removed, False if not found.
        """
        conn = self._get_connection()
        cursor = conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
        conn.commit()
        return cursor.rowcount > 0

    def remove_by_tag(self, tag: str) -> int:
        """Remove all entries with a specific tag.

        Args:
            tag: Tag to match.

        Returns:
            Number of entries removed.
        """
        conn = self._get_connection()
        # Use JSON contains pattern
        cursor = conn.execute(
            """
            DELETE FROM cache_entries
            WHERE tags_json LIKE ?
            """,
            (f'%"{tag}"%',),
        )
        conn.commit()
        return cursor.rowcount

    def remove_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            DELETE FROM cache_entries
            WHERE (? - created_at) > ttl_seconds
            """,
            (time.time(),),
        )
        conn.commit()
        return cursor.rowcount

    def clear(self) -> None:
        """Clear all entries."""
        conn = self._get_connection()
        conn.execute("DELETE FROM cache_entries")
        conn.commit()

    def keys(self) -> list[str]:
        """Get all cache keys."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT key FROM cache_entries")
        return [row["key"] for row in cursor.fetchall()]

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT
                COUNT(*) as entries,
                SUM(size_bytes) as total_bytes,
                AVG(access_count) as avg_access_count
            FROM cache_entries
            """
        )
        row = cursor.fetchone()
        return {
            "entries": row["entries"] or 0,
            "total_bytes": row["total_bytes"] or 0,
            "avg_access_count": row["avg_access_count"] or 0,
        }

    def _serialize(self, value: Any) -> tuple[bytes, str]:
        """Serialize value to bytes.

        Args:
            value: Value to serialize.

        Returns:
            Tuple of (serialized bytes, type string).
        """
        if isinstance(value, np.ndarray):
            # Efficient numpy serialization
            buffer = value.tobytes()
            # Prepend shape and dtype info
            shape_bytes = struct.pack(f"{len(value.shape)}I", *value.shape)
            dtype_bytes = str(value.dtype).encode()
            header = struct.pack("II", len(shape_bytes), len(dtype_bytes))
            return header + shape_bytes + dtype_bytes + buffer, "numpy"
        elif isinstance(value, (dict, list)):
            return json.dumps(value).encode(), "json"
        elif isinstance(value, str):
            return value.encode(), "str"
        elif isinstance(value, bytes):
            return value, "bytes"
        else:
            # Convert to JSON-compatible representation (no pickle for security)
            try:
                return json.dumps({"__repr__": repr(value)}).encode(), "json"
            except (TypeError, ValueError):
                return json.dumps({"__repr__": str(value)}).encode(), "json"

    def _deserialize(self, data: bytes, value_type: str) -> Any:
        """Deserialize bytes to value.

        Args:
            data: Serialized bytes.
            value_type: Type string.

        Returns:
            Deserialized value.
        """
        if value_type == "numpy":
            # Parse header
            shape_len, dtype_len = struct.unpack("II", data[:8])
            shape = struct.unpack(f"{shape_len // 4}I", data[8 : 8 + shape_len])
            dtype_str = data[8 + shape_len : 8 + shape_len + dtype_len].decode()
            buffer = data[8 + shape_len + dtype_len :]
            return np.frombuffer(buffer, dtype=np.dtype(dtype_str)).reshape(shape)
        elif value_type == "json":
            return json.loads(data.decode())
        elif value_type == "str":
            return data.decode()
        elif value_type == "bytes":
            return data
        elif value_type == "pickle":
            # Legacy pickle data: clear entry instead of deserializing (security risk)
            logger.warning("Refusing to deserialize pickle data (security risk). Returning None.")
            return None
        else:
            logger.warning("Unknown value_type '%s', returning None.", value_type)
            return None


class L3Cache:
    """Disk-based cache with TTL support (<50ms access).

    Stores large items as individual files for efficient access.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        max_bytes: int = 1024 * 1024 * 1024,  # 1GB default
    ) -> None:
        """Initialize L3 cache.

        Args:
            cache_dir: Directory for cache files.
            max_bytes: Maximum total size in bytes.
        """
        self._cache_dir = cache_dir or (CACHE_DIR / "l3")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_bytes
        self._lock = threading.RLock()
        self._metadata_path = self._cache_dir / "_metadata.json"
        self._metadata: dict[str, dict] = self._load_metadata()

    def _load_metadata(self) -> dict[str, dict]:
        """Load metadata from disk."""
        if self._metadata_path.exists():
            try:
                return json.loads(self._metadata_path.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        try:
            self._metadata_path.write_text(json.dumps(self._metadata))
        except OSError as e:
            logger.warning(f"Failed to save L3 cache metadata: {e}")

    def _key_to_path(self, key: str) -> Path:
        """Convert cache key to file path."""
        # Use hash to avoid filesystem issues with special characters
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self._cache_dir / f"{key_hash}.cache"

    def get(self, key: str) -> CacheEntry | None:
        """Get entry from cache.

        Args:
            key: Cache key.

        Returns:
            CacheEntry if found and not expired, None otherwise.
        """
        with self._lock:
            if key not in self._metadata:
                return None

            meta = self._metadata[key]
            created_at = meta.get("created_at", 0)
            ttl_seconds = meta.get("ttl_seconds", 300)

            if time.time() - created_at > ttl_seconds:
                self.remove(key)
                return None

            file_path = self._key_to_path(key)
            if not file_path.exists():
                del self._metadata[key]
                self._save_metadata()
                return None

            try:
                with open(file_path, "rb") as f:
                    data = f.read()
                value = self._deserialize(data, meta.get("value_type", "json"))
            except (OSError, json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to read L3 cache entry {key}: {e}")
                self.remove(key)
                return None

            # Parse tags
            tags = meta.get("tags", [])

            entry = CacheEntry(
                key=key,
                value=value,
                tier=CacheTier.L3,
                created_at=meta.get("created_at", time.time()),
                accessed_at=meta.get("accessed_at", time.time()),
                ttl_seconds=meta.get("ttl_seconds", 300),
                access_count=meta.get("access_count", 0),
                size_bytes=meta.get("size_bytes", 0),
                tags=tags,
            )

            # Update metadata
            meta["accessed_at"] = time.time()
            meta["access_count"] = meta.get("access_count", 0) + 1
            self._save_metadata()

            entry.touch()
            return entry

    def set(self, key: str, entry: CacheEntry) -> None:
        """Set entry in cache.

        Args:
            key: Cache key.
            entry: Cache entry.
        """
        with self._lock:
            # Check space and evict if needed
            total_bytes = sum(m.get("size_bytes", 0) for m in self._metadata.values())
            while total_bytes + entry.size_bytes > self._max_bytes and self._metadata:
                # Evict least recently accessed
                oldest_key = min(
                    self._metadata.keys(),
                    key=lambda k: self._metadata[k].get("accessed_at", 0),
                )
                self.remove(oldest_key)
                total_bytes = sum(m.get("size_bytes", 0) for m in self._metadata.values())

            # Serialize and write
            file_path = self._key_to_path(key)
            data, value_type = self._serialize(entry.value)

            try:
                with open(file_path, "wb") as f:
                    f.write(data)
            except OSError as e:
                logger.warning(f"Failed to write L3 cache entry {key}: {e}")
                return

            # Update metadata
            self._metadata[key] = {
                "created_at": entry.created_at,
                "accessed_at": entry.accessed_at,
                "ttl_seconds": entry.ttl_seconds,
                "access_count": entry.access_count,
                "size_bytes": len(data),
                "value_type": value_type,
                "tags": entry.tags,
            }
            self._save_metadata()

    def remove(self, key: str) -> bool:
        """Remove entry from cache.

        Args:
            key: Cache key.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            if key not in self._metadata:
                return False

            file_path = self._key_to_path(key)
            try:
                file_path.unlink(missing_ok=True)
            except OSError:
                pass

            del self._metadata[key]
            self._save_metadata()
            return True

    def remove_by_tag(self, tag: str) -> int:
        """Remove all entries with a specific tag.

        Args:
            tag: Tag to match.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            keys_to_remove = [
                k for k, meta in self._metadata.items() if tag in meta.get("tags", [])
            ]
            for key in keys_to_remove:
                self.remove(key)
            return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            for key in list(self._metadata.keys()):
                self.remove(key)

    def keys(self) -> list[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self._metadata.keys())

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_bytes = sum(m.get("size_bytes", 0) for m in self._metadata.values())
            return {
                "entries": len(self._metadata),
                "total_bytes": total_bytes,
                "max_bytes": self._max_bytes,
            }

    def _serialize(self, value: Any) -> tuple[bytes, str]:
        """Serialize value to bytes."""
        if isinstance(value, np.ndarray):
            buffer = value.tobytes()
            shape_bytes = struct.pack(f"{len(value.shape)}I", *value.shape)
            dtype_bytes = str(value.dtype).encode()
            header = struct.pack("II", len(shape_bytes), len(dtype_bytes))
            return header + shape_bytes + dtype_bytes + buffer, "numpy"
        elif isinstance(value, (dict, list)):
            return json.dumps(value).encode(), "json"
        elif isinstance(value, str):
            return value.encode(), "str"
        elif isinstance(value, bytes):
            return value, "bytes"
        else:
            # Convert to JSON-compatible representation (no pickle for security)
            try:
                return json.dumps({"__repr__": repr(value)}).encode(), "json"
            except (TypeError, ValueError):
                return json.dumps({"__repr__": str(value)}).encode(), "json"

    def _deserialize(self, data: bytes, value_type: str) -> Any:
        """Deserialize bytes to value."""
        if value_type == "numpy":
            shape_len, dtype_len = struct.unpack("II", data[:8])
            shape = struct.unpack(f"{shape_len // 4}I", data[8 : 8 + shape_len])
            dtype_str = data[8 + shape_len : 8 + shape_len + dtype_len].decode()
            buffer = data[8 + shape_len + dtype_len :]
            return np.frombuffer(buffer, dtype=np.dtype(dtype_str)).reshape(shape)
        elif value_type == "json":
            return json.loads(data.decode())
        elif value_type == "str":
            return data.decode()
        elif value_type == "bytes":
            return data
        elif value_type == "pickle":
            # Legacy pickle data: clear entry instead of deserializing (security risk)
            logger.warning("Refusing to deserialize pickle data (security risk). Returning None.")
            return None
        else:
            logger.warning("Unknown value_type '%s', returning None.", value_type)
            return None


class MultiTierCache:
    """Multi-tier cache with automatic tier promotion/demotion.

    Provides hierarchical caching with:
    - L1: In-memory LRU (hot data, <1ms)
    - L2: SQLite (warm data, <10ms)
    - L3: Disk files (cold data, <50ms)

    Items are automatically promoted to faster tiers on access
    and demoted to slower tiers when evicted.
    """

    def __init__(
        self,
        l1_maxsize: int = 1000,
        l1_max_bytes: int = 100 * 1024 * 1024,  # 100MB
        l2_db_path: Path | None = None,
        l3_cache_dir: Path | None = None,
        l3_max_bytes: int = 1024 * 1024 * 1024,  # 1GB
        auto_promote: bool = True,
        promote_threshold: int = 3,  # Promote after N accesses
    ) -> None:
        """Initialize multi-tier cache.

        Args:
            l1_maxsize: Maximum L1 entries.
            l1_max_bytes: Maximum L1 size in bytes.
            l2_db_path: Path to L2 SQLite database.
            l3_cache_dir: Directory for L3 cache files.
            l3_max_bytes: Maximum L3 size in bytes.
            auto_promote: Whether to auto-promote on access.
            promote_threshold: Number of accesses before promotion.
        """
        self._l1 = L1Cache(maxsize=l1_maxsize, max_bytes=l1_max_bytes)
        self._l2 = L2Cache(db_path=l2_db_path)
        self._l3 = L3Cache(cache_dir=l3_cache_dir, max_bytes=l3_max_bytes)
        self._auto_promote = auto_promote
        self._promote_threshold = promote_threshold
        self._stats = CacheStats()
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        """Get value from cache, checking all tiers.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found.
        """
        with self._lock:
            # Check L1
            entry = self._l1.get(key)
            if entry is not None:
                self._stats.l1_hits += 1
                return entry.value

            self._stats.l1_misses += 1

            # Check L2
            entry = self._l2.get(key)
            if entry is not None:
                self._stats.l2_hits += 1
                # Maybe promote to L1
                if self._auto_promote and entry.access_count >= self._promote_threshold:
                    self._promote(entry, CacheTier.L1)
                return entry.value

            self._stats.l2_misses += 1

            # Check L3
            entry = self._l3.get(key)
            if entry is not None:
                self._stats.l3_hits += 1
                # Maybe promote to L2
                if self._auto_promote and entry.access_count >= self._promote_threshold:
                    self._promote(entry, CacheTier.L2)
                return entry.value

            self._stats.l3_misses += 1
            return None

    def set(
        self,
        key: str,
        value: Any,
        tier: CacheTier = CacheTier.L2,
        ttl_seconds: float = 300.0,
        tags: list[str] | None = None,
    ) -> None:
        """Set value in cache at specified tier.

        Args:
            key: Cache key.
            value: Value to cache.
            tier: Initial tier (L1, L2, or L3).
            ttl_seconds: Time-to-live in seconds.
            tags: Optional tags for grouping/invalidation.
        """
        with self._lock:
            # Estimate size
            size_bytes = self._estimate_size(value)

            entry = CacheEntry(
                key=key,
                value=value,
                tier=tier,
                ttl_seconds=ttl_seconds,
                size_bytes=size_bytes,
                tags=tags or [],
            )

            self._stats.total_bytes += size_bytes

            if tier == CacheTier.L1:
                self._l1.set(key, entry)
            elif tier == CacheTier.L2:
                self._l2.set(key, entry)
            else:
                self._l3.set(key, entry)

    def remove(self, key: str) -> bool:
        """Remove entry from all tiers.

        Args:
            key: Cache key.

        Returns:
            True if removed from any tier.
        """
        with self._lock:
            removed = self._l1.remove(key)
            removed |= self._l2.remove(key)
            removed |= self._l3.remove(key)
            return removed

    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with a specific tag.

        Args:
            tag: Tag to match.

        Returns:
            Number of entries invalidated.
        """
        with self._lock:
            # L1 doesn't support tag filtering directly, so check all keys
            count = 0
            for key in self._l1.keys():
                entry = self._l1.get(key)
                if entry and tag in entry.tags:
                    self._l1.remove(key)
                    count += 1

            count += self._l2.remove_by_tag(tag)
            count += self._l3.remove_by_tag(tag)
            return count

    def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate entries matching a key pattern.

        Args:
            pattern: Key prefix to match.

        Returns:
            Number of entries invalidated.
        """
        with self._lock:
            count = 0
            # Check all tiers
            for key in self._l1.keys():
                if key.startswith(pattern):
                    self._l1.remove(key)
                    count += 1

            for key in self._l2.keys():
                if key.startswith(pattern):
                    self._l2.remove(key)
                    count += 1

            for key in self._l3.keys():
                if key.startswith(pattern):
                    self._l3.remove(key)
                    count += 1

            return count

    def clear(self) -> None:
        """Clear all caches."""
        with self._lock:
            self._l1.clear()
            self._l2.clear()
            self._l3.clear()
            self._stats = CacheStats()

    def cleanup_expired(self) -> int:
        """Remove expired entries from all tiers.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            count = 0
            # L1 - check all entries
            for key in self._l1.keys():
                entry = self._l1.get(key)
                if entry is None:  # Expired during get
                    count += 1

            # L2 has built-in expired removal
            count += self._l2.remove_expired()

            # L3 - check metadata
            for key in self._l3.keys():
                entry = self._l3.get(key)
                if entry is None:
                    count += 1

            self._stats.evictions += count
            return count

    def stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            return {
                "l1": self._l1.stats(),
                "l2": self._l2.stats(),
                "l3": self._l3.stats(),
                "hits": {
                    "l1": self._stats.l1_hits,
                    "l2": self._stats.l2_hits,
                    "l3": self._stats.l3_hits,
                },
                "misses": {
                    "l1": self._stats.l1_misses,
                    "l2": self._stats.l2_misses,
                    "l3": self._stats.l3_misses,
                },
                "hit_rates": {
                    "l1": self._stats.l1_hit_rate,
                    "l2": self._stats.l2_hit_rate,
                    "overall": self._stats.overall_hit_rate,
                },
                "promotions": self._stats.promotions,
                "demotions": self._stats.demotions,
                "evictions": self._stats.evictions,
            }

    def _promote(self, entry: CacheEntry, to_tier: CacheTier) -> None:
        """Promote entry to a faster tier.

        Args:
            entry: Entry to promote.
            to_tier: Target tier.
        """
        if to_tier >= entry.tier:
            return  # Already at or above target

        self._stats.promotions += 1

        if to_tier == CacheTier.L1:
            self._l1.set(entry.key, entry)
        elif to_tier == CacheTier.L2:
            self._l2.set(entry.key, entry)

    def _demote(self, entry: CacheEntry, to_tier: CacheTier) -> None:
        """Demote entry to a slower tier.

        Args:
            entry: Entry to demote.
            to_tier: Target tier.
        """
        if to_tier <= entry.tier:
            return  # Already at or below target

        self._stats.demotions += 1

        if to_tier == CacheTier.L2:
            self._l2.set(entry.key, entry)
            self._l1.remove(entry.key)
        elif to_tier == CacheTier.L3:
            self._l3.set(entry.key, entry)
            self._l2.remove(entry.key)

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        if isinstance(value, np.ndarray):
            return value.nbytes
        elif isinstance(value, bytes):
            return len(value)
        elif isinstance(value, str):
            return len(value.encode())
        elif isinstance(value, (dict, list)):
            return len(json.dumps(value))
        else:
            # Rough estimate for other objects
            try:
                return len(json.dumps(str(value)).encode())
            except (TypeError, ValueError):
                return 1024  # Default estimate


# Singleton instance
_cache: MultiTierCache | None = None
_cache_lock = threading.Lock()


def get_cache() -> MultiTierCache:
    """Get or create singleton cache instance."""
    global _cache
    with _cache_lock:
        if _cache is None:
            _cache = MultiTierCache()
        return _cache


def reset_cache() -> None:
    """Reset singleton cache (clears all entries)."""
    global _cache
    with _cache_lock:
        if _cache is not None:
            _cache.clear()
        _cache = None
