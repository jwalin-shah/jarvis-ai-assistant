"""Core database base class with connection management and schema initialization."""

import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from jarvis.db.models import JARVIS_DB_PATH
from jarvis.db.schema import (
    CURRENT_SCHEMA_VERSION,
    EXPECTED_INDICES,
    SCHEMA_SQL,
)
from jarvis.infrastructure.cache import TTLCache
from jarvis.utils.resources import safe_close

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Version-specific migration functions
# ---------------------------------------------------------------------------


def _migrate_v16_to_v17(conn: sqlite3.Connection) -> None:
    """Migration v16 -> v17: Add conversation_segments tables and segment_id column."""
    # Tables are created by SCHEMA_SQL; only need ALTER for existing contact_facts
    try:
        conn.execute("ALTER TABLE contact_facts ADD COLUMN segment_id INTEGER")
        logger.info("Added segment_id column to contact_facts table")
    except sqlite3.OperationalError as e:
        if "duplicate column" in str(e).lower():
            logger.debug("segment_id column already exists")
        else:
            logger.error("Migration v16->v17 failed: %s", e)
            raise


def _migrate_v20_to_v21(conn: sqlite3.Connection) -> None:
    """Migration v20 -> v21: Add relationship_reasoning column to contacts."""
    try:
        conn.execute("ALTER TABLE contacts ADD COLUMN relationship_reasoning TEXT")
        logger.info("Added relationship_reasoning column to contacts table")
    except sqlite3.OperationalError as e:
        if "duplicate column" in str(e).lower():
            logger.debug("relationship_reasoning column already exists")
        else:
            logger.error("Migration v20->v21 failed: %s", e)
            raise


# Maps (max_version_inclusive) -> migration callable.
# A migration runs when current_version > 0 and current_version <= max_version.
# Entries with exact_version=True run only when current_version == that version.
# Order matters: applied sequentially from lowest to highest version.
_MIGRATIONS: list[tuple[int, bool, Any]] = [
    # (max_version, exact_match_only, callable)
    (16, True, _migrate_v16_to_v17),
    (20, True, _migrate_v20_to_v21),
]


class JarvisDBBase:
    """Base class for JARVIS database with connection management and schema init.

    Thread-safe connection management with context manager support.
    Includes TTL-based caching for frequently accessed data.

    Connection pooling limits prevent unbounded growth with thread creation.
    """

    # Maximum number of concurrent connections to prevent unbounded growth
    _MAX_CONNECTIONS = 20

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize database manager.

        Args:
            db_path: Path to database file. Uses default if None.
        """
        self.db_path = db_path or JARVIS_DB_PATH
        self._local = threading.local()
        self._ensure_directory()

        # Query result caches with 30-second TTL
        self._contact_cache = TTLCache(maxsize=256, ttl_seconds=30.0)
        self._stats_cache = TTLCache(maxsize=16, ttl_seconds=60.0)
        self._trigger_pattern_cache = TTLCache(maxsize=128, ttl_seconds=30.0)

        # Track all thread-local connections for cleanup
        self._all_connections: set[sqlite3.Connection] = set()
        self._connections_lock = threading.Lock()
        self._connection_semaphore = threading.Semaphore(self._MAX_CONNECTIONS)

    def _ensure_directory(self) -> None:
        """Ensure the database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a thread-local connection.

        Connections are reused per-thread for efficiency. SQLite pragmas are
        set for optimal read performance while maintaining data integrity.

        Uses semaphore to limit total connections and prevent unbounded growth.

        Returns:
            SQLite connection with row_factory set to sqlite3.Row.
        """
        if not hasattr(self._local, "connection") or self._local.connection is None:
            # Acquire semaphore to limit total connections (blocks if at max)
            self._connection_semaphore.acquire()

            # Clean up stale connections before creating a new one
            self._cleanup_stale_connections()

            self._local.connection = sqlite3.connect(
                str(self.db_path),
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                timeout=30.0,
                check_same_thread=False,
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable foreign keys
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            # Optimize for read-heavy workloads
            self._local.connection.execute("PRAGMA journal_mode = WAL")
            self._local.connection.execute("PRAGMA synchronous = NORMAL")
            self._local.connection.execute("PRAGMA cache_size = -4000")  # 4MB cache
            self._local.connection.execute("PRAGMA mmap_size = 134217728")  # 128MB memory-mapped
            self._local.connection.execute("PRAGMA temp_store = MEMORY")
            # Load sqlite-vec extension for vector search
            try:
                self._local.connection.enable_load_extension(True)
                import sqlite_vec

                sqlite_vec.load(self._local.connection)
                self._local.connection.enable_load_extension(False)
            except (ImportError, sqlite3.Error) as e:
                logger.debug("sqlite-vec extension not available: %s", e)

            # Track this connection for cleanup
            with self._connections_lock:
                self._all_connections.add(self._local.connection)
        return cast(sqlite3.Connection, self._local.connection)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection (reuses thread-local connection).

        Yields:
            SQLite connection with row_factory set to sqlite3.Row.
        """
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _cleanup_stale_connections(self) -> None:
        """Remove connections for threads that are no longer alive.

        Called periodically during new connection creation to prevent
        the connection set from growing unbounded.
        """
        with self._connections_lock:
            # Track which connections are still valid
            active_connections: set[sqlite3.Connection] = set()

            for conn in self._all_connections:
                try:
                    # Try to execute a simple query to check if connection is still valid
                    conn.execute("SELECT 1")
                    active_connections.add(conn)
                except sqlite3.DatabaseError as e:
                    # Connection is stale or broken, close it
                    logger.debug("Stale connection detected, closing: %s", e)
                    safe_close(conn, name="stale db connection")

            self._all_connections = active_connections

    def close(self) -> None:
        """Close all thread-local connections."""
        # Close all tracked connections from all threads
        with self._connections_lock:
            for conn in self._all_connections:
                safe_close(conn, name="db connection shutdown")
                # Release semaphore for each closed connection
                try:
                    self._connection_semaphore.release()
                except ValueError:
                    # Semaphore already at max value
                    pass
            self._all_connections.clear()

        # Clear current thread's reference
        if hasattr(self._local, "connection"):
            self._local.connection = None

        # Clear caches on close
        self._contact_cache.clear()
        self._stats_cache.clear()
        self._trigger_pattern_cache.clear()

    def clear_caches(self) -> None:
        """Clear all query result caches.

        Call this after bulk modifications to ensure fresh data.
        """
        self._contact_cache.clear()
        self._stats_cache.clear()
        self._trigger_pattern_cache.clear()

    def get_cache_stats(self) -> dict[str, dict[str, int]]:
        """Get statistics about cache usage.

        Returns:
            Dictionary with cache names and their stats.
        """
        return {
            "contact_cache": self._contact_cache.stats(),
            "stats_cache": self._stats_cache.stats(),
            "trigger_pattern_cache": self._trigger_pattern_cache.stats(),
        }

    def _ensure_vec_tables(self, conn: sqlite3.Connection) -> None:
        """Ensure sqlite-vec virtual tables exist (idempotent).

        Called as a safety net when schema version is already current but
        vec tables may have been skipped (e.g., sqlite-vec wasn't loaded).
        Creates each table independently so partial failures don't block others.
        """
        try:
            existing = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name IN ('vec_chunks', 'vec_messages', 'vec_binary', 'vec_facts')"
                ).fetchall()
            }

            if "vec_chunks" not in existing:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                        embedding int8[384] distance_metric=L2,
                        contact_id integer partition key,
                        chat_id text,
                        source_timestamp float,
                        +context_text text,
                        +reply_text text,
                        +topic_label text,
                        +message_count integer
                    )
                """
                )
                logger.info("Created vec_chunks table")

            if "vec_messages" not in existing:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_messages USING vec0(
                        embedding int8[384] distance_metric=L2,
                        chat_id text partition key,
                        +text_preview text,
                        +sender text,
                        +timestamp integer,
                        +is_from_me integer
                    )
                """
                )
                logger.info("Created vec_messages table")

            if "vec_binary" not in existing:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_binary USING vec0(
                        embedding bit[384],
                        +chunk_rowid integer,
                        +embedding_int8 blob
                    )
                """
                )
                logger.info("Created vec_binary table")

            if "vec_facts" not in existing:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_facts USING vec0(
                        embedding int8[384] distance_metric=L2,
                        contact_id text,
                        +fact_id INTEGER,
                        +fact_text TEXT
                    )
                """
                )
                logger.info("Created vec_facts table")

        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            logger.warning("Could not create vec tables (sqlite-vec unavailable): %s", e)

    def _ensure_contact_facts_columns(self, conn: sqlite3.Connection) -> None:
        """Backfill contact_facts columns for partially migrated databases.

        Some environments can have an older contact_facts table present even when
        schema_version is behind. Ensure newer columns exist before SCHEMA_SQL
        creates indices that depend on them.
        """
        try:
            table_exists = (
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='contact_facts'"
                ).fetchone()
                is not None
            )
            if not table_exists:
                return

            existing_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(contact_facts)").fetchall()
            }
            required_columns = {
                "linked_contact_id": "TEXT",
                "valid_from": "TIMESTAMP",
                "valid_until": "TIMESTAMP",
                "attribution": "TEXT DEFAULT 'contact'",
                "segment_id": "INTEGER",
            }

            for col_name, col_type in required_columns.items():
                if col_name in existing_columns:
                    continue
                conn.execute(f"ALTER TABLE contact_facts ADD COLUMN {col_name} {col_type}")
                logger.info("Added %s column to contact_facts table", col_name)
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e).lower():
                logger.debug("contact_facts migration encountered existing column: %s", e)
            else:
                logger.error("contact_facts migration failed: %s", e)
                raise

    def init_schema(self) -> bool:
        """Initialize database schema.

        Creates all tables if they don't exist.

        Returns:
            True if schema was created/updated, False if already current.
        """
        with self.connection() as conn:
            # Check current schema version
            try:
                cursor = conn.execute(
                    "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
                )
                row = cursor.fetchone()
                current_version = row["version"] if row else 0
            except sqlite3.OperationalError:
                # Table doesn't exist yet
                current_version = 0

            if current_version >= CURRENT_SCHEMA_VERSION:
                # Ensure all tables from SCHEMA_SQL exist (e.g. if one was dropped)
                conn.executescript(SCHEMA_SQL)
                # Ensure vec tables exist even if version is current
                # (they may have been skipped if sqlite-vec wasn't loaded initially)
                self._ensure_vec_tables(conn)
                logger.debug("Schema already at version %d", current_version)
                return False

            # Apply migrations for existing databases only.
            # For new databases (current_version == 0), SCHEMA_SQL below creates
            # all tables with the latest columns, so no ALTER needed.
            if current_version > 0:
                for max_version, exact_only, migrate_fn in _MIGRATIONS:
                    if exact_only:
                        should_run = current_version == max_version
                    else:
                        should_run = current_version <= max_version
                    if should_run:
                        migrate_fn(conn)

            # Handle partially-migrated DBs where contact_facts exists without newer columns.
            self._ensure_contact_facts_columns(conn)

            # Apply schema
            conn.executescript(SCHEMA_SQL)

            # Ensure vec tables exist (sqlite-vec virtual tables can't be
            # created via executescript, need separate CREATE statements)
            self._ensure_vec_tables(conn)

            # Update version
            conn.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (CURRENT_SCHEMA_VERSION,),
            )

            logger.info(
                "Schema updated from version %d to %d",
                current_version,
                CURRENT_SCHEMA_VERSION,
            )
            return True

    def exists(self) -> bool:
        """Check if the database file exists."""
        return self.db_path.exists()

    def optimize(self) -> None:
        """Optimize the database (VACUUM and REINDEX)."""
        with self.connection() as conn:
            conn.execute("REINDEX")
            conn.execute("VACUUM")
            logger.info("Database optimization completed (REINDEX, VACUUM)")

    def verify_indices(self, create_missing: bool = True) -> dict[str, Any]:
        """Verify that required indices exist and optionally create missing ones.

        Args:
            create_missing: If True, create any missing indices.

        Returns:
            Dictionary with verification results:
            - existing: Set of existing index names
            - missing: Set of missing index names
            - created: Set of newly created index names (if create_missing=True)
        """
        with self.connection() as conn:
            # Get existing indices
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
            )
            existing_indices = {row["name"] for row in cursor}

            missing_indices = EXPECTED_INDICES - existing_indices
            created_indices: set[str] = set()

            if create_missing and missing_indices:
                # Re-run the schema SQL which has CREATE INDEX IF NOT EXISTS
                conn.executescript(SCHEMA_SQL)

                # Check what was created
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
                )
                new_existing = {row["name"] for row in cursor}
                created_indices = new_existing - existing_indices
                existing_indices = new_existing
                missing_indices = EXPECTED_INDICES - existing_indices

                if created_indices:
                    logger.info("Created missing indices: %s", created_indices)

            return {
                "existing": existing_indices,
                "missing": missing_indices,
                "created": created_indices,
                "all_present": len(missing_indices) == 0,
            }
