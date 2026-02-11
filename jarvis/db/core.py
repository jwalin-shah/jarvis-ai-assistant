"""Core database base class with connection management and schema initialization."""

import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from jarvis.cache import TTLCache
from jarvis.db.models import JARVIS_DB_PATH
from jarvis.db.schema import (
    CURRENT_SCHEMA_VERSION,
    EXPECTED_INDICES,
    SCHEMA_SQL,
    VALID_COLUMN_TYPES,
    VALID_MIGRATION_COLUMNS,
)

logger = logging.getLogger(__name__)


class JarvisDBBase:
    """Base class for JARVIS database with connection management and schema init.

    Thread-safe connection management with context manager support.
    Includes TTL-based caching for frequently accessed data.
    """

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

    def _ensure_directory(self) -> None:
        """Ensure the database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a thread-local connection.

        Connections are reused per-thread for efficiency. SQLite pragmas are
        set for optimal read performance while maintaining data integrity.

        Returns:
            SQLite connection with row_factory set to sqlite3.Row.
        """
        if not hasattr(self._local, "connection") or self._local.connection is None:
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
            self._local.connection.execute("PRAGMA cache_size = -8000")  # 8MB cache
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
        return self._local.connection

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
                    try:
                        conn.close()
                    except sqlite3.Error as close_err:
                        logger.debug("Error closing stale connection: %s", close_err)

            self._all_connections = active_connections

    def close(self) -> None:
        """Close all thread-local connections."""
        # Close all tracked connections from all threads
        with self._connections_lock:
            for conn in self._all_connections:
                try:
                    conn.close()
                except sqlite3.Error as e:
                    logger.debug("Error closing connection during shutdown: %s", e)
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
                    "AND name IN ('vec_chunks', 'vec_messages', 'vec_binary')"
                ).fetchall()
            }

            if "vec_chunks" not in existing:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                        embedding int8[384] distance_metric=L2,
                        contact_id integer partition key,
                        chat_id text,
                        response_da_type text,
                        quality_score float,
                        source_timestamp float,
                        +topic_label text,
                        +trigger_text text,
                        +response_text text,
                        +formatted_text text,
                        +keywords_json text,
                        +message_count integer,
                        +response_da_conf float,
                        +source_type text,
                        +source_id text
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
                # Ensure vec tables exist even if version is current
                # (they may have been skipped if sqlite-vec wasn't loaded initially)
                self._ensure_vec_tables(conn)
                logger.debug("Schema already at version %d", current_version)
                return False

            # Apply migrations for existing databases only.
            # For new databases (current_version == 0), SCHEMA_SQL below creates
            # all tables with the latest columns, so no ALTER needed.
            if current_version > 0 and current_version == 2:
                # Migration v2 -> v3: Add context_text column to pairs
                try:
                    conn.execute("ALTER TABLE pairs ADD COLUMN context_text TEXT")
                    logger.info("Added context_text column to pairs table")
                except sqlite3.OperationalError as e:
                    if "duplicate column" in str(e).lower():
                        logger.debug("context_text column already exists")
                    else:
                        logger.error("Migration v2->v3 failed: %s", e)
                        raise

            if current_version > 0 and current_version <= 3:
                # Migration v3 -> v4: Add is_group column to pairs
                try:
                    conn.execute("ALTER TABLE pairs ADD COLUMN is_group BOOLEAN DEFAULT FALSE")
                    logger.info("Added is_group column to pairs table")
                except sqlite3.OperationalError as e:
                    if "duplicate column" in str(e).lower():
                        logger.debug("is_group column already exists")
                    else:
                        logger.error("Migration v3->v4 failed: %s", e)
                        raise

            if current_version > 0 and current_version <= 4:
                # Migration v4 -> v5: Add is_holdout column for train/test split
                try:
                    conn.execute("ALTER TABLE pairs ADD COLUMN is_holdout BOOLEAN DEFAULT FALSE")
                    logger.info("Added is_holdout column to pairs table")
                except sqlite3.OperationalError as e:
                    if "duplicate column" in str(e).lower():
                        logger.debug("is_holdout column already exists")
                    else:
                        logger.error("Migration v4->v5 failed: %s", e)
                        raise

            if current_version > 0 and current_version <= 5:
                # Migration v5 -> v6: Add validity gate columns and split tables
                # Add gate columns to pairs table
                gate_columns = [
                    ("gate_a_passed", "BOOLEAN"),
                    ("gate_b_score", "REAL"),
                    ("gate_c_verdict", "TEXT"),
                    ("validity_status", "TEXT"),
                ]
                for col_name, col_type in gate_columns:
                    if col_name not in VALID_MIGRATION_COLUMNS:
                        raise ValueError(f"Invalid migration column name: {col_name}")
                    if col_type not in VALID_COLUMN_TYPES:
                        raise ValueError(f"Invalid migration column type: {col_type}")
                    try:
                        # SECURITY: f-string is safe here because both col_name and col_type
                        # are validated against strict allow-lists (VALID_MIGRATION_COLUMNS
                        # and VALID_COLUMN_TYPES). SQLite's ALTER TABLE doesn't support
                        # parameterized column names/types, so validation is the correct approach.
                        conn.execute(f"ALTER TABLE pairs ADD COLUMN {col_name} {col_type}")
                        logger.info("Added %s column to pairs table", col_name)
                    except sqlite3.OperationalError as e:
                        if "duplicate column" in str(e).lower():
                            logger.debug("%s column already exists", col_name)
                        else:
                            logger.error("Migration v5->v6 failed adding %s: %s", col_name, e)
                            raise

                # Create new tables (handled by SCHEMA_SQL, but ensure they exist)
                # pair_artifacts and contact_style_targets are created by executescript

            if current_version > 0 and current_version <= 6:
                # Migration v6 -> v7: Add dialogue act classification and cluster columns
                da_columns = [
                    ("trigger_da_type", "TEXT"),
                    ("trigger_da_conf", "REAL"),
                    ("response_da_type", "TEXT"),
                    ("response_da_conf", "REAL"),
                    ("cluster_id", "INTEGER"),
                ]
                for col_name, col_type in da_columns:
                    if col_name not in VALID_MIGRATION_COLUMNS:
                        raise ValueError(f"Invalid migration column name: {col_name}")
                    if col_type not in VALID_COLUMN_TYPES:
                        raise ValueError(f"Invalid migration column type: {col_type}")
                    try:
                        # SECURITY: f-string is safe here because both col_name and col_type
                        # are validated against strict allow-lists (VALID_MIGRATION_COLUMNS
                        # and VALID_COLUMN_TYPES). SQLite's ALTER TABLE doesn't support
                        # parameterized column names/types, so validation is the correct approach.
                        conn.execute(f"ALTER TABLE pairs ADD COLUMN {col_name} {col_type}")
                        logger.info("Added %s column to pairs table", col_name)
                    except sqlite3.OperationalError as e:
                        if "duplicate column" in str(e).lower():
                            logger.debug("%s column already exists", col_name)
                        else:
                            logger.error("Migration v6->v7 failed adding %s: %s", col_name, e)
                            raise

            # Migration v7 -> v8: Add scheduling tables
            # Tables are created by SCHEMA_SQL with CREATE TABLE IF NOT EXISTS
            # No column migrations needed since these are new tables

            # Migration v8 -> v9: Add content_hash for text-based deduplication
            if current_version > 0 and current_version < 9:
                try:
                    conn.execute("ALTER TABLE pairs ADD COLUMN content_hash TEXT")
                    logger.info("Added content_hash column to pairs table")
                except sqlite3.OperationalError as e:
                    if "duplicate column" in str(e).lower():
                        logger.debug("content_hash column already exists")
                    else:
                        logger.error("Migration v8->v9 failed: %s", e)
                        raise

            # Migration v9 -> v10: sqlite-vec virtual tables for vector search
            if current_version < 10:
                try:
                    conn.execute(
                        """
                        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                            embedding int8[384] distance_metric=L2,
                            contact_id integer partition key,
                            chat_id text,
                            response_da_type text,
                            quality_score float,
                            source_timestamp float,
                            +topic_label text,
                            +trigger_text text,
                            +response_text text,
                            +formatted_text text,
                            +keywords_json text,
                            +message_count integer,
                            +response_da_conf float,
                            +source_type text,
                            +source_id text
                        )
                    """
                    )
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
                    conn.execute(
                        """
                        CREATE VIRTUAL TABLE IF NOT EXISTS vec_binary USING vec0(
                            embedding bit[384],
                            +chunk_rowid integer,
                            +embedding_int8 blob
                        )
                    """
                    )
                    logger.info(
                        "Created sqlite-vec virtual tables (vec_chunks, vec_messages, vec_binary)"
                    )
                except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                    if "already exists" in str(e).lower():
                        logger.debug("vec tables already exist")
                    else:
                        logger.warning(
                            "sqlite-vec migration skipped (extension unavailable): %s", e
                        )

            # Migration v10 -> v11: Recreate vec_binary with aux columns
            if current_version == 10:
                try:
                    # Old vec_binary had no aux columns; drop and recreate
                    conn.execute("DROP TABLE IF EXISTS vec_binary")
                    conn.execute(
                        """
                        CREATE VIRTUAL TABLE IF NOT EXISTS vec_binary USING vec0(
                            embedding bit[384],
                            +chunk_rowid integer,
                            +embedding_int8 blob
                        )
                    """
                    )
                    logger.info("Recreated vec_binary with chunk_rowid + embedding_int8 columns")
                except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                    if "already exists" in str(e).lower():
                        logger.debug("vec_binary already has new schema")
                    else:
                        logger.warning(
                            "vec_binary migration skipped (extension unavailable): %s", e
                        )

            # Migration v11 -> v12: contact_facts table for knowledge graph
            # Table is created by SCHEMA_SQL with CREATE TABLE IF NOT EXISTS
            # No column migrations needed since this is a new table

            # Handle partially-migrated DBs where contact_facts exists without newer columns.
            self._ensure_contact_facts_columns(conn)

            # Apply schema
            conn.executescript(SCHEMA_SQL)

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
