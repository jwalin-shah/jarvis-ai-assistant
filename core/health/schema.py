"""Schema detection for iMessage chat.db.

Implements the SchemaDetector protocol from contracts/health.py.
Detects chat.db schema versions and provides appropriate SQL queries.

Workstream 7 implementation.
"""

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

from contracts.health import SchemaInfo

logger = logging.getLogger(__name__)

# Known schema versions and their characteristics
KNOWN_SCHEMAS: dict[str, dict[str, Any]] = {
    "v14": {
        "description": "macOS 14 (Sonoma) and earlier",
        "required_tables": ["message", "chat", "handle", "chat_message_join"],
        "optional_tables": ["chat_handle_join", "attachment", "message_attachment_join"],
    },
    "v15": {
        "description": "macOS 15 (Sequoia) and later",
        "required_tables": ["message", "chat", "handle", "chat_message_join"],
        "optional_tables": ["chat_handle_join", "attachment", "message_attachment_join"],
        "distinguishing_columns": {"chat": ["service_name"]},
    },
}

# Base SQL queries shared between versions
_BASE_QUERIES = {
    "conversations": """
        SELECT
            chat.ROWID as chat_rowid,
            chat.guid as chat_id,
            chat.display_name,
            chat.chat_identifier,
            (
                SELECT GROUP_CONCAT(handle.id, ', ')
                FROM chat_handle_join
                JOIN handle ON chat_handle_join.handle_id = handle.ROWID
                WHERE chat_handle_join.chat_id = chat.ROWID
            ) as participants,
            (
                SELECT COUNT(*)
                FROM chat_message_join
                WHERE chat_message_join.chat_id = chat.ROWID
            ) as message_count,
            (
                SELECT MAX(message.date)
                FROM chat_message_join
                JOIN message ON chat_message_join.message_id = message.ROWID
                WHERE chat_message_join.chat_id = chat.ROWID
            ) as last_message_date
        FROM chat
        WHERE message_count > 0
        {since_filter}
        ORDER BY last_message_date DESC
        LIMIT ?
    """,
    "messages": """
        SELECT
            message.ROWID as id,
            chat.guid as chat_id,
            COALESCE(handle.id, 'me') as sender,
            CASE
                WHEN message.text IS NOT NULL AND message.text != ''
                THEN message.text
                ELSE NULL
            END as text,
            message.attributedBody,
            message.date as date,
            message.is_from_me,
            message.thread_originator_guid as reply_to_guid
        FROM message
        JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
        JOIN chat ON chat_message_join.chat_id = chat.ROWID
        LEFT JOIN handle ON message.handle_id = handle.ROWID
        WHERE chat.guid = ?
        {before_filter}
        ORDER BY message.date DESC
        LIMIT ?
    """,
    "search": """
        SELECT
            message.ROWID as id,
            chat.guid as chat_id,
            COALESCE(handle.id, 'me') as sender,
            message.text,
            message.attributedBody,
            message.date,
            message.is_from_me,
            message.thread_originator_guid as reply_to_guid
        FROM message
        JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
        JOIN chat ON chat_message_join.chat_id = chat.ROWID
        LEFT JOIN handle ON message.handle_id = handle.ROWID
        WHERE message.text LIKE ? ESCAPE '\\'
        ORDER BY message.date DESC
        LIMIT ?
    """,
    "context": """
        SELECT
            message.ROWID as id,
            chat.guid as chat_id,
            COALESCE(handle.id, 'me') as sender,
            message.text,
            message.attributedBody,
            message.date,
            message.is_from_me,
            message.thread_originator_guid as reply_to_guid
        FROM message
        JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
        JOIN chat ON chat_message_join.chat_id = chat.ROWID
        LEFT JOIN handle ON message.handle_id = handle.ROWID
        WHERE chat.guid = ?
        ORDER BY ABS(message.ROWID - ?)
        LIMIT ?
    """,
    "tables": """
        SELECT name FROM sqlite_master
        WHERE type='table'
        ORDER BY name
    """,
    "table_info": """
        PRAGMA table_info({table_name})
    """,
}

# Version-specific query overrides (currently identical, will diverge as needed)
QUERIES = {
    "v14": _BASE_QUERIES.copy(),
    "v15": _BASE_QUERIES.copy(),
}


class ChatDBSchemaDetector:
    """Thread-safe iMessage chat.db schema detector.

    Detects schema versions and provides appropriate SQL queries.
    Implements the SchemaDetector protocol.
    """

    def __init__(self) -> None:
        """Initialize the schema detector."""
        self._lock = threading.Lock()
        self._schema_cache: dict[str, SchemaInfo] = {}
        logger.info("ChatDBSchemaDetector initialized")

    def _get_tables(self, conn: sqlite3.Connection) -> list[str]:
        """Get list of tables in the database.

        Args:
            conn: SQLite connection

        Returns:
            List of table names
        """
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        return [row[0] for row in cursor.fetchall()]

    def _get_table_columns(
        self,
        conn: sqlite3.Connection,
        table_name: str,
    ) -> set[str]:
        """Get column names for a table.

        Args:
            conn: SQLite connection
            table_name: Name of the table

        Returns:
            Set of column names
        """
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")  # noqa: S608
        return {row[1] for row in cursor.fetchall()}

    def _detect_version(self, conn: sqlite3.Connection) -> str:
        """Detect schema version based on table structure.

        Args:
            conn: SQLite connection

        Returns:
            Schema version string ("v14", "v15", or "unknown")
        """
        try:
            # Check for columns that indicate specific macOS versions
            message_columns = self._get_table_columns(conn, "message")

            # Check if thread_originator_guid exists (v14+)
            if "thread_originator_guid" not in message_columns:
                # Very old schema, treat as v14
                return "v14"

            # Check for macOS 15+ specific columns in chat table
            chat_columns = self._get_table_columns(conn, "chat")
            if "service_name" in chat_columns:
                return "v15"

            return "v14"

        except sqlite3.Error as e:
            logger.warning("Error detecting schema version: %s", e)
            return "unknown"

    def _check_required_tables(
        self,
        tables: list[str],
        version: str,
    ) -> bool:
        """Check if all required tables exist for a schema version.

        Args:
            tables: List of tables in the database
            version: Schema version to check against

        Returns:
            True if all required tables exist
        """
        if version not in KNOWN_SCHEMAS:
            return False

        required = set(KNOWN_SCHEMAS[version]["required_tables"])
        return required.issubset(set(tables))

    def detect(self, db_path: str) -> SchemaInfo:
        """Detect schema version and compatibility.

        Args:
            db_path: Path to the chat.db file

        Returns:
            SchemaInfo with detected version and compatibility status
        """
        with self._lock:
            # Check cache first
            if db_path in self._schema_cache:
                logger.debug("Using cached schema info for %s", db_path)
                return self._schema_cache[db_path]

            path = Path(db_path)

            # Check if file exists
            if not path.exists():
                logger.warning("Database file not found: %s", db_path)
                return SchemaInfo(
                    version="unknown",
                    tables=[],
                    compatible=False,
                    migration_needed=False,
                    known_schema=False,
                )

            try:
                # Open database read-only
                uri = f"file:{db_path}?mode=ro"
                conn = sqlite3.connect(uri, uri=True, timeout=5.0)

                try:
                    tables = self._get_tables(conn)
                    version = self._detect_version(conn)
                    known_schema = version in KNOWN_SCHEMAS
                    compatible = self._check_required_tables(tables, version)

                    # Migration might be needed if we detect v15 features
                    # but queries are still v14-compatible
                    migration_needed = version == "v15" and not compatible

                    info = SchemaInfo(
                        version=version,
                        tables=tables,
                        compatible=compatible,
                        migration_needed=migration_needed,
                        known_schema=known_schema,
                    )

                    # Cache the result
                    self._schema_cache[db_path] = info

                    logger.info(
                        "Detected schema version %s for %s (compatible=%s)",
                        version,
                        db_path,
                        compatible,
                    )

                    return info

                finally:
                    conn.close()

            except sqlite3.Error as e:
                logger.error("Error reading database %s: %s", db_path, e)
                return SchemaInfo(
                    version="unknown",
                    tables=[],
                    compatible=False,
                    migration_needed=False,
                    known_schema=False,
                )

    def get_query(
        self,
        query_name: str,
        schema_version: str,
        *,
        with_since_filter: bool = False,
        with_before_filter: bool = False,
    ) -> str:
        """Get appropriate SQL query for the detected schema.

        Args:
            query_name: Name of the query (conversations, messages, search, context)
            schema_version: Schema version (v14, v15)
            with_since_filter: If True, include AND last_message_date > ? clause
            with_before_filter: If True, include AND message.date < ? clause

        Returns:
            SQL query string with filters applied

        Raises:
            KeyError: If query name not found
        """
        # Fall back to v14 if version unknown
        version = schema_version if schema_version in QUERIES else "v14"

        if query_name not in QUERIES[version]:
            msg = f"Query '{query_name}' not found for schema version '{version}'"
            raise KeyError(msg)

        query = QUERIES[version][query_name]

        # Build filter clauses from boolean flags (never from user input)
        since_filter = "AND last_message_date > ?" if with_since_filter else ""
        before_filter = "AND message.date < ?" if with_before_filter else ""

        # Apply filters if the query has placeholders for them
        if "{since_filter}" in query or "{before_filter}" in query:
            query = query.format(
                since_filter=since_filter,
                before_filter=before_filter,
            )

        return query

    def get_available_queries(self, schema_version: str) -> list[str]:
        """Get list of available query names for a schema version.

        Args:
            schema_version: Schema version (v14, v15)

        Returns:
            List of available query names
        """
        version = schema_version if schema_version in QUERIES else "v14"
        return list(QUERIES[version].keys())

    def clear_cache(self) -> None:
        """Clear the schema detection cache."""
        with self._lock:
            self._schema_cache.clear()
            logger.info("Schema cache cleared")

    def is_version_supported(self, version: str) -> bool:
        """Check if a schema version is supported.

        Args:
            version: Schema version string

        Returns:
            True if the version is known and supported
        """
        return version in KNOWN_SCHEMAS


# Module-level singleton
_detector: ChatDBSchemaDetector | None = None
_detector_lock = threading.Lock()


def get_schema_detector() -> ChatDBSchemaDetector:
    """Get the singleton schema detector instance.

    Returns:
        The shared ChatDBSchemaDetector instance
    """
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = ChatDBSchemaDetector()
    return _detector


def reset_schema_detector() -> None:
    """Reset the singleton schema detector.

    Useful for testing or reinitializing the system.
    """
    global _detector
    with _detector_lock:
        _detector = None
        logger.info("Schema detector singleton reset")
