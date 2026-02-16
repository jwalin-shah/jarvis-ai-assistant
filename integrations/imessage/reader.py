"""Read-only iMessage chat.db access.

Implements the iMessageReader protocol from contracts/imessage.py.
"""

import logging
import queue
import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, Self, TypeVar

from contracts.imessage import Attachment, Conversation, Message, Reaction
from jarvis.core.exceptions import (
    ErrorCode,
    imessage_db_not_found,
    imessage_permission_denied,
    iMessageAccessError,
    iMessageQueryError,
)
from jarvis.utils.latency_tracker import track_latency
from jarvis.utils.sqlite_retry import sqlite_retry

from .avatar import ContactAvatarData, get_contact_avatar
from .parser import (
    categorize_attachment_type,
    datetime_to_apple_timestamp,
    extract_text_from_row,
    normalize_phone_number,
    parse_apple_timestamp,
    parse_attachments,
    parse_reactions,
)
from .queries import detect_schema_version, get_query

logger = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """Simple LRU cache with bounded size.

    Uses OrderedDict to maintain insertion/access order.
    Oldest entries are evicted when maxsize is exceeded.
    Thread-safe via internal lock.
    """

    def __init__(self, maxsize: int = 1000) -> None:
        """Initialize the LRU cache.

        Args:
            maxsize: Maximum number of items to store (default 1000)
        """
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: K) -> V | None:
        """Get a value from the cache, returning None if not found.

        Moves the accessed item to the end (most recently used).
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def set(self, key: K, value: V) -> None:
        """Set a value in the cache, evicting oldest if at capacity."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def __contains__(self, key: K) -> bool:
        """Check if key is in cache."""
        with self._lock:
            return key in self._cache

    def clear(self) -> None:
        """Clear all items from the cache."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        """Return the number of items in the cache."""
        with self._lock:
            return len(self._cache)


# Path to macOS AddressBook database for contact name resolution
ADDRESSBOOK_DB_PATH = Path.home() / "Library" / "Application Support" / "AddressBook" / "Sources"

# Default path to iMessage database
CHAT_DB_PATH = Path.home() / "Library" / "Messages" / "chat.db"

# Database connection timeout (handles SQLITE_BUSY from concurrent iMessage app access)
DB_TIMEOUT_SECONDS = 5.0

# Connection pool configuration
DEFAULT_POOL_SIZE = 5
MAX_POOL_SIZE = 10


class ConnectionPool:
    """Thread-safe connection pool for SQLite read-only connections.

    Uses queue.Queue for thread-safe connection management with bounded size.
    Connections are lazily created on first use and reused across requests.

    Thread Safety:
        This class is fully thread-safe. Multiple threads can safely acquire
        and release connections concurrently.

    Example:
        pool = ConnectionPool(db_path, max_connections=5)
        with pool.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chat LIMIT 10")
    """

    def __init__(
        self,
        db_path: Path,
        max_connections: int = DEFAULT_POOL_SIZE,
        timeout: float = DB_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize the connection pool.

        Args:
            db_path: Path to the SQLite database file
            max_connections: Maximum number of connections in the pool (default 5, max 10)
            timeout: Connection timeout in seconds
        """
        self.db_path = db_path
        self.max_connections = min(max_connections, MAX_POOL_SIZE)
        self.timeout = timeout
        self._pool: queue.Queue[sqlite3.Connection] = queue.Queue(maxsize=self.max_connections)
        self._checked_out: set[sqlite3.Connection] = set()
        self._created_count = 0
        self._lock = threading.Lock()
        self._cache_lock = threading.Lock()  # Lock for cache initialization
        self._closed = False
        self._schema_version: str | None = None
        # Shared caches for all reader instances using this pool
        self._contacts_cache: dict[str, str] | None = None
        self._guid_to_rowid_cache: LRUCache[str, int] = LRUCache(maxsize=10000)

    @sqlite_retry()
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new read-only database connection.

        Returns:
            SQLite connection with Row factory

        Raises:
            iMessageAccessError: If connection fails due to permissions or missing file.
        """
        db_path_str = str(self.db_path)

        if not self.db_path.exists():
            raise imessage_db_not_found(db_path_str)

        try:
            uri = f"file:{self.db_path}?mode=ro"
            conn = sqlite3.connect(
                uri,
                uri=True,
                timeout=self.timeout,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row

            # Detect schema version on first connection
            if self._schema_version is None:
                self._schema_version = detect_schema_version(conn)
                logger.debug(f"Detected chat.db schema version: {self._schema_version}")

            return conn
        except PermissionError as e:
            raise imessage_permission_denied(db_path_str) from e
        except (sqlite3.OperationalError, sqlite3.InterfaceError) as e:
            if "unable to open database" in str(e).lower():
                raise imessage_permission_denied(db_path_str) from e
            raise iMessageQueryError(
                f"Failed to connect to database: {e}",
                db_path=db_path_str,
                cause=e,
            ) from e
        except OSError as e:
            raise iMessageAccessError(
                f"Failed to connect to database (I/O error): {e}",
                db_path=db_path_str,
                cause=e,
            ) from e

    def acquire(self, timeout: float | None = None) -> sqlite3.Connection:
        """Acquire a connection from the pool.

        If no connection is available and pool is not at capacity, creates a new one.
        If pool is at capacity, blocks until a connection is available.

        Args:
            timeout: Maximum time to wait for a connection (None = wait forever)

        Returns:
            SQLite connection

        Raises:
            iMessageAccessError: If pool is closed or connection fails
            queue.Empty: If timeout expires without getting a connection
        """
        if self._closed:
            raise iMessageAccessError(
                "Connection pool is closed",
                db_path=str(self.db_path),
            )

        # Try to get from pool without blocking first
        try:
            conn = self._pool.get_nowait()
            with self._lock:
                self._checked_out.add(conn)
            return conn
        except queue.Empty:
            pass

        # Check if we can create a new connection
        with self._lock:
            if self._created_count < self.max_connections:
                self._created_count += 1
                try:
                    conn = self._create_connection()
                    self._checked_out.add(conn)
                    return conn
                except Exception:
                    # Broad catch: _create_connection raises our custom error hierarchy
                    # (iMessageAccessError, iMessageQueryError) which don't share a
                    # common non-Exception base. Must decrement count for any failure.
                    self._created_count -= 1
                    raise

        # Pool is at capacity, wait for a connection
        try:
            conn = self._pool.get(timeout=timeout)
            with self._lock:
                self._checked_out.add(conn)
            return conn
        except queue.Empty:
            raise iMessageAccessError(
                f"Timeout waiting for database connection (pool size: {self.max_connections})",
                db_path=str(self.db_path),
            ) from None

    def release(self, conn: sqlite3.Connection) -> None:
        """Release a connection back to the pool.

        Args:
            conn: Connection to release
        """
        with self._lock:
            self._checked_out.discard(conn)

        if self._closed:
            try:
                conn.close()
            except (sqlite3.Error, OSError):
                pass
            return

        try:
            self._pool.put_nowait(conn)
        except queue.Full:
            # Pool is full (shouldn't happen), close the connection
            try:
                conn.close()
            except (sqlite3.Error, OSError):
                pass

    @contextmanager
    def connection(self, timeout: float | None = None) -> Iterator[sqlite3.Connection]:
        """Context manager for acquiring and releasing connections.

        Args:
            timeout: Maximum time to wait for a connection

        Yields:
            SQLite connection

        Example:
            with pool.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM chat")
        """
        conn = self.acquire(timeout=timeout)
        try:
            yield conn
        finally:
            self.release(conn)

    @property
    def schema_version(self) -> str | None:
        """Get the detected schema version."""
        return self._schema_version

    def close(self) -> None:
        """Close all connections in the pool (both queued and checked-out)."""
        self._closed = True

        # Close all connections in the queue
        while True:
            try:
                conn = self._pool.get_nowait()
                try:
                    conn.close()
                except (sqlite3.Error, OSError):
                    pass
            except queue.Empty:
                break

        # Close checked-out connections and warn if any remain
        with self._lock:
            if self._checked_out:
                logger.warning(
                    "Closing pool with %d checked-out connections",
                    len(self._checked_out),
                )
                for conn in self._checked_out:
                    try:
                        conn.close()
                    except (sqlite3.Error, OSError):
                        pass
                self._checked_out.clear()
            self._created_count = 0
            self._schema_version = None

    def stats(self) -> dict[str, Any]:
        """Get pool statistics.

        Returns:
            Dictionary with pool stats (created, available, max)
        """
        return {
            "created": self._created_count,
            "available": self._pool.qsize(),
            "checked_out": len(self._checked_out),
            "max_connections": self.max_connections,
            "closed": self._closed,
        }


# Module-level connection pool singleton
_connection_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def get_connection_pool(
    db_path: Path | None = None,
    max_connections: int = DEFAULT_POOL_SIZE,
) -> ConnectionPool:
    """Get or create the module-level connection pool singleton.

    Args:
        db_path: Path to chat.db (defaults to ~/Library/Messages/chat.db)
        max_connections: Maximum connections in pool (default 5, max 10)

    Returns:
        ConnectionPool instance
    """
    global _connection_pool

    if _connection_pool is None:
        with _pool_lock:
            if _connection_pool is None:
                path = db_path or CHAT_DB_PATH
                _connection_pool = ConnectionPool(path, max_connections=max_connections)

    return _connection_pool


def reset_connection_pool() -> None:
    """Reset the module-level connection pool (for testing or reconnection)."""
    global _connection_pool

    with _pool_lock:
        if _connection_pool is not None:
            _connection_pool.close()
            _connection_pool = None


class ChatDBReader:
    """Read-only access to iMessage chat.db.

    Implements iMessageReader protocol from contracts/imessage.py.

    Thread Safety:
        This class uses a module-level connection pool for thread-safe access.
        Multiple ChatDBReader instances can safely be used across threads.
        Each method acquires a connection from the pool for its operation and
        releases it back when done.

    Example:
        # Preferred: use as context manager for automatic cleanup
        with ChatDBReader() as reader:
            if reader.check_access():
                conversations = reader.get_conversations(limit=10)
                for conv in conversations:
                    messages = reader.get_messages(conv.chat_id, limit=50)

        # Alternative: manual lifecycle management
        reader = ChatDBReader()
        try:
            if reader.check_access():
                conversations = reader.get_conversations(limit=10)
        finally:
            reader.close()

        # For high-concurrency scenarios, use the connection pool directly:
        pool = get_connection_pool()
        with pool.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chat LIMIT 10")
    """

    def __init__(self, db_path: Path | None = None, use_pool: bool = True):
        """Initialize the reader.

        Args:
            db_path: Path to chat.db. Defaults to ~/Library/Messages/chat.db
            use_pool: If True, use the shared connection pool. If False, use a
                dedicated connection (for backwards compatibility or isolation).
        """
        self.db_path = db_path or CHAT_DB_PATH
        self._use_pool = use_pool
        # Connection (from pool or dedicated)
        self._connection: sqlite3.Connection | None = None
        self._connection_from_pool = False  # Track if connection came from pool
        self._schema_version: str | None = None
        # Cache for contact name lookups (lazy-loaded)
        self.__contacts_cache: dict[str, str] | None = None
        # LRU cache for GUID to ROWID mappings (lazy-loaded)
        self.__guid_to_rowid_cache: LRUCache[str, int] | None = None
        # Pool reference (lazy initialized)
        self._pool: ConnectionPool | None = None

    @property
    def _contacts_cache(self) -> dict[str, str] | None:
        """Get the contacts cache (shared if using pool)."""
        if self._use_pool:
            return self._get_pool()._contacts_cache
        return self.__contacts_cache

    @_contacts_cache.setter
    def _contacts_cache(self, value: dict[str, str] | None) -> None:
        """Set the contacts cache (shared if using pool)."""
        if self._use_pool:
            self._get_pool()._contacts_cache = value
        else:
            self.__contacts_cache = value

    @property
    def _guid_to_rowid_cache(self) -> LRUCache[str, int]:
        """Get the GUID to ROWID cache (shared if using pool).

        Thread-safe: Uses pool's cache lock when in pool mode to prevent race conditions.
        """
        if self._use_pool:
            pool = self._get_pool()
            # Use cache lock to ensure atomic initialization
            with pool._cache_lock:
                if pool._guid_to_rowid_cache is None:
                    pool._guid_to_rowid_cache = LRUCache(maxsize=10000)
                return pool._guid_to_rowid_cache
        # Non-pool mode: single-threaded, no lock needed
        if self.__guid_to_rowid_cache is None:
            self.__guid_to_rowid_cache = LRUCache(maxsize=10000)
        return self.__guid_to_rowid_cache

    def _get_pool(self) -> ConnectionPool:
        """Get the connection pool (lazy initialization).

        Returns:
            ConnectionPool instance
        """
        if self._pool is None:
            self._pool = get_connection_pool(self.db_path)
        return self._pool

    @contextmanager
    def _connection_context(self) -> Iterator[sqlite3.Connection]:
        """Context manager for thread-safe connection access.

        In pool mode, acquires a fresh connection from the pool and releases it when done.
        In legacy mode, uses the dedicated connection.

        Yields:
            SQLite connection
        """
        if self._use_pool:
            pool = self._get_pool()
            with pool.connection() as conn:
                # Set schema version from pool
                self._schema_version = pool.schema_version
                yield conn
        else:
            yield self._get_connection()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a read-only database connection.

        When using pool mode (default), acquires a connection from the shared pool
        and holds it for the lifetime of this reader instance (released on close()).
        When using legacy mode (use_pool=False), maintains a dedicated connection.

        Returns:
            SQLite connection with Row factory

        Raises:
            iMessageAccessError: If connection fails due to permissions or missing file.
            iMessageQueryError: If connection fails for other reasons.
        """
        # Return existing connection if we have one
        if self._connection is not None:
            return self._connection

        if self._use_pool:
            # Pool mode: acquire from shared pool and hold
            pool = self._get_pool()
            self._connection = pool.acquire()
            self._connection_from_pool = True
            self._schema_version = pool.schema_version
            return self._connection

        # Legacy mode: create dedicated connection
        if self._connection is None:
            db_path_str = str(self.db_path)

            if not self.db_path.exists():
                raise imessage_db_not_found(db_path_str)

            try:
                # Open in read-only mode using URI
                uri = f"file:{self.db_path}?mode=ro"
                self._connection = sqlite3.connect(
                    uri,
                    uri=True,
                    timeout=DB_TIMEOUT_SECONDS,
                    check_same_thread=False,  # Safe for async contexts
                )
                self._connection.row_factory = sqlite3.Row

                # Detect schema version
                self._schema_version = detect_schema_version(self._connection)
                logger.debug(f"Detected chat.db schema version: {self._schema_version}")
            except PermissionError as e:
                raise imessage_permission_denied(db_path_str) from e
            except (sqlite3.OperationalError, sqlite3.InterfaceError) as e:
                if "unable to open database" in str(e).lower():
                    raise imessage_permission_denied(db_path_str) from e
                raise iMessageQueryError(
                    f"Failed to connect to database: {e}",
                    db_path=db_path_str,
                    cause=e,
                ) from e
            except OSError as e:
                # File system errors (e.g., file locked, disk I/O issues)
                raise iMessageAccessError(
                    f"Failed to connect to database (I/O error): {e}",
                    db_path=db_path_str,
                    cause=e,
                ) from e
            except (ValueError, TypeError) as e:
                # Configuration or type errors during connection setup
                raise iMessageAccessError(
                    f"Failed to connect to database (configuration error): {e}",
                    db_path=db_path_str,
                    cause=e,
                ) from e
            except Exception as e:
                # Intentionally broad: catches MemoryError, threading issues, and any
                # other truly unexpected errors during connection setup. Wraps them in
                # our error hierarchy so callers get consistent iMessageAccessError.
                logger.exception("Unexpected error connecting to database: %s", db_path_str)
                raise iMessageAccessError(
                    f"Failed to connect to database: {e}",
                    db_path=db_path_str,
                    cause=e,
                ) from e

        return self._connection

    def close(self) -> None:
        """Close or release the database connection.

        In pool mode, releases the connection back to the pool.
        In legacy mode, closes the dedicated connection.
        Call this when done to release resources.
        """
        if self._connection is not None:
            if self._connection_from_pool and self._pool is not None:
                # Pool mode: release connection back to pool
                try:
                    self._pool.release(self._connection)
                except (OSError, sqlite3.Error) as e:
                    logger.debug("Error releasing connection to pool: %s", e)
            else:
                # Legacy mode: close dedicated connection
                try:
                    self._connection.close()
                except sqlite3.Error:
                    # SQLite errors during close (e.g., connection already closed)
                    # are expected and can be safely ignored during cleanup.
                    pass
                except OSError:
                    # File system errors during close are recoverable and can be ignored.
                    logger.debug("I/O error closing database connection", exc_info=True)
                except Exception:
                    # Intentionally broad: cleanup must never raise to avoid masking
                    # the original error in context manager __exit__. sqlite3.Error and
                    # OSError are handled above; this catches truly unexpected errors
                    # (e.g., threading race conditions during interpreter shutdown).
                    logger.debug("Unexpected error closing database connection", exc_info=True)

            self._connection = None
            self._connection_from_pool = False

        self._schema_version = None
        # Only clear internal caches if not using shared pool
        if not self._use_pool and self.__guid_to_rowid_cache is not None:
            self.__guid_to_rowid_cache.clear()

        self.__contacts_cache = None
        self.__guid_to_rowid_cache = None

    def _get_attachments_for_message(self, message_id: int) -> list[Attachment]:
        """Fetch attachments for a specific message.

        Args:
            message_id: The message ROWID

        Returns:
            List of Attachment objects
        """
        # Guard against None or invalid message_id
        if message_id is None or not isinstance(message_id, int):
            return []

        with self._connection_context() as conn:
            cursor = conn.cursor()

            query = get_query("attachments", self._schema_version or "v14")

            try:
                cursor.execute(query, (message_id,))
                rows = cursor.fetchall()
                # Convert rows to dicts, handling potential malformed rows
                row_dicts = []
                for row in rows:
                    try:
                        row_dicts.append(dict(row))
                    except (IndexError, TypeError):
                        # Skip malformed rows
                        continue
                return parse_attachments(row_dicts)
            except (sqlite3.OperationalError, sqlite3.InterfaceError) as e:
                logger.debug(f"Error fetching attachments for message {message_id}: {e}")
                return []

    def _get_reactions_for_message(self, message_guid: str) -> list[Reaction]:
        """Fetch reactions for a specific message.

        Args:
            message_guid: The message GUID (used for associated_message_guid lookups)

        Returns:
            List of Reaction objects
        """
        # Guard against None or invalid guid
        if not message_guid or not isinstance(message_guid, str):
            return []

        with self._connection_context() as conn:
            cursor = conn.cursor()

            query = get_query("reactions", self._schema_version or "v14")

            try:
                cursor.execute(query, (message_guid,))
                rows = cursor.fetchall()
                reactions = parse_reactions([dict(row) for row in rows])

                # Resolve sender names for reactions
                for reaction in reactions:
                    if reaction.sender != "me":
                        reaction.sender_name = self._resolve_contact_name(reaction.sender)

                return reactions
            except (sqlite3.OperationalError, sqlite3.InterfaceError) as e:
                logger.debug(f"Error fetching reactions for message {message_guid}: {e}")
                return []

    def _get_message_rowid_by_guid(self, guid: str) -> int | None:
        """Get message ROWID from GUID.

        Args:
            guid: The message GUID

        Returns:
            Message ROWID, or None if not found
        """
        # Check cache first (LRU cache with bounded size)
        cached = self._guid_to_rowid_cache.get(guid)
        if cached is not None:
            return cached

        with self._connection_context() as conn:
            cursor = conn.cursor()

            query = get_query("message_by_guid", self._schema_version or "v14")

            try:
                cursor.execute(query, (guid,))
                row = cursor.fetchone()
                if row:
                    rowid: int = int(row["id"])
                    self._guid_to_rowid_cache.set(guid, rowid)
                    return rowid
                return None
            except (sqlite3.OperationalError, sqlite3.InterfaceError) as e:
                logger.debug(f"Error fetching message by GUID {guid}: {e}")
                return None

    def get_user_name(self) -> str:
        """Attempt to find the system user's name from AddressBook."""
        if self._contacts_cache is None:
            self._load_contacts_cache()

        # Look for common user name patterns in cache
        if self._contacts_cache is not None:
            for name in self._contacts_cache.values():
                if "Jwalin" in name:
                    return name.split()[0]  # Just first name for the prompt

        return "Me"

    def _resolve_contact_name(self, identifier: str) -> str | None:
        """Resolve a phone number or email to a contact name.

        Queries the macOS AddressBook database if available.

        Args:
            identifier: Phone number or email address

        Returns:
            Contact name if found, None otherwise
        """
        if not identifier or identifier == "me":
            return None

        # Lazily load contacts cache
        if self._contacts_cache is None:
            if self._use_pool:
                # Use pool's cache lock to prevent multiple threads from loading simultaneously
                with self._get_pool()._cache_lock:
                    if self._get_pool()._contacts_cache is None:
                        self._load_contacts_cache()
            else:
                self._load_contacts_cache()

        # Normalize the identifier for lookup
        normalized = normalize_phone_number(identifier)
        if normalized is None:
            return None

        # Cache is guaranteed to be initialized after _load_contacts_cache
        cache = self._contacts_cache
        if cache is None:
            return None
        return cache.get(normalized)

    def _load_contacts_cache(self) -> None:
        """Load contacts from AddressBook database into cache.

        This attempts to read from ALL macOS AddressBook SQLite databases
        (iCloud, Google, On My Mac, etc.) in the Sources directory.
        If unavailable (no Full Disk Access or different OS), cache remains empty.

        Includes timeout protection to prevent hanging on large contact databases.
        Uses ThreadPoolExecutor for thread-safe timeout (works on any thread).
        """
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        # Load into a local dict first to avoid partial cache state
        new_cache: dict[str, str] = {}

        # Try to find AddressBook database
        if not ADDRESSBOOK_DB_PATH.exists():
            logger.debug("AddressBook path not found, contacts resolution disabled")
            self._contacts_cache = new_cache
            return

        def _do_load_contacts() -> int:
            loaded_count = 0
            for source_dir in ADDRESSBOOK_DB_PATH.iterdir():
                if not source_dir.is_dir():
                    continue
                ab_db = source_dir / "AddressBook-v22.abcddb"
                if ab_db.exists():
                    self._load_contacts_from_db_to_dict(ab_db, new_cache)
                    loaded_count += 1
            return loaded_count

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_load_contacts)
                try:
                    loaded_count = future.result(timeout=5.0)
                    logger.debug(f"Loaded contacts from {loaded_count} AddressBook sources")
                except FuturesTimeoutError:
                    logger.warning(
                        "Contact loading timed out after 5s. Using partial cache with %d contacts.",
                        len(new_cache),
                    )
        except PermissionError:
            logger.debug("Permission denied accessing AddressBook directory")
        except OSError as e:
            logger.debug("I/O error accessing AddressBook: %s", e)

        # Atomically update the cache (sets on pool if in pool mode)
        self._contacts_cache = new_cache

    def _load_contacts_from_db_to_dict(self, db_path: Path, cache: dict[str, str]) -> None:
        """Load contacts from a specific AddressBook database into the provided dict.

        Args:
            db_path: Path to the AddressBook database
            cache: Dictionary to load contacts into
        """
        try:
            uri = f"file:{db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=DB_TIMEOUT_SECONDS)
            try:
                conn.row_factory = sqlite3.Row

                # Query for phone numbers and emails with associated names
                # AddressBook schema: ZABCDRECORD has names, ZABCDPHONENUMBER has phones,
                # ZABCDEMAILADDRESS has emails
                cursor = conn.cursor()

                # Load phone numbers with names
                try:
                    cursor.execute(
                        """
                        SELECT
                            ZABCDPHONENUMBER.ZFULLNUMBER as identifier,
                            ZABCDRECORD.ZFIRSTNAME as first_name,
                            ZABCDRECORD.ZLASTNAME as last_name
                        FROM ZABCDPHONENUMBER
                        JOIN ZABCDRECORD ON ZABCDPHONENUMBER.ZOWNER = ZABCDRECORD.Z_PK
                        WHERE ZABCDPHONENUMBER.ZFULLNUMBER IS NOT NULL
                    """
                    )
                    for row in cursor.fetchall():
                        identifier = normalize_phone_number(row["identifier"])
                        name = self._format_name(row["first_name"], row["last_name"])
                        if identifier and name:
                            cache[identifier] = name
                except (sqlite3.OperationalError, sqlite3.InterfaceError):
                    pass  # Table structure may differ

                # Load email addresses with names
                try:
                    cursor.execute(
                        """
                        SELECT
                            ZABCDEMAILADDRESS.ZADDRESS as identifier,
                            ZABCDRECORD.ZFIRSTNAME as first_name,
                            ZABCDRECORD.ZLASTNAME as last_name
                        FROM ZABCDEMAILADDRESS
                        JOIN ZABCDRECORD ON ZABCDEMAILADDRESS.ZOWNER = ZABCDRECORD.Z_PK
                        WHERE ZABCDEMAILADDRESS.ZADDRESS IS NOT NULL
                    """
                    )
                    for row in cursor.fetchall():
                        identifier = row["identifier"]
                        name = self._format_name(row["first_name"], row["last_name"])
                        if identifier and name:
                            cache[identifier.lower()] = name
                except (sqlite3.OperationalError, sqlite3.InterfaceError):
                    pass  # Table structure may differ

                logger.debug(f"Loaded {len(cache)} contacts from AddressBook")
            finally:
                conn.close()

        except (sqlite3.Error, OSError) as e:
            logger.debug(f"Error loading contacts: {e}")

    @staticmethod
    def _format_name(first_name: str | None, last_name: str | None) -> str | None:
        """Format first and last name into a display name.

        Args:
            first_name: First name or None
            last_name: Last name or None

        Returns:
            Formatted name, or None if both are empty
        """
        parts = []
        if first_name:
            parts.append(first_name)
        if last_name:
            parts.append(last_name)
        return " ".join(parts) if parts else None

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context manager and close connection."""
        self.close()

    def check_access(self) -> bool:
        """Check if we have permission to read chat.db.

        Tests Full Disk Access by attempting to open the database file.

        Returns:
            True if access is granted, False otherwise

        Note:
            For error details, use require_access() which raises
            iMessageAccessError with specific information.
        """
        if not self.db_path.exists():
            logger.warning(f"chat.db not found at {self.db_path}")
            return False

        try:
            # Try to open the database
            with self._connection_context() as conn:
                # Execute a simple query to verify read access
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM chat LIMIT 1")
                cursor.fetchone()
            return True
        except PermissionError:
            logger.warning(
                f"Permission denied for {self.db_path}. "
                "Grant Full Disk Access to Terminal/IDE in System Settings."
            )
            return False
        except (sqlite3.OperationalError, sqlite3.InterfaceError) as e:
            if "unable to open database" in str(e).lower():
                logger.warning(f"Cannot open {self.db_path}. Ensure Full Disk Access is granted.")
            else:
                logger.warning(f"Database error: {e}")
            return False
        except OSError as e:
            # File system errors (e.g., file locked, disk I/O issues)
            logger.warning("I/O error checking access: %s", e)
            return False
        except (ValueError, TypeError) as e:
            # Configuration or type errors during access check
            logger.warning("Configuration error checking access: %s", e)
            return False
        except sqlite3.Error:
            # Catch sqlite3.Error subclasses not covered above (e.g., DatabaseError,
            # InternalError). PermissionError and OSError are already handled.
            logger.exception("Unexpected DB error checking access to %s", self.db_path)
            return False

    def require_access(self) -> None:
        """Verify access to chat.db, raising an exception if access is denied.

        Use this method when you want detailed error information about
        why access failed. For simple boolean checks, use check_access().

        Raises:
            iMessageAccessError: If database is not found or permission is denied.
        """
        db_path_str = str(self.db_path)

        if not self.db_path.exists():
            raise imessage_db_not_found(db_path_str)

        try:
            # Try to open the database
            with self._connection_context() as conn:
                # Execute a simple query to verify read access
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM chat LIMIT 1")
                cursor.fetchone()
        except PermissionError as e:
            raise imessage_permission_denied(db_path_str) from e
        except (sqlite3.OperationalError, sqlite3.InterfaceError) as e:
            if "unable to open database" in str(e).lower():
                raise imessage_permission_denied(db_path_str) from e
            else:
                raise iMessageAccessError(
                    f"Database error: {e}",
                    db_path=db_path_str,
                    code=ErrorCode.MSG_QUERY_FAILED,
                    cause=e,
                ) from e
        except OSError as e:
            # File system errors (e.g., file locked, disk I/O issues)
            raise iMessageAccessError(
                f"I/O error checking access: {e}",
                db_path=db_path_str,
                cause=e,
            ) from e
        except (ValueError, TypeError) as e:
            # Configuration or type errors during access check
            raise iMessageAccessError(
                f"Configuration error checking access: {e}",
                db_path=db_path_str,
                cause=e,
            ) from e
        except sqlite3.Error as e:
            # Catch sqlite3.Error subclasses not covered above (e.g., DatabaseError,
            # InternalError). PermissionError and OSError are already handled.
            logger.exception("Unexpected DB error checking access to %s", db_path_str)
            raise iMessageAccessError(
                f"Unexpected error checking access: {e}",
                db_path=db_path_str,
                cause=e,
            ) from e

    def get_conversations(
        self,
        limit: int = 50,
        since: datetime | None = None,
        before: datetime | None = None,
    ) -> list[Conversation]:
        """Get recent conversations.

        Args:
            limit: Maximum number of conversations to return
            since: Only return conversations with messages after this date
            before: Only return conversations with last message before this date (for pagination)

        Returns:
            List of Conversation objects, sorted by last message date (newest first)
        """
        with track_latency("conversations_fetch", limit=limit):
            with self._connection_context() as conn:
                cursor = conn.cursor()

                # Build params list based on filters
                params: list[Any] = []
                if since is not None:
                    params.append(datetime_to_apple_timestamp(since))
                if before is not None:
                    params.append(datetime_to_apple_timestamp(before))
                params.append(limit)

                # USE OPTIMIZED QUERY
                query = get_query(
                    "conversations_light",
                    self._schema_version or "v14",
                    with_since_filter=since is not None,
                    with_conversations_before_filter=before is not None,
                )

                try:
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                except (sqlite3.OperationalError, sqlite3.InterfaceError) as e:
                    logger.warning(f"Query error in get_conversations: {e}")
                    return []

            if not rows:
                return []

            # --- OPTIMIZATION: Batch Fetch Secondary Data ---
            chat_rowids = [row["chat_id"] for row in rows]
            
            # 1. Batch fetch participants
            participants_map = self._get_participants_batch(chat_rowids)
            
            # 2. Batch fetch last message text
            last_msg_map = self._get_last_messages_batch(chat_rowids)

            # 3. Warm AddressBook cache
            self._resolve_contact_name("none")

            conversations = []
            for row in rows:
                chat_rowid = row["chat_id"]
                chat_guid = row["chat_guid"]
                
                # Use batch-fetched data
                participants = participants_map.get(chat_rowid, [])
                last_message_text = last_msg_map.get(chat_rowid)
                
                # Determine if group chat
                is_group = len(participants) > 1

                # Parse last message date
                last_message_date = parse_apple_timestamp(row["last_message_date"])

                # Resolve display name from database or contacts
                display_name = row["display_name"] or None

                if not display_name:
                    if not is_group and len(participants) == 1:
                        display_name = self._resolve_contact_name(participants[0])
                    elif is_group and participants:
                        # For groups, try to build a name from participants
                        resolved_names = []
                        for p in participants[:3]:
                            name = self._resolve_contact_name(p)
                            if name:
                                resolved_names.append(name.split()[0])

                        if resolved_names:
                            suffix = "..." if len(participants) > len(resolved_names) else ""
                            display_name = ", ".join(resolved_names) + suffix

                conversations.append(
                    Conversation(
                        chat_id=chat_guid,
                        participants=participants if participants else ["unknown"],
                        display_name=display_name,
                        last_message_date=last_message_date,
                        message_count=row["message_count"],
                        is_group=is_group,
                        last_message_text=last_message_text,
                    )
                )

            return conversations

    def get_conversation(self, chat_id: str) -> Conversation | None:
        """Get a single conversation by chat_id.

        Direct lookup instead of fetching all conversations and filtering.

        Args:
            chat_id: The conversation ID (chat.guid)

        Returns:
            Conversation object if found, None otherwise
        """
        if not chat_id or not isinstance(chat_id, str):
            return None

        with self._connection_context() as conn:
            cursor = conn.cursor()

            query = get_query(
                "conversation_by_chat_id",
                self._schema_version or "v14",
            )

            try:
                cursor.execute(query, (chat_id, chat_id))
                row = cursor.fetchone()
            except (sqlite3.OperationalError, sqlite3.InterfaceError) as e:
                logger.warning(f"Query error in get_conversation: {e}")
                return None

        if row is None:
            return None

        # Parse participants
        participants_str = row["participants"] or ""
        participants = [
            normalized
            for p in participants_str.split(",")
            if p.strip() and (normalized := normalize_phone_number(p.strip())) is not None
        ]

        is_group = len(participants) > 1
        last_message_date = parse_apple_timestamp(row["last_message_date"])

        display_name = row["display_name"] or None
        if not display_name:
            if not is_group and len(participants) == 1:
                display_name = self._resolve_contact_name(participants[0])
            elif is_group and participants:
                resolved_names = []
                for p in participants[:3]:
                    name = self._resolve_contact_name(p)
                    if name:
                        resolved_names.append(name.split()[0])
                if resolved_names:
                    suffix = "..." if len(participants) > len(resolved_names) else ""
                    display_name = ", ".join(resolved_names) + suffix

        row_keys = row.keys()
        last_message_text = row["last_message_text"] if "last_message_text" in row_keys else None
        if not last_message_text and "last_message_attributed_body" in row_keys:
            attributed_body = row["last_message_attributed_body"]
            if attributed_body:
                from .parser import parse_attributed_body

                last_message_text = parse_attributed_body(attributed_body)

        return Conversation(
            chat_id=row["chat_id"],
            participants=participants,
            display_name=display_name,
            last_message_date=last_message_date,
            message_count=row["message_count"],
            is_group=is_group,
            last_message_text=last_message_text,
        )

    def _get_participants_batch(self, chat_ids: list[int]) -> dict[int, list[str]]:
        """Batch fetch participants for a list of chat ROWIDs."""
        if not chat_ids:
            return {}

        placeholders = ", ".join(["?"] * len(chat_ids))
        query = f"""
            SELECT chat_handle_join.chat_id, handle.id
            FROM chat_handle_join
            JOIN handle ON chat_handle_join.handle_id = handle.ROWID
            WHERE chat_handle_join.chat_id IN ({placeholders})
        """

        result: dict[int, list[str]] = {}
        with self._connection_context() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query, chat_ids)
                for row in cursor.fetchall():
                    cid = row[0]
                    handle = normalize_phone_number(row[1]) or row[1]
                    if cid not in result:
                        result[cid] = []
                    result[cid].append(handle)
            except (sqlite3.OperationalError, sqlite3.InterfaceError) as e:
                logger.debug(f"Error batch-fetching participants: {e}")

        return result

    def _get_last_messages_batch(self, chat_ids: list[int]) -> dict[int, str]:
        """Batch fetch last message text for a list of chat ROWIDs."""
        if not chat_ids:
            return {}

        # Using ROW_NUMBER() over partitions is expensive, so we use a simpler
        # join on the message table using the chat_message_join last entry
        placeholders = ", ".join(["?"] * len(chat_ids))
        query = f"""
            SELECT cmj.chat_id, m.text, m.attributedBody
            FROM chat_message_join cmj
            JOIN message m ON cmj.message_id = m.ROWID
            WHERE cmj.chat_id IN ({placeholders})
            AND cmj.message_date = (
                SELECT MAX(message_date)
                FROM chat_message_join cmj2
                WHERE cmj2.chat_id = cmj.chat_id
            )
        """

        result: dict[int, str] = {}
        with self._connection_context() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query, chat_ids)
                for row in cursor.fetchall():
                    cid = row[0]
                    text = row[1]
                    if not text and row[2]:
                        from .parser import parse_attributed_body
                        text = parse_attributed_body(row[2])
                    
                    if text:
                        result[cid] = text
            except (sqlite3.OperationalError, sqlite3.InterfaceError) as e:
                logger.debug(f"Error batch-fetching last messages: {e}")

        return result

    def get_messages(
        self,
        chat_id: str,
        limit: int = 100,
        before: datetime | None = None,
        after: datetime | None = None,
    ) -> list[Message]:
        """Get messages from a conversation.

        Args:
            chat_id: The conversation ID (chat.guid)
            limit: Maximum number of messages to return
            before: Only return messages before this date
            after: Only return messages after this date

        Returns:
            List of Message objects, sorted by date (newest first)
        """
        # Validate chat_id
        if not chat_id or not isinstance(chat_id, str):
            logger.debug(f"Invalid chat_id: {chat_id}")
            return []

        with track_latency("message_load", chat_id=chat_id, limit=limit):
            with self._connection_context() as conn:
                cursor = conn.cursor()

                # Build params list based on filters
                params: list[Any] = [chat_id]
                if after is not None:
                    params.append(datetime_to_apple_timestamp(after))
                if before is not None:
                    params.append(datetime_to_apple_timestamp(before))
                params.append(limit)

                query = get_query(
                    "messages",
                    self._schema_version or "v14",
                    with_after_filter=after is not None,
                    with_before_filter=before is not None,
                )

                rows = self._execute_with_fallback(cursor, query, params)

            # Warm AddressBook cache before processing rows
            self._resolve_contact_name("none")

            return self._rows_to_messages(rows, chat_id)

    @sqlite_retry()
    def get_messages_batch(
        self,
        chat_ids: list[str],
        limit_per_chat: int | None = 10000,
    ) -> dict[str, list[Message]]:
        """Batch-fetch messages for multiple conversations in fewer queries.

        Instead of N separate queries (one per chat_id), this method processes
        chat_ids in chunks of 900 (SQLite variable limit) and fetches all
        messages per chunk in a single query, then groups by chat_id.

        Args:
            chat_ids: List of conversation IDs (chat.guid).
            limit_per_chat: Maximum messages per conversation.

        Returns:
            Dict mapping chat_id -> list of Message objects (newest first).
        """
        if not chat_ids:
            return {}

        result: dict[str, list[Message]] = {cid: [] for cid in chat_ids}
        chunk_size = 900  # SQLite max variable limit is 999

        # If no limit, use a very large number effectively meaning "all"
        effective_limit = limit_per_chat if limit_per_chat is not None else 1_000_000_000

        with self._connection_context() as conn:
            cursor = conn.cursor()

            # Build a batch query with IN clause and ROW_NUMBER for per-chat limit
            base_query = """
                WITH ranked AS (
                    SELECT
                        message.ROWID as id,
                        message.guid,
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
                        message.thread_originator_guid as reply_to_guid,
                        message.date_delivered,
                        message.date_read,
                        message.group_action_type,
                        affected_handle.id as affected_handle_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY chat.guid ORDER BY message.date DESC
                        ) as rn
                    FROM message
                    JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
                    JOIN chat ON chat_message_join.chat_id = chat.ROWID
                    LEFT JOIN handle ON message.handle_id = handle.ROWID
                    LEFT JOIN handle AS affected_handle
                        ON message.other_handle = affected_handle.ROWID
                    WHERE chat.guid IN ({placeholders})
                )
                SELECT id, guid, chat_id, sender, text, attributedBody, date,
                       is_from_me, reply_to_guid, date_delivered, date_read,
                       group_action_type, affected_handle_id
                FROM ranked
                WHERE rn <= ?
                ORDER BY chat_id, date DESC
            """

            for i in range(0, len(chat_ids), chunk_size):
                chunk = chat_ids[i : i + chunk_size]
                placeholders = ",".join("?" * len(chunk))
                query = base_query.format(placeholders=placeholders)
                params: list[Any] = list(chunk) + [effective_limit]

                try:
                    rows = self._execute_with_fallback(cursor, query, params)
                except Exception as e:
                    logger.warning(f"Batch message fetch failed for chunk: {e}")
                    # Fall back to individual queries for this chunk
                    for cid in chunk:
                        result[cid] = self.get_messages(cid, limit=limit_per_chat or 10000)
                    continue

                # Group rows by chat_id
                rows_by_chat: dict[str, list[sqlite3.Row]] = {}
                for row in rows:
                    cid = row["chat_id"]
                    if cid not in rows_by_chat:
                        rows_by_chat[cid] = []
                    rows_by_chat[cid].append(row)

                # Batch-prefetch attachments and reactions across all messages in chunk
                all_message_ids = []
                all_id_guid_map: dict[int, str | None] = {}
                for row in rows:
                    try:
                        mid = row["id"]
                        all_message_ids.append(mid)
                        all_id_guid_map[mid] = row["guid"] if "guid" in row.keys() else None
                    except (IndexError, KeyError):
                        continue

                attachments_map = self._prefetch_attachments(all_message_ids)
                reactions_map = self._prefetch_reactions(
                    all_message_ids, id_guid_map=all_id_guid_map
                )

                # Convert rows to Message objects per chat
                for cid, chat_rows in rows_by_chat.items():
                    messages = []
                    for row in chat_rows:
                        try:
                            msg = self._row_to_message(
                                row,
                                cid,
                                prefetched_attachments=attachments_map,
                                prefetched_reactions=reactions_map,
                            )
                            if msg:
                                messages.append(msg)
                        except (IndexError, KeyError, TypeError) as e:
                            logger.debug(f"Skipping malformed row in batch: {e}")
                            continue
                    result[cid] = messages

        return result

    def get_messages_after(
        self,
        message_id: int,
        chat_id: str,
        limit: int = 10,
    ) -> list[Message]:
        """Get messages that occurred after a specific message ID.

        Args:
            message_id: The ROWID to start after
            chat_id: The conversation ID
            limit: Maximum messages to return

        Returns:
            List of Message objects, sorted by date (oldest first)
        """
        with self._connection_context() as conn:
            cursor = conn.cursor()

            query = get_query("messages_after", self._schema_version or "v14")

            try:
                cursor.execute(query, (message_id, chat_id, limit))
                rows = cursor.fetchall()
                return self._rows_to_messages(rows, chat_id)
            except (sqlite3.OperationalError, sqlite3.InterfaceError) as e:
                logger.warning(f"Query error in get_messages_after: {e}")
                return []

    def search(
        self,
        query: str,
        limit: int = 50,
        sender: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        chat_id: str | None = None,
        has_attachments: bool | None = None,
    ) -> list[Message]:
        """Full-text search across messages with optional filters.

        Args:
            query: Search query string
            limit: Maximum number of results
            sender: Filter by sender phone number or email (use "me" for own messages)
            after: Filter for messages after this datetime
            before: Filter for messages before this datetime
            chat_id: Filter by conversation ID (chat.guid)
            has_attachments: Filter for messages with (True) or without (False) attachments

        Returns:
            List of Message objects matching the query and filters
        """
        with self._connection_context() as conn:
            cursor = conn.cursor()

            # Escape LIKE wildcards (query uses ESCAPE '\\' clause)
            escaped_query = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            like_pattern = f"%{escaped_query}%"

            # Build params list based on filters
            # Order: like_pattern, [sender, sender], [after], [before], [chat_id], limit
            params: list[Any] = [like_pattern]

            # Normalize sender if provided
            normalized_sender: str | None = None
            if sender is not None:
                normalized_sender = sender if sender == "me" else normalize_phone_number(sender)
                # If normalization returned None (invalid input), skip the filter
                if normalized_sender is not None:
                    # Add sender param twice for the OR condition in the query
                    params.append(normalized_sender)
                    params.append(normalized_sender)
                else:
                    # Treat as no sender filter if normalization failed
                    sender = None

            if after is not None:
                params.append(datetime_to_apple_timestamp(after))

            if before is not None:
                params.append(datetime_to_apple_timestamp(before))

            if chat_id is not None:
                params.append(chat_id)

            params.append(limit)

            sql = get_query(
                "search",
                self._schema_version or "v14",
                with_sender_filter=sender is not None,
                with_after_filter=after is not None,
                with_search_before_filter=before is not None,
                with_chat_id_filter=chat_id is not None,
                with_has_attachments_filter=has_attachments,
            )

            try:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
            except (sqlite3.OperationalError, sqlite3.InterfaceError) as e:
                logger.warning(f"Query error in search: {e}")
                return []

            # Collect message IDs and GUIDs for batch prefetch
            message_ids = []
            id_guid_map: dict[int, str | None] = {}
            for row in rows:
                try:
                    mid = row["id"]
                    message_ids.append(mid)
                    guid = row["guid"] if "guid" in row.keys() else None
                    id_guid_map[mid] = guid
                except (IndexError, KeyError):
                    continue

            # Batch-fetch attachments and reactions (2 queries instead of 2*N)
            attachments_map = self._prefetch_attachments(message_ids)
            reactions_map = self._prefetch_reactions(message_ids, id_guid_map=id_guid_map)

            messages = []
            for row in rows:
                msg = self._row_to_message(
                    row,
                    row["chat_id"],
                    prefetched_attachments=attachments_map,
                    prefetched_reactions=reactions_map,
                )
                if msg:
                    messages.append(msg)

            return messages

    def get_conversation_context(
        self,
        chat_id: str,
        around_message_id: int,
        context_messages: int = 5,
    ) -> list[Message]:
        """Get messages around a specific message for context.

        Args:
            chat_id: The conversation ID
            around_message_id: The message ROWID to center around
            context_messages: Number of messages before and after

        Returns:
            List of Message objects around the target message
        """
        with self._connection_context() as conn:
            cursor = conn.cursor()

            # Get 2*context + 1 messages centered on the target
            total_limit = context_messages * 2 + 1

            query = get_query("context", self._schema_version or "v14")

            rows = self._execute_with_fallback(
                cursor,
                query,
                (chat_id, around_message_id, total_limit),
            )

            messages = self._rows_to_messages(rows, chat_id)

        # Sort by date for proper ordering
        messages.sort(key=lambda m: m.date)

        return messages

    @sqlite_retry()
    def _execute_with_fallback(
        self,
        cursor: sqlite3.Cursor,
        query: str,
        params: list[Any] | tuple[Any, ...],
    ) -> list[sqlite3.Row]:
        """Execute a query with automatic fallback for missing columns.

        If the query fails due to a missing column, replaces optional columns
        (group_action_type, affected_handle_id, date_delivered, date_read) with
        NULL and retries.

        Args:
            cursor: Database cursor
            query: SQL query string
            params: Query parameters

        Returns:
            List of result rows, empty list on failure
        """
        try:
            cursor.execute(query, params)
            return cursor.fetchall()
        except (sqlite3.OperationalError, sqlite3.InterfaceError) as e:
            error_str = str(e).lower()
            if "no such column" not in error_str:
                logger.warning(f"Query error: {e}")
                return []

            logger.debug(f"Query failed with missing column ({e}), using fallback")
            fallback_query = (
                query.replace(
                    "message.date_delivered,",
                    "NULL as date_delivered,",
                )
                .replace(
                    "message.date_read,",
                    "NULL as date_read,",
                )
                .replace(
                    "message.group_action_type,",
                    "NULL as group_action_type,",
                )
                .replace(
                    "affected_handle.id as affected_handle_id",
                    "NULL as affected_handle_id",
                )
                .replace(
                    "LEFT JOIN handle AS affected_handle "
                    "ON message.other_handle = affected_handle.ROWID",
                    "",
                )
            )

            try:
                cursor.execute(fallback_query, params)
                return cursor.fetchall()
            except (sqlite3.OperationalError, sqlite3.InterfaceError) as e2:
                logger.warning(f"Query error after column fallback: {e2}")
                return []

    def _prefetch_attachments(
        self,
        message_ids: list[int],
    ) -> dict[int, list[Attachment]]:
        """Batch-fetch attachments for multiple messages in one query.

        Processes in chunks of 500 to avoid SQLite parameter limits.

        Args:
            message_ids: List of message ROWIDs

        Returns:
            Dict mapping message_id -> list of Attachment objects
        """
        if not message_ids:
            return {}

        with self._connection_context() as conn:
            cursor = conn.cursor()
            version = self._schema_version or "v14"

            result: dict[int, list[Attachment]] = {}

            # Process in chunks of 500 to avoid SQLite limits (max ~999 parameters)
            chunk_size = 500
            for i in range(0, len(message_ids), chunk_size):
                chunk = message_ids[i : i + chunk_size]
                try:
                    placeholders = ",".join("?" * len(chunk))
                    query = get_query("attachments_batch", version).format(
                        placeholders=placeholders,
                    )
                    cursor.execute(query, chunk)
                    rows = cursor.fetchall()
                    for row in rows:
                        try:
                            row_dict = dict(row)
                            mid = row_dict.pop("message_id")
                            attachments = parse_attachments([row_dict])
                            if mid not in result:
                                result[mid] = []
                            result[mid].extend(attachments)
                        except (IndexError, TypeError, KeyError):
                            continue
                except (sqlite3.OperationalError, sqlite3.InterfaceError) as e:
                    logger.debug(f"Error batch-fetching attachments (chunk {i}): {e}")

            return result

    def _prefetch_reactions(
        self,
        message_ids: list[int],
        id_guid_map: dict[int, str | None] | None = None,
    ) -> dict[int, list[Reaction]]:
        """Batch-fetch reactions for multiple messages in one query.

        If id_guid_map is provided (from rows that already include message.guid),
        skips the extra GUID lookup query. Otherwise falls back to batch-fetching
        GUIDs first.

        Args:
            message_ids: List of message ROWIDs
            id_guid_map: Optional pre-built mapping of message_id -> guid

        Returns:
            Dict mapping message_id -> list of Reaction objects
        """
        if not message_ids:
            return {}

        with self._connection_context() as conn:
            cursor = conn.cursor()
            version = self._schema_version or "v14"

            result: dict[int, list[Reaction]] = {}

            # Process in chunks of 500 to avoid SQLite limits (max ~999 parameters)
            chunk_size = 500

            # Step 1: Build GUID maps (skip DB query if already provided)
            id_to_guid: dict[int, str] = {}
            guid_to_id: dict[str, int] = {}

            if id_guid_map and len(id_guid_map) == len(message_ids):
                # All GUIDs provided from query results - skip extra DB query
                id_to_guid = {k: v for k, v in id_guid_map.items() if v is not None}
                guid_to_id = {v: k for k, v in id_to_guid.items()}
            else:
                # Fallback: batch-fetch GUIDs from DB
                for i in range(0, len(message_ids), chunk_size):
                    chunk = message_ids[i : i + chunk_size]
                    try:
                        placeholders = ",".join("?" * len(chunk))
                        guid_query = get_query("message_guids_batch", version).format(
                            placeholders=placeholders,
                        )
                        cursor.execute(guid_query, chunk)
                        guid_rows = cursor.fetchall()

                        for row in guid_rows:
                            mid = row["id"]
                            guid = row["guid"]
                            if guid:
                                id_to_guid[mid] = guid
                                guid_to_id[guid] = mid
                    except (sqlite3.OperationalError, sqlite3.InterfaceError) as e:
                        logger.debug(f"Error batch-fetching GUIDs (chunk {i}): {e}")

            if not guid_to_id:
                return result

            # Step 2: Batch-fetch reactions for all GUIDs in chunks
            all_guids = list(guid_to_id.keys())
            for i in range(0, len(all_guids), chunk_size):
                chunk_guids = all_guids[i : i + chunk_size]
                try:
                    placeholders = ",".join("?" * len(chunk_guids))
                    rx_query = get_query("reactions_batch", version).format(
                        placeholders=placeholders,
                    )
                    cursor.execute(rx_query, chunk_guids)
                    rx_rows = cursor.fetchall()

                    for row in rx_rows:
                        row_dict = dict(row)
                        assoc_guid = row_dict.get("associated_message_guid")
                        if assoc_guid and assoc_guid in guid_to_id:
                            mid = guid_to_id[assoc_guid]
                            reactions = parse_reactions([row_dict])
                            # Resolve sender names
                            for reaction in reactions:
                                if reaction.sender != "me":
                                    reaction.sender_name = self._resolve_contact_name(
                                        reaction.sender,
                                    )
                            if mid not in result:
                                result[mid] = []
                            result[mid].extend(reactions)
                except (sqlite3.OperationalError, sqlite3.InterfaceError) as e:
                    logger.debug(f"Error batch-fetching reactions (chunk {i}): {e}")

        return result

    def _prefetch_reply_rowids(self, guids: Sequence[str]) -> dict[str, int]:
        """Batch-lookup message row IDs for a list of GUIDs.

        Args:
            guids: List of message GUIDs to resolve

        Returns:
            Dict mapping GUID -> row ID
        """
        if not guids:
            return {}

        # Remove duplicates and filter out GUIDs already in cache
        unique_guids = list(set(guids))
        result: dict[str, int] = {}
        missing_guids = []

        for guid in unique_guids:
            cached = self._guid_to_rowid_cache.get(guid)
            if cached is not None:
                result[guid] = cached
            else:
                missing_guids.append(guid)

        if not missing_guids:
            return result

        # Fetch missing GUIDs in chunks
        chunk_size = 900
        for i in range(0, len(missing_guids), chunk_size):
            chunk = missing_guids[i : i + chunk_size]
            placeholders = ", ".join(["?"] * len(chunk))
            query = f"SELECT ROWID, guid FROM message WHERE guid IN ({placeholders})"

            with self._connection_context() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(query, chunk)
                    for row in cursor.fetchall():
                        rid = int(row[0])
                        guid = str(row[1])
                        result[guid] = rid
                        self._guid_to_rowid_cache.set(guid, rid)
                except (sqlite3.OperationalError, sqlite3.InterfaceError) as e:
                    logger.debug(f"Error batch-fetching reply rowids: {e}")

        return result

    def _rows_to_messages(self, rows: Sequence[sqlite3.Row], chat_id: str) -> list[Message]:
        """Convert database rows to Message objects.

        Pre-fetches attachments and reactions in batch to avoid N+1 queries.

        Args:
            rows: Sequence of sqlite3.Row objects
            chat_id: The conversation ID

        Returns:
            List of Message objects
        """
        # Collect message IDs and GUIDs for batch prefetch
        message_ids = []
        id_guid_map: dict[int, str | None] = {}
        for row in rows:
            try:
                mid = row["id"]
                message_ids.append(mid)
                # Extract GUID if available (avoids extra batch query in _prefetch_reactions)
                guid = row["guid"] if "guid" in row.keys() else None
                id_guid_map[mid] = guid
            except (IndexError, KeyError):
                continue

        # Batch-fetch attachments and reactions (2 queries instead of 2*N)
        attachments_map = self._prefetch_attachments(message_ids)
        reactions_map = self._prefetch_reactions(message_ids, id_guid_map=id_guid_map)
        
        # Prefetch reply row IDs
        reply_guids = [
            row["reply_to_guid"] 
            for row in rows 
            if "reply_to_guid" in row.keys() and row["reply_to_guid"]
        ]
        reply_map = self._prefetch_reply_rowids(reply_guids)

        messages = []
        for row in rows:
            try:
                msg = self._row_to_message(
                    row,
                    chat_id,
                    prefetched_attachments=attachments_map,
                    prefetched_reactions=reactions_map,
                    prefetched_replies=reply_map,
                )
                if msg:
                    messages.append(msg)
            except (IndexError, KeyError, TypeError) as e:
                # Skip malformed rows
                logger.debug(f"Skipping malformed row: {e}")
                continue
        return messages

    def _row_to_message(
        self,
        row: sqlite3.Row,
        chat_id: str,
        prefetched_attachments: dict[int, list[Attachment]] | None = None,
        prefetched_reactions: dict[int, list[Reaction]] | None = None,
        prefetched_replies: dict[str, int] | None = None,
    ) -> Message | None:
        """Convert a database row to a Message object.

        Args:
            row: sqlite3.Row object
            chat_id: The conversation ID
            prefetched_attachments: Pre-fetched attachments keyed by message ID
            prefetched_reactions: Pre-fetched reactions keyed by message ID
            prefetched_replies: Pre-fetched reply row IDs keyed by GUID

        Returns:
            Message object, or None if message has no text and no attachments
        """
        message_id = row["id"]
        row_dict = dict(row)

        # Get sender and resolve name from contacts
        sender = normalize_phone_number(row["sender"]) or row["sender"]
        sender_name = None
        if not row["is_from_me"] and sender:
            sender_name = self._resolve_contact_name(sender)

        # Check for group events (system messages)
        group_action_type = row_dict.get("group_action_type", 0) or 0
        if group_action_type != 0:
            return self._parse_system_message(
                row,
                row_dict,
                message_id,
                chat_id,
                sender,
                sender_name,
                group_action_type,
            )

        # Extract text (tries text column, falls back to attributedBody)
        text = extract_text_from_row(row_dict) or ""

        attachments = self._resolve_attachments(message_id, prefetched_attachments)

        # Skip messages with no text AND no attachments (and not a system message)
        if not text and not attachments:
            return None

        # Parse reply_to_id from thread_originator_guid
        reply_to_id = None
        reply_to_guid = row_dict.get("reply_to_guid")
        if reply_to_guid:
            if prefetched_replies and reply_to_guid in prefetched_replies:
                reply_to_id = prefetched_replies[reply_to_guid]
            else:
                reply_to_id = self._get_message_rowid_by_guid(reply_to_guid)

        reactions = self._resolve_reactions(message_id, prefetched_reactions)
        date_delivered, date_read = self._parse_receipts(row, row_dict)

        return Message(
            id=message_id,
            chat_id=chat_id,
            sender=sender,
            sender_name=sender_name,
            text=text,
            date=parse_apple_timestamp(row["date"]),
            is_from_me=bool(row["is_from_me"]),
            attachments=attachments,
            reply_to_id=reply_to_id,
            reactions=reactions,
            date_delivered=date_delivered,
            date_read=date_read,
            is_system_message=False,
        )

    def _parse_system_message(
        self,
        row: sqlite3.Row,
        row_dict: dict[str, Any],
        message_id: int,
        chat_id: str,
        sender: str,
        sender_name: str | None,
        group_action_type: int,
    ) -> Message:
        """Build a Message for a group event (system message).

        Args:
            row: Original database row
            row_dict: Row as dictionary
            message_id: The message ROWID
            chat_id: The conversation ID
            sender: Normalized sender identifier
            sender_name: Resolved contact name
            group_action_type: The group_action_type value from the database

        Returns:
            Message with is_system_message=True
        """
        text = self._generate_group_event_text(
            group_action_type,
            sender,
            sender_name,
            row_dict.get("affected_handle_id"),
            bool(row["is_from_me"]),
        )
        return Message(
            id=message_id,
            chat_id=chat_id,
            sender=sender,
            sender_name=sender_name,
            text=text,
            date=parse_apple_timestamp(row["date"]),
            is_from_me=bool(row["is_from_me"]),
            attachments=[],
            reply_to_id=None,
            reactions=[],
            is_system_message=True,
        )

    def _resolve_attachments(
        self,
        message_id: int,
        prefetched: dict[int, list[Attachment]] | None,
    ) -> list[Attachment]:
        """Get attachments from prefetched map or fall back to per-message query.

        Args:
            message_id: The message ROWID
            prefetched: Pre-fetched attachments keyed by message ID, or None

        Returns:
            List of Attachment objects
        """
        if prefetched is not None:
            return prefetched.get(message_id, [])
        return self._get_attachments_for_message(message_id)

    def _resolve_reactions(
        self,
        message_id: int,
        prefetched: dict[int, list[Reaction]] | None,
    ) -> list[Reaction]:
        """Get reactions from prefetched map or fall back to per-message query.

        Args:
            message_id: The message ROWID
            prefetched: Pre-fetched reactions keyed by message ID, or None

        Returns:
            List of Reaction objects
        """
        if prefetched is not None:
            return prefetched.get(message_id, [])
        return self._get_reactions_for_message_id(message_id)

    def _parse_receipts(
        self,
        row: sqlite3.Row,
        row_dict: dict[str, Any],
    ) -> tuple[datetime | None, datetime | None]:
        """Parse delivery and read receipt timestamps.

        Only meaningful for messages sent by the current user.

        Args:
            row: Original database row
            row_dict: Row as dictionary

        Returns:
            Tuple of (date_delivered, date_read), either may be None
        """
        date_delivered = None
        date_read = None
        if row["is_from_me"]:
            if row_dict.get("date_delivered"):
                date_delivered = parse_apple_timestamp(row_dict["date_delivered"])
            if row_dict.get("date_read"):
                date_read = parse_apple_timestamp(row_dict["date_read"])
        return date_delivered, date_read

    def _generate_group_event_text(
        self,
        action_type: int,
        actor: str,
        actor_name: str | None,
        affected_handle_id: str | None,
        is_from_me: bool,
    ) -> str:
        """Generate human-readable text for group events.

        Args:
            action_type: The group_action_type from the database
                1 = left/removed, 2 = name changed, 3 = joined/added
            actor: The handle ID of who performed the action
            actor_name: Resolved contact name for actor
            affected_handle_id: The handle ID of who was affected (if different)
            is_from_me: Whether the current user performed the action

        Returns:
            Human-readable description of the group event
        """
        # Get display names
        if is_from_me:
            actor_display = "You"
        else:
            actor_display = actor_name or actor

        affected_display = None
        if affected_handle_id:
            normalized = normalize_phone_number(affected_handle_id) or affected_handle_id
            affected_display = self._resolve_contact_name(normalized) or normalized

        # Generate text based on action type
        if action_type == 1:  # Left or removed
            if affected_display and affected_display != actor_display:
                return f"{actor_display} removed {affected_display} from the group"
            else:
                return f"{actor_display} left the group"
        elif action_type == 2:  # Name changed
            return f"{actor_display} changed the group name"
        elif action_type == 3:  # Joined or added
            if affected_display and affected_display != actor_display:
                return f"{actor_display} added {affected_display} to the group"
            else:
                return f"{actor_display} joined the group"
        else:
            return f"Group event (type {action_type})"

    def _get_reactions_for_message_id(self, message_id: int) -> list[Reaction]:
        """Fetch reactions for a message by its ROWID.

        First looks up the message GUID, then queries for associated reactions.

        Args:
            message_id: The message ROWID

        Returns:
            List of Reaction objects
        """
        # Guard against None or invalid message_id
        if message_id is None or not isinstance(message_id, int):
            return []

        with self._connection_context() as conn:
            cursor = conn.cursor()

            # First get the message GUID
            try:
                cursor.execute("SELECT guid FROM message WHERE ROWID = ?", (message_id,))
                row = cursor.fetchone()
                if not row or not row["guid"]:
                    return []

                message_guid = row["guid"]
                return self._get_reactions_for_message(message_guid)
            except (sqlite3.OperationalError, sqlite3.InterfaceError) as e:
                logger.debug(f"Error fetching GUID for message {message_id}: {e}")
                return []

    def get_attachments(
        self,
        chat_id: str | None = None,
        attachment_type: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get attachments with optional filtering.

        Args:
            chat_id: Optional conversation ID to filter by
            attachment_type: Filter by type ("images", "videos", "audio", "documents")
            after: Filter for attachments after this datetime
            before: Filter for attachments before this datetime
            limit: Maximum number of attachments to return

        Returns:
            List of attachment dictionaries with extended metadata
        """
        with self._connection_context() as conn:
            cursor = conn.cursor()

            # Build params list based on filters
            params: list[Any] = []
            if chat_id is not None:
                params.append(chat_id)
            if after is not None:
                params.append(datetime_to_apple_timestamp(after))
            if before is not None:
                params.append(datetime_to_apple_timestamp(before))
            params.append(limit)

            query = get_query(
                "all_attachments",
                self._schema_version or "v14",
                with_attachment_chat_filter=chat_id is not None,
                with_attachment_type_filter=attachment_type,
                with_attachment_date_after_filter=after is not None,
                with_attachment_date_before_filter=before is not None,
            )

            try:
                cursor.execute(query, params)
                rows = cursor.fetchall()
            except (sqlite3.OperationalError, sqlite3.InterfaceError) as e:
                logger.warning(f"Query error in get_attachments: {e}")
                return []

        results = []
        for row in rows:
            row_dict = dict(row)
            # Parse attachment metadata
            attachments = parse_attachments([row_dict])
            if attachments:
                attachment = attachments[0]
                # Add message context
                raw_sender = row_dict.get("sender") or ""
                sender = normalize_phone_number(raw_sender) or raw_sender
                sender_name = None
                if not row_dict.get("is_from_me") and raw_sender:
                    sender_name = self._resolve_contact_name(raw_sender)
                results.append(
                    {
                        "attachment": attachment,
                        "message_id": row_dict.get("message_id"),
                        "message_date": parse_apple_timestamp(row_dict.get("message_date")),
                        "chat_id": row_dict.get("chat_id"),
                        "sender": sender,
                        "sender_name": sender_name,
                        "is_from_me": bool(row_dict.get("is_from_me")),
                    }
                )

        return results

    def get_attachment_stats(self, chat_id: str) -> dict[str, Any]:
        """Get attachment statistics for a conversation.

        Args:
            chat_id: The conversation ID

        Returns:
            Dictionary with total count, total size, and breakdown by type
        """
        with self._connection_context() as conn:
            cursor = conn.cursor()

            query = get_query("attachment_stats", self._schema_version or "v14")

            try:
                cursor.execute(query, (chat_id,))
                rows = cursor.fetchall()
            except (sqlite3.OperationalError, sqlite3.InterfaceError) as e:
                logger.warning(f"Query error in get_attachment_stats: {e}")
                return {
                    "total_count": 0,
                    "total_size_bytes": 0,
                    "by_type": {},
                    "size_by_type": {},
                }

        total_count = 0
        total_size = 0
        by_type: dict[str, int] = {}
        size_by_type: dict[str, int] = {}

        for row in rows:
            row_dict = dict(row)
            count = row_dict.get("total_count", 0)
            size = row_dict.get("total_size", 0)
            mime_type = row_dict.get("mime_type")

            total_count += count
            total_size += size

            # Categorize by type
            category = categorize_attachment_type(mime_type)

            by_type[category] = by_type.get(category, 0) + count
            size_by_type[category] = size_by_type.get(category, 0) + size

        return {
            "total_count": total_count,
            "total_size_bytes": total_size,
            "by_type": by_type,
            "size_by_type": size_by_type,
        }

    def get_storage_by_conversation(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get storage usage breakdown by conversation.

        Args:
            limit: Maximum number of conversations to return

        Returns:
            List of dictionaries with chat_id, display_name, attachment_count, and total_size
        """
        with self._connection_context() as conn:
            cursor = conn.cursor()

            query = get_query("storage_by_conversation", self._schema_version or "v14")

            try:
                cursor.execute(query, (limit,))
                rows = cursor.fetchall()
            except (sqlite3.OperationalError, sqlite3.InterfaceError) as e:
                logger.warning(f"Query error in get_storage_by_conversation: {e}")
                return []

        results = []
        for row in rows:
            row_dict = dict(row)
            results.append(
                {
                    "chat_id": row_dict.get("chat_id"),
                    "display_name": row_dict.get("display_name"),
                    "attachment_count": row_dict.get("attachment_count", 0),
                    "total_size_bytes": row_dict.get("total_size", 0),
                }
            )

        return results

    def get_attachment_thumbnail_path(self, file_path: str) -> str | None:
        """Get the thumbnail path for an attachment if it exists.

        iMessage stores thumbnails for some attachments in a parallel directory.

        Args:
            file_path: The full path to the attachment

        Returns:
            Path to thumbnail if it exists, None otherwise
        """
        if not file_path:
            return None

        # iMessage thumbnails are stored in a similar path with _t suffix
        # e.g., ~/Library/Messages/Attachments/.../IMG_1234.jpg
        #   -> ~/Library/Messages/Attachments/.../IMG_1234_t.jpg
        path = Path(file_path)
        if not path.exists():
            return None

        # Try common thumbnail patterns
        stem = path.stem
        suffix = path.suffix

        # Pattern 1: _t suffix before extension
        thumb_path = path.parent / f"{stem}_t{suffix}"
        if thumb_path.exists():
            return str(thumb_path)

        # Pattern 2: .thumbnail suffix
        thumb_path = path.parent / f"{path.name}.thumbnail"
        if thumb_path.exists():
            return str(thumb_path)

        return None

    def get_contact_avatar(self, identifier: str) -> ContactAvatarData | None:
        """Get contact avatar data for a phone number or email.

        Queries the macOS AddressBook database for the contact's thumbnail
        image and name information.

        Args:
            identifier: Phone number (e.g., "+15551234567") or email address

        Returns:
            ContactAvatarData with image bytes and name info, or None if not found
        """
        return get_contact_avatar(identifier)
