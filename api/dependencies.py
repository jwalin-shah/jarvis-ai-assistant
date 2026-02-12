"""Shared dependencies for API endpoints.

Provides singleton instances of iMessage reader and other shared resources.
Uses a connection pool for thread-safe database access.
"""

from collections.abc import Iterator

from fastapi import HTTPException

from integrations.imessage import ChatDBReader, reset_connection_pool
from jarvis.cache import TTLCache

# Cache access check result to avoid per-request DB queries.
# TTL of 60s balances freshness with performance.
_access_cache = TTLCache(ttl_seconds=60.0, maxsize=1)


def get_imessage_reader() -> Iterator[ChatDBReader]:
    """Get a thread-safe iMessage reader for the current request.

    Uses a connection pool to acquire a fresh connection for each request.
    The connection is automatically released back to the pool when the
    request is finished.

    Yields:
        ChatDBReader instance

    Raises:
        HTTPException: 403 if Full Disk Access is not granted
    """
    reader = ChatDBReader()

    found, has_access = _access_cache.get("imessage_access")
    if not found:
        has_access = reader.check_access()
        if has_access:
            _access_cache.set("imessage_access", True)
        else:
            _access_cache.invalidate("imessage_access")

    if not has_access:
        reader.close()
        raise HTTPException(
            status_code=403,
            detail=(
                "Full Disk Access is required to read iMessages. "
                "Open System Settings > Privacy & Security > Full Disk Access, "
                "add and enable your terminal application, then restart the JARVIS API server."
            ),
        )

    try:
        yield reader
    finally:
        reader.close()


def reset_imessage_reader() -> None:
    """Reset the connection pool (for testing or reconnection)."""
    reset_connection_pool()
    _access_cache.invalidate()
