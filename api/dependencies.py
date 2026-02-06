"""Shared dependencies for API endpoints.

Provides singleton instances of iMessage reader and other shared resources.
Uses a connection pool for thread-safe database access.
"""

from collections.abc import Iterator

from fastapi import HTTPException

from integrations.imessage import ChatDBReader, reset_connection_pool


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

    if not reader.check_access():
        reader.close()
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Permission denied",
                "message": "Full Disk Access is required to read iMessages.",
                "instructions": [
                    "Open System Settings",
                    "Go to Privacy & Security > Full Disk Access",
                    "Add and enable your terminal application",
                    "Restart the JARVIS API server",
                ],
            },
        )

    try:
        yield reader
    finally:
        reader.close()


def reset_imessage_reader() -> None:
    """Reset the connection pool (for testing or reconnection)."""
    # Reset the underlying connection pool
    reset_connection_pool()
