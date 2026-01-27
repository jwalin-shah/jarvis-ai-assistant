"""Shared dependencies for API endpoints.

Provides singleton instances of iMessage reader and other shared resources.
"""

import threading

from fastapi import HTTPException

from integrations.imessage import ChatDBReader

# Singleton iMessage reader instance
_reader: ChatDBReader | None = None
_reader_lock = threading.Lock()


def get_imessage_reader() -> ChatDBReader:
    """Get or create the singleton iMessage reader.

    Uses double-check locking pattern to prevent race conditions in
    multi-threaded async environments.

    Returns:
        ChatDBReader instance

    Raises:
        HTTPException: 403 if Full Disk Access is not granted
    """
    global _reader

    # First check without lock (fast path)
    if _reader is None:
        # Acquire lock for initialization
        with _reader_lock:
            # Double-check after acquiring lock
            if _reader is None:
                _reader = ChatDBReader()

    if not _reader.check_access():
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

    return _reader


def reset_imessage_reader() -> None:
    """Reset the singleton reader (for testing or reconnection)."""
    global _reader
    if _reader is not None:
        _reader.close()
        _reader = None
