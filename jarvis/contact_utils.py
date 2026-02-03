"""Contact utilities - Shared functions for contact handling.

Provides common utilities used across relationship and embedding modules.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache


@lru_cache(maxsize=1000)
def hash_contact_id(contact_id: str) -> str:
    """Create a stable hash for contact ID storage.

    Uses LRU cache for frequently accessed contact IDs.

    Args:
        contact_id: Phone number, email, or chat_id.

    Returns:
        SHA-256 hash prefix (first 16 chars) for filename safety.
    """
    return hashlib.sha256(contact_id.encode("utf-8")).hexdigest()[:16]


__all__ = ["hash_contact_id"]
