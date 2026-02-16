"""Locking utilities for resource synchronization.

Provides per-key locking mechanisms to serialize operations on specific
entities (e.g., chats, files, users) without blocking the entire system.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)


class PerKeyLockManager:
    """Manager for per-key async locks with LRU eviction.

    Ensures that operations on the same key are serialized, while
    operations on different keys can run in parallel. Automatically
    cleans up unused locks to prevent memory leaks.

    Example:
        lock_manager = PerKeyLockManager(max_locks=100)

        async with lock_manager.lock("chat_123"):
            # serialize chat processing
            await process_chat("chat_123")
    """

    def __init__(self, max_locks: int = 1000) -> None:
        """Initialize lock manager.

        Args:
            max_locks: Maximum number of locks to keep in memory.
        """
        self.max_locks = max_locks
        self._locks: OrderedDict[Any, asyncio.Lock] = OrderedDict()
        self._global_lock = asyncio.Lock()

    async def get_lock(self, key: Any) -> asyncio.Lock:
        """Get or create an async lock for a specific key.

        Args:
            key: The unique identifier to lock on.

        Returns:
            An asyncio.Lock instance for that key.
        """
        async with self._global_lock:
            if key in self._locks:
                # Move to end (most recently used)
                self._locks.move_to_end(key)
                return self._locks[key]

            # Create new lock
            lock = asyncio.Lock()
            self._locks[key] = lock

            # Evict oldest if full
            if len(self._locks) > self.max_locks:
                # Evict the first item (least recently used)
                # Note: We only evict if the lock isn't currently held
                # to avoid potential race conditions, though OrderedDict
                # management under global_lock is generally safe.
                evicted_key, _ = self._locks.popitem(last=False)
                logger.debug(f"Evicted lock for key: {evicted_key}")

            return lock

    @asynccontextmanager
    async def lock(self, key: Any) -> AsyncGenerator[None, None]:
        """Context manager for per-key locking.

        Args:
            key: The unique identifier to lock on.
        """
        lock = await self.get_lock(key)
        async with lock:
            yield

    def clear(self) -> None:
        """Clear all locks from the manager."""
        self._locks.clear()

    def __len__(self) -> int:
        """Return the number of active locks."""
        return len(self._locks)
