"""Adaptive Memory Layer using Mem0.

Provides long-term, self-improving user memory that persists across
conversations and sessions.
"""

from __future__ import annotations

import logging
import threading

from mem0 import Memory

__all__ = ["JARVISMemory", "get_memory"]

logger = logging.getLogger(__name__)


class JARVISMemory:
    """Memory layer wrapper for JARVIS using Mem0."""

    def __init__(self, user_id: str = "default_user") -> None:
        self.user_id = user_id
        # Configure Mem0 for local operation
        # Note: By default it uses local storage if no API key provided
        self.memory = Memory()

    def add_interaction(self, user_msg: str, assistant_msg: str) -> bool:
        """Record an interaction to learn from it.

        Returns:
            True if successful, False if failed.
        """
        try:
            self.memory.add(
                f"User: {user_msg}\nAssistant: {assistant_msg}",
                user_id=self.user_id,
            )
            return True
        except Exception:
            logger.exception("Failed to add interaction to memory")
            return False

    def get_relevant_facts(self, query: str) -> list[str]:
        """Retrieve learned facts relevant to the current query."""
        try:
            results = self.memory.search(query, user_id=self.user_id)
            return [res["fact"] for res in results]
        except Exception:
            logger.exception("Failed to retrieve facts from memory")
            return []

    def delete_all(self) -> None:
        """Clear the memory for the user."""
        self.memory.delete_all(user_id=self.user_id)


# Global memory instance
_memory: JARVISMemory | None = None
_memory_lock = threading.Lock()


def get_memory() -> JARVISMemory:
    """Get or create singleton memory instance."""
    global _memory
    if _memory is None:
        with _memory_lock:
            if _memory is None:
                _memory = JARVISMemory()
    return _memory
