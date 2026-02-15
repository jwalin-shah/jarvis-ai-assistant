from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from jarvis.handlers.base import BaseHandler, rpc_handler

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class PrefetchHandler(BaseHandler):
    """Handler for prefetch-related RPC methods."""

    def register(self) -> None:
        """Register prefetch-related RPC methods."""
        self.server.register("prefetch_stats", self._prefetch_stats)
        self.server.register("prefetch_invalidate", self._prefetch_invalidate)
        self.server.register("prefetch_focus", self._prefetch_focus)
        self.server.register("prefetch_hover", self._prefetch_hover)

    @rpc_handler("Failed to get prefetch stats")
    async def _prefetch_stats(self) -> dict[str, Any]:
        """Get prefetch manager statistics."""
        if not self.server._prefetch_manager:
            return {"enabled": False}
        return self.server._prefetch_manager.get_stats()

    @rpc_handler("Failed to invalidate prefetch")
    async def _prefetch_invalidate(self, chat_id: str | None = None) -> dict[str, bool]:
        """Invalidate prefetch cache."""
        if not self.server._prefetch_manager:
            return {"success": False}
        self.server._prefetch_manager.invalidate(chat_id)
        return {"success": True}

    @rpc_handler("Failed to record focus")
    async def _prefetch_focus(self, chat_id: str) -> dict[str, bool]:
        """Record UI focus event for prefetching."""
        if not self.server._prefetch_manager:
            return {"success": False}
        self.server._prefetch_manager.record_event("focus", chat_id)
        return {"success": True}

    @rpc_handler("Failed to record hover")
    async def _prefetch_hover(self, chat_id: str) -> dict[str, bool]:
        """Record UI hover event for prefetching."""
        if not self.server._prefetch_manager:
            return {"success": False}
        self.server._prefetch_manager.record_event("hover", chat_id)
        return {"success": True}
