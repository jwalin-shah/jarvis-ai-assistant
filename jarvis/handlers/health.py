from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from jarvis.handlers.base import BaseHandler, rpc_handler

if TYPE_CHECKING:
    pass


class HealthHandler(BaseHandler):
    """Handler for health-related RPC methods."""

    def register(self) -> None:
        """Register health-related RPC methods."""
        self.server.register("ping", self._ping)

    @rpc_handler("Ping failed")
    async def _ping(self) -> dict[str, Any]:
        """Health check endpoint.

        Returns:
            Dict with status, timestamp, and model readiness.
        """
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "models_ready": self.server.models_ready,
        }
