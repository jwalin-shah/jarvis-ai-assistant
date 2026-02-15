from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from jarvis.handlers.base import BaseHandler, rpc_handler

if TYPE_CHECKING:
    from jarvis.socket_server import JarvisSocketServer

logger = logging.getLogger(__name__)


class MetricsHandler(BaseHandler):
    """Handler for metrics-related RPC methods."""

    def register(self) -> None:
        """Register metrics-related RPC methods."""
        self.server.register("get_routing_metrics", self._get_routing_metrics)
        self.server.register("get_performance_slo", self._get_performance_slo)
        self.server.register("get_draft_metrics", self._get_draft_metrics)

    @rpc_handler("Failed to get routing metrics")
    async def _get_routing_metrics(self) -> dict[str, Any]:
        """Get model routing statistics."""
        from jarvis.router import get_reply_router
        return get_reply_router().get_metrics()

    @rpc_handler("Failed to get performance SLOs")
    async def _get_performance_slo(self) -> dict[str, Any]:
        """Get performance SLO compliance data."""
        from jarvis.utils.latency_tracker import get_tracker
        return get_tracker().get_report()

    @rpc_handler("Failed to get draft metrics")
    async def _get_draft_metrics(self) -> dict[str, Any]:
        """Get draft generation and gating metrics."""
        from jarvis.metrics import get_draft_metrics
        return get_draft_metrics().get_report()
