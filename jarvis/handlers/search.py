from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from jarvis.handlers.base import BaseHandler, rpc_handler

if TYPE_CHECKING:
    from jarvis.socket_server import JarvisSocketServer

logger = logging.getLogger(__name__)


class SearchHandler(BaseHandler):
    """Handler for search-related RPC methods."""

    def register(self) -> None:
        """Register search-related RPC methods."""
        self.server.register("semantic_search", self._semantic_search)

    @rpc_handler("Search failed")
    async def _semantic_search(
        self,
        query: str,
        limit: int = 20,
        threshold: float = 0.3,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Semantic search across messages."""
        from datetime import datetime
        from jarvis.search.vec_search import get_vec_searcher

        chat_id = filters.get("chat_id") if filters else None
        sender_filter = filters.get("sender") if filters else None

        searcher = get_vec_searcher()
        results = searcher.search(query, chat_id=chat_id, limit=limit)

        # Post-filter by sender if requested
        if sender_filter:
            results = [r for r in results if r.sender == sender_filter]

        # Filter by threshold
        results = [r for r in results if r.score >= threshold]

        return {
            "results": [
                {
                    "message": {
                        "id": r.rowid,
                        "chat_id": r.chat_id,
                        "text": r.text,
                        "sender": r.sender or "",
                        "date": (
                            datetime.fromtimestamp(r.timestamp).isoformat()
                            if r.timestamp
                            else ""
                        ),
                    },
                    "similarity": round(r.score, 4),
                }
                for r in results
            ],
            "total_results": len(results),
        }
