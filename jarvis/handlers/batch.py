from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from jarvis.handlers.base import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    BaseHandler,
    JsonRpcError,
    rpc_handler,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BatchHandler(BaseHandler):
    """Handler for batch RPC requests."""

    def register(self) -> None:
        """Register batch RPC methods."""
        self.server.register("batch", self._batch)

    @rpc_handler("Batch execution failed")
    async def _batch(
        self,
        requests: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute multiple RPC calls in a single request."""
        if not requests:
            return {"results": []}

        if len(requests) > 50:
            raise JsonRpcError(INVALID_PARAMS, "Maximum 50 requests per batch")

        async def execute_single(req: dict[str, Any]) -> dict[str, Any]:
            """Execute a single request from the batch."""
            req_id = req.get("id")
            method_name = req.get("method")
            params = req.get("params", {})

            if not method_name:
                return {
                    "id": req_id,
                    "error": {"code": INVALID_REQUEST, "message": "Missing method"},
                }

            # Get the handler from the server
            handler = self.server._methods.get(method_name)
            if not handler:
                return {
                    "id": req_id,
                    "error": {
                        "code": METHOD_NOT_FOUND,
                        "message": f"Method not found: {method_name}",
                    },
                }

            try:
                # Call the handler
                result = await handler(**params)
                return {"id": req_id, "result": result}
            except JsonRpcError as e:
                return {
                    "id": req_id,
                    "error": {"code": e.code, "message": e.message, "data": e.data},
                }
            except Exception as e:
                logger.exception("Error in batch element %s", method_name)
                return {
                    "id": req_id,
                    "error": {"code": -32603, "message": str(e)},
                }

        import asyncio

        results = await asyncio.gather(*[execute_single(r) for r in requests])
        return {"results": list(results)}
