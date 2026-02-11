"""JSON-RPC protocol helpers and constants.

Contains WebSocketWriter adapter, JsonRpcError exception,
standard error codes, and response formatting functions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from websockets.server import ServerConnection

# Socket configuration
SOCKET_PATH = Path.home() / ".jarvis" / "jarvis.sock"
WS_TOKEN_PATH = Path.home() / ".jarvis" / "ws_token"
WEBSOCKET_HOST = "127.0.0.1"
WEBSOCKET_PORT = 8743
MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB max message size
MAX_WS_CONNECTIONS = 10


class WebSocketWriter:
    """Wrapper to make WebSocket connection compatible with StreamWriter interface.

    This allows the same streaming code to work for both Unix sockets and WebSockets.
    """

    def __init__(self, websocket: ServerConnection) -> None:
        self._websocket = websocket
        self._buffer = ""

    def write(self, data: bytes) -> None:
        """Buffer data to send."""
        self._buffer += data.decode("utf-8", errors="replace")

    async def drain(self) -> None:
        """Send buffered data over WebSocket."""
        if self._buffer:
            # WebSocket doesn't need newline delimiters, but we keep them for consistency
            await self._websocket.send(self._buffer.rstrip("\n"))
            self._buffer = ""


class JsonRpcError(Exception):
    """JSON-RPC error with code and data."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


# Standard JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def success_response(request_id: Any, result: Any) -> str:
    """Build a JSON-RPC success response."""
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "result": result,
            "id": request_id,
        }
    )


def error_response(
    request_id: Any,
    code: int,
    message: str,
    data: Any = None,
) -> str:
    """Build a JSON-RPC error response."""
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data

    return json.dumps(
        {
            "jsonrpc": "2.0",
            "error": error,
            "id": request_id,
        }
    )
