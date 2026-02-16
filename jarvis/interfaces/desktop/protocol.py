from __future__ import annotations

import asyncio
import json
from typing import Any

from websockets.server import ServerConnection


class WebSocketWriter:
    """Wrapper to make WebSocket connection compatible with StreamWriter interface.

    This allows the same streaming code to work for both Unix sockets and WebSockets.
    """

    def __init__(self, websocket: ServerConnection) -> None:
        self._websocket = websocket
        self._parts: list[bytes] = []

    def write(self, data: bytes) -> None:
        """Buffer data to send (O(1) append instead of string concat)."""
        self._parts.append(data)

    async def drain(self) -> None:
        """Send buffered data over WebSocket."""
        if self._parts:
            # Join once, decode once (avoids O(n^2) string concat per token)
            combined = b"".join(self._parts).decode("utf-8", errors="replace")
            await self._websocket.send(combined.rstrip("\n"))
            self._parts.clear()


def success_response(request_id: Any, result: Any) -> str:
    """Build a success response."""
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
    """Build an error response."""
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


async def send_stream_token(
    writer: asyncio.StreamWriter | WebSocketWriter,
    token: str,
    token_index: int,
    is_final: bool = False,
    request_id: Any = None,
) -> None:
    """Send a streaming token notification to a client.

    Args:
        writer: Client stream writer
        token: The token text
        token_index: Index of this token in the stream
        is_final: Whether this is the last token
        request_id: Request ID for correlating tokens with requests
    """
    notification = json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "stream.token",
            "params": {
                "token": token,
                "index": token_index,
                "final": is_final,
                "request_id": request_id,
            },
        }
    )
    if isinstance(writer, WebSocketWriter):
        writer.write(notification.encode())
        await writer.drain()
    else:
        writer.write(notification.encode() + b"\n")
        await writer.drain()


async def send_stream_response(
    writer: asyncio.StreamWriter | WebSocketWriter,
    request_id: Any,
    result: dict[str, Any],
) -> None:
    """Send the final response after streaming completes.

    Args:
        writer: Client stream writer
        request_id: Original request ID
        result: Final result data
    """
    response = success_response(request_id, result)
    if isinstance(writer, WebSocketWriter):
        writer.write(response.encode())
        await writer.drain()
    else:
        writer.write(response.encode() + b"\n")
        await writer.drain()
