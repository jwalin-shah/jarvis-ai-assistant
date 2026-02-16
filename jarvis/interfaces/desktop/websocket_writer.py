from __future__ import annotations

from websockets.asyncio.server import ServerConnection


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
