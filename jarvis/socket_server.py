"""JARVIS Socket Server Facade.

Note: This module is now a facade for the decomposed socket server in jarvis.interfaces.desktop.
New code should import from jarvis.interfaces.desktop directly.
"""

from __future__ import annotations

from jarvis.handlers.base import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    JsonRpcError,
)
from jarvis.interfaces.desktop.constants import (
    MAX_MESSAGE_SIZE,
    MAX_WS_CONNECTIONS,
    SOCKET_PATH,
    WEBSOCKET_PORT,
    WS_TOKEN_PATH,
)
from jarvis.interfaces.desktop.limiter import RateLimiter
from jarvis.interfaces.desktop.protocol import WebSocketWriter
from jarvis.interfaces.desktop.server import JarvisSocketServer

__all__ = [
    "JarvisSocketServer",
    "WebSocketWriter",
    "RateLimiter",
    "JsonRpcError",
    "INTERNAL_ERROR",
    "INVALID_PARAMS",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "PARSE_ERROR",
    "SOCKET_PATH",
    "WS_TOKEN_PATH",
    "WEBSOCKET_PORT",
    "MAX_MESSAGE_SIZE",
    "MAX_WS_CONNECTIONS",
]
