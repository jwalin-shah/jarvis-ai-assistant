"""JARVIS Socket Server package.

JSON-RPC server over Unix socket AND WebSocket for the desktop app.
Provides LLM generation, search, and classification with streaming support.

Submodules:
    protocol  - JSON-RPC protocol helpers and constants
    handlers  - RPC method handler implementations
    server    - JarvisSocketServer class and main entry point
"""

from jarvis.socket_server.protocol import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    MAX_MESSAGE_SIZE,
    MAX_WS_CONNECTIONS,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    SOCKET_PATH,
    WEBSOCKET_HOST,
    WEBSOCKET_PORT,
    WS_TOKEN_PATH,
    JsonRpcError,
    WebSocketWriter,
)
from jarvis.socket_server.server import JarvisSocketServer, main

__all__ = [
    # Server
    "JarvisSocketServer",
    "main",
    # Protocol
    "JsonRpcError",
    "WebSocketWriter",
    # Constants
    "SOCKET_PATH",
    "WS_TOKEN_PATH",
    "WEBSOCKET_HOST",
    "WEBSOCKET_PORT",
    "MAX_MESSAGE_SIZE",
    "MAX_WS_CONNECTIONS",
    "PARSE_ERROR",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "INVALID_PARAMS",
    "INTERNAL_ERROR",
]
