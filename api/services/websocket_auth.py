"""WebSocket authentication helpers."""

from __future__ import annotations

import logging
import secrets

from fastapi import WebSocket

logger = logging.getLogger(__name__)


def load_or_generate_token(env_token: str | None) -> str:
    """Return configured auth token or generate a session token."""
    if env_token:
        return env_token
    token = secrets.token_urlsafe(32)
    logger.info("Generated WebSocket auth token (set JARVIS_WS_TOKEN to persist)")
    return token


def validate_websocket_auth(websocket: WebSocket, expected_token: str) -> bool:
    """Validate WebSocket authentication token.

    Accepted token locations:
    1. Header: X-WS-Token
    2. Header: Sec-WebSocket-Protocol
    3. Query param: token (deprecated; warning logged)
    """
    token_header = websocket.headers.get("x-ws-token")
    if token_header and secrets.compare_digest(token_header, expected_token):
        return True

    ws_protocol = websocket.headers.get("sec-websocket-protocol")
    if ws_protocol:
        protocols = [p.strip() for p in ws_protocol.split(",")]
        for protocol in protocols:
            if secrets.compare_digest(protocol, expected_token):
                return True

    token = websocket.query_params.get("token")
    if token and secrets.compare_digest(token, expected_token):
        logger.warning(
            "WebSocket auth via query parameter is DEPRECATED and will be removed. "
            "Use X-WS-Token header or Sec-WebSocket-Protocol instead. "
            "Query params are logged by proxies/servers. Client: %s",
            websocket.client.host if websocket.client else "unknown",
        )
        return True

    return False
