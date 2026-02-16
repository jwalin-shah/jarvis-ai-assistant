"""Socket server service implementation."""

from __future__ import annotations

import logging
import socket
from pathlib import Path

from .base import Service, ServiceConfig

logger = logging.getLogger(__name__)


class SocketService(Service):
    """Socket server service (Unix socket + WebSocket)."""

    def __init__(
        self,
        socket_path: Path | None = None,
        websocket_port: int = 8743,
        venv_path: Path | None = None,
    ) -> None:
        if socket_path is None:
            socket_path = Path.home() / ".jarvis" / "jarvis.sock"

        config = ServiceConfig(
            name="socket",
            venv_path=venv_path,
            command=["python", "-m", "jarvis.interfaces.desktop"],
            health_check_socket=socket_path,
            startup_timeout=30.0,
            dependencies=[],  # Embedding service is optional
        )
        super().__init__(config)
        self.socket_path = socket_path
        self.websocket_port = websocket_port

    def _perform_health_check(self) -> bool:
        """Check if socket server is responding."""
        if not self.config.health_check_socket:
            return super()._perform_health_check()

        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.settimeout(self.config.health_check_timeout)
                sock.connect(str(self.config.health_check_socket))

                # Try a simple ping
                sock.send(b'{"jsonrpc": "2.0", "method": "ping", "id": 1}\n')
                response = sock.recv(1024)

                # Check if we got a response
                return len(response) > 0
            finally:
                sock.close()
        except (OSError, ConnectionError):
            logger.debug("Socket health check failed", exc_info=True)
            return False
