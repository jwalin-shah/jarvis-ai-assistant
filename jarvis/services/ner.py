"""NER service implementation."""

from __future__ import annotations

import socket
from pathlib import Path

from .base import Service, ServiceConfig


class NERService(Service):
    """Named Entity Recognition service using spaCy."""

    def __init__(
        self,
        socket_path: Path | None = None,
        venv_path: Path | None = None,
        script_path: Path | None = None,
    ) -> None:
        if socket_path is None:
            socket_path = Path.home() / ".jarvis" / "jarvis-ner.sock"
        if venv_path is None:
            venv_path = Path.home() / ".jarvis" / "venvs" / "ner"
        if script_path is None:
            script_path = Path("scripts/ner_server.py")

        config = ServiceConfig(
            name="ner",
            venv_path=venv_path,
            command=["python", str(script_path)],
            health_check_socket=socket_path,
            startup_timeout=30.0,
            optional=True,
            dependencies=[],  # No dependencies
        )
        super().__init__(config)
        self.socket_path = socket_path

    def _perform_health_check(self) -> bool:
        """Check if NER service is responding."""
        if not self.config.health_check_socket:
            return super()._perform_health_check()

        sock = None
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.config.health_check_timeout)
            sock.connect(str(self.config.health_check_socket))

            # Send a length-prefixed test request
            payload = b'{"text": "test"}'
            length_bytes = len(payload).to_bytes(4, "big")
            sock.sendall(length_bytes + payload)

            # Read length-prefixed response
            response_length = sock.recv(4)
            if len(response_length) != 4:
                return False

            length = int.from_bytes(response_length, "big")
            response = sock.recv(length)

            # Check if we got a response
            return len(response) > 0
        except Exception:
            return False
        finally:
            if sock is not None:
                sock.close()
