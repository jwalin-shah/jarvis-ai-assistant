"""FastAPI service implementation."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from .base import Service, ServiceConfig


class FastAPIService(Service):
    """FastAPI backend service."""

    def __init__(
        self,
        port: int = 8742,
        host: str = "127.0.0.1",
        venv_path: Path | None = None,
    ) -> None:
        config = ServiceConfig(
            name="api",
            venv_path=venv_path,
            command=["uvicorn", "api.main:app", "--host", host, "--port", str(port)],
            health_check_url=f"http://{host}:{port}/health",
            startup_timeout=30.0,
            dependencies=[],  # No dependencies
        )
        super().__init__(config)
        self.port = port
        self.host = host

    def _perform_health_check(self) -> bool:
        """Check if FastAPI is responding."""
        if not self.config.health_check_url:
            return super()._perform_health_check()

        try:
            req = urllib.request.Request(self.config.health_check_url)
            with urllib.request.urlopen(req, timeout=self.config.health_check_timeout) as response:
                status = int(getattr(response, "status", 0))
                return status == 200
        except Exception:
            return False
