"""FastAPI service implementation."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from .base import Service, ServiceConfig

logger = logging.getLogger(__name__)


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
            parsed = urlparse(self.config.health_check_url)
            if parsed.scheme not in {"http", "https"}:
                logger.warning(
                    "Refusing non-http(s) health check URL: %s",
                    self.config.health_check_url,
                )
                return False

            req = urllib.request.Request(self.config.health_check_url)
            with urllib.request.urlopen(req, timeout=self.config.health_check_timeout) as response:  # nosec B310
                status = int(getattr(response, "status", 0))
                return status == 200
        except (OSError, ConnectionError):
            logger.debug("API health check failed", exc_info=True)
            return False
