"""MLX embedding service - in-process implementation.

Embeddings now run in-process via models.bert_embedder, so no external
service process is needed. This service class is kept for API compatibility
with the ServiceManager but all process management is a no-op.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .base import Service, ServiceConfig, ServiceStatus

logger = logging.getLogger(__name__)


class EmbeddingService(Service):
    """MLX embedding service (in-process, no external process needed)."""

    def __init__(
        self,
        port: int = 8766,
        socket_path: Path | None = None,
        venv_path: Path | None = None,
        service_dir: Path | None = None,
    ) -> None:
        if socket_path is None:
            socket_path = Path.home() / ".jarvis" / "jarvis-embed.sock"
        if service_dir is None:
            service_dir = Path.home() / ".jarvis" / "venvs" / "embedding"
        if venv_path is None:
            venv_path = service_dir

        config = ServiceConfig(
            name="embedding",
            venv_path=venv_path,
            command=["python", "server.py"],
            working_dir=service_dir,
            health_check_socket=socket_path,
            startup_timeout=60.0,
            optional=True,
            dependencies=[],
        )
        super().__init__(config)
        self.port = port
        self.socket_path = socket_path
        self._can_run_without_process = True

    def _start_process(self) -> None:
        """No-op: embeddings run in-process via models.bert_embedder."""
        self._status = ServiceStatus.HEALTHY

    def _perform_health_check(self) -> bool:
        """Check if in-process MLX embedder is available."""
        try:
            from models.embeddings import is_mlx_available

            return is_mlx_available()
        except Exception:
            return False
