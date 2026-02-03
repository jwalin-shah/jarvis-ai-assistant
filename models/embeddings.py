"""MLX Embedding Service Client - Low-level GPU-accelerated embeddings.

This is the LOW-LEVEL MLX embedding service client. It communicates with
a separate MLX microservice via Unix socket for GPU-accelerated embeddings.
Unix sockets provide ~10-50x lower latency than HTTP for local IPC.

IMPORTANT: Most code should NOT import from this module directly.
Instead, use the unified interface in jarvis/embedding_adapter.py:

    from jarvis.embedding_adapter import get_embedder
    embedder = get_embedder()  # Auto-selects MLX or CPU fallback

Architecture (3-layer embedding stack):
    1. jarvis/embeddings.py       - Embedding STORAGE (SQLite-backed message search)
    2. jarvis/embedding_adapter.py - UNIFIED INTERFACE (MLX-first with CPU fallback)
    3. models/embeddings.py       - MLX SERVICE CLIENT (this file, low-level)

This module is appropriate for:
- Direct MLX service health checks
- Starting/managing the MLX embedding service
- Low-level MLX-specific operations

Key Features:
- GPU acceleration on Apple Silicon via MLX
- Unix socket communication (JSON-RPC 2.0 protocol)
- Thread-safe singleton pattern
- Automatic service health checking

Service Setup:
    The MLX embedding service runs separately at ~/.jarvis/mlx-embed-service/
    Start it with: cd ~/.jarvis/mlx-embed-service && uv run python server.py
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from jarvis.errors import ErrorCode, JarvisError

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Embedding dimension is 384 for all supported models
MLX_EMBEDDING_DIM = 384
MLX_EMBED_SERVICE_SOCKET = "/tmp/jarvis-embed.sock"
MLX_EMBED_SERVICE_DIR = Path.home() / ".jarvis" / "mlx-embed-service"

# Legacy constant for backwards compatibility
DEFAULT_MLX_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def _get_mlx_model_name() -> str:
    """Get the MLX model name from config.

    Returns:
        MLX model name (e.g., "bge-small", "gte-tiny").
    """
    try:
        from jarvis.embedding_adapter import get_model_info

        _, mlx_model_name = get_model_info()
        return mlx_model_name
    except Exception:
        # Fallback to legacy model if config unavailable
        return "bge-small"


# =============================================================================
# Exceptions
# =============================================================================


class MLXEmbeddingError(JarvisError):
    """Raised when MLX embedding operations fail."""

    default_message = "MLX embedding operation failed"
    default_code = ErrorCode.MDL_LOAD_FAILED


class MLXServiceNotAvailableError(MLXEmbeddingError):
    """Raised when MLX embedding service is not running."""

    default_message = "MLX embedding service is not available"


class MLXModelLoadError(MLXEmbeddingError):
    """Raised when MLX embedding model fails to load."""

    default_message = "Failed to load MLX embedding model"


# For backwards compatibility
MLXModelNotAvailableError = MLXServiceNotAvailableError


# =============================================================================
# MLX Embedder (Service Client)
# =============================================================================


class MLXEmbedder:
    """Client for MLX embedding microservice via Unix socket.

    Communicates with a separate MLX embedding service using JSON-RPC 2.0
    over Unix sockets for low-latency GPU-accelerated embeddings.

    Thread Safety:
        This class is thread-safe for concurrent encode() calls.

    Example:
        >>> embedder = MLXEmbedder()
        >>> embeddings = embedder.encode(["Hello", "World"])
        >>> print(embeddings.shape)
        (2, 384)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MLX_EMBEDDING_MODEL,
        socket_path: str = MLX_EMBED_SERVICE_SOCKET,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the MLX embedder client.

        Args:
            model_name: Name/path of the embedding model to use.
            socket_path: Path to the Unix socket for the MLX embedding service.
            timeout: Request timeout in seconds.
        """
        self.model_name = model_name
        self.socket_path = socket_path
        self.timeout = timeout
        self._lock = threading.Lock()
        self._service_available: bool | None = None
        self._last_health_check: float = 0
        self._request_id = 0

    def _next_request_id(self) -> int:
        """Get next request ID for JSON-RPC."""
        self._request_id += 1
        return self._request_id

    def _send_request(
        self, method: str, params: dict | None = None, timeout: float | None = None
    ) -> dict:
        """Send a JSON-RPC request over Unix socket.

        Args:
            method: JSON-RPC method name.
            params: Optional parameters dict.
            timeout: Optional timeout override.

        Returns:
            Result dict from the response.

        Raises:
            MLXServiceNotAvailableError: If socket connection fails.
            MLXEmbeddingError: If the request fails.
        """
        import json
        import socket

        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self._next_request_id(),
        }

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout or self.timeout)

        try:
            sock.connect(self.socket_path)
            # Send newline-delimited JSON
            sock.sendall(json.dumps(request).encode() + b"\n")

            # Read response (newline-delimited)
            response_data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response_data += chunk
                if b"\n" in response_data:
                    break

            if not response_data:
                raise MLXServiceNotAvailableError("Empty response from MLX service")

            response = json.loads(response_data.decode().strip())

            if "error" in response:
                error = response["error"]
                raise MLXEmbeddingError(
                    f"MLX service error: {error.get('message', 'Unknown error')}"
                )

            return response.get("result", {})

        except FileNotFoundError:
            raise MLXServiceNotAvailableError(
                f"MLX embedding service socket not found at {self.socket_path}. "
                "Start it with: cd ~/.jarvis/mlx-embed-service && uv run python server.py"
            )
        except ConnectionRefusedError:
            raise MLXServiceNotAvailableError(
                "MLX embedding service is not running. "
                "Start it with: cd ~/.jarvis/mlx-embed-service && uv run python server.py"
            )
        except TimeoutError:
            raise MLXEmbeddingError("MLX service request timed out")
        finally:
            sock.close()

    def _check_service_health(self, force: bool = False) -> bool:
        """Check if the embedding service is healthy.

        Args:
            force: Force a fresh health check even if cached.

        Returns:
            True if service is healthy, False otherwise.
        """
        # Cache health check for 30 seconds
        now = time.time()
        if not force and self._service_available is not None:
            if now - self._last_health_check < 30:
                return self._service_available

        try:
            result = self._send_request("ping", timeout=5)
            if result.get("status") == "ok":
                self._service_available = True
                self._last_health_check = now
                return True
        except Exception as e:
            logger.debug("Health check failed: %s", e)

        self._service_available = False
        self._last_health_check = now
        return False

    def is_available(self) -> bool:
        """Check if the MLX embedding service is available.

        Returns:
            True if service is running and healthy.
        """
        return self._check_service_health()

    def encode(
        self,
        texts: list[str] | str,
        normalize: bool = True,
    ) -> NDArray[np.float32]:
        """Encode texts into embedding vectors.

        Args:
            texts: Single text string or list of texts to encode.
            normalize: If True (default), L2-normalize the embeddings.

        Returns:
            NumPy array of shape (n_texts, embedding_dim).

        Raises:
            MLXServiceNotAvailableError: If service is not running.
            MLXEmbeddingError: If encoding fails.

        Example:
            >>> embedder = MLXEmbedder()
            >>> vecs = embedder.encode(["Hello", "World"])
            >>> similarity = np.dot(vecs[0], vecs[1])  # Cosine similarity
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, MLX_EMBEDDING_DIM)

        # Check service availability
        if not self._check_service_health():
            raise MLXServiceNotAvailableError(
                "MLX embedding service is not running. "
                "Start it with: cd ~/.jarvis/mlx-embed-service && uv run python server.py"
            )

        try:
            result = self._send_request(
                "embed",
                {
                    "texts": texts,
                    "normalize": normalize,
                    "model": self.model_name,
                },
            )

            # Convert to numpy array
            embeddings = np.array(result["embeddings"], dtype=np.float32)
            return embeddings

        except MLXServiceNotAvailableError:
            raise
        except MLXEmbeddingError:
            raise
        except Exception as e:
            logger.exception("Failed to encode texts via MLX service")
            raise MLXEmbeddingError(
                f"Failed to encode texts: {e}",
                cause=e,
            ) from e

    def is_loaded(self) -> bool:
        """Check if the model is loaded in the service.

        Returns:
            True if model is loaded in the service.
        """
        if not self._check_service_health():
            return False

        try:
            result = self._send_request("health", timeout=5)
            return bool(result.get("model_loaded", False))
        except Exception:
            return False

    def unload(self) -> None:
        """Request the service to unload the model.

        Sends unload request to free GPU memory.
        """
        if not self._check_service_health():
            return

        try:
            self._send_request("unload", timeout=5)
            logger.info("Requested model unload from MLX service")
        except Exception as e:
            logger.warning("Failed to unload model: %s", e)

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension.

        Returns:
            The dimensionality of the embedding vectors.
        """
        return MLX_EMBEDDING_DIM


# =============================================================================
# Singleton Access
# =============================================================================

_mlx_embedder: MLXEmbedder | None = None
_mlx_embedder_lock = threading.Lock()


def get_mlx_embedder(model_name: str | None = None) -> MLXEmbedder:
    """Get or create the singleton MLX embedder instance.

    Args:
        model_name: Optional model name. If provided and different from
                   the current model, a new embedder will be created.
                   If None, uses the model from config.embedding.model_name.

    Returns:
        The shared MLXEmbedder instance.

    Example:
        >>> embedder = get_mlx_embedder()
        >>> embeddings = embedder.encode(["Hello, world!"])
    """
    global _mlx_embedder

    # Get model from config if not provided
    effective_model = model_name or _get_mlx_model_name()

    # Fast path: singleton exists and model matches
    if _mlx_embedder is not None:
        if effective_model == _mlx_embedder.model_name:
            return _mlx_embedder

    with _mlx_embedder_lock:
        # Double-check after acquiring lock
        if _mlx_embedder is not None:
            if effective_model == _mlx_embedder.model_name:
                return _mlx_embedder

        # Create new embedder
        _mlx_embedder = MLXEmbedder(effective_model)
        return _mlx_embedder


def reset_mlx_embedder() -> None:
    """Reset the singleton MLX embedder.

    Clears the singleton. A new instance will be created on
    the next get_mlx_embedder() call.
    """
    global _mlx_embedder

    with _mlx_embedder_lock:
        if _mlx_embedder is not None:
            _mlx_embedder.unload()
        _mlx_embedder = None


def is_mlx_available() -> bool:
    """Check if MLX embedding service is available.

    Returns:
        True if the MLX embedding service is running.
    """
    embedder = get_mlx_embedder()
    return embedder.is_available()


def start_mlx_service() -> subprocess.Popen[bytes] | None:
    """Start the MLX embedding service if not running.

    Returns:
        Popen object if service was started, None if already running
        or if service directory doesn't exist.
    """
    # Check if already running
    embedder = get_mlx_embedder()
    if embedder.is_available():
        logger.info("MLX embedding service already running")
        return None

    # Check if service directory exists
    if not MLX_EMBED_SERVICE_DIR.exists():
        logger.warning(
            "MLX embedding service not installed at %s",
            MLX_EMBED_SERVICE_DIR,
        )
        return None

    # Start the service
    logger.info("Starting MLX embedding service...")
    try:
        process = subprocess.Popen(
            ["uv", "run", "python", "server.py"],
            cwd=MLX_EMBED_SERVICE_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait a bit for startup
        time.sleep(2)

        # Check if it started successfully
        if embedder._check_service_health(force=True):
            logger.info("MLX embedding service started successfully")
            return process
        else:
            logger.warning("MLX embedding service failed to start")
            process.terminate()
            return None

    except Exception as e:
        logger.exception("Failed to start MLX embedding service: %s", e)
        return None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    "DEFAULT_MLX_EMBEDDING_MODEL",
    "MLX_EMBEDDING_DIM",
    "MLX_EMBED_SERVICE_SOCKET",
    # Exceptions
    "MLXEmbeddingError",
    "MLXServiceNotAvailableError",
    "MLXModelNotAvailableError",  # backwards compat
    "MLXModelLoadError",
    # Class
    "MLXEmbedder",
    # Singleton functions
    "get_mlx_embedder",
    "reset_mlx_embedder",
    "is_mlx_available",
    "start_mlx_service",
]
