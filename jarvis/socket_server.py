"""JARVIS Socket Server.

JSON-RPC server over Unix socket AND WebSocket for the desktop app.
Provides LLM generation, search, and classification with streaming support.

Unix Socket: ~/.jarvis/jarvis.sock (for Tauri app)
WebSocket:   ws://localhost:8743 (for browser/Playwright)

Protocol: JSON-RPC 2.0 over newline-delimited JSON
Request:  {"jsonrpc": "2.0", "method": "...", "params": {...}, "id": 1}
Response: {"jsonrpc": "2.0", "result": {...}, "id": 1}
Error:    {"jsonrpc": "2.0", "error": {"code": -32600, "message": "..."}, "id": 1}

Features speculative prefetching for near-instant responses:
- Predicts what user needs next based on patterns
- Multi-tier caching (L1 memory, L2 SQLite, L3 disk)
- Background prefetch execution with resource awareness
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import secrets
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

import websockets
from websockets.server import ServerConnection

from jarvis.config import get_config
from jarvis.observability.logging import log_event, timed_operation

if TYPE_CHECKING:
    from jarvis.prefetch import PrefetchManager
    from jarvis.watcher import ChatDBWatcher

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter — O(1) per request.

    Each client gets a bucket that refills at ``refill_rate`` tokens/sec up to
    ``max_tokens``.  Every request consumes one token; requests are rejected
    when the bucket is empty.
    """

    def __init__(self, max_requests: int = 100, window_seconds: float = 1.0) -> None:
        # max_requests tokens refill over window_seconds → steady-state rate
        self._max_tokens = float(max_requests)
        self._refill_rate = max_requests / window_seconds  # tokens per second
        # Per-client state: (tokens_remaining, last_refill_time)
        self._buckets: dict[str, list[float]] = {}
        import time as _time

        self._time = _time

    def is_allowed(self, client_id: str) -> bool:
        """Check if a request from client_id is allowed.

        Returns True if under rate limit, False if exceeded.  O(1) per call.
        """
        now = self._time.monotonic()

        bucket = self._buckets.get(client_id)
        if bucket is None:
            # New client: full bucket minus this request
            self._buckets[client_id] = [self._max_tokens - 1.0, now]
            return True

        # Refill tokens based on elapsed time
        elapsed = now - bucket[1]
        tokens = min(self._max_tokens, bucket[0] + elapsed * self._refill_rate)

        if tokens < 1.0:
            # Update timestamp even on rejection so next refill is accurate
            bucket[0] = tokens
            bucket[1] = now
            return False

        bucket[0] = tokens - 1.0
        bucket[1] = now
        return True


# Socket configuration
SOCKET_PATH = Path.home() / ".jarvis" / "jarvis.sock"
WS_TOKEN_PATH = Path.home() / ".jarvis" / "ws_token"
WEBSOCKET_PORT = 8743
MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB max message size
MAX_WS_CONNECTIONS = 10


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


from jarvis.handlers.base import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    JsonRpcError,
)

# Confidence gating: drafts below this threshold are not shown to the user
# Lowered for 0.7B model - small model needs lower threshold to show drafts
DRAFT_CONFIDENCE_THRESHOLD = 0.25


def _record_rpc_latency(method: str, elapsed_ms: float) -> None:
    """Record RPC call latency to the global latency tracker."""
    from jarvis.utils.latency_tracker import OPERATION_BUDGETS, LatencyRecord, get_tracker

    op = f"rpc.{method}"
    budget = OPERATION_BUDGETS.get(op)
    threshold = budget[1] if budget and budget[1] > 0 else None
    exceeded = threshold is not None and elapsed_ms > threshold
    get_tracker()._records.append(
        LatencyRecord(
            operation=op,
            elapsed_ms=elapsed_ms,
            timestamp=time.time(),
            threshold_ms=threshold,
            exceeded=exceeded,
        )
    )
    if exceeded:
        logging.getLogger(__name__).warning(
            f"[RPC Budget] {method} took {elapsed_ms:.1f}ms (budget: {threshold}ms)"
        )


class JarvisSocketServer:
    """JSON-RPC server over Unix socket.

    Handles requests from the Tauri desktop app for:
    - LLM generation (draft replies, summaries) with real token streaming
    - Semantic search
    - Intent classification
    - Smart reply suggestions

    Also runs a file watcher for real-time new message notifications.
    Preloads models at startup for faster first request.

    Streaming Protocol:
        For streaming methods (e.g., generate_draft with stream=true):
        1. Client sends request with "stream": true in params
        2. Server sends token notifications:
        3. Server sends final response: {"jsonrpc": "2.0", "result": {...}, "id": <request_id>}

    Example:
        server = JarvisSocketServer()
        await server.start()
    """

    def __init__(
        self,
        enable_watcher: bool = True,
        preload_models: bool = True,
        wait_for_preload: bool = False,
        preload_timeout: float = 30.0,
        enable_prefetch: bool = True,
    ) -> None:
        self._server: asyncio.Server | None = None
        self._ws_server: websockets.WebSocketServer | None = None
        self._methods: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        self._streaming_methods: set[str] = set()  # Methods that support streaming
        self._clients: set[asyncio.StreamWriter] = set()  # Unix socket clients
        self._ws_clients: set[ServerConnection] = set()  # WebSocket clients
        self._clients_lock = asyncio.Lock()  # Protect client set mutations
        self._ws_auth_token: str | None = None
        self._token_created_at: float = 0.0
        self._previous_ws_auth_token: str | None = None
        self._previous_token_expired_at: float = 0.0
        self._running = False
        self._enable_watcher = enable_watcher
        self._preload_models = preload_models
        self._wait_for_preload = wait_for_preload
        self._preload_timeout = preload_timeout
        self._enable_prefetch = enable_prefetch
        self._watcher: ChatDBWatcher | None = None
        self._watcher_task: asyncio.Task[None] | None = None
        self._preload_task: asyncio.Task[None] | None = None
        self._models_ready = False
        self._models_ready_event = asyncio.Event()

        # Prefetch manager for speculative caching
        self._prefetch_manager: PrefetchManager | None = None

        # Cached iMessage access check (30s TTL)
        self._imessage_access_cache: bool | None = None
        self._imessage_access_cache_time: float = 0.0

        # Rate limiter: 100 req/s per client
        self._rate_limiter = RateLimiter(max_requests=100, window_seconds=1.0)

        # Register built-in methods
        self._register_methods()

    def _register_methods(self) -> None:
        """Register available RPC methods using domain-specific handlers."""
        from jarvis.handlers.batch import BatchHandler
        from jarvis.handlers.contact import ContactHandler
        from jarvis.handlers.health import HealthHandler
        from jarvis.handlers.message import MessageHandler
        from jarvis.handlers.metrics import MetricsHandler
        from jarvis.handlers.prefetch import PrefetchHandler
        from jarvis.handlers.search import SearchHandler

        self._health_handler = HealthHandler(self)
        self._message_handler = MessageHandler(self)
        self._search_handler = SearchHandler(self)
        self._batch_handler = BatchHandler(self)
        self._contact_handler = ContactHandler(self)
        self._metrics_handler = MetricsHandler(self)
        self._prefetch_handler = PrefetchHandler(self)

        self._health_handler.register()
        self._message_handler.register()
        self._search_handler.register()
        self._batch_handler.register()
        self._contact_handler.register()
        self._metrics_handler.register()
        self._prefetch_handler.register()

    # Compatibility methods for tests
    async def _ping(self) -> dict[str, Any]:
        return await self._health_handler._ping()

    async def _batch(self, requests: list[dict[str, Any]]) -> dict[str, Any]:
        return await self._batch_handler._batch(requests)

    def register(
        self,
        name: str,
        handler: Callable[..., Coroutine[Any, Any, Any]],
        streaming: bool = False,
    ) -> None:
        """Register an RPC method.

        Args:
            name: Method name
            handler: Async handler function
            streaming: Whether this method supports streaming responses
        """
        self._methods[name] = handler
        if streaming:
            self._streaming_methods.add(name)

    async def start(self) -> None:
        """Start the socket server (Unix + WebSocket).

        Creates ~/.jarvis/jarvis.sock and ws://localhost:8743 for connections.
        """
        # Ensure socket directory exists with user-only permissions
        SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        # Remove existing socket file
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()

        # Start Unix socket server
        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(SOCKET_PATH),
        )

        # Set socket permissions (readable/writable by user only)
        os.chmod(SOCKET_PATH, 0o600)

        # Generate WebSocket auth token and write to file
        self._ws_auth_token = secrets.token_urlsafe(32)
        self._token_created_at = time.monotonic()
        try:
            fd = os.open(WS_TOKEN_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(self._ws_auth_token)
        except OSError as e:
            logger.error("Failed to write WebSocket auth token to %s: %s", WS_TOKEN_PATH, e)
            raise

        # Start WebSocket server for browser clients
        server_cfg = get_config().server
        self._ws_server = await websockets.serve(
            self._handle_websocket_client,
            server_cfg.websocket_host,
            WEBSOCKET_PORT,
            max_size=MAX_MESSAGE_SIZE,
            origins=server_cfg.cors_origins,
        )

        self._running = True
        log_event(
            logger,
            "server.start",
            socket_path=str(SOCKET_PATH),
            ws_host=server_cfg.websocket_host,
            ws_port=WEBSOCKET_PORT,
        )

        # Start the file watcher for real-time new message detection
        if self._enable_watcher:
            from jarvis.watcher import ChatDBWatcher

            self._watcher = ChatDBWatcher(self)
            self._watcher_task = asyncio.create_task(self._watcher.start())
            logger.info("Started chat.db watcher for real-time notifications")

        # Start model warmer to manage LLM lifecycle and free memory when idle
        from jarvis.model_warmer import get_model_warmer

        get_model_warmer().start()

        # Preload models in background for faster first request
        # NOTE: Must complete BEFORE prefetch manager starts. Concurrent MLX
        # model loads crash the Metal GPU (assertion failures / malloc errors).
        if self._preload_models:
            self._preload_task = asyncio.create_task(self._preload_models_async())

            # Wait for preload: always required when prefetch is enabled (its
            # prediction loop immediately routes messages that trigger model
            # loading, racing with the preload). Also wait if user requested.
            if self._wait_for_preload or self._enable_prefetch:
                logger.info(f"Waiting up to {self._preload_timeout}s for models to preload...")
                ready = await self.wait_for_models()
                if ready:
                    logger.info("Models ready, accepting connections")
                else:
                    logger.warning("Preload timeout, accepting connections anyway")

        # Start prefetch manager for speculative caching
        # Models are already loaded by preload above, so skip redundant warmup.
        if self._enable_prefetch:
            try:
                from jarvis.prefetch import get_prefetch_manager

                self._prefetch_manager = get_prefetch_manager()
                if self._preload_models:
                    self._prefetch_manager._warmup_on_start = False
                self._prefetch_manager.start()
                logger.info("Started prefetch manager for speculative caching")
            except Exception as e:
                logger.warning(f"Prefetch manager failed to start: {e}")
                self._prefetch_manager = None

        # Run both servers
        async with self._server:
            await self._server.serve_forever()

    async def _preload_models_async(self) -> None:
        """Preload LLM, embeddings, classifiers, and vec search index in background.

        This runs at startup so the first user request doesn't have to wait
        for model loading (which can take several seconds).

        Loads in parallel for faster startup.
        """
        try:
            log_event(logger, "model.preload.start")

            # Load models sequentially to avoid memory spike on 8GB systems
            for loader in [
                self._preload_llm,
                self._preload_embeddings,
                self._preload_cross_encoder,
                self._preload_vec_index,
                self._preload_category_classifier,
            ]:
                try:
                    with timed_operation(logger, f"model.preload.{loader.__name__}"):
                        await asyncio.to_thread(loader)
                except Exception as e:
                    logger.warning(f"Preload failed for {loader.__name__}: {e}")

            self._models_ready = True
            self._models_ready_event.set()
            log_event(logger, "model.preload.complete")

        except Exception as e:
            logger.warning(f"Model preloading failed (will load on demand): {e}")
            # Still set the event so waiters don't hang forever
            self._models_ready_event.set()

    async def wait_for_models(self, timeout: float | None = None) -> bool:
        """Wait for models to finish preloading.

        Args:
            timeout: Maximum seconds to wait (None = use default preload_timeout)

        Returns:
            True if models are ready, False if timeout reached
        """
        if self._models_ready:
            return True

        wait_timeout = timeout if timeout is not None else self._preload_timeout
        try:
            await asyncio.wait_for(self._models_ready_event.wait(), timeout=wait_timeout)
            return self._models_ready
        except TimeoutError:
            logger.warning(f"Model preload wait timed out after {wait_timeout}s")
            return False

    def _preload_llm(self) -> None:
        """Preload the LLM model (sync).

        Uses the shared singleton model loader (from get_model()) which is
        now also used by the generator. This ensures the model is loaded
        exactly once into memory.
        """
        try:
            from models.loader import get_model

            # Load the singleton model loader
            model = get_model()
            if model and not model.is_loaded():
                model.load()
                logger.debug(f"LLM model preloaded: {model.config.display_name}")

            # Note: get_generator() now shares the same loader instance,
            # so no separate preload is needed for the generator.

        except (ImportError, OSError, RuntimeError) as e:
            logger.debug(f"LLM preload skipped: {e}")

    def _preload_embeddings(self) -> None:
        """Preload the embeddings model (sync)."""
        try:
            from jarvis.embedding_adapter import get_embedder

            embedder = get_embedder()
            if embedder:
                # Warm up with a test embedding
                embedder.encode(["test"])
                logger.debug("Embeddings model preloaded")
        except (ImportError, OSError, RuntimeError) as e:
            logger.debug(f"Embeddings preload skipped: {e}")

    def _preload_cross_encoder(self) -> None:
        """Preload the cross-encoder reranker model (sync)."""
        try:
            from models.cross_encoder import get_cross_encoder

            ce = get_cross_encoder()
            if ce and not ce.is_loaded:
                ce.load_model()
                logger.debug("Cross-encoder model preloaded")
        except (ImportError, OSError, RuntimeError) as e:
            logger.debug(f"Cross-encoder preload skipped: {e}")

    def _preload_vec_index(self) -> None:
        """Preload vec searcher and backfill vec_binary if needed (sync)."""
        try:
            from jarvis.search.vec_search import get_vec_searcher

            searcher = get_vec_searcher()
            if searcher:
                # Backfill vec_binary from existing vec_chunks (no-op if already populated)
                searcher.backfill_vec_binary()
                logger.debug("Vec searcher preloaded")

        except (ImportError, OSError, RuntimeError) as e:
            logger.debug(f"Vec searcher preload skipped: {e}")

    def _preload_category_classifier(self) -> None:
        """Preload the category classifier pipeline (sync).

        Triggers joblib.load() of the LightGBM model so the first classify()
        call doesn't incur a 5-15s cold-start penalty.
        """
        try:
            from jarvis.classifiers.category_classifier import get_classifier

            classifier = get_classifier()
            classifier._load_pipeline()
            logger.debug("Category classifier preloaded")
        except (ImportError, OSError, RuntimeError) as e:
            logger.debug(f"Category classifier preload skipped: {e}")

    async def stop(self) -> None:
        """Stop the socket server."""
        self._running = False

        # Stop model warmer
        from jarvis.model_warmer import get_model_warmer

        try:
            get_model_warmer().stop()
        except Exception as e:
            logger.debug(f"Error stopping warmer: {e}")

        # Stop the watcher
        if self._watcher:
            try:
                await self._watcher.stop()
            except Exception as e:
                logger.debug(f"Error stopping watcher: {e}")
            self._watcher = None
        if self._watcher_task:
            self._watcher_task.cancel()
            try:
                await self._watcher_task
            except asyncio.CancelledError:
                pass
            self._watcher_task = None

        # Stop prefetch manager
        if self._prefetch_manager:
            try:
                self._prefetch_manager.stop()
            except Exception as e:
                logger.debug(f"Error stopping prefetch manager: {e}")
            self._prefetch_manager = None

        # Cancel preload if still running
        if self._preload_task:
            self._preload_task.cancel()
            try:
                await self._preload_task
            except asyncio.CancelledError:
                pass
            self._preload_task = None

        # Close all Unix socket client connections
        async with self._clients_lock:
            for writer in self._clients.copy():
                try:
                    writer.close()
                    await writer.wait_closed()
                except OSError as e:
                    logger.debug(f"Failed to close Unix socket connection: {e}")
            self._clients.clear()

            # Close all WebSocket client connections
            for ws in self._ws_clients.copy():
                try:
                    await ws.close()
                except OSError as e:
                    logger.debug(f"Failed to close WebSocket connection: {e}")
            self._ws_clients.clear()

        # Stop Unix socket server
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        # Stop WebSocket server
        if self._ws_server:
            self._ws_server.close()
            await self._ws_server.wait_closed()
            self._ws_server = None

        # Remove socket file
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()

        # Remove WebSocket auth token file
        if WS_TOKEN_PATH.exists():
            WS_TOKEN_PATH.unlink()

        logger.info("Socket server stopped")

    def _rotate_ws_token(self) -> None:
        """Rotate the WebSocket auth token.

        Generates a new token and writes it to the token file. The previous
        token remains valid for a 60-second grace period so existing
        connections can re-authenticate without being immediately killed.
        """
        # Keep the old token valid for a grace period
        self._previous_ws_auth_token = self._ws_auth_token
        self._previous_token_expired_at = time.monotonic() + 60.0

        # Generate and persist the new token
        self._ws_auth_token = secrets.token_urlsafe(32)
        self._token_created_at = time.monotonic()
        try:
            fd = os.open(WS_TOKEN_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(self._ws_auth_token)
        except OSError as e:
            logger.warning("Failed to write rotated token to %s: %s", WS_TOKEN_PATH, e)

        log_event(logger, "websocket.token_rotated")

    def _verify_ws_token(self, client_token: str) -> bool:
        """Verify a client-provided WebSocket auth token.

        Accepts the current token, or the previous token if still within
        the 60-second grace period after rotation.

        Args:
            client_token: Token from the client query params.

        Returns:
            True if the token is valid.
        """
        # Check current token
        if self._ws_auth_token and hmac.compare_digest(client_token, self._ws_auth_token):
            return True

        # Check previous token within grace period
        if (
            self._previous_ws_auth_token
            and time.monotonic() < self._previous_token_expired_at
            and hmac.compare_digest(client_token, self._previous_ws_auth_token)
        ):
            return True

        return False

    async def broadcast(self, method: str, params: dict[str, Any]) -> None:
        """Broadcast a notification to all connected clients (Unix + WebSocket).

        Args:
            method: Notification method name
            params: Notification parameters
        """
        notification = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            }
        )
        # Pre-encode once for all Unix socket clients
        notification_bytes = notification.encode() + b"\n"

        # Broadcast to Unix socket clients
        async with self._clients_lock:
            for writer in self._clients.copy():
                try:
                    writer.write(notification_bytes)
                    await writer.drain()
                except ConnectionError as e:
                    logger.debug(f"Failed to broadcast to Unix socket client: {e}")
                    self._clients.discard(writer)

            # Broadcast to WebSocket clients
            for ws in self._ws_clients.copy():
                try:
                    await ws.send(notification)
                except ConnectionError as e:
                    logger.debug(f"Failed to broadcast to WebSocket client: {e}")
                    self._ws_clients.discard(ws)

        # Hook into prefetch system for new message events
        if method == "new_message" and self._prefetch_manager:
            try:
                chat_id = params.get("chat_id")
                text = params.get("text", "")
                is_from_me = params.get("is_from_me", False)
                if chat_id:
                    self._prefetch_manager.on_message(chat_id, text, is_from_me)
            except Exception as e:
                logger.debug(f"Prefetch notification failed: {e}")

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a client connection.

        Args:
            reader: Stream reader
            writer: Stream writer
        """
        async with self._clients_lock:
            self._clients.add(writer)
        peer = writer.get_extra_info("peername")
        logger.debug(f"Client connected: {peer}")

        try:
            while self._running:
                try:
                    # Read line (newline-delimited JSON)
                    line = await asyncio.wait_for(
                        reader.readline(),
                        timeout=300.0,  # 5 minute timeout
                    )

                    if not line:
                        break  # Client disconnected

                    if len(line) > MAX_MESSAGE_SIZE:
                        response = self._error_response(None, INVALID_REQUEST, "Message too large")
                        writer.write(response.encode() + b"\n")
                        await writer.drain()
                    elif not self._rate_limiter.is_allowed(str(peer)):
                        response = self._error_response(
                            None, INVALID_REQUEST, "Rate limit exceeded"
                        )
                        writer.write(response.encode() + b"\n")
                        await writer.drain()
                    else:
                        # Multiplexing: process message in a separate task so we can
                        # continue reading from the socket immediately.
                        asyncio.create_task(self._process_and_respond(line.decode(), writer, peer))

                except TimeoutError:
                    break  # Client idle timeout

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"Client error: {e}")

        finally:
            async with self._clients_lock:
                self._clients.discard(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except OSError as e:
                logger.debug(f"Failed to close client writer: {e}")
            logger.debug(f"Client disconnected: {peer}")

    async def _process_and_respond(
        self,
        message: str,
        writer: asyncio.StreamWriter | ServerConnection,
        peer: Any,
    ) -> None:
        """Process a message and write the response to the client.

        This is designed to be run as an independent task for multiplexing.
        """
        try:
            # Wrap WebSockets in WebSocketWriter for streaming compatibility
            actual_writer: Any = writer
            if isinstance(writer, ServerConnection):
                actual_writer = WebSocketWriter(writer)

            response = await self._process_message(message, actual_writer)
            if response:
                if isinstance(writer, ServerConnection):
                    await writer.send(response.rstrip("\n"))
                else:
                    writer.write(response.encode() + b"\n")
                    await writer.drain()
        except Exception as e:
            logger.exception(f"Error in background message processing for {peer}: {e}")

    async def _handle_websocket_client(self, websocket: ServerConnection) -> None:
        """Handle a WebSocket client connection.

        Validates auth token from query params and enforces connection limits.

        Args:
            websocket: WebSocket connection
        """
        # Rotate token if older than 24 hours
        token_max_age_secs = 24 * 3600
        if self._ws_auth_token and (time.monotonic() - self._token_created_at) > token_max_age_secs:
            self._rotate_ws_token()

        # Validate auth token from query params
        if self._ws_auth_token:
            try:
                path = websocket.request.path if websocket.request else ""
                query_params = parse_qs(urlparse(path).query)
                client_token = query_params.get("token", [None])[0]
                if not client_token or not self._verify_ws_token(client_token):
                    log_event(
                        logger,
                        "websocket.auth_failed",
                        level=logging.WARNING,
                        remote=str(websocket.remote_address),
                    )
                    await websocket.close(4001, "Unauthorized")
                    return
            except (AttributeError, IndexError, KeyError, TypeError):
                await websocket.close(4001, "Unauthorized")
                return

        # Enforce connection limit (check inside lock to avoid TOCTOU race)
        async with self._clients_lock:
            if len(self._ws_clients) >= MAX_WS_CONNECTIONS:
                log_event(
                    logger,
                    "websocket.capacity.rejected",
                    level=logging.WARNING,
                    max_connections=MAX_WS_CONNECTIONS,
                )
                await websocket.close(4002, "Too many connections")
                return
            self._ws_clients.add(websocket)
        log_event(
            logger,
            "websocket.connect",
            remote=str(websocket.remote_address),
            active_connections=len(self._ws_clients),
        )

        try:
            async for message in websocket:
                if not self._running:
                    break

                if len(message) > MAX_MESSAGE_SIZE:
                    response = self._error_response(None, INVALID_REQUEST, "Message too large")
                    await websocket.send(response)
                elif not self._rate_limiter.is_allowed(str(websocket.remote_address)):
                    response = self._error_response(None, INVALID_REQUEST, "Rate limit exceeded")
                    await websocket.send(response)
                else:
                    # Multiplexing: process message in a separate task
                    asyncio.create_task(
                        self._process_and_respond(str(message), websocket, websocket.remote_address)
                    )

        except asyncio.CancelledError:
            raise
        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            logger.warning(f"WebSocket client error: {e}")
        finally:
            async with self._clients_lock:
                self._ws_clients.discard(websocket)
            logger.debug(f"WebSocket client disconnected: {websocket.remote_address}")

    async def _process_message(
        self, message: str, writer: asyncio.StreamWriter | WebSocketWriter | None = None
    ) -> str | None:
        """Process a JSON-RPC message.

        Args:
            message: Raw JSON message
            writer: Stream writer for streaming responses

        Returns:
            JSON response string, or None if response was streamed
        """
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            return self._error_response(None, PARSE_ERROR, f"Parse error: {e}")

        # Validate request structure
        if not isinstance(data, dict):
            return self._error_response(None, INVALID_REQUEST, "Invalid request")

        request_id = data.get("id")
        method = data.get("method")
        params = data.get("params", {})

        if not method:
            return self._error_response(request_id, INVALID_REQUEST, "Missing method")

        # Look up handler
        handler = self._methods.get(method)
        if not handler:
            log_event(logger, "rpc.method_not_found", level=logging.WARNING, method=method)
            return self._error_response(request_id, METHOD_NOT_FOUND, f"Method not found: {method}")

        # Check if streaming is requested and supported
        stream_requested = isinstance(params, dict) and params.get("stream", False)
        if stream_requested:
            # Remove stream key in-place (no copy needed, dict is ours)
            params.pop("stream", None)
        supports_streaming = method in self._streaming_methods

        # Call handler
        import time as _time

        _rpc_start = _time.perf_counter()
        try:
            if stream_requested and supports_streaming and writer:
                # Streaming mode: pass writer and request_id to handler
                if isinstance(params, dict):
                    result = await handler(_writer=writer, _request_id=request_id, **params)
                else:
                    result = await handler(_writer=writer, _request_id=request_id)
                _rpc_ms = (_time.perf_counter() - _rpc_start) * 1000
                _record_rpc_latency(method, _rpc_ms)
                log_event(
                    logger,
                    "rpc.complete",
                    method=method,
                    streaming=True,
                    latency_ms=round(_rpc_ms, 1),
                )
                # For streaming, the handler sends the final response
                return None
            else:
                # Normal mode
                if isinstance(params, dict):
                    # Remove streaming params that non-streaming handlers don't expect
                    params.pop("_writer", None)
                    params.pop("_request_id", None)
                    result = await handler(**params)
                elif isinstance(params, list):
                    result = await handler(*params)
                else:
                    result = await handler()

                _rpc_ms = (_time.perf_counter() - _rpc_start) * 1000
                _record_rpc_latency(method, _rpc_ms)
                log_event(
                    logger,
                    "rpc.complete",
                    method=method,
                    streaming=False,
                    latency_ms=round(_rpc_ms, 1),
                )
                return self._success_response(request_id, result)

        except asyncio.CancelledError:
            raise
        except JsonRpcError as e:
            _rpc_ms = (_time.perf_counter() - _rpc_start) * 1000
            log_event(
                logger,
                "rpc.error",
                level=logging.WARNING,
                method=method,
                error_code=e.code,
                error_message=e.message,
                latency_ms=round(_rpc_ms, 1),
            )
            return self._error_response(request_id, e.code, e.message, e.data)

        except TypeError as e:
            return self._error_response(request_id, INVALID_PARAMS, f"Invalid params: {e}")

        except Exception:
            _rpc_ms = (_time.perf_counter() - _rpc_start) * 1000
            log_event(
                logger,
                "rpc.error",
                level=logging.ERROR,
                method=method,
                error_code=INTERNAL_ERROR,
                latency_ms=round(_rpc_ms, 1),
            )
            logger.exception(f"Error handling {method}")
            return self._error_response(request_id, INTERNAL_ERROR, "Internal server error")

    def _success_response(self, request_id: Any, result: Any) -> str:
        """Build a success response."""
        return json.dumps(
            {
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id,
            }
        )

    def _error_response(
        self,
        request_id: Any,
        code: int,
        message: str,
        data: Any = None,
    ) -> str:
        """Build an error response."""
        error: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = data

        return json.dumps(
            {
                "jsonrpc": "2.0",
                "error": error,
                "id": request_id,
            }
        )

    async def _send_stream_token(
        self,
        writer: asyncio.StreamWriter,
        token: str,
        token_index: int,
        is_final: bool = False,
        request_id: Any = None,
    ) -> None:
        """Send a streaming token notification to a client.

        Args:
            writer: Client stream writer
            token: The token text
            token_index: Index of this token in the stream
            is_final: Whether this is the last token
            request_id: Request ID for correlating tokens with requests
        """
        notification = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "stream.token",
                "params": {
                    "token": token,
                    "index": token_index,
                    "final": is_final,
                    "request_id": request_id,
                },
            }
        )
        writer.write(notification.encode() + b"\n")
        await writer.drain()

    async def _send_stream_response(
        self,
        writer: asyncio.StreamWriter,
        request_id: Any,
        result: dict[str, Any],
    ) -> None:
        """Send the final response after streaming completes.

        Args:
            writer: Client stream writer
            request_id: Original request ID
            result: Final result data
        """
        response = json.dumps(
            {
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id,
            }
        )
        writer.write(response.encode() + b"\n")
        await writer.drain()
