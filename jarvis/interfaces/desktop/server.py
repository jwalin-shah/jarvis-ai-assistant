from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import secrets
import time
from collections.abc import Callable, Coroutine, Sequence
from re import Pattern
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import parse_qs, urlparse

import websockets
from websockets.asyncio.server import Server, ServerConnection
from websockets.datastructures import Headers
from websockets.http11 import Request as WebSocketRequest
from websockets.http11 import Response as WebSocketResponse
from websockets.typing import Origin

from jarvis.config import get_config
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
from jarvis.interfaces.desktop.protocol import (
    WebSocketWriter,
    error_response,
    send_stream_response,
    send_stream_token,
    success_response,
)
from jarvis.observability.logging import log_event, timed_operation

if TYPE_CHECKING:
    from jarvis.prefetch import PrefetchManager
    from jarvis.watcher import ChatDBWatcher

logger = logging.getLogger(__name__)


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
    """JSON-RPC server over Unix socket and WebSocket.

    Handles requests from the desktop app by dispatching them to domain handlers.
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
        self._ws_server: Server | None = None
        self._methods: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        self._streaming_methods: set[str] = set()
        self._clients: set[asyncio.StreamWriter] = set()
        self._ws_clients: set[ServerConnection] = set()
        self._clients_lock = asyncio.Lock()
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

        self._prefetch_manager: PrefetchManager | None = None
        self._reply_service: Any | None = None
        self._imessage_access_cache: bool | None = None
        self._imessage_access_cache_time: float = 0.0
        self._rate_limiter = RateLimiter(max_requests=100, window_seconds=1.0)

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

    # Compatibility methods
    async def _ping(self) -> dict[str, Any]:
        return await self._health_handler._ping()

    async def _batch(self, requests: list[dict[str, Any]]) -> dict[str, Any]:
        return await self._batch_handler._batch(requests)

    async def _send_stream_token(
        self,
        writer: asyncio.StreamWriter | WebSocketWriter,
        token: str,
        token_index: int,
        is_final: bool = False,
        request_id: Any = None,
    ) -> None:
        await send_stream_token(writer, token, token_index, is_final, request_id)

    async def _send_stream_response(
        self,
        writer: asyncio.StreamWriter | WebSocketWriter,
        request_id: Any,
        result: dict[str, Any],
    ) -> None:
        await send_stream_response(writer, request_id, result)

    def register(
        self,
        name: str,
        handler: Callable[..., Coroutine[Any, Any, Any]],
        streaming: bool = False,
    ) -> None:
        self._methods[name] = handler
        if streaming:
            self._streaming_methods.add(name)

    @property
    def models_ready(self) -> bool:
        """Expose model readiness without leaking private fields to handlers."""
        return self._models_ready

    def get_prefetch_manager(self) -> PrefetchManager | None:
        """Return prefetch manager for handlers that need optional prefetch features."""
        return self._prefetch_manager

    def get_reply_service(self) -> Any:
        """Return shared reply service instance for handlers."""
        if self._reply_service is None:
            from jarvis.reply_service import get_reply_service

            self._reply_service = get_reply_service()
        return self._reply_service

    def pause_prefetch(self) -> None:
        manager = self.get_prefetch_manager()
        if manager is not None:
            manager.pause()

    def resume_prefetch(self) -> None:
        manager = self.get_prefetch_manager()
        if manager is not None:
            manager.resume()

    def get_rpc_handler(
        self,
        name: str,
    ) -> Callable[..., Coroutine[Any, Any, Any]] | None:
        return self._methods.get(name)

    async def send_stream_token(
        self,
        writer: asyncio.StreamWriter | WebSocketWriter,
        token: str,
        token_index: int,
        is_final: bool = False,
        request_id: Any = None,
    ) -> None:
        await send_stream_token(writer, token, token_index, is_final, request_id)

    async def send_stream_response(
        self,
        writer: asyncio.StreamWriter | WebSocketWriter,
        request_id: Any,
        result: dict[str, Any],
    ) -> None:
        await send_stream_response(writer, request_id, result)

    async def start(self) -> None:
        """Start the socket server."""
        SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()

        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(SOCKET_PATH),
        )
        os.chmod(SOCKET_PATH, 0o600)

        self._ws_auth_token = secrets.token_urlsafe(32)
        self._token_created_at = time.monotonic()
        try:
            fd = os.open(WS_TOKEN_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(self._ws_auth_token)
        except OSError as e:
            logger.error("Failed to write WebSocket auth token to %s: %s", WS_TOKEN_PATH, e)
            raise

        server_cfg = get_config().server
        origins = cast(Sequence[Origin | Pattern[str] | None], server_cfg.cors_origins)

        self._ws_server = await websockets.serve(
            self._handle_websocket_client,
            server_cfg.websocket_host,
            WEBSOCKET_PORT,
            max_size=MAX_MESSAGE_SIZE,
            origins=origins,
            process_request=self._process_websocket_request,
        )

        self._running = True
        log_event(
            logger,
            "server.start",
            socket_path=str(SOCKET_PATH),
            ws_host=server_cfg.websocket_host,
            ws_port=WEBSOCKET_PORT,
        )

        if self._enable_watcher:
            from jarvis.watcher import ChatDBWatcher

            self._watcher = ChatDBWatcher(self)
            self._watcher_task = asyncio.create_task(self._watcher.start())

        from jarvis.model_warmer import get_model_warmer

        get_model_warmer().start()

        if self._preload_models:
            self._preload_task = asyncio.create_task(self._preload_models_async())
            if self._wait_for_preload or self._enable_prefetch:
                await self.wait_for_models()

        if self._enable_prefetch:
            try:
                from jarvis.prefetch import get_prefetch_manager

                self._prefetch_manager = get_prefetch_manager()
                if self._preload_models:
                    self._prefetch_manager._warmup_on_start = False
                self._prefetch_manager.start()
            except Exception as e:
                logger.warning(f"Prefetch manager failed to start: {e}")

        async with self._server:
            await self._server.serve_forever()

    async def _preload_models_async(self) -> None:
        try:
            log_event(logger, "model.preload.start")
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
            logger.warning(f"Model preloading failed: {e}")
            self._models_ready_event.set()

    async def wait_for_models(self, timeout: float | None = None) -> bool:
        if self._models_ready:
            return True
        wait_timeout = timeout if timeout is not None else self._preload_timeout
        try:
            await asyncio.wait_for(self._models_ready_event.wait(), timeout=wait_timeout)
            return self._models_ready
        except TimeoutError:
            return False

    def _preload_llm(self) -> None:
        try:
            from models.loader import get_model

            model = get_model()
            if model and not model.is_loaded():
                model.load()
        except Exception as e:
            logger.debug(f"LLM preload skipped: {e}")

    def _preload_embeddings(self) -> None:
        try:
            from jarvis.embedding_adapter import get_embedder

            embedder = get_embedder()
            if embedder:
                embedder.encode(["test"])
        except Exception as e:
            logger.debug(f"Embeddings preload skipped: {e}")

    def _preload_cross_encoder(self) -> None:
        try:
            from models.cross_encoder import get_cross_encoder

            ce = get_cross_encoder()
            if ce and not ce.is_loaded:
                ce.load_model()
        except Exception as e:
            logger.debug(f"Cross-encoder preload skipped: {e}")

    def _preload_vec_index(self) -> None:
        try:
            from jarvis.search.vec_search import get_vec_searcher

            searcher = get_vec_searcher()
            if searcher:
                searcher.backfill_vec_binary()
        except Exception as e:
            logger.debug(f"Vec searcher preload skipped: {e}")

    def _preload_category_classifier(self) -> None:
        try:
            from jarvis.classifiers.category_classifier import get_classifier

            classifier = get_classifier()
            classifier._load_pipeline()
        except Exception as e:
            logger.debug(f"Category classifier preload skipped: {e}")

    async def stop(self) -> None:
        self._running = False
        from jarvis.model_warmer import get_model_warmer

        try:
            get_model_warmer().stop()
        except Exception as e:
            logger.debug(f"Failed to stop model warmer: {e}")

        if self._watcher:
            try:
                await self._watcher.stop()
            except Exception as e:
                logger.debug(f"Failed to stop watcher: {e}")
        if self._watcher_task:
            self._watcher_task.cancel()
            try:
                await self._watcher_task
            except asyncio.CancelledError:
                pass

        if self._prefetch_manager:
            try:
                self._prefetch_manager.stop()
            except Exception as e:
                logger.debug(f"Failed to stop prefetch manager: {e}")

        if self._preload_task:
            self._preload_task.cancel()
            try:
                await self._preload_task
            except asyncio.CancelledError:
                pass

        async with self._clients_lock:
            for writer in self._clients.copy():
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception as e:
                    logger.debug(f"Failed to close client connection: {e}")
            self._clients.clear()

            for ws in self._ws_clients.copy():
                try:
                    await ws.close()
                except Exception as e:
                    logger.debug(f"Failed to close websocket connection: {e}")
            self._ws_clients.clear()

        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if self._ws_server:
            self._ws_server.close()
            await self._ws_server.wait_closed()

        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()
        if WS_TOKEN_PATH.exists():
            WS_TOKEN_PATH.unlink()

    def _rotate_ws_token(self) -> None:
        self._previous_ws_auth_token = self._ws_auth_token
        self._previous_token_expired_at = time.monotonic() + 60.0
        self._ws_auth_token = secrets.token_urlsafe(32)
        self._token_created_at = time.monotonic()
        try:
            fd = os.open(WS_TOKEN_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(self._ws_auth_token)
        except OSError as e:
            logger.debug(f"Failed to rotate WebSocket token: {e}")

    def _verify_ws_token(self, client_token: str) -> bool:
        if self._ws_auth_token and hmac.compare_digest(client_token, self._ws_auth_token):
            return True
        if (
            self._previous_ws_auth_token
            and time.monotonic() < self._previous_token_expired_at
            and hmac.compare_digest(client_token, self._previous_ws_auth_token)
        ):
            return True
        return False

    @staticmethod
    def _http_error_response(status: int, reason: str, body: str) -> WebSocketResponse:
        payload = body.encode("utf-8")
        headers = Headers(
            [
                ("Content-Type", "text/plain; charset=utf-8"),
                ("Content-Length", str(len(payload))),
            ]
        )
        return WebSocketResponse(status, reason, headers, payload)

    async def _process_websocket_request(
        self,
        connection: ServerConnection,
        request: WebSocketRequest,
    ) -> WebSocketResponse | None:
        del connection  # Not needed; token is in request path query params.

        if self._ws_auth_token and (time.monotonic() - self._token_created_at) > 86400:
            self._rotate_ws_token()
        if not self._ws_auth_token:
            return None

        query_params = parse_qs(urlparse(request.path).query)
        client_token = query_params.get("token", [None])[0]
        if not client_token or not self._verify_ws_token(client_token):
            return self._http_error_response(401, "Unauthorized", "Unauthorized")
        return None

    async def broadcast(self, method: str, params: dict[str, Any]) -> None:
        notification = json.dumps({"jsonrpc": "2.0", "method": method, "params": params})
        notification_bytes = notification.encode() + b"\n"
        async with self._clients_lock:
            for writer in self._clients.copy():
                try:
                    writer.write(notification_bytes)
                    await writer.drain()
                except Exception as e:
                    logger.debug(f"Failed to write to client: {e}")
                    self._clients.discard(writer)
            for ws in self._ws_clients.copy():
                try:
                    await ws.send(notification)
                except Exception as e:
                    logger.debug(f"Failed to send to websocket: {e}")
                    self._ws_clients.discard(ws)

        if method == "new_message" and self._prefetch_manager:
            try:
                chat_id = params.get("chat_id")
                text = params.get("text", "")
                is_from_me = params.get("is_from_me", False)
                if chat_id:
                    self._prefetch_manager.on_message(chat_id, text, is_from_me)
            except Exception as e:
                logger.debug(f"Failed to notify prefetch manager: {e}")

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        async with self._clients_lock:
            self._clients.add(writer)
        peer = writer.get_extra_info("peername")
        try:
            while self._running:
                try:
                    line = await asyncio.wait_for(reader.readline(), timeout=300.0)
                    if not line:
                        break
                    if len(line) > MAX_MESSAGE_SIZE:
                        writer.write(
                            error_response(None, INVALID_REQUEST, "Message too large").encode()
                            + b"\n"
                        )
                        await writer.drain()
                    elif not self._rate_limiter.is_allowed(str(peer)):
                        writer.write(
                            error_response(None, INVALID_REQUEST, "Rate limit exceeded").encode()
                            + b"\n"
                        )
                        await writer.drain()
                    else:
                        asyncio.create_task(self._process_and_respond(line.decode(), writer, peer))
                except TimeoutError:
                    break
        except Exception as e:
            logger.debug(f"Client handler error: {e}")
        finally:
            async with self._clients_lock:
                self._clients.discard(writer)
            writer.close()

    async def _process_and_respond(
        self, message: str, writer: asyncio.StreamWriter | ServerConnection, peer: Any
    ) -> None:
        try:
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
        except Exception:
            logger.exception(f"Error processing message for {peer}")

    async def _handle_websocket_client(self, websocket: ServerConnection) -> None:
        async with self._clients_lock:
            if len(self._ws_clients) >= MAX_WS_CONNECTIONS:
                await websocket.close(4002, "Too many connections")
                return
            self._ws_clients.add(websocket)

        try:
            async for message in websocket:
                if not self._running:
                    break
                if len(message) > MAX_MESSAGE_SIZE:
                    await websocket.send(error_response(None, INVALID_REQUEST, "Message too large"))
                elif not self._rate_limiter.is_allowed(str(websocket.remote_address)):
                    await websocket.send(
                        error_response(None, INVALID_REQUEST, "Rate limit exceeded")
                    )
                else:
                    asyncio.create_task(
                        self._process_and_respond(str(message), websocket, websocket.remote_address)
                    )
        except Exception as e:
            logger.debug(f"Websocket handler error: {e}")
        finally:
            async with self._clients_lock:
                self._ws_clients.discard(websocket)

    async def _process_message(
        self, message: str, writer: asyncio.StreamWriter | WebSocketWriter | None = None
    ) -> str | None:
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            return error_response(None, PARSE_ERROR, f"Parse error: {e}")

        if not isinstance(data, dict):
            return error_response(None, INVALID_REQUEST, "Invalid request")

        request_id = data.get("id")
        method = data.get("method")
        params = data.get("params", {})

        if not method:
            return error_response(request_id, INVALID_REQUEST, "Missing method")

        handler = self._methods.get(method)
        if not handler:
            return error_response(request_id, METHOD_NOT_FOUND, f"Method not found: {method}")

        stream_requested = isinstance(params, dict) and params.get("stream", False)
        if stream_requested:
            params.pop("stream", None)
        supports_streaming = method in self._streaming_methods

        start_time = time.perf_counter()
        try:
            if stream_requested and supports_streaming and writer:
                if isinstance(params, dict):
                    await handler(_writer=writer, _request_id=request_id, **params)
                else:
                    await handler(_writer=writer, _request_id=request_id)
                _record_rpc_latency(method, (time.perf_counter() - start_time) * 1000)
                return None
            else:
                if isinstance(params, dict):
                    params.pop("_writer", None)
                    params.pop("_request_id", None)
                    result = await handler(**params)
                elif isinstance(params, list):
                    result = await handler(*params)
                else:
                    result = await handler()
                _record_rpc_latency(method, (time.perf_counter() - start_time) * 1000)
                return success_response(request_id, result)
        except asyncio.CancelledError:
            raise
        except JsonRpcError as e:
            return error_response(request_id, e.code, e.message, e.data)
        except TypeError as e:
            return error_response(request_id, INVALID_PARAMS, f"Invalid params: {e}")
        except Exception:
            logger.exception(f"Error handling {method}")
            return error_response(request_id, INTERNAL_ERROR, "Internal server error")
