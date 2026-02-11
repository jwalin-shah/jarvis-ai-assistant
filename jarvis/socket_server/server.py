"""JARVIS Socket Server.

JSON-RPC server over Unix socket AND WebSocket for the desktop app.
Provides LLM generation, search, and classification with streaming support.

Unix Socket: ~/.jarvis/jarvis.sock (for Tauri app)
WebSocket:   ws://localhost:8743 (for browser/Playwright)

Protocol: JSON-RPC 2.0 over newline-delimited JSON
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import secrets
import signal
from collections.abc import Callable, Coroutine
from typing import Any
from urllib.parse import parse_qs, urlparse

import websockets
from websockets.server import ServerConnection

from jarvis.observability.logging import log_event, timed_operation
from jarvis.socket_server import handlers
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
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)


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
        2. Server sends token notifications
        3. Server sends final response

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
        self._streaming_methods: set[str] = set()
        self._clients: set[asyncio.StreamWriter] = set()
        self._ws_clients: set[ServerConnection] = set()
        self._clients_lock = asyncio.Lock()
        self._ws_auth_token: str | None = None
        self._running = False
        self._enable_watcher = enable_watcher
        self._preload_models = preload_models
        self._wait_for_preload = wait_for_preload
        self._preload_timeout = preload_timeout
        self._enable_prefetch = enable_prefetch
        self._watcher: Any = None
        self._watcher_task: asyncio.Task[None] | None = None
        self._preload_task: asyncio.Task[None] | None = None
        self._models_ready = False
        self._models_ready_event = asyncio.Event()

        # Prefetch manager for speculative caching
        self._prefetch_manager: Any = None

        # Register built-in methods
        self._register_methods()

    def _register_methods(self) -> None:
        """Register available RPC methods."""
        # Health check
        self.register("ping", self._ping)

        # LLM methods (with streaming support)
        self.register("generate_draft", self._generate_draft, streaming=True)
        self.register("summarize", self._summarize, streaming=True)
        self.register("get_smart_replies", self._get_smart_replies)

        # Search methods
        self.register("semantic_search", self._semantic_search)

        # Batch operations
        self.register("batch", self._batch)

        # Contact resolution
        self.register("resolve_contacts", self._resolve_contacts)

        # Conversation methods
        self.register("list_conversations", self._list_conversations)

        # Metrics
        self.register("get_routing_metrics", self._get_routing_metrics)

        # Prefetch/cache operations
        self.register("prefetch_stats", self._prefetch_stats)
        self.register("prefetch_invalidate", self._prefetch_invalidate)
        self.register("prefetch_focus", self._prefetch_focus)
        self.register("prefetch_hover", self._prefetch_hover)

    def register(
        self,
        name: str,
        handler: Callable[..., Coroutine[Any, Any, Any]],
        streaming: bool = False,
    ) -> None:
        """Register an RPC method."""
        self._methods[name] = handler
        if streaming:
            self._streaming_methods.add(name)

    async def start(self) -> None:
        """Start the socket server (Unix + WebSocket)."""
        SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()

        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(SOCKET_PATH),
        )

        os.chmod(SOCKET_PATH, 0o600)

        self._ws_auth_token = secrets.token_urlsafe(32)
        WS_TOKEN_PATH.write_text(self._ws_auth_token)
        os.chmod(WS_TOKEN_PATH, 0o600)

        self._ws_server = await websockets.serve(
            self._handle_websocket_client,
            WEBSOCKET_HOST,
            WEBSOCKET_PORT,
            max_size=MAX_MESSAGE_SIZE,
            origins=["tauri://localhost", "http://localhost", "http://127.0.0.1"],
        )

        self._running = True
        log_event(
            logger,
            "server.start",
            socket_path=str(SOCKET_PATH),
            ws_host=WEBSOCKET_HOST,
            ws_port=WEBSOCKET_PORT,
        )

        if self._enable_watcher:
            from jarvis.watcher import ChatDBWatcher

            self._watcher = ChatDBWatcher(self)
            self._watcher_task = asyncio.create_task(self._watcher.start())
            logger.info("Started chat.db watcher for real-time notifications")

        if self._preload_models:
            self._preload_task = asyncio.create_task(self._preload_models_async())

            if self._wait_for_preload or self._enable_prefetch:
                logger.info(f"Waiting up to {self._preload_timeout}s for models to preload...")
                ready = await self.wait_for_models()
                if ready:
                    logger.info("Models ready, accepting connections")
                else:
                    logger.warning("Preload timeout, accepting connections anyway")

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

        async with self._server:
            await self._server.serve_forever()

    async def _preload_models_async(self) -> None:
        """Preload LLM, embeddings, classifiers, and vec search index in background."""
        try:
            log_event(logger, "model.preload.start")

            for loader in [
                self._preload_llm,
                self._preload_embeddings,
                self._preload_cross_encoder,
                self._preload_vec_index,
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
            self._models_ready_event.set()

    async def wait_for_models(self, timeout: float | None = None) -> bool:
        """Wait for models to finish preloading."""
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
        """Preload the LLM model (sync)."""
        try:
            from models import get_generator
            from models.loader import get_model

            model = get_model()
            if model and not model.is_loaded():
                model.load()
                logger.debug(f"LLM model preloaded: {model.config.display_name}")

            generator = get_generator()
            if generator._loader and not generator._loader.is_loaded():
                generator._loader.load()
                logger.debug("Generator LLM loader preloaded")

        except Exception as e:
            logger.debug(f"LLM preload skipped: {e}")

    def _preload_embeddings(self) -> None:
        """Preload the embeddings model (sync)."""
        try:
            from jarvis.embedding_adapter import get_embedder

            embedder = get_embedder()
            if embedder:
                embedder.encode(["test"])
                logger.debug("Embeddings model preloaded")
        except Exception as e:
            logger.debug(f"Embeddings preload skipped: {e}")

    def _preload_cross_encoder(self) -> None:
        """Preload the cross-encoder reranker model (sync)."""
        try:
            from models.cross_encoder import get_cross_encoder

            ce = get_cross_encoder()
            if ce and not ce.is_loaded:
                ce.load_model()
                logger.debug("Cross-encoder model preloaded")
        except Exception as e:
            logger.debug(f"Cross-encoder preload skipped: {e}")

    def _preload_vec_index(self) -> None:
        """Preload vec searcher and backfill vec_binary if needed (sync)."""
        try:
            from jarvis.search.vec_search import get_vec_searcher

            searcher = get_vec_searcher()
            if searcher:
                searcher.backfill_vec_binary()
                logger.debug("Vec searcher preloaded")

        except Exception as e:
            logger.debug(f"Vec searcher preload skipped: {e}")

    async def stop(self) -> None:
        """Stop the socket server."""
        self._running = False

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

        if self._prefetch_manager:
            try:
                self._prefetch_manager.stop()
            except Exception as e:
                logger.debug(f"Error stopping prefetch manager: {e}")
            self._prefetch_manager = None

        if self._preload_task:
            self._preload_task.cancel()
            try:
                await self._preload_task
            except asyncio.CancelledError:
                pass
            self._preload_task = None

        async with self._clients_lock:
            for writer in self._clients.copy():
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass
            self._clients.clear()

            for ws in self._ws_clients.copy():
                try:
                    await ws.close()
                except Exception:
                    pass
            self._ws_clients.clear()

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        if self._ws_server:
            self._ws_server.close()
            await self._ws_server.wait_closed()
            self._ws_server = None

        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()

        if WS_TOKEN_PATH.exists():
            WS_TOKEN_PATH.unlink()

        logger.info("Socket server stopped")

    async def broadcast(self, method: str, params: dict[str, Any]) -> None:
        """Broadcast a notification to all connected clients (Unix + WebSocket)."""
        notification = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            }
        )

        async with self._clients_lock:
            for writer in self._clients.copy():
                try:
                    writer.write(notification.encode() + b"\n")
                    await writer.drain()
                except Exception:
                    self._clients.discard(writer)

            for ws in self._ws_clients.copy():
                try:
                    await ws.send(notification)
                except Exception:
                    self._ws_clients.discard(ws)

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
        """Handle a client connection."""
        async with self._clients_lock:
            self._clients.add(writer)
        peer = writer.get_extra_info("peername")
        logger.debug(f"Client connected: {peer}")

        try:
            while self._running:
                try:
                    line = await asyncio.wait_for(
                        reader.readline(),
                        timeout=300.0,
                    )

                    if not line:
                        break

                    if len(line) > MAX_MESSAGE_SIZE:
                        response = error_response(None, INVALID_REQUEST, "Message too large")
                        writer.write(response.encode() + b"\n")
                        await writer.drain()
                    else:
                        response = await self._process_message(line.decode(), writer)
                        if response:
                            writer.write(response.encode() + b"\n")
                            await writer.drain()

                except TimeoutError:
                    break

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
            except Exception:
                pass
            logger.debug(f"Client disconnected: {peer}")

    async def _handle_websocket_client(self, websocket: ServerConnection) -> None:
        """Handle a WebSocket client connection."""
        if self._ws_auth_token:
            try:
                path = websocket.request.path if websocket.request else ""
                query_params = parse_qs(urlparse(path).query)
                client_token = query_params.get("token", [None])[0]
                if client_token != self._ws_auth_token:
                    log_event(
                        logger,
                        "websocket.auth_failed",
                        level=logging.WARNING,
                        remote=str(websocket.remote_address),
                    )
                    await websocket.close(4001, "Unauthorized")
                    return
            except Exception:
                await websocket.close(4001, "Unauthorized")
                return

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
                    response = error_response(None, INVALID_REQUEST, "Message too large")
                    await websocket.send(response)
                else:
                    ws_writer = WebSocketWriter(websocket)
                    response = await self._process_message(str(message), ws_writer)
                    if response:
                        await websocket.send(response)

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
        """Process a JSON-RPC message."""
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
            log_event(logger, "rpc.method_not_found", level=logging.WARNING, method=method)
            return error_response(request_id, METHOD_NOT_FOUND, f"Method not found: {method}")

        if isinstance(params, dict):
            params = copy.deepcopy(params)
        stream_requested = isinstance(params, dict) and params.pop("stream", False)
        supports_streaming = method in self._streaming_methods

        import time as _time

        _rpc_start = _time.perf_counter()
        try:
            if stream_requested and supports_streaming and writer:
                if isinstance(params, dict):
                    result = await handler(_writer=writer, _request_id=request_id, **params)
                else:
                    result = await handler(_writer=writer, _request_id=request_id)
                _rpc_ms = (_time.perf_counter() - _rpc_start) * 1000
                log_event(
                    logger,
                    "rpc.complete",
                    method=method,
                    streaming=True,
                    latency_ms=round(_rpc_ms, 1),
                )
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

                _rpc_ms = (_time.perf_counter() - _rpc_start) * 1000
                log_event(
                    logger,
                    "rpc.complete",
                    method=method,
                    streaming=False,
                    latency_ms=round(_rpc_ms, 1),
                )
                return success_response(request_id, result)

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
            return error_response(request_id, e.code, e.message, e.data)

        except TypeError as e:
            return error_response(request_id, INVALID_PARAMS, f"Invalid params: {e}")

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
            return error_response(request_id, INTERNAL_ERROR, "Internal server error")

    # ========== Streaming Helpers ==========

    async def _send_stream_token(
        self,
        writer: asyncio.StreamWriter,
        token: str,
        token_index: int,
        is_final: bool = False,
        request_id: Any = None,
    ) -> None:
        """Send a streaming token notification to a client."""
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
        """Send the final response after streaming completes."""
        response = json.dumps(
            {
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id,
            }
        )
        writer.write(response.encode() + b"\n")
        await writer.drain()

    # ========== RPC Method Delegates ==========

    async def _ping(self) -> dict[str, str | bool]:
        return await handlers.handle_ping(self._models_ready)

    async def _generate_draft(self, **kwargs: Any) -> dict[str, Any]:
        return await handlers.handle_generate_draft(
            prefetch_manager=self._prefetch_manager,
            send_stream_token=self._send_stream_token,
            send_stream_response=self._send_stream_response,
            **kwargs,
        )

    async def _summarize(self, **kwargs: Any) -> dict[str, Any]:
        return await handlers.handle_summarize(
            send_stream_token=self._send_stream_token,
            send_stream_response=self._send_stream_response,
            **kwargs,
        )

    async def _get_smart_replies(self, **kwargs: Any) -> dict[str, Any]:
        return await handlers.handle_get_smart_replies(**kwargs)

    async def _semantic_search(self, **kwargs: Any) -> dict[str, Any]:
        return await handlers.handle_semantic_search(**kwargs)

    async def _batch(self, **kwargs: Any) -> dict[str, Any]:
        return await handlers.handle_batch(methods=self._methods, **kwargs)

    async def _resolve_contacts(self, **kwargs: Any) -> dict[str, str | None]:
        return await handlers.handle_resolve_contacts(**kwargs)

    async def _list_conversations(self, **kwargs: Any) -> dict[str, Any]:
        return await handlers.handle_list_conversations(**kwargs)

    async def _get_routing_metrics(self, **kwargs: Any) -> dict[str, Any]:
        return await handlers.handle_get_routing_metrics(**kwargs)

    async def _prefetch_stats(self) -> dict[str, Any]:
        return await handlers.handle_prefetch_stats(self._prefetch_manager)

    async def _prefetch_invalidate(self, **kwargs: Any) -> dict[str, Any]:
        return await handlers.handle_prefetch_invalidate(
            prefetch_manager=self._prefetch_manager, **kwargs
        )

    async def _prefetch_focus(self, chat_id: str) -> dict[str, Any]:
        return await handlers.handle_prefetch_focus(self._prefetch_manager, chat_id)

    async def _prefetch_hover(self, chat_id: str) -> dict[str, Any]:
        return await handlers.handle_prefetch_hover(self._prefetch_manager, chat_id)


async def main(preload_models: bool = True) -> None:
    """Run the socket server."""
    from jarvis.observability.logging import configure_structured_logging

    configure_structured_logging(level=logging.INFO)

    logger.info(f"Starting socket server (preload_models={preload_models})...")
    server = JarvisSocketServer(enable_watcher=True, preload_models=preload_models)

    loop = asyncio.get_event_loop()

    def shutdown_handler() -> None:
        logger.info("Shutdown signal received")
        asyncio.create_task(server.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown_handler)

    try:
        await server.start()
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop()
