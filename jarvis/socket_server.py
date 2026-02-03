"""JARVIS Socket Server.

JSON-RPC server over Unix socket AND WebSocket for the desktop app.
Provides LLM generation, search, and classification with streaming support.

Unix Socket: /tmp/jarvis.sock (for Tauri app)
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

import asyncio
import json
import logging
import os
import signal
import sys
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import websockets
from websockets.server import ServerConnection

logger = logging.getLogger(__name__)

# Socket configuration
SOCKET_PATH = Path("/tmp/jarvis.sock")
WEBSOCKET_HOST = "127.0.0.1"
WEBSOCKET_PORT = 8743
MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB max message size


class WebSocketWriter:
    """Wrapper to make WebSocket connection compatible with StreamWriter interface.

    This allows the same streaming code to work for both Unix sockets and WebSockets.
    """

    def __init__(self, websocket: ServerConnection) -> None:
        self._websocket = websocket
        self._buffer = ""

    def write(self, data: bytes) -> None:
        """Buffer data to send."""
        self._buffer += data.decode()

    async def drain(self) -> None:
        """Send buffered data over WebSocket."""
        if self._buffer:
            # WebSocket doesn't need newline delimiters, but we keep them for consistency
            await self._websocket.send(self._buffer.rstrip("\n"))
            self._buffer = ""


class JsonRpcError(Exception):
    """JSON-RPC error with code and data."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


# Standard JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


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
        2. Server sends token notifications: {"jsonrpc": "2.0", "method": "stream.token", "params": {...}}
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
        self._running = False
        self._enable_watcher = enable_watcher
        self._preload_models = preload_models
        self._wait_for_preload = wait_for_preload
        self._preload_timeout = preload_timeout
        self._enable_prefetch = enable_prefetch
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

        # Classification methods
        self.register("classify_intent", self._classify_intent)

        # Batch operations
        self.register("batch", self._batch)

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

        Creates /tmp/jarvis.sock and ws://localhost:8743 for connections.
        """
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

        # Start WebSocket server for browser clients
        self._ws_server = await websockets.serve(
            self._handle_websocket_client,
            WEBSOCKET_HOST,
            WEBSOCKET_PORT,
            max_size=MAX_MESSAGE_SIZE,
        )

        self._running = True
        logger.info(f"Unix socket listening on {SOCKET_PATH}")
        logger.info(f"WebSocket listening on ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")

        # Start the file watcher for real-time new message detection
        if self._enable_watcher:
            from jarvis.watcher import ChatDBWatcher

            watcher = ChatDBWatcher(self)
            self._watcher_task = asyncio.create_task(watcher.start())
            logger.info("Started chat.db watcher for real-time notifications")

        # Start prefetch manager for speculative caching
        if self._enable_prefetch:
            try:
                from jarvis.prefetch import get_prefetch_manager

                self._prefetch_manager = get_prefetch_manager()
                self._prefetch_manager.start()
                logger.info("Started prefetch manager for speculative caching")
            except Exception as e:
                logger.warning(f"Prefetch manager failed to start: {e}")
                self._prefetch_manager = None

        # Preload models in background for faster first request
        if self._preload_models:
            self._preload_task = asyncio.create_task(self._preload_models_async())

            # Optionally wait for preload to complete before accepting connections
            # This eliminates cold-start latency for the first request
            if self._wait_for_preload:
                logger.info(
                    f"Waiting up to {self._preload_timeout}s for models to preload..."
                )
                ready = await self.wait_for_models()
                if ready:
                    logger.info("Models ready, accepting connections")
                else:
                    logger.warning("Preload timeout, accepting connections anyway")

        # Run both servers
        async with self._server:
            await self._server.serve_forever()

    async def _preload_models_async(self) -> None:
        """Preload LLM, embeddings, classifiers, and FAISS index in background.

        This runs at startup so the first user request doesn't have to wait
        for model loading (which can take several seconds).

        Loads in parallel for faster startup.
        """
        try:
            logger.info("Preloading models in background...")

            # Run all preloading in parallel for faster startup
            await asyncio.gather(
                asyncio.to_thread(self._preload_llm),
                asyncio.to_thread(self._preload_embeddings),
                asyncio.to_thread(self._preload_classifiers),
                asyncio.to_thread(self._preload_faiss_index),
                return_exceptions=True,  # Don't fail if one preload fails
            )

            self._models_ready = True
            self._models_ready_event.set()
            logger.info("Models preloaded and ready")

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
        except asyncio.TimeoutError:
            logger.warning(f"Model preload wait timed out after {wait_timeout}s")
            return False

    def _preload_llm(self) -> None:
        """Preload the LLM model (sync)."""
        try:
            from models.loader import get_model

            # This triggers model loading
            model = get_model()
            if model:
                logger.debug(f"LLM model preloaded: {model.model_id}")
        except Exception as e:
            logger.debug(f"LLM preload skipped: {e}")

    def _preload_embeddings(self) -> None:
        """Preload the embeddings model (sync)."""
        try:
            from jarvis.embeddings import get_embedder

            embedder = get_embedder()
            if embedder:
                # Warm up with a test embedding
                embedder.encode(["test"])
                logger.debug("Embeddings model preloaded")
        except Exception as e:
            logger.debug(f"Embeddings preload skipped: {e}")

    def _preload_classifiers(self) -> None:
        """Preload intent classifier (sync)."""
        try:
            from jarvis.intent import get_intent_classifier

            # Load intent classifier (triggers SVM model load)
            intent_clf = get_intent_classifier()
            if intent_clf:
                logger.debug("Intent classifier preloaded")

        except Exception as e:
            logger.debug(f"Classifier preload skipped: {e}")

    def _preload_faiss_index(self) -> None:
        """Preload the FAISS index for fast first search (sync)."""
        try:
            from jarvis.db import get_db
            from jarvis.index import TriggerIndexSearcher

            db = get_db()
            db.init_schema()

            # Check if index exists before trying to load
            active_index = db.get_active_index()
            if active_index:
                searcher = TriggerIndexSearcher(db)
                # Access the index property to trigger loading
                _ = searcher.index
                logger.debug(f"FAISS index preloaded (v{active_index.version_id})")
            else:
                logger.debug("No FAISS index to preload")

        except Exception as e:
            logger.debug(f"FAISS preload skipped: {e}")

    async def stop(self) -> None:
        """Stop the socket server."""
        self._running = False

        # Stop the watcher
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
        for writer in self._clients.copy():
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
        self._clients.clear()

        # Close all WebSocket client connections
        for ws in self._ws_clients.copy():
            try:
                await ws.close()
            except Exception:
                pass
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

        logger.info("Socket server stopped")

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

        # Broadcast to Unix socket clients
        for writer in self._clients.copy():
            try:
                writer.write(notification.encode() + b"\n")
                await writer.drain()
            except Exception:
                self._clients.discard(writer)

        # Broadcast to WebSocket clients
        for ws in self._ws_clients.copy():
            try:
                await ws.send(notification)
            except Exception:
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
                        response = self._error_response(
                            None, INVALID_REQUEST, "Message too large"
                        )
                        writer.write(response.encode() + b"\n")
                        await writer.drain()
                    else:
                        # Pass writer for streaming support
                        response = await self._process_message(line.decode(), writer)
                        if response:  # May be None for streaming (already sent)
                            writer.write(response.encode() + b"\n")
                            await writer.drain()

                except asyncio.TimeoutError:
                    break  # Client idle timeout

        except Exception as e:
            logger.debug(f"Client error: {e}")

        finally:
            self._clients.discard(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            logger.debug(f"Client disconnected: {peer}")

    async def _handle_websocket_client(self, websocket: ServerConnection) -> None:
        """Handle a WebSocket client connection.

        Args:
            websocket: WebSocket connection
        """
        self._ws_clients.add(websocket)
        logger.debug(f"WebSocket client connected: {websocket.remote_address}")

        try:
            async for message in websocket:
                if not self._running:
                    break

                if len(message) > MAX_MESSAGE_SIZE:
                    response = self._error_response(
                        None, INVALID_REQUEST, "Message too large"
                    )
                    await websocket.send(response)
                else:
                    # Create a WebSocket writer wrapper for streaming
                    ws_writer = WebSocketWriter(websocket)
                    response = await self._process_message(str(message), ws_writer)
                    if response:  # May be None for streaming (already sent)
                        await websocket.send(response)

        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            logger.debug(f"WebSocket client error: {e}")
        finally:
            self._ws_clients.discard(websocket)
            logger.debug(f"WebSocket client disconnected: {websocket.remote_address}")

    async def _process_message(
        self, message: str, writer: "asyncio.StreamWriter | WebSocketWriter | None" = None
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
            return self._error_response(
                request_id, METHOD_NOT_FOUND, f"Method not found: {method}"
            )

        # Check if streaming is requested and supported
        stream_requested = isinstance(params, dict) and params.pop("stream", False)
        supports_streaming = method in self._streaming_methods

        # Call handler
        try:
            if stream_requested and supports_streaming and writer:
                # Streaming mode: pass writer and request_id to handler
                if isinstance(params, dict):
                    result = await handler(
                        _writer=writer, _request_id=request_id, **params
                    )
                else:
                    result = await handler(
                        _writer=writer, _request_id=request_id
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

                return self._success_response(request_id, result)

        except JsonRpcError as e:
            return self._error_response(request_id, e.code, e.message, e.data)

        except TypeError as e:
            return self._error_response(
                request_id, INVALID_PARAMS, f"Invalid params: {e}"
            )

        except Exception as e:
            logger.exception(f"Error handling {method}")
            return self._error_response(request_id, INTERNAL_ERROR, str(e))

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
    ) -> None:
        """Send a streaming token notification to a client.

        Args:
            writer: Client stream writer
            token: The token text
            token_index: Index of this token in the stream
            is_final: Whether this is the last token
        """
        notification = json.dumps({
            "jsonrpc": "2.0",
            "method": "stream.token",
            "params": {
                "token": token,
                "index": token_index,
                "final": is_final,
            },
        })
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
        response = json.dumps({
            "jsonrpc": "2.0",
            "result": result,
            "id": request_id,
        })
        writer.write(response.encode() + b"\n")
        await writer.drain()

    # ========== RPC Method Handlers ==========

    async def _ping(self) -> dict[str, str | bool]:
        """Health check with model readiness status."""
        return {
            "status": "ok",
            "models_ready": self._models_ready,
        }

    async def _generate_draft(
        self,
        chat_id: str,
        instruction: str | None = None,
        context_messages: int = 20,
        _writer: asyncio.StreamWriter | None = None,
        _request_id: Any = None,
        skip_cache: bool = False,
    ) -> dict[str, Any]:
        """Generate draft replies for a conversation.

        Supports streaming: if _writer is provided, streams tokens in real-time.
        Uses prefetch cache for near-instant responses when available.

        Args:
            chat_id: Conversation ID
            instruction: Optional custom instruction
            context_messages: Number of context messages to include
            _writer: Stream writer for streaming mode (injected by dispatcher)
            _request_id: Request ID for streaming mode (injected by dispatcher)
            skip_cache: Skip prefetch cache lookup

        Returns:
            Dict with suggestions and context_used
        """
        try:
            # Check prefetch cache first for instant response (non-streaming only)
            if not skip_cache and _writer is None and self._prefetch_manager:
                cached_draft = self._prefetch_manager.get_draft(chat_id)
                if cached_draft and "suggestions" in cached_draft:
                    logger.debug(f"Serving prefetched draft for {chat_id}")
                    cached_draft["from_cache"] = True
                    return cached_draft

            # Get context from iMessage
            from integrations.imessage import ChatDBReader

            with ChatDBReader() as reader:
                messages = reader.get_messages(chat_id, limit=context_messages)

            if not messages:
                raise JsonRpcError(INVALID_PARAMS, "No messages found in conversation")

            # Build context from messages
            context = []
            participants: set[str] = set()
            for msg in reversed(messages):  # Oldest first
                sender = msg.sender_name or msg.sender
                participants.add(sender)
                prefix = "Me" if msg.is_from_me else sender
                if msg.text:
                    context.append(f"{prefix}: {msg.text}")

            # Get the last incoming message to respond to
            last_incoming = None
            for msg in messages:
                if not msg.is_from_me and msg.text:
                    last_incoming = msg.text
                    break

            if not last_incoming:
                raise JsonRpcError(INVALID_PARAMS, "No message to respond to")

            context_used = {
                "num_messages": len(messages),
                "participants": list(participants),
                "last_message": messages[0].text if messages else None,
            }

            # Check if streaming is requested
            if _writer is not None:
                return await self._generate_draft_streaming(
                    last_incoming=last_incoming,
                    context=context,
                    chat_id=chat_id,
                    instruction=instruction,
                    writer=_writer,
                    request_id=_request_id,
                    context_used=context_used,
                )

            # Non-streaming: use router
            from jarvis.router import get_reply_router

            router = get_reply_router()
            result = router.route(
                incoming=last_incoming,
                thread=context[-10:] if context else None,
                chat_id=chat_id,
            )

            response_text = result.get("response", "")
            confidence = 0.8 if result.get("confidence") == "high" else 0.6

            return {
                "suggestions": [{"text": response_text, "confidence": confidence}],
                "context_used": context_used,
            }

        except JsonRpcError:
            raise
        except Exception as e:
            logger.exception("Error generating draft")
            raise JsonRpcError(INTERNAL_ERROR, f"Generation failed: {e}") from e

    async def _generate_draft_streaming(
        self,
        last_incoming: str,
        context: list[str],
        chat_id: str,
        instruction: str | None,
        writer: asyncio.StreamWriter,
        request_id: Any,
        context_used: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate draft with real token streaming.

        Streams tokens to the client as they're generated using a queue
        to bridge sync generator and async sending.
        """
        import queue
        import threading

        from models.loader import get_model

        model = get_model()
        if not model:
            raise JsonRpcError(INTERNAL_ERROR, "Model not available")

        # Ensure model is loaded before streaming
        if not model.is_loaded():
            await asyncio.to_thread(model.load)

        # Build prompt
        thread_text = "\n".join(context[-10:]) if context else ""
        if instruction:
            prompt = f"""Conversation:
{thread_text}

Last message: {last_incoming}

Instruction: {instruction}

Write a helpful reply:"""
        else:
            prompt = f"""Conversation:
{thread_text}

Last message: {last_incoming}

Write a brief, helpful reply:"""

        # Use a queue to bridge sync generator and async sending
        token_queue: queue.Queue[tuple[str, int, bool] | None] = queue.Queue()
        generation_error: list[Exception] = []

        def producer():
            """Run in thread: generate tokens and put in queue."""
            try:
                for stream_token in model.generate_stream(prompt, max_tokens=150):
                    token_queue.put((
                        stream_token.token,
                        stream_token.token_index,
                        stream_token.is_final,
                    ))
                token_queue.put(None)  # Signal completion
            except Exception as e:
                generation_error.append(e)
                token_queue.put(None)

        # Start generation in background thread
        gen_thread = threading.Thread(target=producer, daemon=True)
        gen_thread.start()

        # Consume tokens and send to client
        full_response = ""
        token_count = 0

        try:
            while True:
                # Check queue with timeout to stay responsive
                try:
                    item = await asyncio.to_thread(token_queue.get, timeout=30.0)
                except Exception:
                    break

                if item is None:
                    break  # Generation complete

                token_text, token_index, is_final = item
                full_response += token_text
                token_count += 1

                # Send token to client immediately
                await self._send_stream_token(writer, token_text, token_index, is_final)

            # Wait for thread to finish
            gen_thread.join(timeout=5.0)

            # Check for errors
            if generation_error:
                raise generation_error[0]

        except Exception as e:
            logger.exception("Streaming generation failed")
            raise JsonRpcError(INTERNAL_ERROR, f"Streaming failed: {e}") from e

        # Send final response
        result = {
            "suggestions": [{"text": full_response.strip(), "confidence": 0.8}],
            "context_used": context_used,
            "streamed": True,
            "tokens_generated": token_count,
        }
        await self._send_stream_response(writer, request_id, result)
        return result

    async def _summarize(
        self,
        chat_id: str,
        num_messages: int = 50,
        _writer: asyncio.StreamWriter | None = None,
        _request_id: Any = None,
    ) -> dict[str, Any]:
        """Summarize a conversation.

        Supports streaming: if _writer is provided, streams tokens in real-time.

        Args:
            chat_id: Conversation ID
            num_messages: Number of messages to summarize
            _writer: Stream writer for streaming mode (injected by dispatcher)
            _request_id: Request ID for streaming mode (injected by dispatcher)

        Returns:
            Dict with summary, key_points, and message_count
        """
        try:
            # Get messages from iMessage
            from integrations.imessage import ChatDBReader

            with ChatDBReader() as reader:
                messages = reader.get_messages(chat_id, limit=num_messages)

            if not messages:
                raise JsonRpcError(INVALID_PARAMS, "No messages found")

            # Build conversation text
            conversation = []
            for msg in reversed(messages):
                sender = msg.sender_name or msg.sender
                prefix = "Me" if msg.is_from_me else sender
                if msg.text:
                    conversation.append(f"{prefix}: {msg.text}")

            if not conversation:
                return {
                    "summary": "No text messages found in conversation",
                    "key_points": [],
                    "message_count": len(messages),
                }

            from models.loader import get_model

            model = get_model()
            if not model:
                raise JsonRpcError(INTERNAL_ERROR, "Model not available")

            conversation_text = "\n".join(conversation[-30:])  # Limit for context window
            prompt = f"""Summarize this conversation in 2-3 sentences, then list key points:

{conversation_text}

Summary:"""

            # Check if streaming is requested
            if _writer is not None:
                return await self._summarize_streaming(
                    model=model,
                    prompt=prompt,
                    message_count=len(messages),
                    writer=_writer,
                    request_id=_request_id,
                )

            # Non-streaming generation - ensure model is loaded first
            if not model.is_loaded():
                model.load()
            result = model.generate_sync(prompt, max_tokens=300)
            response_text = result.text

            # Parse response - simple extraction
            lines = response_text.strip().split("\n")
            summary = lines[0] if lines else "Conversation summary unavailable"

            # Extract bullet points
            key_points = [
                line.lstrip("•-*0123456789.)").strip()
                for line in lines[1:]
                if line.strip() and len(line.strip()) > 5
            ][:5]  # Max 5 key points

            return {
                "summary": summary,
                "key_points": key_points or ["See full conversation for details"],
                "message_count": len(messages),
            }

        except JsonRpcError:
            raise
        except Exception as e:
            logger.exception("Error summarizing conversation")
            raise JsonRpcError(INTERNAL_ERROR, f"Summarization failed: {e}") from e

    async def _summarize_streaming(
        self,
        model: Any,
        prompt: str,
        message_count: int,
        writer: asyncio.StreamWriter,
        request_id: Any,
    ) -> dict[str, Any]:
        """Summarize with real token streaming."""
        import queue
        import threading

        # Ensure model is loaded before streaming
        if not model.is_loaded():
            await asyncio.to_thread(model.load)

        token_queue: queue.Queue[tuple[str, int, bool] | None] = queue.Queue()
        generation_error: list[Exception] = []

        def producer():
            try:
                for stream_token in model.generate_stream(prompt, max_tokens=300):
                    token_queue.put((
                        stream_token.token,
                        stream_token.token_index,
                        stream_token.is_final,
                    ))
                token_queue.put(None)
            except Exception as e:
                generation_error.append(e)
                token_queue.put(None)

        gen_thread = threading.Thread(target=producer, daemon=True)
        gen_thread.start()

        full_response = ""
        token_count = 0

        try:
            while True:
                try:
                    item = await asyncio.to_thread(token_queue.get, timeout=30.0)
                except Exception:
                    break

                if item is None:
                    break

                token_text, token_index, is_final = item
                full_response += token_text
                token_count += 1

                await self._send_stream_token(writer, token_text, token_index, is_final)

            gen_thread.join(timeout=5.0)

            if generation_error:
                raise generation_error[0]

        except Exception as e:
            logger.exception("Streaming summarization failed")
            raise JsonRpcError(INTERNAL_ERROR, f"Streaming failed: {e}") from e

        # Parse the streamed response
        lines = full_response.strip().split("\n")
        summary = lines[0] if lines else "Conversation summary unavailable"
        key_points = [
            line.lstrip("•-*0123456789.)").strip()
            for line in lines[1:]
            if line.strip() and len(line.strip()) > 5
        ][:5]

        result = {
            "summary": summary,
            "key_points": key_points or ["See full conversation for details"],
            "message_count": message_count,
            "streamed": True,
            "tokens_generated": token_count,
        }
        await self._send_stream_response(writer, request_id, result)
        return result

    async def _get_smart_replies(
        self,
        last_message: str,
        num_suggestions: int = 3,
    ) -> dict[str, Any]:
        """Get smart reply suggestions.

        Args:
            last_message: The last message to respond to
            num_suggestions: Number of suggestions

        Returns:
            Dict with suggestions list
        """
        try:
            # Use the router to generate a response
            from jarvis.router import get_reply_router

            router = get_reply_router()
            result = router.route(incoming=last_message)

            # Get the main response
            response_text = result.get("response", "")

            suggestions = []
            if response_text:
                suggestions.append({
                    "text": response_text,
                    "score": 0.9 if result.get("confidence") == "high" else 0.7,
                })

            # For additional suggestions, we could generate variations
            # For now, just return the single best response
            return {"suggestions": suggestions}

        except Exception as e:
            logger.exception("Error getting smart replies")
            raise JsonRpcError(INTERNAL_ERROR, f"Smart replies failed: {e}") from e

    async def _semantic_search(
        self,
        query: str,
        limit: int = 20,
        threshold: float = 0.3,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Semantic search across messages.

        Args:
            query: Search query
            limit: Maximum results
            threshold: Minimum similarity threshold
            filters: Optional filters (chat_id, sender, after, before)

        Returns:
            Dict with results and total_results
        """
        try:
            from jarvis.index import semantic_search

            # Build filter args
            filter_kwargs: dict[str, Any] = {"limit": limit, "threshold": threshold}
            if filters:
                if "chat_id" in filters:
                    filter_kwargs["chat_id"] = filters["chat_id"]
                if "sender" in filters:
                    filter_kwargs["sender"] = filters["sender"]

            results = semantic_search(query, **filter_kwargs)

            return {
                "results": [
                    {
                        "message": {
                            "id": r["message_id"],
                            "chat_id": r["chat_id"],
                            "text": r["text"],
                            "sender": r["sender"],
                            "date": r["date"],
                        },
                        "similarity": r["similarity"],
                    }
                    for r in results
                ],
                "total_results": len(results),
            }

        except Exception as e:
            logger.exception("Error in semantic search")
            raise JsonRpcError(INTERNAL_ERROR, f"Search failed: {e}") from e

    async def _classify_intent(self, text: str) -> dict[str, Any]:
        """Classify message intent.

        Args:
            text: Message text to classify

        Returns:
            Dict with intent, confidence, requires_response
        """
        try:
            from jarvis.intent import get_intent_classifier

            classifier = get_intent_classifier()
            result = classifier.classify(text)

            return {
                "intent": result.intent.value if hasattr(result.intent, "value") else str(result.intent),
                "confidence": result.confidence,
                "requires_response": result.requires_response,
            }

        except Exception as e:
            logger.exception("Error classifying intent")
            raise JsonRpcError(INTERNAL_ERROR, f"Classification failed: {e}") from e

    async def _batch(
        self,
        requests: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute multiple RPC calls in a single request.

        Runs requests in parallel where possible for maximum efficiency.

        Args:
            requests: List of request objects, each with:
                - method: The method name
                - params: Optional parameters dict
                - id: Optional request ID for correlating responses

        Returns:
            Dict with results list, each containing:
                - id: Request ID (if provided)
                - result: Method result (if successful)
                - error: Error info (if failed)

        Example:
            {"requests": [
                {"method": "classify_intent", "params": {"text": "hello"}, "id": 1},
                {"method": "classify_intent", "params": {"text": "thanks"}, "id": 2},
            ]}
        """
        if not requests:
            return {"results": []}

        if len(requests) > 50:
            raise JsonRpcError(INVALID_PARAMS, "Maximum 50 requests per batch")

        async def execute_single(req: dict[str, Any]) -> dict[str, Any]:
            """Execute a single request from the batch."""
            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params", {})

            if not method:
                return {
                    "id": req_id,
                    "error": {"code": INVALID_REQUEST, "message": "Missing method"},
                }

            handler = self._methods.get(method)
            if not handler:
                return {
                    "id": req_id,
                    "error": {"code": METHOD_NOT_FOUND, "message": f"Method not found: {method}"},
                }

            try:
                if isinstance(params, dict):
                    # Remove streaming params - batch doesn't support streaming
                    params.pop("stream", None)
                    params.pop("_writer", None)
                    params.pop("_request_id", None)
                    result = await handler(**params)
                elif isinstance(params, list):
                    result = await handler(*params)
                else:
                    result = await handler()

                return {"id": req_id, "result": result}

            except JsonRpcError as e:
                return {
                    "id": req_id,
                    "error": {"code": e.code, "message": e.message},
                }
            except Exception as e:
                logger.exception(f"Batch error for {method}")
                return {
                    "id": req_id,
                    "error": {"code": INTERNAL_ERROR, "message": str(e)},
                }

        # Execute all requests in parallel
        results = await asyncio.gather(
            *[execute_single(req) for req in requests],
            return_exceptions=False,
        )

        return {"results": list(results)}

    # ========== Prefetch RPC Methods ==========

    async def _prefetch_stats(self) -> dict[str, Any]:
        """Get prefetch system statistics.

        Returns:
            Dict with cache, executor, and invalidator stats
        """
        if not self._prefetch_manager:
            return {
                "enabled": False,
                "error": "Prefetch system not enabled",
            }

        try:
            stats = self._prefetch_manager.stats()
            stats["enabled"] = True
            return stats
        except Exception as e:
            logger.exception("Error getting prefetch stats")
            raise JsonRpcError(INTERNAL_ERROR, f"Stats failed: {e}") from e

    async def _prefetch_invalidate(
        self,
        chat_id: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Invalidate prefetch cache entries.

        Args:
            chat_id: Chat ID to invalidate
            tags: Tags to invalidate

        Returns:
            Dict with count of invalidated entries
        """
        if not self._prefetch_manager:
            return {"invalidated": 0, "error": "Prefetch system not enabled"}

        try:
            count = self._prefetch_manager.invalidate(chat_id=chat_id, tags=tags)
            return {"invalidated": count}
        except Exception as e:
            logger.exception("Error invalidating cache")
            raise JsonRpcError(INTERNAL_ERROR, f"Invalidation failed: {e}") from e

    async def _prefetch_focus(self, chat_id: str) -> dict[str, Any]:
        """Signal that user focused on a chat (triggers high-priority prefetch).

        Args:
            chat_id: Chat ID that was focused

        Returns:
            Dict with status and any prefetched data
        """
        if not self._prefetch_manager:
            return {"status": "disabled"}

        try:
            self._prefetch_manager.on_focus(chat_id)

            # Check if we have a prefetched draft
            draft = self._prefetch_manager.get_draft(chat_id)
            if draft:
                return {
                    "status": "ok",
                    "prefetched": True,
                    "draft": draft,
                }
            return {"status": "ok", "prefetched": False}
        except Exception as e:
            logger.debug(f"Prefetch focus error: {e}")
            return {"status": "error", "error": str(e)}

    async def _prefetch_hover(self, chat_id: str) -> dict[str, Any]:
        """Signal that user hovered over a chat (triggers low-priority prefetch).

        Args:
            chat_id: Chat ID that was hovered

        Returns:
            Dict with status
        """
        if not self._prefetch_manager:
            return {"status": "disabled"}

        try:
            self._prefetch_manager.on_hover(chat_id)
            return {"status": "ok"}
        except Exception as e:
            logger.debug(f"Prefetch hover error: {e}")
            return {"status": "error", "error": str(e)}


async def main() -> None:
    """Run the socket server."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    server = JarvisSocketServer()

    # Handle shutdown signals
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


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
