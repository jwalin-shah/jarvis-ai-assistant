"""WebSocket router for real-time communication.

Provides WebSocket endpoints for:
- Real-time conversation updates
- Streaming model generation responses (token-by-token)
- Health/connection status updates
- Graceful reconnection handling
"""

import asyncio
import json
import logging
import os
import secrets
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from contracts.models import GenerationRequest
from models import get_generator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

# SECURITY: WebSocket authentication token
# Set JARVIS_WS_TOKEN environment variable or generate a random token
_WS_AUTH_TOKEN: str | None = os.getenv("JARVIS_WS_TOKEN")
if _WS_AUTH_TOKEN is None:
    # Generate a random token for this session (log it so it can be used)
    _WS_AUTH_TOKEN = secrets.token_urlsafe(32)
    logger.info("Generated WebSocket auth token (set JARVIS_WS_TOKEN to persist)")


def _validate_websocket_auth(websocket: WebSocket) -> bool:
    """Validate WebSocket authentication token.

    SECURITY: Checks for auth token in query parameters to prevent unauthorized access.
    Token can be provided via:
    - Query parameter: ?token=<token>
    - Header: X-WS-Token: <token>

    For localhost connections, authentication can be bypassed if JARVIS_WS_REQUIRE_AUTH=false.

    Args:
        websocket: The WebSocket connection to validate

    Returns:
        True if authenticated, False otherwise
    """
    # Allow bypass for localhost if explicitly disabled
    require_auth = os.getenv("JARVIS_WS_REQUIRE_AUTH", "true").lower() != "false"
    if not require_auth:
        client_host = websocket.client.host if websocket.client else None
        if client_host in ("127.0.0.1", "localhost", "::1"):
            return True

    # Check query parameters
    token = websocket.query_params.get("token")
    if token and secrets.compare_digest(token, _WS_AUTH_TOKEN):
        return True

    # Check headers
    token_header = websocket.headers.get("x-ws-token")
    if token_header and secrets.compare_digest(token_header, _WS_AUTH_TOKEN):
        return True

    return False


class MessageType(str, Enum):
    """WebSocket message types."""

    # Client -> Server
    GENERATE = "generate"
    GENERATE_STREAM = "generate_stream"
    SUBSCRIBE_HEALTH = "subscribe_health"
    UNSUBSCRIBE_HEALTH = "unsubscribe_health"
    PING = "ping"
    CANCEL = "cancel"

    # Server -> Client
    CONNECTED = "connected"
    TOKEN = "token"
    GENERATION_START = "generation_start"
    GENERATION_COMPLETE = "generation_complete"
    GENERATION_ERROR = "generation_error"
    HEALTH_UPDATE = "health_update"
    PONG = "pong"
    ERROR = "error"


@dataclass
class WebSocketClient:
    """Represents a connected WebSocket client."""

    websocket: WebSocket
    client_id: str
    connected_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    subscribed_to_health: bool = False
    active_generation_id: str | None = None
    # Per-client rate limiting: track last 5 generation request timestamps
    generation_request_times: list[float] = field(default_factory=list)


class ConnectionManager:
    """Manages WebSocket connections and broadcasts.

    Thread-safe connection tracking with support for:
    - Individual client messaging
    - Broadcast to all clients
    - Health update subscriptions
    - Active generation tracking for cancellation
    - Periodic cleanup of stale connections
    """

    # Connection limits and TTL settings
    MAX_CONNECTIONS = 100  # Maximum concurrent connections
    CONNECTION_TTL_SECONDS = 3600  # 1 hour TTL for inactive connections
    CLEANUP_INTERVAL_SECONDS = 300  # Run cleanup every 5 minutes

    def __init__(self) -> None:
        """Initialize the connection manager."""
        self._clients: dict[str, WebSocketClient] = {}
        self._lock = asyncio.Lock()
        self._health_task: asyncio.Task[None] | None = None
        self._cleanup_task: asyncio.Task[None] | None = None

    @property
    def active_connections(self) -> int:
        """Return count of active connections."""
        return len(self._clients)

    async def connect(self, websocket: WebSocket) -> WebSocketClient:
        """Accept and register a new WebSocket connection.

        Args:
            websocket: The WebSocket connection to accept.

        Returns:
            WebSocketClient instance for the new connection.
        """
        await websocket.accept()
        client_id = str(uuid.uuid4())
        client = WebSocketClient(websocket=websocket, client_id=client_id)

        async with self._lock:
            self._clients[client_id] = client

        logger.info("WebSocket client connected: %s", client_id)
        return client

    async def disconnect(self, client_id: str) -> None:
        """Remove a client from the connection pool.

        Args:
            client_id: The ID of the client to disconnect.
        """
        async with self._lock:
            if client_id in self._clients:
                del self._clients[client_id]
                logger.info("WebSocket client disconnected: %s", client_id)

    async def send_message(
        self, client_id: str, message_type: MessageType, data: dict[str, Any] | None = None
    ) -> bool:
        """Send a message to a specific client.

        Args:
            client_id: Target client ID.
            message_type: Type of message to send.
            data: Optional message payload.

        Returns:
            True if message was sent successfully, False otherwise.
        """
        async with self._lock:
            client = self._clients.get(client_id)
            if not client:
                return False

        try:
            message = {"type": message_type.value, "data": data or {}}
            await client.websocket.send_json(message)
            return True
        except Exception as e:
            logger.warning("Failed to send message to %s: %s", client_id, e)
            return False

    async def broadcast(
        self, message_type: MessageType, data: dict[str, Any] | None = None
    ) -> None:
        """Broadcast a message to all connected clients.

        Uses asyncio.gather() for parallel message sending.

        Args:
            message_type: Type of message to broadcast.
            data: Optional message payload.
        """
        async with self._lock:
            client_ids = list(self._clients.keys())

        # Send to all clients in parallel for better performance
        await asyncio.gather(
            *[self.send_message(client_id, message_type, data) for client_id in client_ids],
            return_exceptions=True,
        )

    async def broadcast_health_update(self, health_data: dict[str, Any]) -> None:
        """Broadcast health update to subscribed clients.

        Uses asyncio.gather() for parallel message sending.

        Args:
            health_data: Health status data to broadcast.
        """
        async with self._lock:
            subscribed_clients = [
                c.client_id for c in self._clients.values() if c.subscribed_to_health
            ]

        # Send to all subscribed clients in parallel
        await asyncio.gather(
            *[
                self.send_message(client_id, MessageType.HEALTH_UPDATE, health_data)
                for client_id in subscribed_clients
            ],
            return_exceptions=True,
        )

    def get_client(self, client_id: str) -> WebSocketClient | None:
        """Get a client by ID.

        Note: dict.get() is atomic in CPython (GIL-protected), so no lock needed
        for read-only access. Mutations use self._lock elsewhere.

        Args:
            client_id: The client ID to look up.

        Returns:
            WebSocketClient if found, None otherwise.
        """
        return self._clients.get(client_id)

    async def set_health_subscription(self, client_id: str, subscribed: bool) -> None:
        """Update a client's health subscription status.

        Args:
            client_id: The client ID to update.
            subscribed: Whether to subscribe or unsubscribe.
        """
        async with self._lock:
            client = self._clients.get(client_id)
            if client:
                client.subscribed_to_health = subscribed

    async def set_active_generation(self, client_id: str, generation_id: str | None) -> None:
        """Set the active generation ID for a client.

        Args:
            client_id: The client ID to update.
            generation_id: The generation ID, or None to clear.
        """
        async with self._lock:
            client = self._clients.get(client_id)
            if client:
                client.active_generation_id = generation_id

    def get_all_client_ids(self) -> list[str]:
        """Get all connected client IDs.

        Returns:
            List of client IDs.
        """
        return list(self._clients.keys())

    async def update_activity(self, client_id: str) -> None:
        """Update the last activity timestamp for a client.

        Args:
            client_id: The client ID to update.
        """
        async with self._lock:
            client = self._clients.get(client_id)
            if client:
                client.last_activity = time.time()

    async def cleanup_stale_connections(self) -> int:
        """Remove stale connections that have exceeded TTL.

        Returns:
            Number of connections removed.
        """
        now = time.time()
        stale_clients: list[tuple[str, WebSocketClient]] = []

        # Store references while holding lock to avoid race condition
        async with self._lock:
            for client_id, client in self._clients.items():
                if now - client.last_activity > self.CONNECTION_TTL_SECONDS:
                    stale_clients.append((client_id, client))

        # Close and remove stale connections outside the lock
        for client_id, stale_client in stale_clients:
            logger.info("Removing stale WebSocket connection: %s", client_id)
            try:
                await stale_client.websocket.close(code=1000, reason="Connection timeout")
            except Exception as e:
                logger.debug("Error closing stale connection %s: %s", client_id, e)
            await self.disconnect(client_id)

        if stale_clients:
            logger.info("Cleaned up %d stale WebSocket connections", len(stale_clients))

        return len(stale_clients)

    async def start_cleanup_task(self) -> None:
        """Start the periodic cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info("Started WebSocket cleanup task")

    async def stop_cleanup_task(self) -> None:
        """Stop the periodic cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped WebSocket cleanup task")

    async def _periodic_cleanup(self) -> None:
        """Periodically clean up stale connections."""
        while True:
            try:
                await asyncio.sleep(self.CLEANUP_INTERVAL_SECONDS)
                await self.cleanup_stale_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in cleanup task: %s", e)

    def is_at_capacity(self) -> bool:
        """Check if the connection manager is at capacity.

        Returns:
            True if at or over MAX_CONNECTIONS, False otherwise.
        """
        return len(self._clients) >= self.MAX_CONNECTIONS


# Global connection manager instance
manager = ConnectionManager()


def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager instance.

    Returns:
        The singleton ConnectionManager instance.
    """
    return manager


def _log_task_exception(task: asyncio.Task[None], client_id: str) -> None:
    """Log exceptions from background tasks.

    Args:
        task: The completed task to check for exceptions
        client_id: The client ID for logging context
    """
    if task.done() and not task.cancelled():
        exc = task.exception()
        if exc:
            logger.error("Background task failed for client %s: %s", client_id, exc, exc_info=exc)


async def _check_client_rate_limit(client: WebSocketClient) -> bool:
    """Check if client is within generation rate limit.

    Rate limit: 5 generation requests per minute per client.

    Args:
        client: The client to check.

    Returns:
        True if within limit, False if exceeded.
    """
    now = time.time()
    # Remove timestamps older than 60 seconds
    client.generation_request_times = [t for t in client.generation_request_times if now - t < 60]

    # Check if limit exceeded (5 requests per minute)
    if len(client.generation_request_times) >= 5:
        return False

    # Add current request timestamp
    client.generation_request_times.append(now)
    return True


async def _handle_generate(
    client: WebSocketClient, data: dict[str, Any], stream: bool = False
) -> None:
    """Handle a generation request.

    Args:
        client: The client making the request.
        data: Request data including prompt, context, etc.
        stream: Whether to stream tokens or return complete response.
    """
    generation_id = str(uuid.uuid4())
    await manager.set_active_generation(client.client_id, generation_id)

    try:
        # Extract request parameters
        prompt = data.get("prompt", "")
        if not prompt:
            await manager.send_message(
                client.client_id,
                MessageType.GENERATION_ERROR,
                {"error": "Prompt is required", "generation_id": generation_id},
            )
            return

        context_documents = data.get("context_documents", [])
        few_shot_examples = data.get("few_shot_examples", [])
        max_tokens = data.get("max_tokens", 100)
        temperature = data.get("temperature", 0.7)
        stop_sequences = data.get("stop_sequences")

        # Validate parameter types and ranges
        if not isinstance(max_tokens, int) or max_tokens < 1 or max_tokens > 4096:
            await manager.send_message(
                client.client_id,
                MessageType.GENERATION_ERROR,
                {
                    "error": "max_tokens must be an integer between 1 and 4096",
                    "generation_id": generation_id,
                },
            )
            return
        if not isinstance(temperature, (int, float)) or temperature < 0.0 or temperature > 2.0:
            await manager.send_message(
                client.client_id,
                MessageType.GENERATION_ERROR,
                {
                    "error": "temperature must be a number between 0.0 and 2.0",
                    "generation_id": generation_id,
                },
            )
            return
        if not isinstance(context_documents, list):
            await manager.send_message(
                client.client_id,
                MessageType.GENERATION_ERROR,
                {"error": "context_documents must be a list", "generation_id": generation_id},
            )
            return
        if not isinstance(few_shot_examples, list):
            await manager.send_message(
                client.client_id,
                MessageType.GENERATION_ERROR,
                {"error": "few_shot_examples must be a list", "generation_id": generation_id},
            )
            return

        # Convert few_shot_examples to list of tuples
        examples = [
            (ex.get("input", ""), ex.get("output", ""))
            for ex in few_shot_examples
            if isinstance(ex, dict)
        ]

        # Notify generation start
        await manager.send_message(
            client.client_id,
            MessageType.GENERATION_START,
            {
                "generation_id": generation_id,
                "streaming": stream,
            },
        )

        # Create generation request
        request = GenerationRequest(
            prompt=prompt,
            context_documents=context_documents,
            few_shot_examples=examples,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
        )

        # Get generator
        try:
            generator = get_generator()
        except Exception as e:
            logger.error("Failed to get generator: %s", e)
            await manager.send_message(
                client.client_id,
                MessageType.GENERATION_ERROR,
                {
                    "error": "Model service unavailable",
                    "generation_id": generation_id,
                },
            )
            return

        if stream:
            # Use streaming generation
            await _stream_generation(client, generator, request, generation_id)
        else:
            # Use regular generation - run in threadpool to avoid blocking event loop
            start_time = time.perf_counter()
            response = await run_in_threadpool(generator.generate, request)
            generation_time = (time.perf_counter() - start_time) * 1000

            await manager.send_message(
                client.client_id,
                MessageType.GENERATION_COMPLETE,
                {
                    "generation_id": generation_id,
                    "text": response.text,
                    "tokens_used": response.tokens_used,
                    "generation_time_ms": generation_time,
                    "model_name": response.model_name,
                    "used_template": response.used_template,
                    "template_name": response.template_name,
                    "finish_reason": response.finish_reason,
                },
            )

    except Exception as e:
        # Log detailed error server-side, send generic message to client
        logger.exception("Generation failed for client %s: %s", client.client_id, e)
        await manager.send_message(
            client.client_id,
            MessageType.GENERATION_ERROR,
            {"error": "Generation failed. Please try again.", "generation_id": generation_id},
        )
    finally:
        await manager.set_active_generation(client.client_id, None)


async def _stream_generation(
    client: WebSocketClient,
    generator: Any,
    request: GenerationRequest,
    generation_id: str,
) -> None:
    """Stream generation tokens to the client.

    Args:
        client: The client to stream to.
        generator: The generator instance.
        request: The generation request.
        generation_id: Unique ID for this generation.
    """
    start_time = time.perf_counter()
    tokens_sent = 0
    full_text = ""

    try:
        # Check if generator supports streaming
        if hasattr(generator, "generate_stream"):
            async for token_data in generator.generate_stream(request):
                # Check for cancellation
                current_client = manager.get_client(client.client_id)
                if not current_client or current_client.active_generation_id != generation_id:
                    logger.info("Generation %s was cancelled", generation_id)
                    return

                token = token_data.get("token", "")
                full_text += token
                tokens_sent += 1

                await manager.send_message(
                    client.client_id,
                    MessageType.TOKEN,
                    {
                        "generation_id": generation_id,
                        "token": token,
                        "token_index": tokens_sent,
                    },
                )

            generation_time = (time.perf_counter() - start_time) * 1000

            model_name = "unknown"
            if hasattr(generator, "config"):
                model_name = generator.config.model_path

            await manager.send_message(
                client.client_id,
                MessageType.GENERATION_COMPLETE,
                {
                    "generation_id": generation_id,
                    "text": full_text,
                    "tokens_used": tokens_sent,
                    "generation_time_ms": generation_time,
                    "model_name": model_name,
                    "used_template": False,
                    "template_name": None,
                    "finish_reason": "stop",
                },
            )
        else:
            # Fallback: use regular generation and simulate streaming
            # Run in threadpool to avoid blocking event loop
            response = await run_in_threadpool(generator.generate, request)
            generation_time = (time.perf_counter() - start_time) * 1000

            # Send tokens one at a time with small delays to simulate streaming
            words = response.text.split()
            for i, word in enumerate(words):
                # Check for cancellation
                current_client = manager.get_client(client.client_id)
                if not current_client or current_client.active_generation_id != generation_id:
                    logger.info("Generation %s was cancelled", generation_id)
                    return

                token = word + (" " if i < len(words) - 1 else "")
                await manager.send_message(
                    client.client_id,
                    MessageType.TOKEN,
                    {
                        "generation_id": generation_id,
                        "token": token,
                        "token_index": i,
                    },
                )
                # Small delay between tokens for visual effect
                await asyncio.sleep(0.02)

            await manager.send_message(
                client.client_id,
                MessageType.GENERATION_COMPLETE,
                {
                    "generation_id": generation_id,
                    "text": response.text,
                    "tokens_used": response.tokens_used,
                    "generation_time_ms": generation_time,
                    "model_name": response.model_name,
                    "used_template": response.used_template,
                    "template_name": response.template_name,
                    "finish_reason": response.finish_reason,
                },
            )

    except Exception as e:
        # Log detailed error server-side, send generic message to client
        logger.exception("Streaming generation failed for client %s: %s", client.client_id, e)
        await manager.send_message(
            client.client_id,
            MessageType.GENERATION_ERROR,
            {"error": "Generation failed. Please try again.", "generation_id": generation_id},
        )


async def _handle_message(client: WebSocketClient, message: dict[str, Any]) -> None:
    """Handle an incoming WebSocket message.

    Args:
        client: The client that sent the message.
        message: The parsed message data.
    """
    msg_type = message.get("type", "")
    data = message.get("data", {})

    if msg_type == MessageType.PING.value:
        await manager.send_message(
            client.client_id,
            MessageType.PONG,
            {"timestamp": time.time()},
        )

    elif msg_type == MessageType.GENERATE.value:
        # Check per-client rate limit (5 requests per minute)
        if not await _check_client_rate_limit(client):
            await manager.send_message(
                client.client_id,
                MessageType.ERROR,
                {"error": "Rate limit exceeded. Maximum 5 generation requests per minute."},
            )
        else:
            # Run generation in background to not block message handling
            task = asyncio.create_task(_handle_generate(client, data, stream=False))
            task.add_done_callback(lambda t: _log_task_exception(t, client.client_id))

    elif msg_type == MessageType.GENERATE_STREAM.value:
        # Check per-client rate limit (5 requests per minute)
        if not await _check_client_rate_limit(client):
            await manager.send_message(
                client.client_id,
                MessageType.ERROR,
                {"error": "Rate limit exceeded. Maximum 5 generation requests per minute."},
            )
        else:
            task = asyncio.create_task(_handle_generate(client, data, stream=True))
            task.add_done_callback(lambda t: _log_task_exception(t, client.client_id))

    elif msg_type == MessageType.SUBSCRIBE_HEALTH.value:
        await manager.set_health_subscription(client.client_id, True)
        await manager.send_message(
            client.client_id,
            MessageType.HEALTH_UPDATE,
            {"subscribed": True},
        )

    elif msg_type == MessageType.UNSUBSCRIBE_HEALTH.value:
        await manager.set_health_subscription(client.client_id, False)

    elif msg_type == MessageType.CANCEL.value:
        # Cancel active generation by clearing the generation ID
        await manager.set_active_generation(client.client_id, None)

    else:
        await manager.send_message(
            client.client_id,
            MessageType.ERROR,
            {"error": f"Unknown message type: {msg_type}"},
        )


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time communication.

    Handles bidirectional communication with clients including:
    - Generation requests (streaming and non-streaming)
    - Health status subscriptions
    - Connection management

    Authentication:
        - Requires auth token via query parameter (?token=<token>) or X-WS-Token header
        - Token is set via JARVIS_WS_TOKEN environment variable or auto-generated
        - Localhost can bypass auth if JARVIS_WS_REQUIRE_AUTH=false

    Message format (JSON):
        {
            "type": "<message_type>",
            "data": { ... }
        }

    Supported message types (client -> server):
        - ping: Keep-alive ping
        - generate: Non-streaming generation
        - generate_stream: Streaming generation (token-by-token)
        - subscribe_health: Subscribe to health updates
        - unsubscribe_health: Unsubscribe from health updates
        - cancel: Cancel active generation

    Server -> client message types:
        - connected: Connection established
        - pong: Response to ping
        - generation_start: Generation started
        - token: Single token (streaming)
        - generation_complete: Generation finished
        - generation_error: Generation failed
        - health_update: Health status update
        - error: Generic error
    """
    # SECURITY: Validate authentication before accepting connection
    if not _validate_websocket_auth(websocket):
        logger.warning(
            "WebSocket connection rejected: invalid or missing auth token from %s",
            websocket.client.host if websocket.client else "unknown",
        )
        await websocket.close(code=1008, reason="Authentication required")
        return

    # Check connection limit before accepting
    if manager.is_at_capacity():
        logger.warning("WebSocket connection rejected: at capacity (%d)", manager.MAX_CONNECTIONS)
        await websocket.close(code=1013, reason="Server at capacity")
        return

    # Ensure cleanup task is running
    await manager.start_cleanup_task()

    client = await manager.connect(websocket)

    # Send connection confirmation
    await manager.send_message(
        client.client_id,
        MessageType.CONNECTED,
        {
            "client_id": client.client_id,
            "timestamp": client.connected_at,
        },
    )

    try:
        while True:
            # Receive message
            try:
                raw_message = await websocket.receive_text()
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                await manager.send_message(
                    client.client_id,
                    MessageType.ERROR,
                    {"error": "Invalid JSON"},
                )
                continue

            # Update activity timestamp on each message
            await manager.update_activity(client.client_id)

            # Handle message
            await _handle_message(client, message)

    except WebSocketDisconnect:
        logger.info("WebSocket client %s disconnected", client.client_id)
    except Exception as e:
        logger.exception("WebSocket error for client %s: %s", client.client_id, e)
    finally:
        await manager.disconnect(client.client_id)


@router.get(
    "/ws/status",
    summary="Get WebSocket connection status",
    response_description="Current WebSocket server status",
    responses={
        200: {
            "description": "WebSocket server status",
            "content": {
                "application/json": {
                    "example": {
                        "active_connections": 5,
                        "health_subscribers": 2,
                        "status": "operational",
                    }
                }
            },
        }
    },
)
def get_websocket_status() -> dict[str, Any]:
    """Get the current WebSocket server status.

    Returns:
        Dictionary with connection counts and status.
    """
    client_ids = manager.get_all_client_ids()
    health_subscribers = sum(
        1 for cid in client_ids if (c := manager.get_client(cid)) and c.subscribed_to_health
    )

    return {
        "active_connections": manager.active_connections,
        "health_subscribers": health_subscribers,
        "status": "operational",
    }
