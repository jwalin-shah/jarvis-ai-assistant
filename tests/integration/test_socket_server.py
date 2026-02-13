"""Integration tests for the JARVIS socket server.

Tests the JSON-RPC protocol, method handling, batching, and error cases.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.socket_server import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    JarvisSocketServer,
    JsonRpcError,
    WebSocketWriter,
)


class TestJsonRpcProtocol:
    """Tests for JSON-RPC protocol handling."""

    @pytest.fixture
    def server(self):
        """Create a socket server instance for testing."""
        return JarvisSocketServer(
            enable_watcher=False,
            preload_models=False,
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_parse_error_on_invalid_json(self, server):
        """Invalid JSON returns parse error."""
        response = await server._process_message("not valid json")
        data = json.loads(response)

        assert data["jsonrpc"] == "2.0"
        assert "error" in data
        assert data["error"]["code"] == PARSE_ERROR
        assert "Parse error" in data["error"]["message"]

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_invalid_request_when_not_dict(self, server):
        """Non-dict request returns invalid request error."""
        response = await server._process_message('"just a string"')
        data = json.loads(response)

        assert data["error"]["code"] == INVALID_REQUEST
        assert "Invalid request" in data["error"]["message"]

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_invalid_request_when_missing_method(self, server):
        """Missing method returns invalid request error."""
        response = await server._process_message('{"jsonrpc": "2.0", "id": 1}')
        data = json.loads(response)

        assert data["error"]["code"] == INVALID_REQUEST
        assert "Missing method" in data["error"]["message"]

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_method_not_found(self, server):
        """Unknown method returns method not found error."""
        request = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "nonexistent_method",
                "id": 1,
            }
        )
        response = await server._process_message(request)
        data = json.loads(response)

        assert data["error"]["code"] == METHOD_NOT_FOUND
        assert "Method not found" in data["error"]["message"]
        assert "nonexistent_method" in data["error"]["message"]

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_success_response_format(self, server):
        """Successful request returns properly formatted response."""
        request = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "ping",
                "id": 42,
            }
        )
        response = await server._process_message(request)
        data = json.loads(response)

        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 42
        assert "result" in data
        assert data["result"]["status"] in ("healthy", "degraded", "unhealthy")

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_request_id_preserved(self, server):
        """Request ID is preserved in response."""
        request = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "ping",
                "id": "unique-id-123",
            }
        )
        response = await server._process_message(request)
        data = json.loads(response)

        assert data["id"] == "unique-id-123"


class TestPingMethod:
    """Tests for the ping health check method."""

    @pytest.fixture
    def server(self):
        """Create a socket server instance for testing."""
        return JarvisSocketServer(
            enable_watcher=False,
            preload_models=False,
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_ping_returns_status(self, server):
        """Ping returns valid health status."""
        result = await server._ping()

        assert result["status"] in ("healthy", "degraded", "unhealthy")
        assert "models_ready" in result

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_ping_models_ready_initially_false(self, server):
        """Ping shows models_ready as False initially."""
        result = await server._ping()

        assert result["models_ready"] is False

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_ping_via_rpc(self, server):
        """Ping works via JSON-RPC."""
        request = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "ping",
                "params": {},
                "id": 1,
            }
        )
        response = await server._process_message(request)
        data = json.loads(response)

        assert data["result"]["status"] in ("healthy", "degraded", "unhealthy")


class TestBatchOperations:
    """Tests for batch RPC operations."""

    @pytest.fixture
    def server(self):
        """Create a socket server instance for testing."""
        return JarvisSocketServer(
            enable_watcher=False,
            preload_models=False,
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_batch_empty_requests(self, server):
        """Empty batch returns empty results."""
        result = await server._batch(requests=[])

        assert result == {"results": []}

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_batch_multiple_pings(self, server):
        """Batch multiple ping requests."""
        result = await server._batch(
            requests=[
                {"method": "ping", "id": 1},
                {"method": "ping", "id": 2},
                {"method": "ping", "id": 3},
            ]
        )

        assert len(result["results"]) == 3
        for r in result["results"]:
            assert "result" in r
            assert r["result"]["status"] in ("healthy", "degraded", "unhealthy")

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_batch_preserves_ids(self, server):
        """Batch preserves request IDs."""
        result = await server._batch(
            requests=[
                {"method": "ping", "id": "first"},
                {"method": "ping", "id": "second"},
            ]
        )

        ids = {r["id"] for r in result["results"]}
        assert ids == {"first", "second"}

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_batch_handles_errors(self, server):
        """Batch handles individual errors."""
        result = await server._batch(
            requests=[
                {"method": "ping", "id": 1},
                {"method": "nonexistent", "id": 2},
                {"method": "ping", "id": 3},
            ]
        )

        assert len(result["results"]) == 3
        # First and third should succeed
        assert "result" in result["results"][0]
        assert "result" in result["results"][2]
        # Second should fail
        assert "error" in result["results"][1]
        assert result["results"][1]["error"]["code"] == METHOD_NOT_FOUND

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_batch_missing_method_error(self, server):
        """Batch returns error for missing method."""
        result = await server._batch(
            requests=[
                {"id": 1},  # Missing method
            ]
        )

        assert len(result["results"]) == 1
        assert "error" in result["results"][0]
        assert result["results"][0]["error"]["code"] == INVALID_REQUEST

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_batch_max_requests_limit(self, server):
        """Batch rejects more than 50 requests."""
        with pytest.raises(JsonRpcError) as exc_info:
            await server._batch(requests=[{"method": "ping"}] * 51)

        assert exc_info.value.code == INVALID_PARAMS
        assert "50" in exc_info.value.message

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_batch_runs_in_parallel(self, server):
        """Batch executes requests in parallel."""
        # Track order of execution
        execution_order = []

        async def slow_handler():
            execution_order.append("start")
            await asyncio.sleep(0.05)
            execution_order.append("end")
            return {"ok": True}

        server.register("slow", slow_handler)

        result = await server._batch(
            requests=[
                {"method": "slow", "id": 1},
                {"method": "slow", "id": 2},
            ]
        )

        assert len(result["results"]) == 2
        # If parallel, we'd see start, start, end, end
        # If sequential, we'd see start, end, start, end
        # Verify parallel execution: both handlers started before either finished
        assert execution_order[0] == "start"
        assert execution_order[1] == "start"
        assert execution_order.count("start") == 2
        assert execution_order.count("end") == 2


class TestMethodRegistration:
    """Tests for custom method registration."""

    @pytest.fixture
    def server(self):
        """Create a socket server instance for testing."""
        return JarvisSocketServer(
            enable_watcher=False,
            preload_models=False,
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_register_custom_method(self, server):
        """Register and call a custom method."""

        async def custom_handler(name: str = "World"):
            return {"greeting": f"Hello, {name}!"}

        server.register("greet", custom_handler)

        request = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "greet",
                "params": {"name": "Alice"},
                "id": 1,
            }
        )
        response = await server._process_message(request)
        data = json.loads(response)

        assert data["result"]["greeting"] == "Hello, Alice!"

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_register_streaming_method(self, server):
        """Register a streaming method."""

        async def stream_handler(
            data: str,
            _writer: Any = None,
            _request_id: Any = None,
        ):
            return {"streamed": True, "data": data}

        server.register("stream_test", stream_handler, streaming=True)

        assert "stream_test" in server._streaming_methods

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_method_invalid_params_error(self, server):
        """Invalid params returns appropriate error."""

        async def handler(required_param: str):
            return {"value": required_param}

        server.register("needs_param", handler)

        request = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "needs_param",
                "params": {},  # Missing required param
                "id": 1,
            }
        )
        response = await server._process_message(request)
        data = json.loads(response)

        assert data["error"]["code"] == INVALID_PARAMS

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_method_internal_error(self, server):
        """Handler exception returns internal error."""

        async def failing_handler():
            raise RuntimeError("Something went wrong")

        server.register("failing", failing_handler)

        request = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "failing",
                "id": 1,
            }
        )
        response = await server._process_message(request)
        data = json.loads(response)

        assert data["error"]["code"] == INTERNAL_ERROR
        assert "Internal server error" in data["error"]["message"]


class TestWebSocketWriter:
    """Tests for the WebSocket writer wrapper."""

    def test_websocket_writer_buffers_data(self):
        """WebSocketWriter buffers written data."""
        mock_ws = MagicMock()
        writer = WebSocketWriter(mock_ws)

        writer.write(b"Hello ")
        writer.write(b"World")

        assert writer._parts == [b"Hello ", b"World"]

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_websocket_writer_drain_sends_data(self):
        """WebSocketWriter sends data on drain."""
        mock_ws = AsyncMock()
        writer = WebSocketWriter(mock_ws)

        writer.write(b"Test message\n")
        await writer.drain()

        mock_ws.send.assert_called_once_with("Test message")
        assert writer._parts == []

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_websocket_writer_drain_empty_buffer(self):
        """WebSocketWriter drain with empty buffer is no-op."""
        mock_ws = AsyncMock()
        writer = WebSocketWriter(mock_ws)

        await writer.drain()

        mock_ws.send.assert_not_called()


class TestBroadcast:
    """Tests for broadcast functionality."""

    @pytest.fixture
    def server(self):
        """Create a socket server instance for testing."""
        return JarvisSocketServer(
            enable_watcher=False,
            preload_models=False,
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_broadcast_to_no_clients(self, server):
        """Broadcast with no clients does nothing."""
        # Should not raise
        await server.broadcast("test_event", {"data": "value"})

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_broadcast_format(self, server):
        """Broadcast creates properly formatted notification."""
        # Add a mock client
        mock_writer = AsyncMock()
        mock_writer.write = MagicMock()
        server._clients.add(mock_writer)

        await server.broadcast("new_message", {"text": "Hello"})

        # Check what was written
        mock_writer.write.assert_called_once()
        written_data = mock_writer.write.call_args[0][0].decode()
        notification = json.loads(written_data.strip())

        assert notification["jsonrpc"] == "2.0"
        assert notification["method"] == "new_message"
        assert notification["params"]["text"] == "Hello"
        assert "id" not in notification  # Notifications have no id


class TestStreamingSupport:
    """Tests for streaming response support."""

    @pytest.fixture
    def server(self):
        """Create a socket server instance for testing."""
        return JarvisSocketServer(
            enable_watcher=False,
            preload_models=False,
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_send_stream_token(self, server):
        """Send stream token creates proper notification."""
        mock_writer = AsyncMock()
        mock_writer.write = MagicMock()

        await server._send_stream_token(mock_writer, "Hello", 0, False)

        mock_writer.write.assert_called_once()
        written_data = mock_writer.write.call_args[0][0].decode()
        notification = json.loads(written_data.strip())

        assert notification["method"] == "stream.token"
        assert notification["params"]["token"] == "Hello"
        assert notification["params"]["index"] == 0
        assert notification["params"]["final"] is False

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_send_stream_response(self, server):
        """Send stream response creates proper response."""
        mock_writer = AsyncMock()
        mock_writer.write = MagicMock()

        await server._send_stream_response(
            mock_writer,
            request_id=123,
            result={"text": "Complete response"},
        )

        mock_writer.write.assert_called_once()
        written_data = mock_writer.write.call_args[0][0].decode()
        response = json.loads(written_data.strip())

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 123
        assert response["result"]["text"] == "Complete response"

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_streaming_removes_stream_param(self, server):
        """Stream param is removed before passing to handler."""
        received_params = []

        async def tracking_handler(**kwargs):
            received_params.append(kwargs)
            return {"ok": True}

        server.register("track", tracking_handler)

        request = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "track",
                "params": {"stream": True, "data": "test"},
                "id": 1,
            }
        )
        await server._process_message(request)

        # stream param should be removed
        assert "stream" not in received_params[0]
        assert received_params[0]["data"] == "test"


class TestServerLifecycle:
    """Tests for server start/stop lifecycle."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_server_creates_with_watcher_disabled(self):
        """Server can be created with watcher disabled."""
        server = JarvisSocketServer(
            enable_watcher=False,
            preload_models=False,
        )

        assert server._enable_watcher is False
        assert server._preload_models is False
        assert server._running is False

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_server_stop_clears_state(self):
        """Server stop clears all state."""
        server = JarvisSocketServer(
            enable_watcher=False,
            preload_models=False,
        )

        # Add some mock clients
        mock_writer = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()
        server._clients.add(mock_writer)

        mock_ws = AsyncMock()
        server._ws_clients.add(mock_ws)

        server._running = True

        await server.stop()

        assert server._running is False
        assert len(server._clients) == 0
        assert len(server._ws_clients) == 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_wait_for_models_returns_immediately_if_ready(self):
        """wait_for_models returns True if models already ready."""
        server = JarvisSocketServer(
            enable_watcher=False,
            preload_models=False,
        )
        server._models_ready = True

        result = await server.wait_for_models(timeout=0.1)

        assert result is True

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_wait_for_models_timeout(self):
        """wait_for_models returns False on timeout."""
        server = JarvisSocketServer(
            enable_watcher=False,
            preload_models=False,
        )
        server._models_ready = False
        # Event is not set, so it will timeout

        result = await server.wait_for_models(timeout=0.1)

        assert result is False
