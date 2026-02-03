"""Tests for the JARVIS socket server."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.socket_server import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    JarvisSocketServer,
    JsonRpcError,
)


class TestJsonRpcError:
    """Tests for JsonRpcError."""

    def test_error_attributes(self) -> None:
        """Error has code, message, and optional data."""
        error = JsonRpcError(-32600, "Invalid request", {"detail": "missing method"})
        assert error.code == -32600
        assert error.message == "Invalid request"
        assert error.data == {"detail": "missing method"}

    def test_error_without_data(self) -> None:
        """Error can be created without data."""
        error = JsonRpcError(-32601, "Method not found")
        assert error.code == -32601
        assert error.message == "Method not found"
        assert error.data is None


class TestJarvisSocketServer:
    """Tests for JarvisSocketServer."""

    @pytest.fixture
    def server(self) -> JarvisSocketServer:
        """Create a server instance without watcher or preloading."""
        return JarvisSocketServer(enable_watcher=False, preload_models=False)

    def test_init(self, server: JarvisSocketServer) -> None:
        """Server initializes with default state."""
        assert server._server is None
        assert server._running is False
        assert len(server._methods) > 0  # Has registered methods

    def test_register_method(self, server: JarvisSocketServer) -> None:
        """Can register custom methods."""

        async def custom_handler(arg: str) -> dict[str, str]:
            return {"result": arg}

        server.register("custom_method", custom_handler)
        assert "custom_method" in server._methods

    @pytest.mark.asyncio
    async def test_ping(self, server: JarvisSocketServer) -> None:
        """Ping method returns ok status with models_ready flag."""
        result = await server._ping()
        assert result["status"] == "ok"
        assert "models_ready" in result

    @pytest.mark.asyncio
    async def test_process_valid_request(self, server: JarvisSocketServer) -> None:
        """Valid JSON-RPC request returns result."""
        message = json.dumps({"jsonrpc": "2.0", "method": "ping", "params": {}, "id": 1})
        response = await server._process_message(message)
        data = json.loads(response)

        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 1
        assert data["result"]["status"] == "ok"
        assert "models_ready" in data["result"]
        assert "error" not in data

    @pytest.mark.asyncio
    async def test_process_invalid_json(self, server: JarvisSocketServer) -> None:
        """Invalid JSON returns parse error."""
        response = await server._process_message("not json")
        data = json.loads(response)

        assert data["error"]["code"] == PARSE_ERROR
        assert "Parse error" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_process_missing_method(self, server: JarvisSocketServer) -> None:
        """Request without method returns invalid request."""
        message = json.dumps({"jsonrpc": "2.0", "params": {}, "id": 1})
        response = await server._process_message(message)
        data = json.loads(response)

        assert data["error"]["code"] == INVALID_REQUEST
        assert "Missing method" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_process_unknown_method(self, server: JarvisSocketServer) -> None:
        """Unknown method returns method not found."""
        message = json.dumps(
            {"jsonrpc": "2.0", "method": "unknown_method", "params": {}, "id": 1}
        )
        response = await server._process_message(message)
        data = json.loads(response)

        assert data["error"]["code"] == METHOD_NOT_FOUND
        assert "Method not found" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_process_invalid_params(self, server: JarvisSocketServer) -> None:
        """Invalid params returns error."""

        async def requires_arg(required_arg: str) -> dict[str, str]:
            return {"arg": required_arg}

        server.register("requires_arg", requires_arg)

        message = json.dumps(
            {"jsonrpc": "2.0", "method": "requires_arg", "params": {}, "id": 1}
        )
        response = await server._process_message(message)
        data = json.loads(response)

        assert data["error"]["code"] == INVALID_PARAMS

    @pytest.mark.asyncio
    async def test_broadcast(self, server: JarvisSocketServer) -> None:
        """Broadcast sends notification to all clients."""
        # Create mock writers
        writer1 = AsyncMock()
        writer2 = AsyncMock()
        server._clients = {writer1, writer2}

        await server.broadcast("new_message", {"text": "hello"})

        # Both writers should receive the notification
        assert writer1.write.called
        assert writer2.write.called

        # Check the notification format
        call_args = writer1.write.call_args[0][0]
        notification = json.loads(call_args.decode().strip())
        assert notification["jsonrpc"] == "2.0"
        assert notification["method"] == "new_message"
        assert notification["params"] == {"text": "hello"}

    @pytest.mark.asyncio
    async def test_broadcast_removes_disconnected_clients(
        self, server: JarvisSocketServer
    ) -> None:
        """Broadcast removes clients that fail to receive."""
        writer1 = AsyncMock()
        writer2 = AsyncMock()
        # Make drain fail to trigger removal
        writer2.drain.side_effect = ConnectionError("disconnected")
        server._clients = {writer1, writer2}

        await server.broadcast("test", {})

        # writer2 should be removed due to drain failure
        assert writer1 in server._clients
        assert writer2 not in server._clients

    def test_success_response(self, server: JarvisSocketServer) -> None:
        """Success response has correct format."""
        response = server._success_response(42, {"data": "value"})
        data = json.loads(response)

        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 42
        assert data["result"] == {"data": "value"}
        assert "error" not in data

    def test_error_response(self, server: JarvisSocketServer) -> None:
        """Error response has correct format."""
        response = server._error_response(42, -32600, "Invalid request", {"extra": "info"})
        data = json.loads(response)

        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 42
        assert data["error"]["code"] == -32600
        assert data["error"]["message"] == "Invalid request"
        assert data["error"]["data"] == {"extra": "info"}
        assert "result" not in data


class TestSocketServerMethods:
    """Tests for socket server RPC methods."""

    @pytest.fixture
    def server(self) -> JarvisSocketServer:
        """Create a server instance."""
        return JarvisSocketServer(enable_watcher=False, preload_models=False)

    @pytest.mark.asyncio
    async def test_classify_intent(self, server: JarvisSocketServer) -> None:
        """Intent classification returns expected structure."""
        with patch("jarvis.intent.get_intent_classifier") as mock_get_classifier:
            mock_classifier = MagicMock()
            mock_result = MagicMock()
            mock_result.intent = MagicMock(value="question")
            mock_result.confidence = 0.85
            mock_result.requires_response = True
            mock_classifier.classify.return_value = mock_result
            mock_get_classifier.return_value = mock_classifier

            result = await server._classify_intent("What time is it?")

            assert result["intent"] == "question"
            assert result["confidence"] == 0.85
            assert result["requires_response"] is True

    @pytest.mark.asyncio
    async def test_classify_intent_error(self, server: JarvisSocketServer) -> None:
        """Intent classification handles errors."""
        with patch("jarvis.intent.get_intent_classifier") as mock_get_classifier:
            mock_get_classifier.side_effect = Exception("Classification failed")

            with pytest.raises(JsonRpcError) as exc_info:
                await server._classify_intent("test")

            assert exc_info.value.code == INTERNAL_ERROR

    @pytest.mark.asyncio
    async def test_get_smart_replies(self, server: JarvisSocketServer) -> None:
        """Smart replies returns suggestions."""
        with patch("jarvis.router.get_reply_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.route.return_value = {
                "type": "generated",
                "response": "Sure, that sounds great!",
                "confidence": "high",
            }
            mock_get_router.return_value = mock_router

            result = await server._get_smart_replies("Want to grab lunch?", 3)

            assert "suggestions" in result
            assert len(result["suggestions"]) >= 1
            assert result["suggestions"][0]["text"] == "Sure, that sounds great!"
            assert result["suggestions"][0]["score"] == 0.9  # high confidence
