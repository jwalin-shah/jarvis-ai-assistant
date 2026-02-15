"""Tests for the JARVIS socket server."""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.socket_server import (
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
    @pytest.mark.timeout(10)
    async def test_ping(self, server: JarvisSocketServer) -> None:
        """Ping method returns valid health status with models_ready flag."""
        result = await server._ping()
        assert result["status"] in ("healthy", "degraded", "unhealthy")
        assert "models_ready" in result

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_process_valid_request(self, server: JarvisSocketServer) -> None:
        """Valid JSON-RPC request returns result."""
        message = json.dumps({"jsonrpc": "2.0", "method": "ping", "params": {}, "id": 1})
        response = await server._process_message(message)
        data = json.loads(response)

        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 1
        assert data["result"]["status"] in ("healthy", "degraded", "unhealthy")
        assert "models_ready" in data["result"]
        assert "error" not in data

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_process_invalid_json(self, server: JarvisSocketServer) -> None:
        """Invalid JSON returns parse error."""
        response = await server._process_message("not json")
        data = json.loads(response)

        assert data["error"]["code"] == PARSE_ERROR
        assert "Parse error" in data["error"]["message"]

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_process_missing_method(self, server: JarvisSocketServer) -> None:
        """Request without method returns invalid request."""
        message = json.dumps({"jsonrpc": "2.0", "params": {}, "id": 1})
        response = await server._process_message(message)
        data = json.loads(response)

        assert data["error"]["code"] == INVALID_REQUEST
        assert "Missing method" in data["error"]["message"]

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_process_unknown_method(self, server: JarvisSocketServer) -> None:
        """Unknown method returns method not found."""
        message = json.dumps({"jsonrpc": "2.0", "method": "unknown_method", "params": {}, "id": 1})
        response = await server._process_message(message)
        data = json.loads(response)

        assert data["error"]["code"] == METHOD_NOT_FOUND
        assert "Method not found" in data["error"]["message"]

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_process_invalid_params(self, server: JarvisSocketServer) -> None:
        """Invalid params returns error."""

        async def requires_arg(required_arg: str) -> dict[str, str]:
            return {"arg": required_arg}

        server.register("requires_arg", requires_arg)

        message = json.dumps({"jsonrpc": "2.0", "method": "requires_arg", "params": {}, "id": 1})
        response = await server._process_message(message)
        data = json.loads(response)

        assert data["error"]["code"] == INVALID_PARAMS

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
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
    @pytest.mark.timeout(10)
    async def test_broadcast_removes_disconnected_clients(self, server: JarvisSocketServer) -> None:
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

class TestRotateWsToken:
    """Tests for WebSocket token rotation."""

    def _make_server(self):
        with patch.multiple(
            "jarvis.socket_server",
            WS_TOKEN_PATH=MagicMock(),
        ):
            server = MagicMock(spec=JarvisSocketServer)
            server._ws_auth_token = "old-token"
            server._token_created_at = time.monotonic() - 100
            server._previous_ws_auth_token = None
            server._previous_token_expired_at = 0.0
            server._rotate_ws_token = JarvisSocketServer._rotate_ws_token.__get__(server)
            server._verify_ws_token = JarvisSocketServer._verify_ws_token.__get__(server)
            return server

    def _patch_token_write(self):
        """Patch os.open/os.fdopen used for atomic token file writes."""
        MagicMock()
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        return patch("jarvis.socket_server.os.open", return_value=42), patch(
            "jarvis.socket_server.os.fdopen", return_value=mock_file
        )

    def test_generates_new_token(self):
        server = self._make_server()
        p1, p2 = self._patch_token_write()
        with p1, p2:
            server._rotate_ws_token()
        assert server._ws_auth_token != "old-token"
        assert len(server._ws_auth_token) > 0

    def test_saves_previous_token(self):
        server = self._make_server()
        p1, p2 = self._patch_token_write()
        with p1, p2:
            server._rotate_ws_token()
        assert server._previous_ws_auth_token == "old-token"

    def test_sets_grace_period(self):
        server = self._make_server()
        p1, p2 = self._patch_token_write()
        with p1, p2:
            server._rotate_ws_token()
        # Grace period should be ~60s from now
        remaining = server._previous_token_expired_at - time.monotonic()
        assert 55 < remaining < 65


class TestVerifyWsToken:
    """Tests for WebSocket token verification."""

    def _make_server(self):
        server = MagicMock(spec=JarvisSocketServer)
        server._ws_auth_token = "current-token"
        server._previous_ws_auth_token = "old-token"
        server._previous_token_expired_at = time.monotonic() + 60.0
        server._verify_ws_token = JarvisSocketServer._verify_ws_token.__get__(server)
        return server

    def test_accepts_current_token(self):
        server = self._make_server()
        assert server._verify_ws_token("current-token") is True

    def test_accepts_previous_within_grace(self):
        server = self._make_server()
        assert server._verify_ws_token("old-token") is True

    def test_rejects_previous_after_grace(self):
        server = self._make_server()
        server._previous_token_expired_at = time.monotonic() - 1.0
        assert server._verify_ws_token("old-token") is False

    def test_rejects_invalid_token(self):
        server = self._make_server()
        assert server._verify_ws_token("wrong-token") is False

    def test_rejects_when_no_tokens(self):
        server = self._make_server()
        server._ws_auth_token = None
        server._previous_ws_auth_token = None
        assert server._verify_ws_token("any-token") is False
