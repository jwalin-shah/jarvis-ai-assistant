"""TEST-06: Socket server streaming response tests.

Verifies:
1. Token ordering is preserved during streaming
2. Errors during streaming are handled gracefully
3. WebSocketWriter buffers and flushes correctly
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestStreamTokenOrdering:
    """Verify token ordering is preserved in streaming."""

    @pytest.mark.asyncio
    async def test_send_stream_token_preserves_order(self):
        """Tokens are sent with correct indices and in order."""
        from jarvis.interfaces.desktop.server import JarvisSocketServer

        server = JarvisSocketServer(
            enable_watcher=False,
            preload_models=False,
            enable_prefetch=False,
        )

        # Create a mock writer that records what's written
        written_data: list[bytes] = []
        writer = AsyncMock()
        writer.write = MagicMock(side_effect=lambda data: written_data.append(data))
        writer.drain = AsyncMock()

        # Send tokens in order
        tokens = ["Hello", " ", "world", "!"]
        for i, token in enumerate(tokens):
            await server._send_stream_token(
                writer,
                token=token,
                token_index=i,
                is_final=(i == len(tokens) - 1),
                request_id=42,
            )

        assert len(written_data) == len(tokens)

        # Parse each notification and verify ordering
        for i, data in enumerate(written_data):
            msg = json.loads(data.decode().strip())
            assert msg["method"] == "stream.token"
            assert msg["params"]["token"] == tokens[i]
            assert msg["params"]["index"] == i
            assert msg["params"]["request_id"] == 42

        # Verify last token is marked final
        last_msg = json.loads(written_data[-1].decode().strip())
        assert last_msg["params"]["final"] is True

        # Non-final tokens should not be marked final
        first_msg = json.loads(written_data[0].decode().strip())
        assert first_msg["params"]["final"] is False

    @pytest.mark.asyncio
    async def test_send_stream_response_after_tokens(self):
        """Final response is sent after all stream tokens."""
        from jarvis.interfaces.desktop.server import JarvisSocketServer

        server = JarvisSocketServer(
            enable_watcher=False,
            preload_models=False,
            enable_prefetch=False,
        )

        written_data: list[bytes] = []
        writer = AsyncMock()
        writer.write = MagicMock(side_effect=lambda data: written_data.append(data))
        writer.drain = AsyncMock()

        # Send a token then the final response
        await server._send_stream_token(writer, "test", 0, False, request_id=1)
        await server._send_stream_response(
            writer,
            request_id=1,
            result={"text": "test", "streamed": True},
        )

        assert len(written_data) == 2

        # First is a token notification
        token_msg = json.loads(written_data[0].decode().strip())
        assert token_msg["method"] == "stream.token"

        # Second is the final JSON-RPC response
        final_msg = json.loads(written_data[1].decode().strip())
        assert final_msg["id"] == 1
        assert final_msg["result"]["streamed"] is True


class TestStreamingErrorHandling:
    """Verify errors during streaming are handled properly."""

    @pytest.mark.asyncio
    async def test_process_message_returns_error_for_invalid_json(self):
        """Invalid JSON returns parse error."""
        from jarvis.handlers.base import PARSE_ERROR
        from jarvis.interfaces.desktop.server import JarvisSocketServer

        server = JarvisSocketServer(
            enable_watcher=False,
            preload_models=False,
            enable_prefetch=False,
        )

        result = await server._process_message("not valid json")
        assert result is not None
        parsed = json.loads(result)
        assert parsed["error"]["code"] == PARSE_ERROR

    @pytest.mark.asyncio
    async def test_process_message_returns_error_for_unknown_method(self):
        """Unknown method returns method_not_found error."""
        from jarvis.handlers.base import METHOD_NOT_FOUND
        from jarvis.interfaces.desktop.server import JarvisSocketServer

        server = JarvisSocketServer(
            enable_watcher=False,
            preload_models=False,
            enable_prefetch=False,
        )

        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "nonexistent_method",
                "params": {},
                "id": 1,
            }
        )
        result = await server._process_message(msg)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["error"]["code"] == METHOD_NOT_FOUND

    @pytest.mark.asyncio
    async def test_process_message_missing_method(self):
        """Missing method field returns invalid request error."""
        from jarvis.handlers.base import INVALID_REQUEST
        from jarvis.interfaces.desktop.server import JarvisSocketServer

        server = JarvisSocketServer(
            enable_watcher=False,
            preload_models=False,
            enable_prefetch=False,
        )

        msg = json.dumps({"jsonrpc": "2.0", "params": {}, "id": 1})
        result = await server._process_message(msg)
        parsed = json.loads(result)
        assert parsed["error"]["code"] == INVALID_REQUEST


class TestWebSocketWriter:
    """Test WebSocketWriter buffering and flushing."""

    @pytest.mark.asyncio
    async def test_write_buffers_data(self):
        """write() buffers data instead of sending immediately."""
        from jarvis.interfaces.desktop.websocket_writer import WebSocketWriter

        mock_ws = AsyncMock()
        writer = WebSocketWriter(mock_ws)

        writer.write(b"hello ")
        writer.write(b"world")

        # Should not have sent yet
        mock_ws.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_drain_sends_combined_data(self):
        """drain() combines buffered data and sends once."""
        from jarvis.interfaces.desktop.websocket_writer import WebSocketWriter

        mock_ws = AsyncMock()
        writer = WebSocketWriter(mock_ws)

        writer.write(b"hello ")
        writer.write(b"world")
        await writer.drain()

        mock_ws.send.assert_called_once_with("hello world")

    @pytest.mark.asyncio
    async def test_drain_clears_buffer(self):
        """drain() clears the buffer after sending."""
        from jarvis.interfaces.desktop.websocket_writer import WebSocketWriter

        mock_ws = AsyncMock()
        writer = WebSocketWriter(mock_ws)

        writer.write(b"first")
        await writer.drain()

        writer.write(b"second")
        await writer.drain()

        assert mock_ws.send.call_count == 2
        calls = mock_ws.send.call_args_list
        assert calls[0][0][0] == "first"
        assert calls[1][0][0] == "second"

    @pytest.mark.asyncio
    async def test_drain_noop_when_empty(self):
        """drain() with empty buffer does nothing."""
        from jarvis.interfaces.desktop.websocket_writer import WebSocketWriter

        mock_ws = AsyncMock()
        writer = WebSocketWriter(mock_ws)

        await writer.drain()
        mock_ws.send.assert_not_called()
