from __future__ import annotations

import json

import pytest

from jarvis.interfaces.desktop.server import JarvisSocketServer


def _build_server_for_dispatch() -> JarvisSocketServer:
    # Avoid full constructor side effects; only seed fields used by _process_message.
    server = object.__new__(JarvisSocketServer)
    server._methods = {}
    server._streaming_methods = set()
    return server


class _WriterStub:
    def __init__(self) -> None:
        self.buffer = bytearray()

    def write(self, data: bytes) -> None:
        self.buffer.extend(data)

    async def drain(self) -> None:
        return None


@pytest.mark.asyncio
async def test_process_message_dispatches_dict_and_list_params() -> None:
    server = _build_server_for_dispatch()

    async def add(a: int, b: int) -> int:
        return a + b

    async def mul(a: int, b: int) -> int:
        return a * b

    server._methods["add"] = add
    server._methods["mul"] = mul

    add_response = await JarvisSocketServer._process_message(
        server,
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "add", "params": {"a": 2, "b": 3}}),
    )
    mul_response = await JarvisSocketServer._process_message(
        server,
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "mul", "params": [4, 5]}),
    )

    assert json.loads(add_response)["result"] == 5
    assert json.loads(mul_response)["result"] == 20


@pytest.mark.asyncio
async def test_process_message_reports_method_not_found() -> None:
    server = _build_server_for_dispatch()
    response = await JarvisSocketServer._process_message(
        server,
        json.dumps({"jsonrpc": "2.0", "id": 42, "method": "does_not_exist", "params": {}}),
    )
    payload = json.loads(response)
    assert payload["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_process_message_streaming_mode_writes_stream_frames() -> None:
    server = _build_server_for_dispatch()
    server._streaming_methods.add("stream_echo")
    writer = _WriterStub()

    async def stream_echo(
        text: str,
        _writer: _WriterStub,
        _request_id: int,
    ) -> dict[str, str]:
        token_frame = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "stream.token",
                "params": {"token": text, "index": 0, "final": True, "request_id": _request_id},
            }
        ).encode() + b"\n"
        final_frame = json.dumps(
            {"jsonrpc": "2.0", "result": {"response": text}, "id": _request_id}
        ).encode() + b"\n"
        _writer.write(token_frame)
        _writer.write(final_frame)
        await _writer.drain()
        return {"response": text}

    server._methods["stream_echo"] = stream_echo

    response = await JarvisSocketServer._process_message(
        server,
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "stream_echo",
                "params": {"text": "hello", "stream": True},
            }
        ),
        writer=writer,
    )

    assert response is None
    written = writer.buffer.decode("utf-8")
    assert '"method": "stream.token"' in written
    assert '"id": 7' in written
