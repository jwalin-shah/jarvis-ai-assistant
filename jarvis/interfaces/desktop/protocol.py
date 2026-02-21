from __future__ import annotations

import asyncio
from typing import Any

import orjson

from jarvis.interfaces.desktop.websocket_writer import WebSocketWriter


def success_response(request_id: Any, result: Any) -> str:
    """Build a success response."""
    return orjson.dumps(
        {
            "jsonrpc": "2.0",
            "result": result,
            "id": request_id,
        }
    ).decode("utf-8")


def error_response(
    request_id: Any,
    code: int,
    message: str,
    data: Any = None,
) -> str:
    """Build an error response."""
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data

    return orjson.dumps(
        {
            "jsonrpc": "2.0",
            "error": error,
            "id": request_id,
        }
    ).decode("utf-8")


async def send_stream_token(
    writer: asyncio.StreamWriter | WebSocketWriter,
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
    notification = orjson.dumps(
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
    if isinstance(writer, WebSocketWriter):
        writer.write(notification)
        await writer.drain()
    else:
        writer.write(notification + b"\n")
        await writer.drain()


async def send_stream_response(
    writer: asyncio.StreamWriter | WebSocketWriter,
    request_id: Any,
    result: dict[str, Any],
) -> None:
    """Send the final response after streaming completes.

    Args:
        writer: Client stream writer
        request_id: Original request ID
        result: Final result data
    """
    response = success_response(request_id, result)
    if isinstance(writer, WebSocketWriter):
        writer.write(response.encode())
        await writer.drain()
    else:
        writer.write(response.encode() + b"\n")
        await writer.drain()


async def send_stream_error(
    writer: asyncio.StreamWriter | WebSocketWriter,
    request_id: Any,
    code: int,
    message: str,
    data: Any = None,
) -> None:
    """Send an error notification during streaming.

    Args:
        writer: Client stream writer
        request_id: Original request ID
        code: Error code
        message: Error message
        data: Optional error data
    """
    notification = orjson.dumps(
        {
            "jsonrpc": "2.0",
            "method": "stream.error",
            "params": {
                "request_id": request_id,
                "code": code,
                "message": message,
                "data": data,
            },
        }
    )
    if isinstance(writer, WebSocketWriter):
        writer.write(notification)
        await writer.drain()
    else:
        writer.write(notification + b"\n")
        await writer.drain()
