from __future__ import annotations

from typing import Any

import pytest

from jarvis.handlers.batch import BatchHandler


class _ServerStub:
    def __init__(self) -> None:
        self._methods: dict[str, Any] = {}

    def register(self, name: str, handler: Any, streaming: bool = False) -> None:
        self._methods[name] = handler

    def get_rpc_handler(self, name: str) -> Any:
        return self._methods.get(name)

    def get_prefetch_manager(self) -> None:
        return None

    def pause_prefetch(self) -> None:
        return None

    def resume_prefetch(self) -> None:
        return None

    @property
    def models_ready(self) -> bool:
        return True

    async def send_stream_token(self, **kwargs: Any) -> None:
        return None

    async def send_stream_response(self, **kwargs: Any) -> None:
        return None


@pytest.mark.asyncio
async def test_batch_supports_list_params() -> None:
    server = _ServerStub()
    handler = BatchHandler(server)  # type: ignore[arg-type]

    async def add(a: int, b: int) -> int:
        return a + b

    server._methods["add"] = add

    result = await handler._batch([{"id": 1, "method": "add", "params": [2, 3]}])
    assert result["results"][0]["result"] == 5


@pytest.mark.asyncio
async def test_batch_supports_dict_params() -> None:
    server = _ServerStub()
    handler = BatchHandler(server)  # type: ignore[arg-type]

    async def concat(left: str, right: str) -> str:
        return left + right

    server._methods["concat"] = concat

    result = await handler._batch(
        [{"id": 2, "method": "concat", "params": {"left": "jar", "right": "vis"}}]
    )
    assert result["results"][0]["result"] == "jarvis"
