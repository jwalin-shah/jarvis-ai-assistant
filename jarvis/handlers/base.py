from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from jarvis.core.exceptions import ErrorCode, JarvisError

if TYPE_CHECKING:
    from collections.abc import Awaitable


class PrefetchManagerProtocol(Protocol):
    def get_draft(self, chat_id: str) -> dict[str, Any] | None: ...
    def stats(self) -> dict[str, Any]: ...
    def invalidate(self, chat_id: str | None = None) -> int: ...
    def on_focus(self, chat_id: str) -> None: ...
    def on_hover(self, chat_id: str) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...


class ContextServiceProtocol(Protocol):
    def fetch_conversation_context(
        self,
        chat_id: str,
        limit: int = 20,
    ) -> tuple[list[str], set[str]]: ...


class ReplyServiceProtocol(Protocol):
    @property
    def context_service(self) -> ContextServiceProtocol: ...

    @property
    def generator(self) -> Any: ...

    def prepare_streaming_context(
        self,
        incoming: str,
        thread: list[str] | None = None,
        chat_id: str | None = None,
        instruction: str | None = None,
    ) -> tuple[Any, dict[str, Any]]: ...

    def route_legacy(
        self,
        incoming: str,
        contact_id: int | None = None,
        thread: list[str] | None = None,
        chat_id: str | None = None,
        conversation_messages: list[Any] | None = None,
        context: Any | None = None,
    ) -> dict[str, Any]: ...


class HandlerServerProtocol(Protocol):
    @property
    def models_ready(self) -> bool: ...

    def register(
        self,
        name: str,
        handler: Callable[..., Coroutine[Any, Any, Any]],
        streaming: bool = False,
    ) -> None: ...

    def get_rpc_handler(self, name: str) -> Callable[..., Awaitable[Any]] | None: ...
    def get_prefetch_manager(self) -> PrefetchManagerProtocol | None: ...
    def get_reply_service(self) -> ReplyServiceProtocol: ...
    def pause_prefetch(self) -> None: ...
    def resume_prefetch(self) -> None: ...
    def set_focused_chat(self, chat_id: str | None) -> None: ...
    def is_generation_stale(self, chat_id: str) -> bool: ...

    async def send_stream_token(
        self,
        writer: Any,
        token: str,
        token_index: int,
        is_final: bool = False,
        request_id: Any = None,
    ) -> None: ...

    async def send_stream_response(
        self,
        writer: Any,
        request_id: Any,
        result: dict[str, Any],
    ) -> None: ...


logger = logging.getLogger(__name__)

# Standard JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# Type var for async RPC handler functions
_AsyncRpcHandler = TypeVar("_AsyncRpcHandler", bound=Callable[..., Coroutine[Any, Any, Any]])


class JsonRpcError(JarvisError):
    """JSON-RPC error with code and data."""

    def __init__(
        self,
        code: int,
        message: str,
        data: Any | None = None,
    ) -> None:
        # We store the JSON-RPC integer code in self.code to keep tests passing
        # and use JarvisError's code for the ErrorCode enum (set to UNKNOWN here).
        super().__init__(message, code=ErrorCode.UNKNOWN)
        self.code: int = code  # type: ignore[assignment]
        self.data: Any | None = data


def rpc_handler(error_msg: str) -> Callable[[_AsyncRpcHandler], _AsyncRpcHandler]:
    """Decorator that wraps async RPC handlers with standard error handling.

    Catches exceptions and converts them to JsonRpcError with a consistent
    pattern, letting JsonRpcError pass through unmodified.

    Args:
        error_msg: User-facing error message for unexpected exceptions.
    """

    def decorator(fn: _AsyncRpcHandler) -> _AsyncRpcHandler:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except JsonRpcError:
                raise
            except Exception as e:
                logger.exception("Error in %s", fn.__name__)
                raise JsonRpcError(INTERNAL_ERROR, error_msg) from e

        return wrapper  # type: ignore[return-value]

    return decorator


class BaseHandler:
    """Base class for all RPC handlers.

    Handlers are domain-specific modules that provide RPC methods for the
    JarvisSocketServer. This helps keep the socket server code clean and
    modular.
    """

    def __init__(self, server: HandlerServerProtocol) -> None:
        """Initialize the handler.

        Args:
            server: The JarvisSocketServer instance.
        """
        self.server: HandlerServerProtocol = server

    def register(self) -> None:
        """Register the handler's methods with the server.

        Subclasses should override this method to call self.server.register()
        for each RPC method they provide.
        """
        raise NotImplementedError("Subclasses must implement register()")

    async def send_stream_token(
        self,
        writer: Any,
        token: str,
        token_index: int,
        is_final: bool = False,
        request_id: Any = None,
    ) -> None:
        """Helper to send a streaming token to the client."""
        await self.server.send_stream_token(
            writer=writer,
            token=token,
            token_index=token_index,
            is_final=is_final,
            request_id=request_id,
        )

    async def send_stream_response(
        self,
        writer: Any,
        request_id: Any,
        result: dict[str, Any],
    ) -> None:
        """Helper to send the final streaming response to the client."""
        await self.server.send_stream_response(
            writer=writer,
            request_id=request_id,
            result=result,
        )
