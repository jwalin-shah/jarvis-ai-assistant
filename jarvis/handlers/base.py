from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from jarvis.socket_server import JarvisSocketServer

logger = logging.getLogger(__name__)

from jarvis.errors import ErrorCode, JarvisError

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
        self.code: int = code
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

    def __init__(self, server: JarvisSocketServer) -> None:
        """Initialize the handler.

        Args:
            server: The JarvisSocketServer instance.
        """
        self.server = server

    def register(self) -> None:
        """Register the handler's methods with the server.

        Subclasses should override this method to call self.server.register()
        for each RPC method they provide.
        """
        raise NotImplementedError("Subclasses must implement register()")
