# Facade Retirement Migration Map

> Last Updated: 2026-02-16

The legacy facade modules were removed:

- `jarvis.errors`
- `jarvis.cache`
- `jarvis.router`
- `jarvis.socket_server`

Use these canonical imports instead.

## Import Mapping

- `from jarvis.errors import ...`
  - `from jarvis.core.exceptions import ...`

- `from jarvis.cache import TTLCache`
  - `from jarvis.infrastructure.cache import TTLCache`

- `from jarvis.router import get_reply_router`
  - `from jarvis.reply_service import get_reply_service`
  - then call `get_reply_service().route_legacy(...)` where legacy dict output is required

- `from jarvis.socket_server import JarvisSocketServer`
  - `from jarvis.interfaces.desktop.server import JarvisSocketServer`

- `from jarvis.socket_server import WebSocketWriter`
  - `from jarvis.interfaces.desktop.protocol import WebSocketWriter`

- `from jarvis.socket_server import RateLimiter`
  - `from jarvis.interfaces.desktop.limiter import RateLimiter`

- JSON-RPC constants and errors formerly from `jarvis.socket_server`
  - `from jarvis.handlers.base import JsonRpcError, PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR`

## Socket Runtime Entrypoint

- Old: `python -m jarvis.socket_server`
- New: `python -m jarvis.interfaces.desktop`
