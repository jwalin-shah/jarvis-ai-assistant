from pathlib import Path

# Socket configuration
SOCKET_PATH = Path.home() / ".jarvis" / "jarvis.sock"
WS_TOKEN_PATH = Path.home() / ".jarvis" / "ws_token"
WEBSOCKET_PORT = 8743
MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB max message size
MAX_WS_CONNECTIONS = 10
