"""JSON-RPC handlers for the JARVIS socket server.

Handlers are organized by domain (messages, contacts, health, etc.)
and registered with the socket server.
"""

from jarvis.handlers.base import BaseHandler, JsonRpcError, rpc_handler
from jarvis.handlers.batch import BatchHandler
from jarvis.handlers.contact import ContactHandler
from jarvis.handlers.health import HealthHandler
from jarvis.handlers.message import MessageHandler
from jarvis.handlers.metrics import MetricsHandler
from jarvis.handlers.prefetch import PrefetchHandler
from jarvis.handlers.search import SearchHandler

__all__ = [
    "BaseHandler",
    "JsonRpcError",
    "rpc_handler",
    "BatchHandler",
    "ContactHandler",
    "HealthHandler",
    "MessageHandler",
    "MetricsHandler",
    "PrefetchHandler",
    "SearchHandler",
]
