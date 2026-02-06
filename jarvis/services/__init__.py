"""JARVIS Service Management System.

Provides unified management for all JARVIS services with consistent
lifecycle management, health checks, and dependency handling.
"""

from .api import FastAPIService
from .base import Service, ServiceStatus
from .context_service import ContextService
from .embedding import EmbeddingService
from .manager import ServiceManager
from .ner import NERService
from .socket import SocketService

__all__ = [
    "ServiceManager",
    "Service",
    "ServiceStatus",
    "FastAPIService",
    "SocketService",
    "EmbeddingService",
    "NERService",
    "ContextService",
]
