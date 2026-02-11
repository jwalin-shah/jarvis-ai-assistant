"""JARVIS Reliability Framework - Offline-mode and resilience utilities.

This module provides utilities for building resilient, offline-capable features:

- OfflineSyncQueue: Queue operations for sync when back online
- OfflineManager: Manage offline/online state transitions
- ResilientClient: HTTP client with automatic retry and fallback
- ConnectivityMonitor: Monitor network and service connectivity

Example:
    >>> from jarvis.reliability import OfflineSyncQueue, SyncOp
    >>> queue = OfflineSyncQueue()
    >>> queue.enqueue(SyncOp.MESSAGE_SEND, {"text": "Hello", "chat_id": "123"})
    >>> # When back online:
    >>> await queue.process_pending()
"""

from __future__ import annotations

from jarvis.reliability.connectivity import (
    ConnectivityMonitor,
    ConnectivityState,
    ServiceHealth,
)
from jarvis.reliability.offline import (
    OfflineManager,
    OfflineSyncQueue,
    SyncOp,
    SyncPriority,
    SyncResult,
    SyncStatus,
)
from jarvis.reliability.resilient_client import (
    FallbackStrategy,
    ResilientClient,
    ResilientClientConfig,
)

__all__ = [
    # Offline management
    "OfflineManager",
    "OfflineSyncQueue",
    "SyncOp",
    "SyncPriority",
    "SyncResult",
    "SyncStatus",
    # Connectivity monitoring
    "ConnectivityMonitor",
    "ConnectivityState",
    "ServiceHealth",
    # Resilient client
    "ResilientClient",
    "ResilientClientConfig",
    "FallbackStrategy",
]
