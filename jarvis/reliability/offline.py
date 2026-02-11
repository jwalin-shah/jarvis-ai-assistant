"""Offline-mode support and synchronization queue.

Provides offline-first capabilities with automatic synchronization
when connectivity is restored.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SyncOp(Enum):
    """Types of syncable operations."""

    MESSAGE_SEND = "message_send"
    MESSAGE_READ = "message_read"
    DRAFT_SAVE = "draft_save"
    SETTING_UPDATE = "setting_update"
    ANALYTICS_EVENT = "analytics_event"
    FEEDBACK_SUBMIT = "feedback_submit"


class SyncPriority(Enum):
    """Priority levels for sync operations."""

    CRITICAL = 0  # User-initiated, must sync
    HIGH = 1  # Important state changes
    NORMAL = 2  # Regular updates
    LOW = 3  # Analytics, can be dropped


class SyncStatus(Enum):
    """Status of a sync operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class SyncResult:
    """Result of a sync operation."""

    success: bool
    operation_id: str
    attempts: int
    error: str | None = None
    synced_at: datetime | None = None


# Priority mapping for operations
OP_PRIORITY: dict[SyncOp, SyncPriority] = {
    SyncOp.MESSAGE_SEND: SyncPriority.CRITICAL,
    SyncOp.MESSAGE_READ: SyncPriority.HIGH,
    SyncOp.DRAFT_SAVE: SyncPriority.HIGH,
    SyncOp.SETTING_UPDATE: SyncPriority.NORMAL,
    SyncOp.FEEDBACK_SUBMIT: SyncPriority.NORMAL,
    SyncOp.ANALYTICS_EVENT: SyncPriority.LOW,
}

# Retry configuration
RETRY_BACKOFF_SECONDS = [1, 5, 15, 60, 300]  # Exponential-ish backoff
MAX_RETRY_ATTEMPTS = len(RETRY_BACKOFF_SECONDS)


@dataclass
class SyncOperation:
    """A single sync operation."""

    id: str
    op_type: SyncOp
    payload: dict[str, Any]
    priority: SyncPriority
    status: SyncStatus = field(default=SyncStatus.PENDING)
    created_at: float = field(default_factory=time.time)
    attempts: int = 0
    last_attempt_at: float | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "op_type": self.op_type.value,
            "payload": self.payload,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "attempts": self.attempts,
            "last_attempt_at": self.last_attempt_at,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SyncOperation:
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            op_type=SyncOp(data["op_type"]),
            payload=data["payload"],
            priority=SyncPriority(data["priority"]),
            status=SyncStatus(data["status"]),
            created_at=data["created_at"],
            attempts=data.get("attempts", 0),
            last_attempt_at=data.get("last_attempt_at"),
            error_message=data.get("error_message"),
        )


class OfflineSyncQueue:
    """Thread-safe queue for offline operations.

    Persists operations to disk and processes them when online.
    Automatically retries failed operations with exponential backoff.

    Example:
        >>> queue = OfflineSyncQueue(persistence_path=Path("~/.jarvis/sync.json"))
        >>> queue.enqueue(SyncOp.MESSAGE_SEND, {"text": "Hello", "chat_id": "123"})
        >>> # Later, when online:
        >>> results = await queue.process_pending()
    """

    DEFAULT_MAX_QUEUE_SIZE = 1000
    DEFAULT_MAX_AGE_HOURS = 24

    def __init__(
        self,
        persistence_path: Path | None = None,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
        max_age_hours: int = DEFAULT_MAX_AGE_HOURS,
        auto_persist: bool = True,
    ) -> None:
        """Initialize the sync queue.

        Args:
            persistence_path: Path to persist queue state
            max_queue_size: Maximum number of operations to queue
            max_age_hours: Maximum age of operations before dropping
            auto_persist: Whether to auto-save on changes
        """
        self._operations: dict[str, SyncOperation] = {}
        self._lock = threading.RLock()
        self._persistence_path = persistence_path
        self._max_queue_size = max_queue_size
        self._max_age_seconds = max_age_hours * 3600
        self._auto_persist = auto_persist
        self._handlers: dict[SyncOp, Callable[[dict[str, Any]], Any]] = {}

        if persistence_path:
            self._load()

    def _load(self) -> None:
        """Load queue state from disk."""
        if not self._persistence_path or not self._persistence_path.exists():
            return

        try:
            data = json.loads(self._persistence_path.read_text())
            for op_data in data.get("operations", []):
                try:
                    op = SyncOperation.from_dict(op_data)
                    # Skip old operations
                    if time.time() - op.created_at < self._max_age_seconds:
                        self._operations[op.id] = op
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping corrupted sync operation: {e}")

            # Reset in-progress operations to pending
            for op in self._operations.values():
                if op.status == SyncStatus.IN_PROGRESS:
                    op.status = SyncStatus.PENDING

            logger.info(f"Loaded {len(self._operations)} sync operations")
        except json.JSONDecodeError:
            logger.warning("Sync queue file corrupted, starting fresh")
        except OSError as e:
            logger.warning(f"Failed to load sync queue: {e}")

    def _persist(self) -> None:
        """Save queue state to disk."""
        if not self._persistence_path:
            return

        try:
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": 1,
                "saved_at": time.time(),
                "operations": [op.to_dict() for op in self._operations.values()],
            }
            # Write atomically
            temp_path = self._persistence_path.with_suffix(".tmp")
            temp_path.write_text(json.dumps(data, indent=2))
            temp_path.replace(self._persistence_path)
        except OSError as e:
            logger.warning(f"Failed to persist sync queue: {e}")

    def register_handler(self, op_type: SyncOp, handler: Callable[[dict[str, Any]], Any]) -> None:
        """Register a handler for an operation type.

        Args:
            op_type: Type of operation to handle
            handler: Function that processes the operation payload
        """
        self._handlers[op_type] = handler

    def enqueue(
        self,
        op_type: SyncOp,
        payload: dict[str, Any],
        priority: SyncPriority | None = None,
    ) -> str | None:
        """Add an operation to the queue.

        Args:
            op_type: Type of operation
            payload: Operation data
            priority: Optional priority override

        Returns:
            Operation ID if queued, None if queue full
        """
        with self._lock:
            # Check queue size limit
            if len(self._operations) >= self._max_queue_size:
                # Try to drop old LOW priority items
                self._drop_old_operations()

                if len(self._operations) >= self._max_queue_size:
                    logger.error("Sync queue full, dropping operation")
                    return None

            op = SyncOperation(
                id=str(uuid.uuid4()),
                op_type=op_type,
                payload=payload,
                priority=priority or OP_PRIORITY.get(op_type, SyncPriority.NORMAL),
            )
            self._operations[op.id] = op

            if self._auto_persist:
                self._persist()

            logger.debug(f"Enqueued {op_type.value} operation: {op.id}")
            return op.id

    def _drop_old_operations(self) -> None:
        """Drop old low-priority operations to make room."""
        now = time.time()
        to_drop = [
            op_id
            for op_id, op in self._operations.items()
            if op.priority == SyncPriority.LOW
            and op.status == SyncStatus.PENDING
            and now - op.created_at > 3600  # Older than 1 hour
        ]
        for op_id in to_drop[:100]:  # Drop max 100 at a time
            del self._operations[op_id]

        if to_drop:
            logger.info(f"Dropped {len(to_drop)} old operations")

    def get_operation(self, op_id: str) -> SyncOperation | None:
        """Get a specific operation by ID."""
        with self._lock:
            return self._operations.get(op_id)

    def get_pending(self) -> list[SyncOperation]:
        """Get all pending operations, sorted by priority."""
        with self._lock:
            return sorted(
                [op for op in self._operations.values() if op.status == SyncStatus.PENDING],
                key=lambda op: (op.priority.value, op.created_at),
            )

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            by_status: dict[str, int] = {}
            by_priority: dict[str, int] = {}

            for op in self._operations.values():
                by_status[op.status.value] = by_status.get(op.status.value, 0) + 1
                by_priority[op.priority.name] = by_priority.get(op.priority.name, 0) + 1

            return {
                "total": len(self._operations),
                "by_status": by_status,
                "by_priority": by_priority,
                "max_size": self._max_queue_size,
            }

    async def process_pending(self) -> list[SyncResult]:
        """Process all pending operations.

        Returns:
            List of results for each operation processed
        """
        results: list[SyncResult] = []
        pending = self.get_pending()

        for op in pending:
            result = await self._process_operation(op)
            results.append(result)

        return results

    async def _process_operation(self, op: SyncOperation) -> SyncResult:
        """Process a single operation."""
        handler = self._handlers.get(op.op_type)

        if not handler:
            logger.warning(f"No handler for {op.op_type.value}")
            return SyncResult(
                success=False,
                operation_id=op.id,
                attempts=op.attempts,
                error="No handler registered",
            )

        with self._lock:
            op.status = SyncStatus.IN_PROGRESS
            op.last_attempt_at = time.time()
            op.attempts += 1

        try:
            await handler(op.payload)

            with self._lock:
                op.status = SyncStatus.COMPLETED
                if self._auto_persist:
                    self._persist()

            return SyncResult(
                success=True,
                operation_id=op.id,
                attempts=op.attempts,
                synced_at=datetime.utcnow(),
            )

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Sync operation {op.id} failed: {error_msg}")

            with self._lock:
                op.error_message = error_msg

                # Determine if we should retry
                if op.attempts < MAX_RETRY_ATTEMPTS:
                    op.status = SyncStatus.RETRYING
                else:
                    op.status = SyncStatus.FAILED

                if self._auto_persist:
                    self._persist()

            return SyncResult(
                success=False,
                operation_id=op.id,
                attempts=op.attempts,
                error=error_msg,
            )

    def retry_failed(self) -> list[str]:
        """Reset failed operations to pending for retry.

        Returns:
            List of operation IDs reset
        """
        reset_ids: list[str] = []

        with self._lock:
            for op in self._operations.values():
                if op.status == SyncStatus.FAILED and op.attempts < MAX_RETRY_ATTEMPTS:
                    op.status = SyncStatus.PENDING
                    op.error_message = None
                    reset_ids.append(op.id)

            if reset_ids and self._auto_persist:
                self._persist()

        logger.info(f"Reset {len(reset_ids)} failed operations for retry")
        return reset_ids

    def clear_completed(self) -> int:
        """Remove completed operations from queue.

        Returns:
            Number of operations removed
        """
        with self._lock:
            to_remove = [
                op_id for op_id, op in self._operations.items() if op.status == SyncStatus.COMPLETED
            ]
            for op_id in to_remove:
                del self._operations[op_id]

            if to_remove and self._auto_persist:
                self._persist()

        return len(to_remove)

    def clear_all(self) -> None:
        """Clear all operations. Use with caution."""
        with self._lock:
            self._operations.clear()
            if self._auto_persist:
                self._persist()


class OfflineManager:
    """Manages offline/online state transitions.

        Coordinates between connectivity monitoring and sync queue
    to provide seamless offline-first experience.

        Example:
            >>> manager = OfflineManager(sync_queue=queue)
            >>> manager.on_online = lambda: print("Back online!")
            >>> manager.on_offline = lambda: print("Gone offline")
            >>> await manager.check_connectivity()
    """

    def __init__(
        self,
        sync_queue: OfflineSyncQueue | None = None,
        check_interval_seconds: float = 30.0,
    ) -> None:
        """Initialize the offline manager.

        Args:
            sync_queue: Queue for pending operations
            check_interval_seconds: How often to check connectivity
        """
        self._sync_queue = sync_queue
        self._check_interval = check_interval_seconds
        self._is_online = True  # Assume online until proven otherwise
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread: threading.Thread | None = None

        # Callbacks
        self.on_online: Callable[[], None] | None = None
        self.on_offline: Callable[[], None] | None = None
        self.on_sync_complete: Callable[[list[SyncResult]], None] | None = None

    @property
    def is_online(self) -> bool:
        """Current connectivity state."""
        with self._lock:
            return self._is_online

    async def check_connectivity(self) -> bool:
        """Check if currently online.

        Returns:
            True if online, False otherwise
        """
        # Try multiple endpoints
        endpoints = [
            ("localhost", 8742),  # API
            ("localhost", 8743),  # WebSocket
        ]

        import socket

        for host, port in endpoints:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((host, port))
                sock.close()
                if result == 0:
                    self._set_online(True)
                    return True
            except Exception:
                continue

        self._set_online(False)
        return False

    def _set_online(self, online: bool) -> None:
        """Update online state and trigger callbacks."""
        with self._lock:
            if self._is_online == online:
                return  # No change

            self._is_online = online

        if online:
            logger.info("Connectivity restored")
            if self.on_online:
                self.on_online()
            # Trigger sync
            if self._sync_queue:
                import asyncio

                asyncio.create_task(self._sync_pending())
        else:
            logger.warning("Connectivity lost")
            if self.on_offline:
                self.on_offline()

    async def _sync_pending(self) -> None:
        """Sync pending operations."""
        if not self._sync_queue:
            return

        results = await self._sync_queue.process_pending()

        if self.on_sync_complete:
            self.on_sync_complete(results)

        # Clear completed operations
        cleared = self._sync_queue.clear_completed()
        logger.info(f"Cleared {cleared} completed sync operations")

    def start_monitoring(self) -> None:
        """Start background connectivity monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Started connectivity monitoring")

    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Stopped connectivity monitoring")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        import asyncio

        while self._monitoring:
            try:
                asyncio.run(self.check_connectivity())
            except Exception as e:
                logger.warning(f"Connectivity check failed: {e}")

            # Wait for next check
            for _ in range(int(self._check_interval)):
                if not self._monitoring:
                    break
                time.sleep(1)
