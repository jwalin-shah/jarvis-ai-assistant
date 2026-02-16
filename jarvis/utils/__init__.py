"""Utility modules for JARVIS."""

from jarvis.utils.memory import (
    MemoryMonitor,
    SwapThresholdExceededError,
    get_memory_info,
    get_swap_info,
    log_memory_snapshot,
)
from jarvis.utils.singleton import thread_safe_singleton
from jarvis.utils.error_handling import silence_exceptions, graceful_shutdown, safe_execution
from jarvis.utils.backoff import ConsecutiveErrorTracker, AsyncConsecutiveErrorTracker, with_retry
from jarvis.utils.locks import PerKeyLockManager
from jarvis.utils.resources import safe_close, managed_resource
from jarvis.utils.async_utils import run_in_thread, sync_to_async, log_task_exception
from jarvis.utils.datetime_utils import parse_apple_timestamp
from jarvis.utils.polling import poll_until, async_poll_until

__all__ = [
    "MemoryMonitor",
    "SwapThresholdExceededError",
    "get_memory_info",
    "get_swap_info",
    "log_memory_snapshot",
    "thread_safe_singleton",
    "silence_exceptions",
    "graceful_shutdown",
    "safe_execution",
    "ConsecutiveErrorTracker",
    "AsyncConsecutiveErrorTracker",
    "with_retry",
    "PerKeyLockManager",
    "safe_close",
    "managed_resource",
    "run_in_thread",
    "sync_to_async",
    "log_task_exception",
    "parse_apple_timestamp",
    "poll_until",
    "async_poll_until",
]
