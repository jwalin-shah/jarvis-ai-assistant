"""Utility modules for JARVIS."""

from jarvis.utils.async_utils import log_task_exception, run_in_thread, sync_to_async
from jarvis.utils.backoff import AsyncConsecutiveErrorTracker, ConsecutiveErrorTracker, with_retry
from jarvis.utils.datetime_utils import parse_apple_timestamp
from jarvis.utils.error_handling import graceful_shutdown, safe_execution, silence_exceptions
from jarvis.utils.locks import PerKeyLockManager
from jarvis.utils.memory import (
    MemoryMonitor,
    SwapThresholdExceededError,
    get_memory_info,
    get_swap_info,
    log_memory_snapshot,
)
from jarvis.utils.polling import async_poll_until, poll_until
from jarvis.utils.resources import managed_resource, safe_close
from jarvis.utils.singleton import thread_safe_singleton

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
