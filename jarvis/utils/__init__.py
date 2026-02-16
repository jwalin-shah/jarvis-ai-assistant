"""Utility modules for JARVIS."""

from jarvis.utils.async_utils import log_task_exception, run_in_thread, sync_to_async
from jarvis.utils.atomic_write import atomic_write_text
from jarvis.utils.backoff import (
    AsyncConsecutiveErrorTracker,
    BackoffConfig,
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitBreakerOpenError,
    ConsecutiveErrorTracker,
    RetryStats,
    with_retry,
)
from jarvis.utils.datetime_utils import parse_apple_timestamp
from jarvis.utils.error_handling import graceful_shutdown, safe_execution, silence_exceptions
from jarvis.utils.locks import PerKeyLockManager
from jarvis.utils.logging import setup_script_logging
from jarvis.utils.memory import (
    MemoryMonitor,
    SwapThresholdExceededError,
    get_memory_info,
    get_swap_info,
    log_memory_snapshot,
)
from jarvis.utils.polling import async_poll_until, poll_until
from jarvis.utils.resources import managed_resource, multi_resource_manager, safe_close
from jarvis.utils.singleton import thread_safe_singleton
from jarvis.utils.sqlite_retry import sqlite_retry

__all__ = [
    "AsyncConsecutiveErrorTracker",
    "BackoffConfig",
    "CircuitBreaker",
    "CircuitBreakerOpen",
    "CircuitBreakerOpenError",
    "ConsecutiveErrorTracker",
    "MemoryMonitor",
    "PerKeyLockManager",
    "RetryStats",
    "SwapThresholdExceededError",
    "async_poll_until",
    "atomic_write_text",
    "get_memory_info",
    "get_swap_info",
    "graceful_shutdown",
    "log_memory_snapshot",
    "log_task_exception",
    "managed_resource",
    "multi_resource_manager",
    "parse_apple_timestamp",
    "poll_until",
    "run_in_thread",
    "safe_close",
    "safe_execution",
    "setup_script_logging",
    "silence_exceptions",
    "sqlite_retry",
    "sync_to_async",
    "thread_safe_singleton",
    "with_retry",
]
