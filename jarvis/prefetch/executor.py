"""Background prefetch executor with resource management.

Executes prefetch predictions in the background with:
- Priority-based scheduling
- Resource awareness (CPU, memory, battery)
- Rate limiting and backpressure
- Metrics and monitoring

Usage:
    executor = PrefetchExecutor(cache=cache)
    executor.start()
    executor.schedule(prediction)
    executor.stop()
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from queue import PriorityQueue
from typing import Any

from jarvis.observability.logging import log_event
from jarvis.prefetch.cache import CacheTier, MultiTierCache, get_cache
from jarvis.prefetch.predictor import Prediction, PredictionPriority, PredictionType
from jarvis.utils.backoff import ConsecutiveErrorTracker

logger = logging.getLogger(__name__)


class ExecutorState(StrEnum):
    """Executor states."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"


@dataclass
class ExecutorStats:
    """Statistics for the executor."""

    predictions_scheduled: int = 0
    predictions_executed: int = 0
    predictions_cached: int = 0
    predictions_skipped: int = 0
    predictions_failed: int = 0
    cache_hits: int = 0
    total_cost_ms: int = 0
    avg_latency_ms: float = 0.0
    queue_size: int = 0
    workers_active: int = 0

    def record_execution(self, cost_ms: int, cached: bool) -> None:
        """Record an execution result."""
        self.predictions_executed += 1
        self.total_cost_ms += cost_ms
        if cached:
            self.predictions_cached += 1
        n = self.predictions_executed
        self.avg_latency_ms = (self.avg_latency_ms * (n - 1) + cost_ms) / n


@dataclass(order=True)
class PrefetchTask:
    """A task to be executed by the prefetch executor."""

    priority: int  # Lower = higher priority (for PriorityQueue)
    created_at: float = field(compare=False)
    prediction: Prediction = field(compare=False)
    retries: int = field(default=0, compare=False)
    max_retries: int = field(default=2, compare=False)


# Type alias for prefetch handlers
PrefetchHandler = Callable[[Prediction], dict[str, Any] | None]


class ResourceManager:
    """Manages system resources for prefetching.

    Monitors CPU, memory, and battery to ensure prefetching
    doesn't impact user experience.
    """

    def __init__(
        self,
        memory_threshold_mb: int = 500,  # Min available memory
        cpu_threshold_percent: float = 80.0,  # Max CPU before throttling
        battery_threshold: float = 0.2,  # Min battery for prefetch
    ) -> None:
        self._memory_threshold = memory_threshold_mb * 1024 * 1024
        self._cpu_threshold = cpu_threshold_percent
        self._battery_threshold = battery_threshold
        self._is_plugged_in = True
        self._battery_level = 1.0
        self._available_memory = 8 * 1024 * 1024 * 1024  # Default 8GB
        self._cpu_usage = 0.0
        self._last_update = 0.0
        self._update_interval = 5.0  # Update every 5 seconds

    def update(self) -> None:
        """Update resource metrics."""
        now = time.time()
        if now - self._last_update < self._update_interval:
            return

        self._last_update = now

        # Try to get memory info
        try:
            import psutil

            mem = psutil.virtual_memory()
            self._available_memory = mem.available
            self._cpu_usage = psutil.cpu_percent(interval=0)

            # Battery info
            battery = psutil.sensors_battery()
            if battery:
                self._battery_level = battery.percent / 100
                self._is_plugged_in = battery.power_plugged
            else:
                self._is_plugged_in = True
                self._battery_level = 1.0
        except ImportError:
            # psutil not available, use defaults
            pass
        except Exception as e:
            logger.debug(f"Error getting resource info: {e}")

    def can_prefetch(self) -> bool:
        """Check if prefetching is allowed.

        Returns:
            True if resources allow prefetching.
        """
        self.update()

        # Check memory
        if self._available_memory < self._memory_threshold:
            log_event(
                logger,
                "prefetch.resource_blocked",
                level=logging.DEBUG,
                resource="memory",
                available_mb=self._available_memory // (1024 * 1024),
            )
            return False

        # Check CPU
        if self._cpu_usage > self._cpu_threshold:
            log_event(
                logger,
                "prefetch.resource_blocked",
                level=logging.DEBUG,
                resource="cpu",
                cpu_percent=self._cpu_usage,
            )
            return False

        # Check battery (only if not plugged in)
        if not self._is_plugged_in and self._battery_level < self._battery_threshold:
            log_event(
                logger,
                "prefetch.resource_blocked",
                level=logging.DEBUG,
                resource="battery",
                battery_level=self._battery_level,
            )
            return False

        return True

    def get_concurrency_limit(self) -> int:
        """Get recommended concurrency based on resources.

        Returns:
            Maximum concurrent prefetch tasks.
        """
        self.update()

        # Base concurrency on CPU and memory
        cpu_cores = os.cpu_count() or 4
        base_concurrency = max(1, cpu_cores // 2)

        # Reduce if resources are constrained
        if self._cpu_usage > 50:
            base_concurrency = max(1, base_concurrency // 2)

        if self._available_memory < self._memory_threshold * 2:
            base_concurrency = max(1, base_concurrency // 2)

        # Battery mode: single worker
        if not self._is_plugged_in and self._battery_level < 0.5:
            base_concurrency = 1

        return base_concurrency

    @property
    def battery_level(self) -> float:
        """Current battery level (0-1)."""
        return self._battery_level

    @property
    def available_memory_mb(self) -> int:
        """Available memory in MB."""
        return self._available_memory // (1024 * 1024)


class PrefetchExecutor:
    """Background executor for prefetch tasks.

    Features:
    - Priority queue for task scheduling
    - Resource-aware execution (default: 2 workers, adjusted 1-4 by ResourceManager)
    - Extensible handler system
    - Metrics and monitoring

    Concurrency Model:
    - Default 2 workers handle IO-bound (DB, cache) and GPU-bound (LLM, embeddings) tasks.
    - GPU operations (encode, generate, load) acquire MLXModelLoader._mlx_load_lock
      internally within the model/embedder classes. No outer locking needed here.
    - Multiple workers benefit IO-bound tasks (contact profiles, search, vec index).
    - GPU-bound tasks (draft replies, embeddings) block on the internal lock regardless.
    - ResourceManager dynamically adjusts 1-4 workers based on CPU/memory/battery.
    """

    def __init__(
        self,
        cache: MultiTierCache | None = None,
        max_workers: int | None = None,
        max_queue_size: int = 100,
        tick_interval: float = 0.1,
    ) -> None:
        """Initialize the executor.

        Args:
            cache: Multi-tier cache for storing prefetched data.
            max_workers: Maximum worker threads. Defaults to 2. ResourceManager
                can recommend 1-4 based on system resources.
            max_queue_size: Maximum pending tasks.
            tick_interval: How often to check for tasks (seconds).
        """
        self._cache = cache or get_cache()
        self._max_workers = max_workers or 2
        self._max_queue_size = max_queue_size
        self._tick_interval = tick_interval

        self._queue: PriorityQueue[PrefetchTask] = PriorityQueue(maxsize=max_queue_size)
        self._state = ExecutorState.STOPPED
        self._stats = ExecutorStats()
        self._resource_manager = ResourceManager()

        self._handlers: dict[PredictionType, PrefetchHandler] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._worker_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        self._lock = threading.RLock()

        # Completion callback: (prediction, result) -> None
        self.on_task_complete: Callable[[Prediction, dict[str, Any]], None] | None = None

        # Active tasks tracking (guarded by self._lock to avoid deadlock from
        # inconsistent lock ordering between _lock and a separate _active_lock)
        self._active_tasks: set[str] = set()
        self._active_drafts: dict[str, str] = {}  # chat_id → active_key for O(1) lookup

        # Shared ChatDBReader for handlers (thread-safe via connection pool).
        # Lazily initialized on first use, closed on stop().
        self._reader: Any = None  # integrations.imessage.ChatDBReader
        self._reader_lock = threading.Lock()

        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default prefetch handlers."""
        from .handlers_impl import (
            ContactProfileHandler,
            DraftReplyHandler,
            EmbeddingHandler,
            ModelWarmHandler,
            SearchResultsHandler,
            VecIndexHandler,
        )

        # Keep concrete handlers as attributes and route through methods so tests
        # can patch `_handle_*` methods without mutating internal registry state.
        self._draft_reply_handler = DraftReplyHandler()
        self._embedding_handler = EmbeddingHandler()
        self._contact_profile_handler = ContactProfileHandler()
        self._model_warm_handler = ModelWarmHandler()
        self._search_results_handler = SearchResultsHandler()
        self._vec_index_handler = VecIndexHandler()

        self.register_handler(PredictionType.DRAFT_REPLY, lambda p: self._handle_draft_reply(p))
        self.register_handler(PredictionType.EMBEDDING, lambda p: self._handle_embedding(p))
        self.register_handler(
            PredictionType.CONTACT_PROFILE, lambda p: self._handle_contact_profile(p)
        )
        self.register_handler(PredictionType.MODEL_WARM, lambda p: self._handle_model_warm(p))
        self.register_handler(
            PredictionType.SEARCH_RESULTS, lambda p: self._handle_search_results(p)
        )
        self.register_handler(PredictionType.VEC_INDEX, lambda p: self._handle_vec_index(p))

    def _handle_draft_reply(self, prediction: Prediction) -> dict[str, Any] | None:
        return self._draft_reply_handler(prediction)

    def _handle_embedding(self, prediction: Prediction) -> dict[str, Any] | None:
        return self._embedding_handler(prediction)

    def _handle_contact_profile(self, prediction: Prediction) -> dict[str, Any] | None:
        return self._contact_profile_handler(prediction)

    def _handle_model_warm(self, prediction: Prediction) -> dict[str, Any] | None:
        return self._model_warm_handler(prediction)

    def _handle_search_results(self, prediction: Prediction) -> dict[str, Any] | None:
        return self._search_results_handler(prediction)

    def _handle_vec_index(self, prediction: Prediction) -> dict[str, Any] | None:
        return self._vec_index_handler(prediction)

    def _get_reader(self) -> Any:
        """Get or create the shared ChatDBReader instance.

        ChatDBReader uses a module-level connection pool and is thread-safe,
        so a single instance can be shared across all worker threads.
        """
        if self._reader is not None:
            return self._reader

        with self._reader_lock:
            if self._reader is None:
                from integrations.imessage import ChatDBReader

                self._reader = ChatDBReader()
            return self._reader

    def register_handler(self, pred_type: PredictionType, handler: PrefetchHandler) -> None:
        """Register a handler for a prediction type.

        Args:
            pred_type: Prediction type to handle.
            handler: Handler function that returns data to cache or None.
        """
        with self._lock:
            self._handlers[pred_type] = handler

    def start(self) -> None:
        """Start the executor."""
        with self._lock:
            if self._state != ExecutorState.STOPPED:
                return

            self._state = ExecutorState.STARTING
            self._shutdown_event.clear()

            # Create thread pool
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="prefetch-",
            )

            # Initialize shared ChatDBReader eagerly (avoids lazy-init latency
            # on first prefetch task and ensures import errors surface early).
            try:
                from integrations.imessage import ChatDBReader

                self._reader = ChatDBReader()
            except Exception as e:
                logger.warning(f"Failed to initialize ChatDBReader at start: {e}")

            # Start worker thread
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="prefetch-scheduler",
                daemon=True,
            )
            self._worker_thread.start()

            self._state = ExecutorState.RUNNING
            log_event(logger, "prefetch.executor.start", workers=self._max_workers)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the executor.

        Args:
            timeout: Maximum seconds to wait for shutdown.
        """
        with self._lock:
            if self._state == ExecutorState.STOPPED:
                return

            self._state = ExecutorState.STOPPING
            self._shutdown_event.set()

        # Wait for worker thread
        if self._worker_thread:
            self._worker_thread.join(timeout=timeout)
            # Only clear reference if thread actually stopped
            if not self._worker_thread.is_alive():
                self._worker_thread = None
            else:
                logger.warning("Worker thread did not stop within timeout, still running")

        # Shutdown thread pool
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

        # Close shared reader
        with self._reader_lock:
            if self._reader is not None:
                try:
                    self._reader.close()
                except Exception:  # nosec B110
                    pass
                self._reader = None

        with self._lock:
            self._state = ExecutorState.STOPPED
            # Clear queue
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Exception:
                    break

        log_event(
            logger,
            "prefetch.executor.stop",
            executed=self._stats.predictions_executed,
            failed=self._stats.predictions_failed,
            avg_latency_ms=round(self._stats.avg_latency_ms, 1),
        )

    def pause(self) -> None:
        """Pause execution (tasks remain in queue)."""
        with self._lock:
            if self._state == ExecutorState.RUNNING:
                self._state = ExecutorState.PAUSED
                logger.info("Prefetch executor paused")

    def resume(self) -> None:
        """Resume execution."""
        with self._lock:
            if self._state == ExecutorState.PAUSED:
                self._state = ExecutorState.RUNNING
                logger.info("Prefetch executor resumed")

    def schedule(self, prediction: Prediction) -> bool:
        """Schedule a prediction for prefetching.

        Args:
            prediction: Prediction to prefetch.

        Returns:
            True if scheduled, False if rejected (queue full, duplicate, etc.).
        """
        with self._lock:
            if self._state not in (ExecutorState.RUNNING, ExecutorState.PAUSED):
                return False

            # Check if already in cache
            if self._cache.get(prediction.key) is not None:
                self._stats.cache_hits += 1
                return False

            # Check if already being processed (uses self._lock, already held)
            if prediction.key in self._active_tasks:
                return False
            # Block duplicate drafts for the same chat (O(1) lookup)
            if prediction.type == PredictionType.DRAFT_REPLY:
                draft_cid = prediction.params.get("chat_id", "")
                if draft_cid and draft_cid in self._active_drafts:
                    return False

        # Create task with inverted priority (lower = higher priority)
        task = PrefetchTask(
            priority=-prediction.priority.value,  # Negate for min-heap
            created_at=time.time(),
            prediction=prediction,
        )

        try:
            self._queue.put_nowait(task)
            self._stats.predictions_scheduled += 1
            self._stats.queue_size = self._queue.qsize()
            return True
        except Exception:
            self._stats.predictions_skipped += 1
            return False

    def schedule_batch(self, predictions: list[Prediction]) -> int:
        """Schedule multiple predictions with single lock acquisition.

        Args:
            predictions: List of predictions to schedule.

        Returns:
            Number of predictions scheduled.
        """
        tasks_to_enqueue = []

        with self._lock:
            if self._state not in (ExecutorState.RUNNING, ExecutorState.PAUSED):
                return 0

            for prediction in predictions:
                # Check if already in cache
                if self._cache.get(prediction.key) is not None:
                    self._stats.cache_hits += 1
                    continue

                # Check if already being processed (uses self._lock, already held)
                if prediction.key in self._active_tasks:
                    continue
                if prediction.type == PredictionType.DRAFT_REPLY:
                    draft_cid = prediction.params.get("chat_id", "")
                    if draft_cid and draft_cid in self._active_drafts:
                        continue

                tasks_to_enqueue.append(
                    PrefetchTask(
                        priority=-prediction.priority.value,
                        created_at=time.time(),
                        prediction=prediction,
                    )
                )

        count = 0
        for task in tasks_to_enqueue:
            try:
                self._queue.put_nowait(task)
                count += 1
            except Exception:  # nosec B110
                pass

        with self._lock:
            self._stats.predictions_scheduled += count
            self._stats.predictions_skipped += len(predictions) - count
            self._stats.queue_size = self._queue.qsize()

        return count

    def stats(self) -> dict[str, Any]:
        """Get executor statistics."""
        with self._lock:
            return {
                "state": self._state.value,
                "scheduled": self._stats.predictions_scheduled,
                "executed": self._stats.predictions_executed,
                "cached": self._stats.predictions_cached,
                "skipped": self._stats.predictions_skipped,
                "failed": self._stats.predictions_failed,
                "cache_hits": self._stats.cache_hits,
                "total_cost_ms": self._stats.total_cost_ms,
                "avg_latency_ms": self._stats.avg_latency_ms,
                "queue_size": self._queue.qsize(),
                "active_tasks": len(self._active_tasks),
                "workers": self._max_workers,
                "resource_manager": {
                    "can_prefetch": self._resource_manager.can_prefetch(),
                    "battery": self._resource_manager.battery_level,
                    "memory_available_mb": self._resource_manager.available_memory_mb,
                },
            }

    def _worker_loop(self) -> None:
        """Main worker loop that processes tasks."""
        tracker = ConsecutiveErrorTracker(
            base_delay=self._tick_interval, max_delay=5.0, max_consecutive=5, name="prefetch-worker"
        )

        while not self._shutdown_event.is_set():
            try:
                # Check state
                if self._state != ExecutorState.RUNNING:
                    time.sleep(self._tick_interval)
                    continue

                # Get task from queue
                try:
                    task = self._queue.get(timeout=self._tick_interval)
                except Exception:  # nosec B112
                    continue

                # Resource guard:
                # - high-priority tasks are allowed through even under pressure
                #   to avoid starvation/timeouts in latency-sensitive flows.
                # - lower-priority tasks are requeued with a small backoff.
                if (
                    task.prediction.priority < PredictionPriority.HIGH
                    and not self._resource_manager.can_prefetch()
                ):
                    try:
                        self._queue.put_nowait(task)
                    except Exception:  # nosec B110
                        self._stats.predictions_skipped += 1
                    time.sleep(self._tick_interval * 2)
                    continue

                # Check if prediction is still valid
                prediction = task.prediction
                age = time.time() - task.created_at
                if age > prediction.ttl_seconds / 2:
                    # Too old, skip
                    self._stats.predictions_skipped += 1
                    continue

                # Execute task
                if self._executor:
                    self._executor.submit(self._execute_task, task)

                # Reset error counter on success
                tracker.reset()

            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                logger.exception(f"Worker loop error: {e}")
                time.sleep(tracker.on_error(log_level=logging.DEBUG))

    def _execute_task(self, task: PrefetchTask) -> None:
        """Execute a single prefetch task.

        Args:
            task: Task to execute.
        """
        prediction = task.prediction
        start_time = time.time()

        # Mark as active
        with self._lock:
            if prediction.key in self._active_tasks:
                return  # Already being processed
            self._active_tasks.add(prediction.key)
            # Track active drafts by chat_id for O(1) duplicate check
            if prediction.type == PredictionType.DRAFT_REPLY:
                draft_cid = prediction.params.get("chat_id", "")
                if draft_cid:
                    self._active_drafts[draft_cid] = prediction.key

        try:
            # Get handler
            handler = self._handlers.get(prediction.type)
            if not handler:
                logger.warning(f"No handler for prediction type: {prediction.type}")
                self._stats.predictions_skipped += 1
                return

            # Execute handler
            try:
                result = handler(prediction)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                logger.exception(f"Handler failed for {prediction.key}: {e}")
                # Retry if possible
                if task.retries < task.max_retries:
                    task.retries += 1
                    task.priority -= 10  # Lower priority on retry
                    try:
                        self._queue.put_nowait(task)
                    except Exception:  # nosec B110
                        pass
                else:
                    self._stats.predictions_failed += 1
                return

            # Cache result if any
            if result is not None:
                tier = self._get_cache_tier(prediction)
                self._cache.set(
                    key=prediction.key,
                    value=result,
                    tier=tier,
                    ttl_seconds=prediction.ttl_seconds,
                    tags=prediction.tags,
                )
                cost_ms = int((time.time() - start_time) * 1000)
                self._stats.record_execution(cost_ms, cached=True)
                log_event(
                    logger,
                    "prefetch.execute.complete",
                    prediction_type=prediction.type.value,
                    key=prediction.key,
                    execution_ms=cost_ms,
                    tier=tier.value,
                )

                # Trigger completion callback
                if self.on_task_complete:
                    try:
                        self.on_task_complete(prediction, result)
                    except Exception as e:
                        logger.debug(f"Error in prefetch completion callback: {e}")
            else:
                self._stats.predictions_skipped += 1

        finally:
            # Mark as inactive
            with self._lock:
                self._active_tasks.discard(prediction.key)
                # Remove from active drafts tracking
                if prediction.type == PredictionType.DRAFT_REPLY:
                    draft_cid = prediction.params.get("chat_id", "")
                    if draft_cid:
                        self._active_drafts.pop(draft_cid, None)
            self._stats.queue_size = self._queue.qsize()

    def _get_cache_tier(self, prediction: Prediction) -> CacheTier:
        """Determine cache tier based on prediction.

        Args:
            prediction: Prediction to cache.

        Returns:
            Appropriate cache tier.
        """
        # High priority items go to L1
        if prediction.priority >= PredictionPriority.HIGH:
            return CacheTier.L1

        # Medium priority to L2
        if prediction.priority >= PredictionPriority.MEDIUM:
            return CacheTier.L2

        # Low priority to L3
        return CacheTier.L3


from jarvis.utils.singleton import thread_safe_singleton  # noqa: E402


@thread_safe_singleton
def get_executor() -> PrefetchExecutor:
    """Get or create singleton executor instance."""
    return PrefetchExecutor()


def reset_executor() -> None:
    """Reset singleton executor (stops if running)."""
    executor = get_executor.peek()  # type: ignore[attr-defined]
    if executor is not None:
        executor.stop()
    get_executor.reset()  # type: ignore[attr-defined]
