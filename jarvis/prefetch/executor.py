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
from enum import Enum
from queue import PriorityQueue
from typing import Any

from jarvis.observability.logging import log_event
from jarvis.prefetch.cache import CacheTier, MultiTierCache, get_cache
from jarvis.prefetch.predictor import Prediction, PredictionPriority, PredictionType

logger = logging.getLogger(__name__)


class ExecutorState(str, Enum):
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
    - MLXModelLoader._mlx_load_lock serializes all GPU operations across workers.
    - Multiple workers benefit IO-bound tasks (contact profiles, search, vec index).
    - GPU-bound tasks (draft replies, embeddings) block on the lock regardless.
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

        # Active tasks tracking (guarded by self._lock, an RLock for re-entrant use)
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
        self.register_handler(PredictionType.DRAFT_REPLY, self._handle_draft_reply)
        self.register_handler(PredictionType.EMBEDDING, self._handle_embedding)
        self.register_handler(PredictionType.CONTACT_PROFILE, self._handle_contact_profile)
        self.register_handler(PredictionType.MODEL_WARM, self._handle_model_warm)
        self.register_handler(PredictionType.SEARCH_RESULTS, self._handle_search_results)
        self.register_handler(PredictionType.VEC_INDEX, self._handle_vec_index)

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
                except Exception:
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

            # Check if already being processed
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

                # Check if already being processed
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
            except Exception:
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
        consecutive_errors = 0

        while not self._shutdown_event.is_set():
            try:
                # Check state
                if self._state != ExecutorState.RUNNING:
                    time.sleep(self._tick_interval)
                    continue

                # Check resources
                if not self._resource_manager.can_prefetch():
                    time.sleep(self._tick_interval * 5)  # Back off
                    continue

                # Get task from queue
                try:
                    task = self._queue.get(timeout=self._tick_interval)
                except Exception:
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
                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                logger.warning(f"Worker loop error (consecutive: {consecutive_errors}): {e}")

                # Add backoff delay if consecutive errors exceed threshold
                if consecutive_errors >= 5:
                    backoff_delay = min((consecutive_errors - 4) * self._tick_interval, 5.0)
                    time.sleep(backoff_delay)
                else:
                    time.sleep(self._tick_interval)

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
            except Exception as e:
                logger.debug(f"Handler failed for {prediction.key}: {e}")
                # Retry if possible
                if task.retries < task.max_retries:
                    task.retries += 1
                    task.priority -= 10  # Lower priority on retry
                    try:
                        self._queue.put_nowait(task)
                    except Exception:
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

    # ========== Default Handlers ==========

    def _handle_draft_reply(self, prediction: Prediction) -> dict[str, Any] | None:
        """Generate and cache a draft reply.

        Args:
            prediction: Prediction with chat_id in params.

        Returns:
            Draft reply data or None.
        """
        chat_id = prediction.params.get("chat_id")
        if not chat_id:
            return None

        try:
            # Import router lazily to avoid circular imports
            from jarvis.router import get_reply_router

            router = get_reply_router()

            # Get recent messages using shared reader (thread-safe via connection pool)
            reader = self._get_reader()
            messages = reader.get_messages(chat_id, limit=10)

            if not messages:
                return None

            # Find last incoming message
            last_incoming = None
            for msg in messages:
                if not msg.is_from_me and msg.text:
                    last_incoming = msg.text
                    break

            if not last_incoming:
                return None

            # Route and generate (acquire MLX lock to serialize GPU ops)
            # Pass conversation_messages so router skips re-fetching from DB
            from models.loader import MLXModelLoader

            with MLXModelLoader._mlx_load_lock:
                result = router.route(
                    incoming=last_incoming,
                    chat_id=chat_id,
                    conversation_messages=messages,
                )

            return {
                "suggestions": [
                    {
                        "text": result.get("response", ""),
                        "confidence": 0.8 if result.get("confidence") == "high" else 0.6,
                    }
                ],
                "prefetched": True,
                "prefetch_time": time.time(),
            }

        except Exception as e:
            logger.debug(f"Draft reply prefetch failed for {chat_id}: {e}")
            return None

    def _handle_embedding(self, prediction: Prediction) -> dict[str, Any] | None:
        """Pre-compute embeddings.

        Args:
            prediction: Prediction with texts in params.

        Returns:
            Embedding data or None.
        """
        texts = prediction.params.get("texts", [])
        if not texts:
            return None

        try:
            from jarvis.embedding_adapter import get_embedder

            embedder = get_embedder()
            from models.loader import MLXModelLoader

            with MLXModelLoader._mlx_load_lock:
                embeddings = embedder.encode(texts)

            return {
                "embeddings": embeddings,
                "texts": texts,
                "prefetched": True,
                "prefetch_time": time.time(),
            }

        except Exception as e:
            logger.debug(f"Embedding prefetch failed: {e}")
            return None

    def _handle_contact_profile(self, prediction: Prediction) -> dict[str, Any] | None:
        """Pre-load contact profile.

        Args:
            prediction: Prediction with chat_id in params.

        Returns:
            Contact data or None.
        """
        chat_id = prediction.params.get("chat_id")
        if not chat_id:
            return None

        try:
            from jarvis.db import get_db

            db = get_db()
            contact = db.get_contact_by_chat_id(chat_id)

            if contact:
                return {
                    "contact": {
                        "id": contact.id,
                        "display_name": contact.display_name,
                        "relationship": contact.relationship,
                        "style_notes": contact.style_notes,
                    },
                    "prefetched": True,
                    "prefetch_time": time.time(),
                }
            return None

        except Exception as e:
            logger.debug(f"Contact profile prefetch failed for {chat_id}: {e}")
            return None

    def _handle_model_warm(self, prediction: Prediction) -> dict[str, Any] | None:
        """Warm up model weights.

        Args:
            prediction: Prediction with model_type in params.

        Returns:
            Warming status or None.
        """
        model_type = prediction.params.get("model_type")
        if not model_type:
            return None

        try:
            from models.loader import MLXModelLoader

            if model_type == "llm":
                from models.loader import get_model

                model = get_model()
                if model and not model.is_loaded():
                    with MLXModelLoader._mlx_load_lock:
                        model.load()
                return {"model": "llm", "warm": True, "prefetch_time": time.time()}

            elif model_type == "embeddings":
                from jarvis.embedding_adapter import get_embedder

                embedder = get_embedder()
                # Warm up with a test embedding
                with MLXModelLoader._mlx_load_lock:
                    embedder.encode(["warmup test"])
                return {"model": "embeddings", "warm": True, "prefetch_time": time.time()}

            return None

        except Exception as e:
            logger.debug(f"Model warming failed for {model_type}: {e}")
            return None

    def _handle_search_results(self, prediction: Prediction) -> dict[str, Any] | None:
        """Pre-compute search results.

        Args:
            prediction: Prediction with query in params.

        Returns:
            Search results or None.
        """
        query = prediction.params.get("query")
        if not query:
            return None

        try:
            from jarvis.db import get_db
            from jarvis.search.vec_search import get_vec_searcher

            db = get_db()
            searcher = get_vec_searcher(db)
            results = searcher.search(query=query, k=10)

            return {
                "query": query,
                "results": [
                    {"trigger": r.last_trigger, "response": r.last_response, "sim": r.similarity}
                    for r in results
                    if r.last_trigger
                ],
                "prefetched": True,
                "prefetch_time": time.time(),
            }

        except Exception as e:
            logger.debug(f"Search prefetch failed for '{query}': {e}")
            return None

    def _handle_vec_index(self, prediction: Prediction) -> dict[str, Any] | None:
        """Pre-load sqlite-vec index.

        Args:
            prediction: Prediction for index loading.

        Returns:
            Index status or None.
        """
        try:
            from jarvis.db import get_db
            from jarvis.search.vec_search import get_vec_searcher

            db = get_db()
            searcher = get_vec_searcher(db)
            if searcher._vec_tables_exist():
                return {
                    "loaded": True,
                    "prefetch_time": time.time(),
                }

            return None

        except Exception as e:
            logger.debug(f"Vec index prefetch failed: {e}")
            return None


from jarvis.utils.singleton import thread_safe_singleton


@thread_safe_singleton
def get_executor() -> PrefetchExecutor:
    """Get or create singleton executor instance."""
    return PrefetchExecutor()


def reset_executor() -> None:
    """Reset singleton executor (stops if running)."""
    instance = get_executor.peek()  # type: ignore[attr-defined]
    if instance is not None:
        instance.stop()
    get_executor.reset()  # type: ignore[attr-defined]
