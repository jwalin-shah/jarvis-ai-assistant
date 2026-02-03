"""Speculative prefetching system for near-instant responses.

This module provides intelligent prediction and caching to minimize
perceived latency in the JARVIS assistant.

Components:
- cache: Multi-tier caching system (L1/L2/L3)
- predictor: Prediction strategies for speculative prefetching
- executor: Background prefetch execution with resource management
- invalidation: Smart cache invalidation

Usage:
    from jarvis.prefetch import (
        get_cache,
        get_predictor,
        get_executor,
        get_invalidator,
        PrefetchManager,
    )

    # Use the unified manager
    manager = PrefetchManager()
    manager.start()

    # Or use individual components
    cache = get_cache()
    predictor = get_predictor()
    executor = get_executor()
    invalidator = get_invalidator()

Metrics target: 80% cache hit rate, 10x perceived latency improvement.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

from jarvis.prefetch.cache import (
    CacheEntry,
    CacheError,
    CacheStats,
    CacheTier,
    L1Cache,
    L2Cache,
    L3Cache,
    MultiTierCache,
    get_cache,
    reset_cache,
)
from jarvis.prefetch.executor import (
    ExecutorState,
    ExecutorStats,
    PrefetchExecutor,
    PrefetchHandler,
    PrefetchTask,
    ResourceManager,
    get_executor,
    reset_executor,
)
from jarvis.prefetch.invalidation import (
    CacheInvalidator,
    DependencyTracker,
    InvalidationEvent,
    InvalidationReason,
    InvalidationRule,
    InvalidationStats,
    get_invalidator,
    reset_invalidator,
)
from jarvis.prefetch.predictor import (
    AccessPattern,
    ContactFrequencyStrategy,
    ConversationContinuationStrategy,
    ModelWarmingStrategy,
    PredictionContext,
    PredictionPriority,
    PredictionStrategy,
    PredictionType,
    Prediction,
    PrefetchPredictor,
    RecentContextStrategy,
    TimeOfDayStrategy,
    UIFocusStrategy,
    get_predictor,
    reset_predictor,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Cache
    "CacheEntry",
    "CacheError",
    "CacheStats",
    "CacheTier",
    "L1Cache",
    "L2Cache",
    "L3Cache",
    "MultiTierCache",
    "get_cache",
    "reset_cache",
    # Predictor
    "AccessPattern",
    "ContactFrequencyStrategy",
    "ConversationContinuationStrategy",
    "ModelWarmingStrategy",
    "PredictionContext",
    "PredictionPriority",
    "PredictionStrategy",
    "PredictionType",
    "Prediction",
    "PrefetchPredictor",
    "RecentContextStrategy",
    "TimeOfDayStrategy",
    "UIFocusStrategy",
    "get_predictor",
    "reset_predictor",
    # Executor
    "ExecutorState",
    "ExecutorStats",
    "PrefetchExecutor",
    "PrefetchHandler",
    "PrefetchTask",
    "ResourceManager",
    "get_executor",
    "reset_executor",
    # Invalidation
    "CacheInvalidator",
    "DependencyTracker",
    "InvalidationEvent",
    "InvalidationReason",
    "InvalidationRule",
    "InvalidationStats",
    "get_invalidator",
    "reset_invalidator",
    # Manager
    "PrefetchManager",
    "get_prefetch_manager",
    "reset_prefetch_manager",
]


class PrefetchManager:
    """Unified manager for the prefetch system.

    Coordinates cache, predictor, executor, and invalidator
    for optimal speculative prefetching.

    Usage:
        manager = PrefetchManager()
        manager.start()

        # Record events
        manager.on_message("chat123", "Hello", is_from_me=False)
        manager.on_focus("chat123")

        # Get cached data (with prefetch scheduling)
        draft = manager.get_draft("chat123")

        manager.stop()
    """

    def __init__(
        self,
        prediction_interval: float = 5.0,  # How often to generate predictions
        cleanup_interval: float = 60.0,  # How often to cleanup expired entries
        warmup_on_start: bool = True,  # Run startup warmup
    ) -> None:
        """Initialize the prefetch manager.

        Args:
            prediction_interval: Seconds between prediction cycles.
            cleanup_interval: Seconds between cache cleanup.
            warmup_on_start: Whether to run startup warming.
        """
        self._cache = get_cache()
        self._predictor = get_predictor()
        self._executor = get_executor()
        self._invalidator = get_invalidator()

        self._prediction_interval = prediction_interval
        self._cleanup_interval = cleanup_interval
        self._warmup_on_start = warmup_on_start

        self._running = False
        self._prediction_thread: threading.Thread | None = None
        self._cleanup_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        self._lock = threading.RLock()

    def start(self) -> None:
        """Start the prefetch system."""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._shutdown_event.clear()

            # Start executor
            self._executor.start()

            # Start prediction loop
            self._prediction_thread = threading.Thread(
                target=self._prediction_loop,
                name="prefetch-prediction",
                daemon=True,
            )
            self._prediction_thread.start()

            # Start cleanup loop
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                name="prefetch-cleanup",
                daemon=True,
            )
            self._cleanup_thread.start()

            # Run startup warmup
            if self._warmup_on_start:
                self._startup_warmup()

            logger.info("Prefetch manager started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the prefetch system.

        Args:
            timeout: Maximum seconds to wait for shutdown.
        """
        with self._lock:
            if not self._running:
                return

            self._running = False
            self._shutdown_event.set()

        # Wait for threads
        if self._prediction_thread:
            self._prediction_thread.join(timeout=timeout)
            self._prediction_thread = None

        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=timeout)
            self._cleanup_thread = None

        # Stop executor
        self._executor.stop(timeout=timeout)

        logger.info("Prefetch manager stopped")

    def on_message(self, chat_id: str, text: str, is_from_me: bool = False) -> None:
        """Handle new message event.

        Args:
            chat_id: Chat identifier.
            text: Message text.
            is_from_me: Whether message was from user.
        """
        # Update predictor
        self._predictor.record_message(chat_id, text, is_from_me)

        # Invalidate stale cache
        self._invalidator.on_new_message(chat_id, text, is_from_me)

        # Trigger immediate prediction for active conversation
        if not is_from_me:
            predictions = self._predictor.predict()
            high_priority = [
                p for p in predictions if p.priority >= PredictionPriority.HIGH
            ]
            self._executor.schedule_batch(high_priority)

    def on_focus(self, chat_id: str) -> None:
        """Handle UI focus event.

        Args:
            chat_id: Chat identifier.
        """
        self._predictor.record_focus(chat_id)

        # Immediately schedule high-priority prefetch
        predictions = self._predictor.predict()
        critical = [p for p in predictions if p.priority == PredictionPriority.CRITICAL]
        self._executor.schedule_batch(critical)

    def on_hover(self, chat_id: str) -> None:
        """Handle UI hover event.

        Args:
            chat_id: Chat identifier.
        """
        self._predictor.record_hover(chat_id)

    def on_search(self, query: str) -> None:
        """Handle search event.

        Args:
            query: Search query.
        """
        self._predictor.record_search(query)

    def get(self, key: str) -> Any | None:
        """Get cached value.

        Args:
            key: Cache key.

        Returns:
            Cached value or None.
        """
        # Record access for prediction
        self._predictor.record_access(key)
        return self._cache.get(key)

    def get_draft(self, chat_id: str) -> dict[str, Any] | None:
        """Get prefetched draft for a chat.

        Args:
            chat_id: Chat identifier.

        Returns:
            Draft data or None if not prefetched.
        """
        # Try multiple possible keys
        for prefix in ["draft:focus:", "draft:cont:", "draft:tod:", "draft:"]:
            result = self._cache.get(f"{prefix}{chat_id}")
            if result is not None:
                return result
        return None

    def get_embedding(self, chat_id: str) -> dict[str, Any] | None:
        """Get prefetched embeddings for a chat.

        Args:
            chat_id: Chat identifier.

        Returns:
            Embedding data or None if not prefetched.
        """
        return self._cache.get(f"embed:ctx:{chat_id}")

    def get_contact(self, chat_id: str) -> dict[str, Any] | None:
        """Get prefetched contact profile.

        Args:
            chat_id: Chat identifier.

        Returns:
            Contact data or None if not prefetched.
        """
        return self._cache.get(f"contact:{chat_id}")

    def schedule_prefetch(self, prediction: Prediction) -> bool:
        """Manually schedule a prefetch.

        Args:
            prediction: Prediction to prefetch.

        Returns:
            True if scheduled.
        """
        return self._executor.schedule(prediction)

    def invalidate(self, chat_id: str | None = None, tags: list[str] | None = None) -> int:
        """Manually invalidate cache entries.

        Args:
            chat_id: Chat ID to invalidate.
            tags: Tags to invalidate.

        Returns:
            Number of entries invalidated.
        """
        keys = []
        all_tags = tags or []

        if chat_id:
            keys.extend([
                f"draft:{chat_id}",
                f"draft:cont:{chat_id}",
                f"draft:focus:{chat_id}",
                f"embed:ctx:{chat_id}",
                f"contact:{chat_id}",
            ])
            all_tags.append(f"chat:{chat_id}")

        return self._invalidator.manual_invalidate(keys=keys, tags=all_tags)

    def stats(self) -> dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "cache": self._cache.stats(),
            "executor": self._executor.stats(),
            "invalidator": self._invalidator.stats(),
            "running": self._running,
        }

    def _prediction_loop(self) -> None:
        """Background prediction loop."""
        while not self._shutdown_event.is_set():
            try:
                # Generate predictions
                predictions = self._predictor.predict()

                # Schedule prefetches
                scheduled = self._executor.schedule_batch(predictions)
                if scheduled > 0:
                    logger.debug(f"Scheduled {scheduled} prefetches")

            except Exception as e:
                logger.warning(f"Prediction loop error: {e}")

            # Wait for next cycle
            self._shutdown_event.wait(self._prediction_interval)

    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while not self._shutdown_event.is_set():
            try:
                # Cleanup expired entries
                count = self._invalidator.cleanup_expired()
                if count > 0:
                    logger.debug(f"Cleaned up {count} expired entries")

            except Exception as e:
                logger.warning(f"Cleanup loop error: {e}")

            # Wait for next cycle
            self._shutdown_event.wait(self._cleanup_interval)

    def _startup_warmup(self) -> None:
        """Run startup warming sequence."""
        logger.info("Running startup warmup...")

        # Schedule model warming
        warmup_predictions = [
            Prediction(
                type=PredictionType.MODEL_WARM,
                priority=PredictionPriority.HIGH,
                confidence=1.0,
                key="warm:llm",
                params={"model_type": "llm"},
                reason="Startup warmup",
                ttl_seconds=600,
                tags=["model", "llm"],
                estimated_cost_ms=2000,
            ),
            Prediction(
                type=PredictionType.MODEL_WARM,
                priority=PredictionPriority.MEDIUM,
                confidence=1.0,
                key="warm:embeddings",
                params={"model_type": "embeddings"},
                reason="Startup warmup",
                ttl_seconds=600,
                tags=["model", "embeddings"],
                estimated_cost_ms=500,
            ),
            Prediction(
                type=PredictionType.FAISS_INDEX,
                priority=PredictionPriority.MEDIUM,
                confidence=1.0,
                key="warm:faiss",
                params={},
                reason="Startup warmup",
                ttl_seconds=1800,
                tags=["faiss"],
                estimated_cost_ms=200,
            ),
        ]

        self._executor.schedule_batch(warmup_predictions)

        # Also run initial predictions
        predictions = self._predictor.predict()
        self._executor.schedule_batch(predictions)


# Singleton instance
_manager: PrefetchManager | None = None
_manager_lock = threading.Lock()


def get_prefetch_manager() -> PrefetchManager:
    """Get or create singleton prefetch manager."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = PrefetchManager()
        return _manager


def reset_prefetch_manager() -> None:
    """Reset singleton prefetch manager (stops if running)."""
    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager.stop()
        _manager = None
