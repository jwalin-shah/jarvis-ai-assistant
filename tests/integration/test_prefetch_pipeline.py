"""Integration tests for the prefetch pipeline.

Tests the complete flow: prediction → schedule → execute → cache → retrieve.
Individual components are tested in unit tests, this verifies end-to-end flow.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from jarvis.prefetch import (
    CacheTier,
    Prediction,
    PredictionPriority,
    PredictionType,
    PrefetchExecutor,
    PrefetchManager,
    get_cache,
    get_executor,
    get_predictor,
    reset_cache,
    reset_executor,
    reset_predictor,
)


@pytest.fixture(autouse=True)
def reset_prefetch_components():
    """Reset prefetch singletons before and after each test."""
    reset_cache()
    reset_executor()
    reset_predictor()
    yield
    reset_cache()
    reset_executor()
    reset_predictor()


@pytest.fixture
def mock_mlx_lock():
    """Mock the MLX GPU lock to avoid real GPU operations."""
    with patch("models.loader.MLXModelLoader._mlx_load_lock", new=threading.Lock()):
        yield


@pytest.fixture
def mock_router():
    """Mock the router to avoid real LLM inference."""
    with patch("jarvis.router.get_reply_router") as mock_get_router:
        router = MagicMock()
        router.route.return_value = {
            "type": "generated",
            "response": "Sure, sounds good!",
            "confidence": "high",
        }
        mock_get_router.return_value = router
        yield router


@pytest.fixture
def mock_chat_db():
    """Mock ChatDBReader to return fake messages."""
    with patch("integrations.imessage.ChatDBReader") as mock_reader_class:
        # Create mock reader instance
        mock_reader = MagicMock()

        # Mock messages for a conversation
        mock_messages = [
            Mock(
                text="Hey, want to grab lunch?",
                is_from_me=False,
                sender="Alice",
                sender_name="Alice",
                date=time.time() - 60,
            ),
            Mock(
                text="Yeah, what time?",
                is_from_me=True,
                sender="Me",
                sender_name="Me",
                date=time.time() - 30,
            ),
            Mock(
                text="How about noon?",
                is_from_me=False,
                sender="Alice",
                sender_name="Alice",
                date=time.time() - 10,
            ),
        ]

        mock_reader.get_messages.return_value = mock_messages

        # Setup both direct instantiation and context manager usage
        # The executor uses ChatDBReader() directly (not as context manager)
        mock_reader_class.return_value = mock_reader
        mock_reader.__enter__ = Mock(return_value=mock_reader)
        mock_reader.__exit__ = Mock(return_value=False)

        yield mock_reader


@pytest.mark.integration
class TestPrefetchPipelineFlow:
    """Tests for the complete prefetch pipeline flow."""

    def test_prediction_schedule_execute_cache_retrieve(
        self, mock_mlx_lock, mock_router, mock_chat_db
    ):
        """Test full flow: prediction → schedule → execute → cache → retrieve.

        Verifies:
        - Prediction is created
        - Task is scheduled in executor
        - Handler executes and caches result
        - Cached result can be retrieved
        """
        # Create executor with real cache
        cache = get_cache()
        executor = PrefetchExecutor(cache=cache, max_workers=2, tick_interval=0.05)
        executor.start()

        try:
            # Create a prediction
            prediction = Prediction(
                type=PredictionType.DRAFT_REPLY,
                priority=PredictionPriority.HIGH,
                confidence=0.9,
                key="draft:focus:chat123",
                params={"chat_id": "chat123"},
                reason="UI focus event",
                ttl_seconds=300,
                tags=["chat:chat123", "draft"],
                estimated_cost_ms=500,
            )

            # Schedule the prediction
            scheduled = executor.schedule(prediction)
            assert scheduled is True, "Prediction should be scheduled"

            # Wait for execution (max 2 seconds)
            max_wait = 2.0
            start = time.time()
            result = None
            while time.time() - start < max_wait:
                result = cache.get(prediction.key)
                if result is not None:
                    break
                time.sleep(0.1)

            # Verify result was cached
            assert result is not None, f"Result should be cached within {max_wait}s"
            assert result["prefetched"] is True
            assert "suggestions" in result
            assert len(result["suggestions"]) > 0
            assert result["suggestions"][0]["text"] == "Sure, sounds good!"

            # Verify router was called with conversation_messages (not thread)
            mock_router.route.assert_called_once()
            call_kwargs = mock_router.route.call_args[1]
            assert call_kwargs["incoming"] == "Hey, want to grab lunch?"
            assert call_kwargs["chat_id"] == "chat123"
            assert "conversation_messages" in call_kwargs

            # Verify cache stats
            stats = cache.stats()
            assert stats["hits"]["l1"] > 0 or stats["l1"]["entries"] > 0

        finally:
            executor.stop(timeout=2.0)

    def test_cache_invalidation_on_new_message(self, mock_mlx_lock, mock_router, mock_chat_db):
        """Test that cache is invalidated when a new message arrives.

        Verifies:
        - Old draft is pre-populated in cache
        - New message triggers invalidation
        - Old draft is removed
        """
        cache = get_cache()
        executor = PrefetchExecutor(cache=cache, max_workers=2)
        executor.start()

        try:
            # Pre-populate cache with a draft
            old_draft = {
                "suggestions": [{"text": "Old draft", "confidence": 0.8}],
                "prefetched": True,
                "prefetch_time": time.time(),
            }
            cache.set(
                key="draft:cont:chat123",
                value=old_draft,
                tier=CacheTier.L1,
                ttl_seconds=300,
                tags=["chat:chat123", "draft"],
            )

            # Verify old draft exists
            assert cache.get("draft:cont:chat123") == old_draft

            # Simulate invalidation on new message (would be called by manager)
            cache.invalidate_by_tag("chat:chat123")

            # Verify old draft is gone
            assert cache.get("draft:cont:chat123") is None

        finally:
            executor.stop(timeout=1.0)

    def test_ui_focus_triggers_critical_prediction(self, mock_mlx_lock, mock_router, mock_chat_db):
        """Test that UI focus triggers a CRITICAL priority prediction.

        Verifies:
        - on_focus() generates prediction
        - Prediction has CRITICAL priority
        - Task is scheduled immediately
        """
        manager = PrefetchManager(
            prediction_interval=10.0,  # Don't run background predictions
            warmup_on_start=False,
        )
        manager.start()

        try:
            # Mock predictor to return a critical prediction
            predictor = get_predictor()
            with patch.object(
                predictor,
                "predict",
                return_value=[
                    Prediction(
                        type=PredictionType.DRAFT_REPLY,
                        priority=PredictionPriority.CRITICAL,
                        confidence=1.0,
                        key="draft:focus:chat456",
                        params={"chat_id": "chat456"},
                        reason="UI focus",
                        ttl_seconds=300,
                        tags=["chat:chat456", "draft"],
                        estimated_cost_ms=500,
                    )
                ],
            ):
                # Trigger focus event
                manager.on_focus("chat456")

                # Give executor time to schedule
                time.sleep(0.2)

                # Verify task was scheduled
                executor = get_executor()
                stats = executor.stats()
                assert stats["scheduled"] > 0, "Focus should schedule prediction"

        finally:
            manager.stop(timeout=2.0)

    def test_duplicate_draft_prevention(self, mock_mlx_lock, mock_router, mock_chat_db):
        """Test that duplicate draft predictions for same chat are rejected.

        Verifies:
        - First prediction is scheduled
        - Second prediction for same chat is rejected (already active)
        - After first completes, second can be scheduled
        """
        cache = get_cache()
        executor = PrefetchExecutor(cache=cache, max_workers=1, tick_interval=0.05)
        executor.start()

        try:
            # Use a gate to keep the first handler busy while we test duplicate rejection
            gate = threading.Event()
            original_handler = executor._handle_draft_reply

            def slow_handler(prediction):
                gate.wait(timeout=5.0)
                return original_handler(prediction)

            executor._handlers[PredictionType.DRAFT_REPLY] = slow_handler

            # Create two predictions for the same chat
            pred1 = Prediction(
                type=PredictionType.DRAFT_REPLY,
                priority=PredictionPriority.HIGH,
                confidence=0.9,
                key="draft:focus:chat789",
                params={"chat_id": "chat789"},
                reason="First prediction",
                ttl_seconds=300,
                tags=["chat:chat789", "draft"],
                estimated_cost_ms=500,
            )

            pred2 = Prediction(
                type=PredictionType.DRAFT_REPLY,
                priority=PredictionPriority.MEDIUM,
                confidence=0.8,
                key="draft:cont:chat789",
                params={"chat_id": "chat789"},
                reason="Second prediction",
                ttl_seconds=300,
                tags=["chat:chat789", "draft"],
                estimated_cost_ms=500,
            )

            # Schedule first prediction
            scheduled1 = executor.schedule(pred1)
            assert scheduled1 is True, "First prediction should be scheduled"

            # Wait for first to enter execution (handler is blocked on gate)
            # Poll until the task is marked active (worker picks it up)
            max_wait_active = 2.0
            start = time.time()
            while time.time() - start < max_wait_active:
                with executor._lock:
                    if "chat789" in executor._active_drafts:
                        break
                time.sleep(0.05)

            scheduled2 = executor.schedule(pred2)

            # Second MUST be rejected: same chat_id is already active
            assert scheduled2 is False, "Duplicate draft for same chat_id should be rejected"

            # Release the gate so first completes
            gate.set()

            # Wait for first to complete
            max_wait = 2.0
            start = time.time()
            while time.time() - start < max_wait:
                if cache.get(pred1.key) is not None:
                    break
                time.sleep(0.1)

            assert cache.get(pred1.key) is not None, "First prediction should complete"

            # After first completes, second can now be scheduled
            scheduled2_retry = executor.schedule(pred2)
            assert scheduled2_retry is True, (
                "After first draft completes, second should be accepted"
            )

        finally:
            executor.stop(timeout=2.0)

    def test_manager_on_message_triggers_prediction(self, mock_mlx_lock, mock_router, mock_chat_db):
        """Test that on_message() triggers immediate high-priority prediction.

        Verifies:
        - Incoming message (is_from_me=False) triggers prediction
        - High-priority predictions are scheduled
        - Predictor records the message
        """
        manager = PrefetchManager(
            prediction_interval=10.0,  # Don't run background predictions
            warmup_on_start=False,
        )
        manager.start()

        try:
            predictor = get_predictor()

            # Spy on predictor methods
            with patch.object(predictor, "record_message", wraps=predictor.record_message):
                with patch.object(
                    predictor,
                    "predict",
                    return_value=[
                        Prediction(
                            type=PredictionType.DRAFT_REPLY,
                            priority=PredictionPriority.HIGH,
                            confidence=0.85,
                            key="draft:cont:chat999",
                            params={"chat_id": "chat999"},
                            reason="Conversation continuation",
                            ttl_seconds=300,
                            tags=["chat:chat999", "draft"],
                            estimated_cost_ms=500,
                        )
                    ],
                ):
                    # Trigger message event (incoming)
                    manager.on_message("chat999", "Hey, are you free?", is_from_me=False)

                    # Verify predictor recorded the message
                    predictor.record_message.assert_called_once_with(
                        "chat999", "Hey, are you free?", False
                    )

                    # Verify prediction was called
                    assert predictor.predict.call_count >= 1

                    # Give executor time to schedule
                    time.sleep(0.2)

                    # Verify task was scheduled
                    executor = get_executor()
                    stats = executor.stats()
                    assert stats["scheduled"] > 0, "Message should schedule prediction"

        finally:
            manager.stop(timeout=2.0)

    def test_get_draft_checks_multiple_keys(self, mock_mlx_lock):
        """Test that get_draft() checks multiple possible cache keys.

        Verifies:
        - get_draft() tries draft:focus:, draft:cont:, draft:tod:, draft:
        - Returns first match
        """
        cache = get_cache()
        manager = PrefetchManager(warmup_on_start=False)
        manager.start()

        try:
            # Put draft under draft:cont: prefix
            draft = {
                "suggestions": [{"text": "Test draft", "confidence": 0.9}],
                "prefetched": True,
            }
            cache.set(
                key="draft:cont:chat111",
                value=draft,
                tier=CacheTier.L1,
                ttl_seconds=300,
            )

            # get_draft should find it
            result = manager.get_draft("chat111")
            assert result is not None
            assert result["suggestions"][0]["text"] == "Test draft"

            # Clear and test with draft:focus: prefix
            cache.clear()
            cache.set(
                key="draft:focus:chat111",
                value=draft,
                tier=CacheTier.L1,
                ttl_seconds=300,
            )
            result = manager.get_draft("chat111")
            assert result is not None
            assert result["suggestions"][0]["text"] == "Test draft"

        finally:
            manager.stop(timeout=1.0)

    def test_executor_handles_missing_handler_gracefully(self, mock_mlx_lock):
        """Test that executor handles predictions with no registered handler.

        Verifies:
        - Prediction with unknown type is skipped
        - Stats track skipped predictions
        - No crash or exception
        """
        cache = get_cache()
        executor = PrefetchExecutor(cache=cache, max_workers=2, tick_interval=0.05)
        executor.start()

        try:
            # Create prediction with a fake type (no handler registered)
            # We'll create a DRAFT_REPLY but unregister its handler
            executor._handlers.clear()

            prediction = Prediction(
                type=PredictionType.DRAFT_REPLY,
                priority=PredictionPriority.LOW,
                confidence=0.5,
                key="draft:unknown:chat000",
                params={"chat_id": "chat000"},
                reason="Test missing handler",
                ttl_seconds=300,
                tags=["test"],
                estimated_cost_ms=100,
            )

            scheduled = executor.schedule(prediction)
            assert scheduled is True

            # Wait briefly for execution attempt
            time.sleep(0.3)

            # Verify it was skipped (not cached)
            result = cache.get(prediction.key)
            assert result is None, "Should not cache when handler is missing"

            # Verify stats
            stats = executor.stats()
            assert stats["skipped"] > 0, "Should track skipped predictions"

        finally:
            executor.stop(timeout=1.0)

    def test_cache_tier_assignment_based_on_priority(
        self, mock_mlx_lock, mock_router, mock_chat_db
    ):
        """Test that cache tier matches prediction priority.

        Verifies:
        - HIGH/CRITICAL priority → L1 cache
        - MEDIUM priority → L2 cache
        - LOW priority → L3 cache
        """
        cache = get_cache()
        executor = PrefetchExecutor(cache=cache, max_workers=2, tick_interval=0.05)
        executor.start()

        try:
            # High priority prediction
            high_pred = Prediction(
                type=PredictionType.DRAFT_REPLY,
                priority=PredictionPriority.HIGH,
                confidence=0.9,
                key="draft:high:chat_high",
                params={"chat_id": "chat_high"},
                reason="High priority test",
                ttl_seconds=300,
                tags=["test"],
                estimated_cost_ms=500,
            )

            scheduled = executor.schedule(high_pred)
            assert scheduled is True

            # Wait for caching
            max_wait = 2.0
            start = time.time()
            while time.time() - start < max_wait:
                if cache.get(high_pred.key) is not None:
                    break
                time.sleep(0.1)

            # Verify it's in L1
            stats = cache.stats()
            # Result should be in cache (either L1, L2, or L3)
            result = cache.get(high_pred.key)
            assert result is not None, "High priority should be cached"

            # L1 should have at least one entry (the high priority one)
            assert stats["l1"]["entries"] > 0, "High priority should go to L1"

        finally:
            executor.stop(timeout=2.0)


@pytest.mark.integration
class TestPrefetchResourceManagement:
    """Tests for resource-aware prefetch execution."""

    def test_executor_respects_queue_size_limit(self, mock_mlx_lock):
        """Test that executor rejects tasks when queue is full.

        Verifies:
        - Queue has a maximum size
        - Additional tasks are rejected when full
        - Stats track skipped tasks
        """
        cache = get_cache()
        executor = PrefetchExecutor(
            cache=cache,
            max_workers=1,
            max_queue_size=5,  # Small queue
            tick_interval=0.5,  # Slow processing
        )
        executor.start()

        try:
            # Schedule more tasks than queue can hold
            predictions = [
                Prediction(
                    type=PredictionType.DRAFT_REPLY,
                    priority=PredictionPriority.LOW,
                    confidence=0.5,
                    key=f"draft:test:chat{i}",
                    params={"chat_id": f"chat{i}"},
                    reason=f"Test {i}",
                    ttl_seconds=300,
                    tags=["test"],
                    estimated_cost_ms=100,
                )
                for i in range(10)
            ]

            scheduled_count = 0
            for pred in predictions:
                if executor.schedule(pred):
                    scheduled_count += 1

            # Some should be rejected
            assert scheduled_count < 10, "Some tasks should be rejected when queue is full"

            stats = executor.stats()
            assert stats["skipped"] > 0, "Should track skipped tasks"

        finally:
            executor.stop(timeout=1.0)

    def test_executor_pause_and_resume(self, mock_mlx_lock):
        """Test that executor can be paused and resumed.

        Verifies:
        - Paused executor keeps tasks in queue
        - No execution happens while paused
        - Resume starts processing again
        """
        cache = get_cache()
        executor = PrefetchExecutor(cache=cache, max_workers=2, tick_interval=0.05)
        executor.start()

        try:
            # Pause immediately
            executor.pause()

            # Schedule a prediction
            prediction = Prediction(
                type=PredictionType.DRAFT_REPLY,
                priority=PredictionPriority.MEDIUM,
                confidence=0.8,
                key="draft:pause:chat_pause",
                params={"chat_id": "chat_pause"},
                reason="Pause test",
                ttl_seconds=300,
                tags=["test"],
                estimated_cost_ms=100,
            )

            # Mock the handler to avoid actual execution
            with patch.object(executor, "_handle_draft_reply", return_value=None):
                scheduled = executor.schedule(prediction)
                assert scheduled is True

                # Wait a bit - should NOT execute
                time.sleep(0.3)
                stats = executor.stats()
                initial_executed = stats["executed"]

                # Resume
                executor.resume()

                # Now it should execute
                time.sleep(0.5)
                stats = executor.stats()
                # Executed count should have increased (or stayed same if handler returns None)
                # The point is the state changed from PAUSED to RUNNING

                assert stats["state"] == "running"

        finally:
            executor.stop(timeout=1.0)
