"""Tests for the prediction strategies."""

import time

from jarvis.prefetch.predictor import (
    AccessPattern,
    ContactFrequencyStrategy,
    ConversationContinuationStrategy,
    ModelWarmingStrategy,
    Prediction,
    PredictionContext,
    PredictionPriority,
    PredictionType,
    PrefetchPredictor,
    RecentContextStrategy,
    UIFocusStrategy,
)


class TestPrediction:
    """Tests for Prediction dataclass."""

    def test_score_calculation(self) -> None:
        """Test prediction score calculation."""
        pred = Prediction(
            type=PredictionType.DRAFT_REPLY,
            priority=PredictionPriority.HIGH,
            confidence=0.8,
            key="test",
        )
        # Score = priority * confidence = 75 * 0.8 = 60
        assert pred.score == 60.0

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        pred = Prediction(
            type=PredictionType.EMBEDDING,
            priority=PredictionPriority.LOW,
            confidence=0.5,
            key="test_key",
        )
        assert pred.params == {}
        assert pred.reason == ""
        assert pred.ttl_seconds == 300.0
        assert pred.tags == []
        assert pred.estimated_cost_ms == 100


class TestAccessPattern:
    """Tests for AccessPattern tracking."""

    def test_record_access(self) -> None:
        """Test recording accesses."""
        pattern = AccessPattern(key="test")
        pattern.record_access()
        pattern.record_access()

        assert pattern.access_count == 2
        assert len(pattern.access_times) == 2
        assert pattern.last_access > 0

    def test_frequency_calculation(self) -> None:
        """Test access frequency calculation."""
        pattern = AccessPattern(key="test")
        # Simulate accesses over 1 second
        pattern.access_times = [time.time() - 1.0, time.time()]
        pattern.access_count = 2

        # 1 access per second = 3600 per hour
        freq = pattern.frequency
        assert 3500 < freq < 3700

    def test_max_access_times(self) -> None:
        """Test that access times are limited to last 100."""
        pattern = AccessPattern(key="test")
        for _ in range(150):
            pattern.record_access()

        assert len(pattern.access_times) == 100


class TestPredictionContext:
    """Tests for PredictionContext."""

    def test_default_values(self) -> None:
        """Test default context values."""
        ctx = PredictionContext()

        assert ctx.current_time is not None
        assert 0 <= ctx.current_hour <= 23
        assert 0 <= ctx.current_day_of_week <= 6
        assert ctx.active_chat_id is None
        assert ctx.recent_chat_ids == []
        assert ctx.recent_searches == []
        assert ctx.battery_level == 1.0
        assert ctx.memory_available_mb == 1000


class TestConversationContinuationStrategy:
    """Tests for conversation continuation prediction."""

    def test_active_conversation_prediction(self) -> None:
        """Test prediction for active conversations."""
        strategy = ConversationContinuationStrategy(
            active_window_minutes=30,
            response_probability_threshold=0.3,
        )

        # Record recent messages
        strategy.record_message("chat123", is_from_me=False)
        strategy.record_message("chat123", is_from_me=False)

        ctx = PredictionContext()
        predictions = strategy.predict(ctx)

        assert len(predictions) > 0
        assert predictions[0].params.get("chat_id") == "chat123"
        assert predictions[0].type == PredictionType.DRAFT_REPLY

    def test_stale_conversation_no_prediction(self) -> None:
        """Test no prediction for stale conversations."""
        strategy = ConversationContinuationStrategy(
            active_window_minutes=1,  # Very short window
            response_probability_threshold=0.5,
        )

        # Record old message
        strategy._recent_messages["chat123"] = [time.time() - 120]  # 2 minutes ago

        ctx = PredictionContext()
        predictions = strategy.predict(ctx)

        # Should be filtered out due to low confidence
        chat_predictions = [p for p in predictions if p.params.get("chat_id") == "chat123"]
        assert len(chat_predictions) == 0


class TestRecentContextStrategy:
    """Tests for recent context prediction."""

    def test_context_embedding_prediction(self) -> None:
        """Test embedding prediction for recent context."""
        strategy = RecentContextStrategy()

        strategy.record_context("chat123", "Hello there")
        strategy.record_context("chat123", "How are you?")

        ctx = PredictionContext(active_chat_id="chat123")
        predictions = strategy.predict(ctx)

        assert len(predictions) > 0
        embed_preds = [p for p in predictions if p.type == PredictionType.EMBEDDING]
        assert len(embed_preds) > 0
        assert embed_preds[0].params.get("chat_id") == "chat123"

    def test_no_context_no_prediction(self) -> None:
        """Test no prediction without context."""
        strategy = RecentContextStrategy()
        ctx = PredictionContext(active_chat_id="chat123")
        predictions = strategy.predict(ctx)

        # No predictions without recorded context
        chat_predictions = [p for p in predictions if p.params.get("chat_id") == "chat123"]
        assert len(chat_predictions) == 0


class TestUIFocusStrategy:
    """Tests for UI focus prediction."""

    def test_focus_prediction(self) -> None:
        """Test prediction on focus event."""
        strategy = UIFocusStrategy(focus_threshold_ms=100)

        strategy.record_focus("chat123")
        time.sleep(0.15)  # Wait past threshold

        ctx = PredictionContext(active_chat_id="chat123")
        predictions = strategy.predict(ctx)

        assert len(predictions) > 0
        assert predictions[0].priority == PredictionPriority.CRITICAL
        assert predictions[0].params.get("chat_id") == "chat123"

    def test_hover_prediction(self) -> None:
        """Test prediction on hover event."""
        strategy = UIFocusStrategy()

        strategy.record_hover("chat456")

        ctx = PredictionContext(active_chat_id=None)  # Not the hovered chat
        predictions = strategy.predict(ctx)

        hover_preds = [p for p in predictions if p.params.get("chat_id") == "chat456"]
        assert len(hover_preds) > 0
        assert hover_preds[0].priority == PredictionPriority.HIGH


class TestModelWarmingStrategy:
    """Tests for model warming prediction."""

    def test_warming_during_active_period(self) -> None:
        """Test model warming during active period."""
        strategy = ModelWarmingStrategy(idle_threshold_seconds=300)
        strategy.record_activity()

        ctx = PredictionContext()
        predictions = strategy.predict(ctx)

        assert len(predictions) > 0
        model_preds = [p for p in predictions if p.type == PredictionType.MODEL_WARM]
        assert len(model_preds) > 0

    def test_no_warming_when_idle(self) -> None:
        """Test no warming when idle."""
        strategy = ModelWarmingStrategy(idle_threshold_seconds=1)
        strategy._last_activity = time.time() - 10  # 10 seconds ago

        ctx = PredictionContext()
        predictions = strategy.predict(ctx)

        assert len(predictions) == 0

    def test_battery_aware_warming(self) -> None:
        """Test battery-aware warming."""
        strategy = ModelWarmingStrategy(idle_threshold_seconds=300)
        strategy.record_activity()

        ctx = PredictionContext(battery_level=0.1)  # Low battery
        predictions = strategy.predict(ctx)

        # Should filter out expensive warming
        expensive_preds = [p for p in predictions if p.estimated_cost_ms >= 500]
        assert len(expensive_preds) == 0


class TestPrefetchPredictor:
    """Tests for the main predictor."""

    def test_predict_returns_sorted_predictions(self) -> None:
        """Test predictions are sorted by score."""
        predictor = PrefetchPredictor(max_predictions=10, min_confidence=0.1)

        # Record activity to generate predictions
        predictor.record_message("chat1", "Hello", is_from_me=False)
        predictor.record_focus("chat1")

        predictions = predictor.predict()

        # Check sorted by score (descending)
        for i in range(len(predictions) - 1):
            assert predictions[i].score >= predictions[i + 1].score

    def test_deduplication_by_key(self) -> None:
        """Test duplicate predictions are deduplicated."""
        predictor = PrefetchPredictor()

        # Record same chat multiple times
        for _ in range(5):
            predictor.record_focus("chat123")

        predictions = predictor.predict()

        # Count predictions for chat123
        chat_keys = [p.key for p in predictions if "chat123" in p.key]
        unique_keys = set(chat_keys)

        # Should not have duplicates with same key
        assert len(chat_keys) == len(unique_keys)

    def test_max_predictions_limit(self) -> None:
        """Test max predictions limit is respected."""
        predictor = PrefetchPredictor(max_predictions=5, min_confidence=0.0)

        # Generate many predictions
        for i in range(20):
            predictor.record_message(f"chat{i}", f"message{i}", is_from_me=False)

        predictions = predictor.predict()
        assert len(predictions) <= 5

    def test_min_confidence_filter(self) -> None:
        """Test minimum confidence filter."""
        predictor = PrefetchPredictor(min_confidence=0.9)

        predictions = predictor.predict()

        for pred in predictions:
            assert pred.confidence >= 0.9

    def test_record_message_updates_context(self) -> None:
        """Test record_message updates recent chat IDs."""
        predictor = PrefetchPredictor()

        predictor.record_message("chat1", "test", is_from_me=False)
        predictor.record_message("chat2", "test", is_from_me=False)

        assert "chat2" in predictor._context.recent_chat_ids
        assert "chat1" in predictor._context.recent_chat_ids
        assert predictor._context.recent_chat_ids[0] == "chat2"  # Most recent first

    def test_record_search(self) -> None:
        """Test search recording."""
        predictor = PrefetchPredictor()

        predictor.record_search("meeting notes")
        predictor.record_search("project plan")

        assert "meeting notes" in predictor._context.recent_searches
        assert "project plan" in predictor._context.recent_searches

    def test_update_context(self) -> None:
        """Test context updates."""
        predictor = PrefetchPredictor()

        predictor.update_context(battery_level=0.5, memory_available_mb=2000)

        assert predictor._context.battery_level == 0.5
        assert predictor._context.memory_available_mb == 2000

    def test_get_strategy(self) -> None:
        """Test getting strategy by name."""
        predictor = PrefetchPredictor()

        strategy = predictor.get_strategy("contact_frequency")
        assert strategy is not None
        assert isinstance(strategy, ContactFrequencyStrategy)

        missing = predictor.get_strategy("nonexistent")
        assert missing is None
