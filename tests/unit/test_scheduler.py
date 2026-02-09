"""Tests for the draft scheduling system.

Tests cover:
- Scheduler models (ScheduledItem, Priority, TimingSuggestion)
- Queue operations (add, get, cancel, reschedule)
- Timing analysis (suggest times, quiet hours)
- Executor (rate limiting, duplicate detection)
- Main scheduler lifecycle
"""

from __future__ import annotations

import tempfile
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from jarvis.scheduler.executor import (
    RateLimitConfig,
    SendExecutor,
    get_executor,
    reset_executor,
)
from jarvis.scheduler.models import (
    ContactTimingPrefs,
    Priority,
    QuietHours,
    ScheduledItem,
    ScheduledStatus,
    SendResult,
    TimingSuggestion,
)
from jarvis.scheduler.queue import SchedulerQueue, get_scheduler_queue, reset_scheduler_queue
from jarvis.scheduler.scheduler import (
    DraftScheduler,
    get_scheduler,
    reset_scheduler,
)
from jarvis.scheduler.timing import (
    TimingAnalyzer,
    get_timing_analyzer,
    reset_timing_analyzer,
)


class TestSchedulerModels:
    """Tests for scheduler data models."""

    def test_priority_weight(self) -> None:
        """Test priority weight ordering."""
        assert Priority.URGENT.weight > Priority.NORMAL.weight
        assert Priority.NORMAL.weight > Priority.LOW.weight

    def test_scheduled_item_creation(self) -> None:
        """Test creating a scheduled item."""
        send_at = datetime.now(UTC) + timedelta(hours=1)
        item = ScheduledItem(
            draft_id="draft1",
            contact_id=1,
            chat_id="chat1",
            message_text="Hello!",
            send_at=send_at,
        )

        assert item.id is not None
        assert item.draft_id == "draft1"
        assert item.status == ScheduledStatus.PENDING
        assert item.priority == Priority.NORMAL
        assert item.expires_at is not None

    def test_scheduled_item_is_due(self) -> None:
        """Test is_due property."""
        # Not due yet
        future = datetime.now(UTC) + timedelta(hours=1)
        item = ScheduledItem(
            draft_id="d1",
            contact_id=1,
            chat_id="c1",
            message_text="Hi",
            send_at=future,
        )
        assert item.is_due is False

        # Due now
        past = datetime.now(UTC) - timedelta(minutes=1)
        item.send_at = past
        assert item.is_due is True

        # Not due if not pending
        item.status = ScheduledStatus.SENT
        assert item.is_due is False

    def test_scheduled_item_is_expired(self) -> None:
        """Test is_expired property."""
        send_at = datetime.now(UTC) - timedelta(hours=25)
        item = ScheduledItem(
            draft_id="d1",
            contact_id=1,
            chat_id="c1",
            message_text="Hi",
            send_at=send_at,
        )
        # Default expiry is 24 hours after send_at
        assert item.is_expired is True

    def test_scheduled_item_is_terminal(self) -> None:
        """Test is_terminal property."""
        item = ScheduledItem(
            draft_id="d1",
            contact_id=1,
            chat_id="c1",
            message_text="Hi",
            send_at=datetime.now(UTC),
        )

        assert item.is_terminal is False

        item.status = ScheduledStatus.SENT
        assert item.is_terminal is True

        item.status = ScheduledStatus.CANCELLED
        assert item.is_terminal is True

        item.status = ScheduledStatus.EXPIRED
        assert item.is_terminal is True

    def test_scheduled_item_retry(self) -> None:
        """Test retry mechanism."""
        item = ScheduledItem(
            draft_id="d1",
            contact_id=1,
            chat_id="c1",
            message_text="Hi",
            send_at=datetime.now(UTC),
            max_retries=3,
        )
        item.mark_failed("Error")

        assert item.can_retry is True
        assert item.retry_count == 1

        item.retry()
        assert item.status == ScheduledStatus.PENDING
        assert item.retry_count == 1

    def test_scheduled_item_serialization(self) -> None:
        """Test serialization round-trip."""
        item = ScheduledItem(
            draft_id="d1",
            contact_id=1,
            chat_id="c1",
            message_text="Hello!",
            send_at=datetime.now(UTC) + timedelta(hours=1),
            priority=Priority.URGENT,
            metadata={"key": "value"},
        )

        data = item.to_dict()
        restored = ScheduledItem.from_dict(data)

        assert restored.id == item.id
        assert restored.draft_id == item.draft_id
        assert restored.priority == item.priority
        assert restored.metadata == item.metadata

    def test_quiet_hours_check(self) -> None:
        """Test quiet hours detection."""
        quiet = QuietHours(start_hour=22, end_hour=8, enabled=True)

        # 11 PM should be quiet
        late_night = datetime(2024, 1, 15, 23, 0)
        assert quiet.is_quiet_time(late_night) is True

        # 3 AM should be quiet
        early_morning = datetime(2024, 1, 15, 3, 0)
        assert quiet.is_quiet_time(early_morning) is True

        # 2 PM should not be quiet
        afternoon = datetime(2024, 1, 15, 14, 0)
        assert quiet.is_quiet_time(afternoon) is False

    def test_quiet_hours_disabled(self) -> None:
        """Test disabled quiet hours."""
        quiet = QuietHours(enabled=False)
        late_night = datetime(2024, 1, 15, 23, 0)
        assert quiet.is_quiet_time(late_night) is False

    def test_quiet_hours_next_allowed(self) -> None:
        """Test next allowed time calculation."""
        quiet = QuietHours(start_hour=22, end_hour=8, enabled=True)

        # At 11 PM, next allowed should be 8 AM tomorrow
        late_night = datetime(2024, 1, 15, 23, 0)
        next_allowed = quiet.next_allowed_time(late_night)
        assert next_allowed.hour == 8
        assert next_allowed.day == 16

        # At 3 AM, next allowed should be 8 AM same day
        early_morning = datetime(2024, 1, 15, 3, 0)
        next_allowed = quiet.next_allowed_time(early_morning)
        assert next_allowed.hour == 8
        assert next_allowed.day == 15

    def test_timing_suggestion_creation(self) -> None:
        """Test creating a timing suggestion."""
        suggestion = TimingSuggestion(
            suggested_time=datetime.now(UTC) + timedelta(hours=2),
            confidence=0.85,
            reason="High engagement at this hour",
            is_optimal=True,
        )

        assert suggestion.confidence == 0.85
        assert suggestion.is_optimal is True

    def test_send_result_creation(self) -> None:
        """Test creating a send result."""
        result = SendResult(success=True, sent_at=datetime.now(UTC), attempts=1)
        assert result.success is True
        assert result.sent_at is not None


class TestSchedulerQueue:
    """Tests for the scheduler queue."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self) -> None:
        """Reset singletons before each test."""
        reset_scheduler_queue()
        reset_timing_analyzer()
        reset_executor()
        reset_scheduler()

    @pytest.fixture
    def temp_queue(self) -> SchedulerQueue:
        """Create a queue with a temporary persistence file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        queue = SchedulerQueue(persistence_path=temp_path, auto_persist=False)
        yield queue

        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

    def test_add_item(self, temp_queue: SchedulerQueue) -> None:
        """Test adding an item to the queue."""
        item = ScheduledItem(
            draft_id="d1",
            contact_id=1,
            chat_id="c1",
            message_text="Hello",
            send_at=datetime.now(UTC) + timedelta(hours=1),
        )

        added = temp_queue.add(item)

        assert added is not None
        assert added.id == item.id

        retrieved = temp_queue.get(item.id)
        assert retrieved is not None
        assert retrieved.message_text == "Hello"

    def test_get_all_filtered(self, temp_queue: SchedulerQueue) -> None:
        """Test getting all items with filters."""
        item1 = ScheduledItem(
            draft_id="d1",
            contact_id=1,
            chat_id="c1",
            message_text="Hi",
            send_at=datetime.now(UTC) + timedelta(hours=1),
            priority=Priority.URGENT,
        )
        item2 = ScheduledItem(
            draft_id="d2",
            contact_id=2,
            chat_id="c2",
            message_text="Hello",
            send_at=datetime.now(UTC) + timedelta(hours=2),
            priority=Priority.LOW,
        )

        temp_queue.add(item1)
        temp_queue.add(item2)

        # Filter by contact
        contact1_items = temp_queue.get_all(contact_id=1)
        assert len(contact1_items) == 1
        assert contact1_items[0].id == item1.id

        # Filter by priority
        urgent_items = temp_queue.get_all(priority=Priority.URGENT)
        assert len(urgent_items) == 1

    def test_get_pending(self, temp_queue: SchedulerQueue) -> None:
        """Test getting pending items in order."""
        now = datetime.now(UTC)
        item1 = ScheduledItem(
            draft_id="d1",
            contact_id=1,
            chat_id="c1",
            message_text="First",
            send_at=now + timedelta(hours=2),
        )
        item2 = ScheduledItem(
            draft_id="d2",
            contact_id=1,
            chat_id="c1",
            message_text="Second",
            send_at=now + timedelta(hours=1),  # Earlier
        )

        temp_queue.add(item1)
        temp_queue.add(item2)

        pending = temp_queue.get_pending()
        assert len(pending) == 2
        # Second should come first (earlier send_at)
        assert pending[0].id == item2.id

    def test_get_due_items(self, temp_queue: SchedulerQueue) -> None:
        """Test getting due items."""
        now = datetime.now(UTC)
        # Past due item
        item1 = ScheduledItem(
            draft_id="d1",
            contact_id=1,
            chat_id="c1",
            message_text="Due",
            send_at=now - timedelta(minutes=5),
        )
        # Future item
        item2 = ScheduledItem(
            draft_id="d2",
            contact_id=1,
            chat_id="c1",
            message_text="Future",
            send_at=now + timedelta(hours=1),
        )

        temp_queue.add(item1)
        temp_queue.add(item2)

        due = temp_queue.get_due_items()
        assert len(due) == 1
        assert due[0].id == item1.id

    def test_cancel_item(self, temp_queue: SchedulerQueue) -> None:
        """Test cancelling an item."""
        item = ScheduledItem(
            draft_id="d1",
            contact_id=1,
            chat_id="c1",
            message_text="Hi",
            send_at=datetime.now(UTC) + timedelta(hours=1),
        )
        temp_queue.add(item)

        success = temp_queue.cancel(item.id)
        assert success is True

        retrieved = temp_queue.get(item.id)
        assert retrieved is not None
        assert retrieved.status == ScheduledStatus.CANCELLED

    def test_reschedule_item(self, temp_queue: SchedulerQueue) -> None:
        """Test rescheduling an item."""
        original_time = datetime.now(UTC) + timedelta(hours=1)
        item = ScheduledItem(
            draft_id="d1",
            contact_id=1,
            chat_id="c1",
            message_text="Hi",
            send_at=original_time,
        )
        temp_queue.add(item)

        new_time = datetime.now(UTC) + timedelta(hours=3)
        rescheduled = temp_queue.reschedule(item.id, new_time)

        assert rescheduled is not None
        assert rescheduled.send_at == new_time

    def test_mark_sent(self, temp_queue: SchedulerQueue) -> None:
        """Test marking an item as sent."""
        item = ScheduledItem(
            draft_id="d1",
            contact_id=1,
            chat_id="c1",
            message_text="Hi",
            send_at=datetime.now(UTC),
        )
        temp_queue.add(item)

        success = temp_queue.mark_sent(item.id)
        assert success is True

        retrieved = temp_queue.get(item.id)
        assert retrieved is not None
        assert retrieved.status == ScheduledStatus.SENT

    def test_mark_failed_with_retry(self, temp_queue: SchedulerQueue) -> None:
        """Test marking an item as failed triggers retry."""
        item = ScheduledItem(
            draft_id="d1",
            contact_id=1,
            chat_id="c1",
            message_text="Hi",
            send_at=datetime.now(UTC),
            max_retries=3,
        )
        temp_queue.add(item)

        success = temp_queue.mark_failed(item.id, "Connection error")
        assert success is True

        retrieved = temp_queue.get(item.id)
        assert retrieved is not None
        # Should be pending again (auto-retry)
        assert retrieved.status == ScheduledStatus.PENDING
        assert retrieved.retry_count == 1

    def test_get_stats(self, temp_queue: SchedulerQueue) -> None:
        """Test getting queue statistics."""
        item1 = ScheduledItem(
            draft_id="d1",
            contact_id=1,
            chat_id="c1",
            message_text="Pending",
            send_at=datetime.now(UTC) + timedelta(hours=1),
        )
        item2 = ScheduledItem(
            draft_id="d2",
            contact_id=2,
            chat_id="c2",
            message_text="Sent",
            send_at=datetime.now(UTC),
        )

        temp_queue.add(item1)
        temp_queue.add(item2)
        temp_queue.mark_sent(item2.id)

        stats = temp_queue.get_stats()

        assert stats["total"] == 2
        assert stats["by_status"]["pending"] == 1
        assert stats["by_status"]["sent"] == 1

    def test_clear_terminal(self, temp_queue: SchedulerQueue) -> None:
        """Test clearing terminal items."""
        item1 = ScheduledItem(
            draft_id="d1",
            contact_id=1,
            chat_id="c1",
            message_text="Sent",
            send_at=datetime.now(UTC),
        )
        item2 = ScheduledItem(
            draft_id="d2",
            contact_id=1,
            chat_id="c1",
            message_text="Pending",
            send_at=datetime.now(UTC) + timedelta(hours=1),
        )

        temp_queue.add(item1)
        temp_queue.add(item2)
        temp_queue.mark_sent(item1.id)

        count = temp_queue.clear_terminal()
        assert count == 1

        # Only pending item should remain
        items = temp_queue.get_all()
        assert len(items) == 1
        assert items[0].id == item2.id


class TestTimingAnalyzer:
    """Tests for the timing analyzer."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self) -> None:
        """Reset singletons before each test."""
        reset_timing_analyzer()

    def test_suggest_time_default(self) -> None:
        """Test default time suggestion."""
        analyzer = TimingAnalyzer()
        suggestions = analyzer.suggest_time(contact_id=1, num_suggestions=3)

        assert len(suggestions) > 0
        assert all(isinstance(s, TimingSuggestion) for s in suggestions)

    def test_suggest_time_with_prefs(self) -> None:
        """Test time suggestion with contact preferences."""
        analyzer = TimingAnalyzer()

        prefs = ContactTimingPrefs(
            contact_id=1,
            timezone="America/New_York",
            preferred_hours=[10, 14, 16],
            optimal_weekdays=[0, 1, 2, 3, 4],  # Mon-Fri
        )
        analyzer.set_contact_prefs(1, prefs)

        suggestions = analyzer.suggest_time(contact_id=1, num_suggestions=1)
        assert len(suggestions) == 1

    def test_suggest_time_respects_quiet_hours(self) -> None:
        """Test that suggestions respect quiet hours."""
        analyzer = TimingAnalyzer()

        prefs = ContactTimingPrefs(
            contact_id=1,
            quiet_hours=QuietHours(start_hour=22, end_hour=8, enabled=True),
        )
        analyzer.set_contact_prefs(1, prefs)

        # Use a fixed morning time to avoid test flakiness when run during quiet hours
        morning = datetime(2025, 6, 15, 10, 0, 0, tzinfo=UTC)
        suggestions = analyzer.suggest_time(
            contact_id=1,
            earliest=morning,
            num_suggestions=5,
        )

        # All suggestions should be outside quiet hours
        for s in suggestions:
            hour = s.suggested_time.hour
            # Should not be in 22-8 range
            assert not (22 <= hour or hour < 8)

    def test_analyze_patterns_empty(self) -> None:
        """Test pattern analysis with no data."""
        analyzer = TimingAnalyzer()
        patterns = analyzer.analyze_patterns(contact_id=999)

        assert patterns["total_interactions"] == 0
        assert patterns["preferred_hours"] == []

    def test_analyze_patterns_with_data(self) -> None:
        """Test pattern analysis with interaction data."""
        analyzer = TimingAnalyzer()

        # Cache some interactions
        interactions = [
            {"timestamp": datetime(2024, 1, 15, 14, 0), "is_from_me": False},
            {"timestamp": datetime(2024, 1, 15, 14, 30), "is_from_me": False},
            {"timestamp": datetime(2024, 1, 16, 14, 0), "is_from_me": False},
            {"timestamp": datetime(2024, 1, 17, 10, 0), "is_from_me": False},
        ]
        analyzer.cache_interactions(1, interactions)

        patterns = analyzer.analyze_patterns(1)

        assert patterns["total_interactions"] == 4
        # Hour 14 should be preferred (3 out of 4 interactions)
        assert 14 in patterns["preferred_hours"]

    def test_is_good_time(self) -> None:
        """Test checking if a time is good."""
        from zoneinfo import ZoneInfo

        analyzer = TimingAnalyzer()
        la_tz = ZoneInfo("America/Los_Angeles")

        # 2 PM on a weekday in LA time should be fine (outside quiet hours 22-8)
        good_time = datetime(2024, 1, 15, 14, 0, tzinfo=la_tz)  # Monday 2PM PT
        is_good, reason = analyzer.is_good_time(1, good_time)
        assert is_good is True

        # 2 AM in LA time should not be good (within default quiet hours)
        bad_time = datetime(2024, 1, 15, 2, 0, tzinfo=la_tz)  # 2AM PT
        is_good, reason = analyzer.is_good_time(1, bad_time)
        assert is_good is False
        assert "quiet hours" in reason.lower()


class TestSendExecutor:
    """Tests for the send executor."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self) -> None:
        """Reset singletons before each test."""
        reset_executor()

    def test_rate_limit_check(self) -> None:
        """Test rate limit checking."""
        executor = SendExecutor(rate_limit=RateLimitConfig(per_minute=2, per_hour=10, per_day=50))

        # First two should be allowed
        # Create item (unused, but demonstrates ScheduledItem structure)
        ScheduledItem(
            draft_id="d1",
            contact_id=1,
            chat_id="c1",
            message_text="Hi",
            send_at=datetime.now(UTC),
        )

        allowed, _ = executor._check_rate_limit(1)
        assert allowed is True

        # Record sends
        executor._record_send(1)
        executor._record_send(1)

        # Third should be blocked
        allowed, reason = executor._check_rate_limit(1)
        assert allowed is False
        assert "per_minute" in reason.lower() or "/min" in reason

    def test_duplicate_detection(self) -> None:
        """Test duplicate message detection."""
        executor = SendExecutor()

        item = ScheduledItem(
            draft_id="d1",
            contact_id=1,
            chat_id="c1",
            message_text="Hello!",
            send_at=datetime.now(UTC),
        )

        # First should not be duplicate
        is_dup, _ = executor._check_duplicate(item)
        assert is_dup is False

        # Record the message
        executor._record_message(item)

        # Same message should be duplicate
        is_dup, reason = executor._check_duplicate(item)
        assert is_dup is True
        assert "duplicate" in reason.lower()

    def test_get_rate_limit_status(self) -> None:
        """Test getting rate limit status."""
        executor = SendExecutor(rate_limit=RateLimitConfig(per_minute=5, per_hour=30, per_day=100))

        executor._record_send(1)
        executor._record_send(1)

        status = executor.get_rate_limit_status(1)

        assert status["minute"]["used"] == 2
        assert status["minute"]["limit"] == 5
        assert status["minute"]["remaining"] == 3


class TestDraftScheduler:
    """Tests for the main draft scheduler."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self) -> None:
        """Reset singletons before each test."""
        reset_scheduler()
        reset_scheduler_queue()
        reset_timing_analyzer()
        reset_executor()

    @pytest.fixture
    def temp_scheduler(self) -> DraftScheduler:
        """Create a scheduler with test queue."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        queue = SchedulerQueue(persistence_path=temp_path, auto_persist=False)
        scheduler = DraftScheduler(queue=queue, check_interval=1)

        yield scheduler

        if scheduler.is_running:
            scheduler.stop()
        if temp_path.exists():
            temp_path.unlink()

    def test_schedule_draft(self, temp_scheduler: DraftScheduler) -> None:
        """Test scheduling a draft."""
        send_at = datetime.now(UTC) + timedelta(hours=1)
        item = temp_scheduler.schedule_draft(
            draft_id="d1",
            contact_id=1,
            chat_id="c1",
            message_text="Hello!",
            send_at=send_at,
            priority=Priority.URGENT,
        )

        assert item is not None
        assert item.draft_id == "d1"
        assert item.priority == Priority.URGENT
        assert item.status == ScheduledStatus.PENDING

    def test_schedule_with_smart_timing(self, temp_scheduler: DraftScheduler) -> None:
        """Test scheduling with smart timing."""
        item, suggestion = temp_scheduler.schedule_with_smart_timing(
            draft_id="d1",
            contact_id=1,
            chat_id="c1",
            message_text="Hello!",
        )

        assert item is not None
        assert suggestion is not None
        assert isinstance(suggestion, TimingSuggestion)

    def test_get_scheduled(self, temp_scheduler: DraftScheduler) -> None:
        """Test getting scheduled items."""
        temp_scheduler.schedule_draft(
            draft_id="d1",
            contact_id=1,
            chat_id="c1",
            message_text="First",
            send_at=datetime.now(UTC) + timedelta(hours=1),
        )
        temp_scheduler.schedule_draft(
            draft_id="d2",
            contact_id=2,
            chat_id="c2",
            message_text="Second",
            send_at=datetime.now(UTC) + timedelta(hours=2),
        )

        # Get all
        all_items = temp_scheduler.get_scheduled()
        assert len(all_items) == 2

        # Filter by contact
        contact1 = temp_scheduler.get_scheduled(contact_id=1)
        assert len(contact1) == 1

    def test_cancel_scheduled(self, temp_scheduler: DraftScheduler) -> None:
        """Test cancelling a scheduled item."""
        item = temp_scheduler.schedule_draft(
            draft_id="d1",
            contact_id=1,
            chat_id="c1",
            message_text="Hello",
            send_at=datetime.now(UTC) + timedelta(hours=1),
        )

        success = temp_scheduler.cancel(item.id)
        assert success is True

        retrieved = temp_scheduler.get_item(item.id)
        assert retrieved is not None
        assert retrieved.status == ScheduledStatus.CANCELLED

    def test_reschedule(self, temp_scheduler: DraftScheduler) -> None:
        """Test rescheduling an item."""
        item = temp_scheduler.schedule_draft(
            draft_id="d1",
            contact_id=1,
            chat_id="c1",
            message_text="Hello",
            send_at=datetime.now(UTC) + timedelta(hours=1),
        )

        new_time = datetime.now(UTC) + timedelta(hours=5)
        rescheduled = temp_scheduler.reschedule(item.id, new_time)

        assert rescheduled is not None
        assert rescheduled.send_at == new_time

    def test_update_message(self, temp_scheduler: DraftScheduler) -> None:
        """Test updating message text."""
        item = temp_scheduler.schedule_draft(
            draft_id="d1",
            contact_id=1,
            chat_id="c1",
            message_text="Original",
            send_at=datetime.now(UTC) + timedelta(hours=1),
        )

        updated = temp_scheduler.update_message(item.id, "Updated message")

        assert updated is not None
        assert updated.message_text == "Updated message"

    def test_scheduler_start_stop(self, temp_scheduler: DraftScheduler) -> None:
        """Test starting and stopping the scheduler."""
        assert temp_scheduler.is_running is False

        temp_scheduler.start()
        assert temp_scheduler.is_running is True

        temp_scheduler.stop()
        assert temp_scheduler.is_running is False

    def test_get_stats(self, temp_scheduler: DraftScheduler) -> None:
        """Test getting scheduler statistics."""
        temp_scheduler.schedule_draft(
            draft_id="d1",
            contact_id=1,
            chat_id="c1",
            message_text="Test",
            send_at=datetime.now(UTC) + timedelta(hours=1),
        )

        stats = temp_scheduler.get_stats()

        assert "running" in stats
        assert "total" in stats
        assert stats["total"] == 1


class TestConcurrency:
    """Tests for concurrent access."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self) -> None:
        """Reset singletons before each test."""
        reset_scheduler_queue()

    def test_concurrent_add(self) -> None:
        """Test concurrent item additions."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            queue = SchedulerQueue(persistence_path=temp_path, auto_persist=False)
            item_ids: list[str] = []
            lock = threading.Lock()

            def add_item() -> None:
                item = ScheduledItem(
                    draft_id=f"d{threading.current_thread().name}",
                    contact_id=1,
                    chat_id="c1",
                    message_text="Test",
                    send_at=datetime.now(UTC) + timedelta(hours=1),
                )
                added = queue.add(item)
                with lock:
                    item_ids.append(added.id)

            threads = [threading.Thread(target=add_item) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(item_ids) == 10
            assert len(set(item_ids)) == 10  # All unique

        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestSingletons:
    """Tests for singleton access."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self) -> None:
        """Reset singletons before each test."""
        reset_scheduler_queue()
        reset_timing_analyzer()
        reset_executor()
        reset_scheduler()

    def test_get_scheduler_queue_singleton(self) -> None:
        """Test that get_scheduler_queue returns the same instance."""
        queue1 = get_scheduler_queue()
        queue2 = get_scheduler_queue()
        assert queue1 is queue2

    def test_get_timing_analyzer_singleton(self) -> None:
        """Test that get_timing_analyzer returns the same instance."""
        analyzer1 = get_timing_analyzer()
        analyzer2 = get_timing_analyzer()
        assert analyzer1 is analyzer2

    def test_get_executor_singleton(self) -> None:
        """Test that get_executor returns the same instance."""
        executor1 = get_executor()
        executor2 = get_executor()
        assert executor1 is executor2

    def test_get_scheduler_singleton(self) -> None:
        """Test that get_scheduler returns the same instance."""
        scheduler1 = get_scheduler()
        scheduler2 = get_scheduler()
        assert scheduler1 is scheduler2
