"""Unit tests for JARVIS Feedback Loop System.

Tests cover feedback store initialization, recording feedback,
retrieving feedback, statistics, analytics, and error handling.
"""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jarvis.core.exceptions import ErrorCode, FeedbackInvalidActionError
from jarvis.eval.feedback import (
    ContactStats,
    DailyStats,
    Feedback,
    FeedbackAction,
    FeedbackStore,
    get_feedback_store,
    reset_feedback_store,
)


class TestFeedbackStoreInitialization:
    """Tests for feedback store initialization and schema creation."""

    def test_init_creates_directory(self, tmp_path: Path) -> None:
        """Test that FeedbackStore creates parent directory if it doesn't exist."""
        db_path = tmp_path / "nested" / "dir" / "jarvis.db"
        store = FeedbackStore(db_path)

        assert store.db_path == db_path
        assert db_path.parent.exists()

    def test_exists_returns_false_when_no_file(self, tmp_path: Path) -> None:
        """Test exists() returns False when database file doesn't exist."""
        db_path = tmp_path / "nonexistent.db"
        store = FeedbackStore(db_path)

        assert store.exists() is False

    def test_exists_returns_true_after_init_schema(self, tmp_path: Path) -> None:
        """Test exists() returns True after schema initialization."""
        db_path = tmp_path / "jarvis.db"
        store = FeedbackStore(db_path)
        store.init_schema()

        assert store.exists() is True

    def test_init_schema_creates_table(self, tmp_path: Path) -> None:
        """Test that init_schema creates the feedback table."""
        db_path = tmp_path / "jarvis.db"
        store = FeedbackStore(db_path)
        store.init_schema()

        with store.connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'"
            )
            result = cursor.fetchone()
            assert result is not None
            assert result["name"] == "feedback"

    def test_init_schema_idempotent(self, tmp_path: Path) -> None:
        """Test that calling init_schema multiple times is safe."""
        db_path = tmp_path / "jarvis.db"
        store = FeedbackStore(db_path)

        # First call should return True (created)
        result1 = store.init_schema()
        assert result1 is True

        # Second call should return False (already exists)
        result2 = store.init_schema()
        assert result2 is False

    def test_init_schema_creates_indices(self, tmp_path: Path) -> None:
        """Test that init_schema creates expected indices."""
        db_path = tmp_path / "jarvis.db"
        store = FeedbackStore(db_path)
        store.init_schema()

        with store.connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indices = {row["name"] for row in cursor}

            expected_indices = {
                "idx_feedback_message",
                "idx_feedback_suggestion",
                "idx_feedback_action",
                "idx_feedback_timestamp",
                "idx_feedback_contact",
                "idx_feedback_timestamp_date",
            }
            assert expected_indices.issubset(indices)

    def test_init_schema_creates_version_table(self, tmp_path: Path) -> None:
        """Test that init_schema creates version tracking table."""
        db_path = tmp_path / "jarvis.db"
        store = FeedbackStore(db_path)
        store.init_schema()

        with store.connection() as conn:
            cursor = conn.execute(
                "SELECT version FROM feedback_schema_version ORDER BY version DESC LIMIT 1"
            )
            row = cursor.fetchone()
            assert row is not None
            assert row["version"] == 2  # Current schema version

    @patch("jarvis.eval.feedback.get_config")
    def test_init_uses_configured_path(self, mock_get_config: MagicMock, tmp_path: Path) -> None:
        """Test that init uses path from config when db_path is None."""
        # Setup mock config
        expected_path = tmp_path / "config_db.db"
        mock_config = MagicMock()
        mock_config.database.feedback_path = str(expected_path)
        mock_get_config.return_value = mock_config

        store = FeedbackStore(db_path=None)

        assert store.db_path == expected_path
        mock_get_config.assert_called_once()


class TestFeedbackAction:
    """Tests for FeedbackAction enum."""

    def test_valid_actions(self) -> None:
        """Test that valid action values work."""
        assert FeedbackAction.ACCEPTED.value == "accepted"
        assert FeedbackAction.REJECTED.value == "rejected"
        assert FeedbackAction.EDITED.value == "edited"

    def test_action_from_string(self) -> None:
        """Test creating action from string value."""
        assert FeedbackAction("accepted") == FeedbackAction.ACCEPTED
        assert FeedbackAction("rejected") == FeedbackAction.REJECTED
        assert FeedbackAction("edited") == FeedbackAction.EDITED

    def test_invalid_action_raises(self) -> None:
        """Test that invalid action string raises ValueError."""
        with pytest.raises(ValueError):
            FeedbackAction("invalid")


class TestFeedbackDataclass:
    """Tests for Feedback dataclass."""

    def test_feedback_creation(self) -> None:
        """Test creating a Feedback instance."""
        now = datetime.now()
        feedback = Feedback(
            id=1,
            message_id="msg_123",
            suggestion_id="sug_456",
            action=FeedbackAction.ACCEPTED,
            timestamp=now,
        )

        assert feedback.id == 1
        assert feedback.message_id == "msg_123"
        assert feedback.suggestion_id == "sug_456"
        assert feedback.action == FeedbackAction.ACCEPTED
        assert feedback.timestamp == now

    def test_feedback_with_new_fields(self) -> None:
        """Test creating a Feedback instance with new v2 fields."""
        now = datetime.now()
        feedback = Feedback(
            id=1,
            message_id="msg_123",
            suggestion_id="sug_456",
            action=FeedbackAction.EDITED,
            timestamp=now,
            contact_id=42,
            original_suggestion="Hello there",
            edited_text="Hello there!",
        )

        assert feedback.contact_id == 42
        assert feedback.original_suggestion == "Hello there"
        assert feedback.edited_text == "Hello there!"

    def test_feedback_to_dict(self) -> None:
        """Test converting Feedback to dictionary."""
        now = datetime.now()
        feedback = Feedback(
            id=1,
            message_id="msg_123",
            suggestion_id="sug_456",
            action=FeedbackAction.EDITED,
            timestamp=now,
            contact_id=42,
            original_suggestion="Hello",
            edited_text="Hello!",
        )

        result = feedback.to_dict()

        assert result["id"] == 1
        assert result["message_id"] == "msg_123"
        assert result["suggestion_id"] == "sug_456"
        assert result["action"] == "edited"
        assert result["timestamp"] == now.isoformat()
        assert result["contact_id"] == 42
        assert result["original_suggestion"] == "Hello"
        assert result["edited_text"] == "Hello!"


class TestDailyStatsDataclass:
    """Tests for DailyStats dataclass."""

    def test_daily_stats_creation(self) -> None:
        """Test creating a DailyStats instance."""
        stats = DailyStats(
            date="2024-01-15",
            total=100,
            accepted=60,
            rejected=30,
            edited=10,
            acceptance_rate=0.6,
        )

        assert stats.date == "2024-01-15"
        assert stats.total == 100
        assert stats.accepted == 60
        assert stats.acceptance_rate == 0.6

    def test_daily_stats_to_dict(self) -> None:
        """Test converting DailyStats to dictionary."""
        stats = DailyStats(
            date="2024-01-15",
            total=100,
            accepted=60,
            rejected=30,
            edited=10,
            acceptance_rate=0.6,
        )

        result = stats.to_dict()

        assert result["date"] == "2024-01-15"
        assert result["total"] == 100
        assert result["acceptance_rate"] == 0.6


class TestContactStatsDataclass:
    """Tests for ContactStats dataclass."""

    def test_contact_stats_creation(self) -> None:
        """Test creating a ContactStats instance."""
        stats = ContactStats(
            contact_id=42,
            total=50,
            accepted=40,
            rejected=5,
            edited=5,
            acceptance_rate=0.8,
        )

        assert stats.contact_id == 42
        assert stats.total == 50
        assert stats.acceptance_rate == 0.8

    def test_contact_stats_to_dict(self) -> None:
        """Test converting ContactStats to dictionary."""
        stats = ContactStats(
            contact_id=42,
            total=50,
            accepted=40,
            rejected=5,
            edited=5,
            acceptance_rate=0.8,
        )

        result = stats.to_dict()

        assert result["contact_id"] == 42
        assert result["total"] == 50
        assert result["acceptance_rate"] == 0.8


class TestRecordFeedback:
    """Tests for recording feedback."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> FeedbackStore:
        """Create a fresh feedback store for each test."""
        db_path = tmp_path / "jarvis.db"
        store = FeedbackStore(db_path)
        store.init_schema()
        return store

    def test_record_feedback_basic(self, store: FeedbackStore) -> None:
        """Test recording basic feedback."""
        feedback = store.record_feedback(
            message_id="msg_123",
            suggestion_id="sug_456",
            action=FeedbackAction.ACCEPTED,
        )

        assert feedback.id is not None
        assert feedback.message_id == "msg_123"
        assert feedback.suggestion_id == "sug_456"
        assert feedback.action == FeedbackAction.ACCEPTED
        assert feedback.timestamp is not None

    def test_record_feedback_with_string_action(self, store: FeedbackStore) -> None:
        """Test recording feedback with string action."""
        feedback = store.record_feedback(
            message_id="msg_123",
            suggestion_id="sug_456",
            action="rejected",
        )

        assert feedback.action == FeedbackAction.REJECTED

    def test_record_feedback_with_timestamp(self, store: FeedbackStore) -> None:
        """Test recording feedback with explicit timestamp."""
        specific_time = datetime(2024, 1, 15, 10, 30, 0)
        feedback = store.record_feedback(
            message_id="msg_123",
            suggestion_id="sug_456",
            action=FeedbackAction.EDITED,
            timestamp=specific_time,
        )

        assert feedback.timestamp == specific_time

    def test_record_feedback_with_metadata(self, store: FeedbackStore) -> None:
        """Test recording feedback with metadata."""
        feedback = store.record_feedback(
            message_id="msg_123",
            suggestion_id="sug_456",
            action=FeedbackAction.EDITED,
            metadata={"edit_distance": 5, "original_text": "Hello"},
        )

        assert feedback.metadata_json is not None
        assert "edit_distance" in feedback.metadata_json

    def test_record_feedback_with_contact_id(self, store: FeedbackStore) -> None:
        """Test recording feedback with contact association."""
        feedback = store.record_feedback(
            message_id="msg_123",
            suggestion_id="sug_456",
            action=FeedbackAction.ACCEPTED,
            contact_id=42,
        )

        assert feedback.contact_id == 42

    def test_record_feedback_with_edited_text(self, store: FeedbackStore) -> None:
        """Test recording feedback with original suggestion and edited text."""
        feedback = store.record_feedback(
            message_id="msg_123",
            suggestion_id="sug_456",
            action=FeedbackAction.EDITED,
            original_suggestion="How about tomorrow?",
            edited_text="How about tomorrow at 3pm?",
        )

        assert feedback.original_suggestion == "How about tomorrow?"
        assert feedback.edited_text == "How about tomorrow at 3pm?"

    def test_record_feedback_updates_existing(self, store: FeedbackStore) -> None:
        """Test that recording feedback for same message/suggestion updates it."""
        # Record initial feedback
        store.record_feedback(
            message_id="msg_123",
            suggestion_id="sug_456",
            action=FeedbackAction.ACCEPTED,
        )

        # Update with different action
        feedback2 = store.record_feedback(
            message_id="msg_123",
            suggestion_id="sug_456",
            action=FeedbackAction.REJECTED,
        )

        # Should have same ID (upsert)
        assert feedback2.action == FeedbackAction.REJECTED

        # Should only have one record
        assert store.count_feedback() == 1

    def test_record_feedback_invalid_action(self, store: FeedbackStore) -> None:
        """Test that invalid action raises FeedbackInvalidActionError."""
        with pytest.raises(FeedbackInvalidActionError) as exc_info:
            store.record_feedback(
                message_id="msg_123",
                suggestion_id="sug_456",
                action="invalid_action",
            )

        assert exc_info.value.code == ErrorCode.FBK_INVALID_ACTION
        assert "invalid_action" in str(exc_info.value)


class TestRecordFeedbackBulk:
    """Tests for bulk feedback recording."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> FeedbackStore:
        """Create a fresh feedback store for each test."""
        db_path = tmp_path / "jarvis.db"
        store = FeedbackStore(db_path)
        store.init_schema()
        return store

    def test_record_feedback_bulk_basic(self, store: FeedbackStore) -> None:
        """Test bulk recording of feedback."""
        items = [
            {"message_id": "msg_1", "suggestion_id": "sug_1", "action": "accepted"},
            {"message_id": "msg_2", "suggestion_id": "sug_2", "action": "rejected"},
            {"message_id": "msg_3", "suggestion_id": "sug_3", "action": "edited"},
        ]

        recorded = store.record_feedback_bulk(items)

        assert recorded == 3
        assert store.count_feedback() == 3

    def test_record_feedback_bulk_with_all_fields(self, store: FeedbackStore) -> None:
        """Test bulk recording with all optional fields."""
        items = [
            {
                "message_id": "msg_1",
                "suggestion_id": "sug_1",
                "action": FeedbackAction.EDITED,
                "contact_id": 42,
                "original_suggestion": "Hello",
                "edited_text": "Hello there!",
            },
        ]

        recorded = store.record_feedback_bulk(items)

        assert recorded == 1
        feedback = store.get_feedback_by_suggestion("sug_1")
        assert feedback is not None
        assert feedback.contact_id == 42
        assert feedback.original_suggestion == "Hello"
        assert feedback.edited_text == "Hello there!"

    def test_record_feedback_bulk_empty_list(self, store: FeedbackStore) -> None:
        """Test bulk recording with empty list."""
        recorded = store.record_feedback_bulk([])
        assert recorded == 0

    def test_record_feedback_bulk_updates_existing(self, store: FeedbackStore) -> None:
        """Test that bulk recording updates existing records."""
        # Record initial
        store.record_feedback(
            message_id="msg_1",
            suggestion_id="sug_1",
            action=FeedbackAction.ACCEPTED,
        )

        # Bulk update
        items = [
            {"message_id": "msg_1", "suggestion_id": "sug_1", "action": "rejected"},
            {"message_id": "msg_2", "suggestion_id": "sug_2", "action": "accepted"},
        ]
        recorded = store.record_feedback_bulk(items)

        assert recorded == 2
        assert store.count_feedback() == 2  # Only 2 records, not 3

        feedback = store.get_feedback_by_suggestion("sug_1")
        assert feedback is not None
        assert feedback.action == FeedbackAction.REJECTED


class TestGetFeedback:
    """Tests for retrieving feedback."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> FeedbackStore:
        """Create a fresh feedback store for each test."""
        db_path = tmp_path / "jarvis.db"
        store = FeedbackStore(db_path)
        store.init_schema()
        return store

    def test_get_feedback_by_id(self, store: FeedbackStore) -> None:
        """Test retrieving feedback by ID."""
        created = store.record_feedback(
            message_id="msg_123",
            suggestion_id="sug_456",
            action=FeedbackAction.ACCEPTED,
            contact_id=42,
            original_suggestion="Test",
        )

        retrieved = store.get_feedback(created.id)  # type: ignore[arg-type]

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.message_id == "msg_123"
        assert retrieved.action == FeedbackAction.ACCEPTED
        assert retrieved.contact_id == 42
        assert retrieved.original_suggestion == "Test"

    def test_get_feedback_returns_none_for_missing(self, store: FeedbackStore) -> None:
        """Test that get_feedback returns None for non-existent ID."""
        result = store.get_feedback(99999)
        assert result is None

    def test_get_feedback_by_suggestion(self, store: FeedbackStore) -> None:
        """Test retrieving feedback by suggestion ID."""
        store.record_feedback(
            message_id="msg_123",
            suggestion_id="unique_sug",
            action=FeedbackAction.REJECTED,
        )

        retrieved = store.get_feedback_by_suggestion("unique_sug")

        assert retrieved is not None
        assert retrieved.suggestion_id == "unique_sug"
        assert retrieved.action == FeedbackAction.REJECTED

    def test_get_feedback_by_suggestion_returns_none(self, store: FeedbackStore) -> None:
        """Test that get_feedback_by_suggestion returns None for non-existent."""
        result = store.get_feedback_by_suggestion("nonexistent")
        assert result is None

    def test_get_feedback_by_message(self, store: FeedbackStore) -> None:
        """Test retrieving all feedback for a message."""
        store.record_feedback(
            message_id="msg_123",
            suggestion_id="sug_1",
            action=FeedbackAction.ACCEPTED,
        )
        store.record_feedback(
            message_id="msg_123",
            suggestion_id="sug_2",
            action=FeedbackAction.REJECTED,
        )
        store.record_feedback(
            message_id="msg_456",
            suggestion_id="sug_3",
            action=FeedbackAction.EDITED,
        )

        results = store.get_feedback_by_message("msg_123")

        assert len(results) == 2
        suggestion_ids = {f.suggestion_id for f in results}
        assert suggestion_ids == {"sug_1", "sug_2"}

    def test_get_feedback_by_message_empty(self, store: FeedbackStore) -> None:
        """Test get_feedback_by_message returns empty list for no matches."""
        results = store.get_feedback_by_message("nonexistent")
        assert results == []

    def test_get_feedback_by_contact(self, store: FeedbackStore) -> None:
        """Test retrieving all feedback for a contact."""
        store.record_feedback(
            message_id="msg_1",
            suggestion_id="sug_1",
            action=FeedbackAction.ACCEPTED,
            contact_id=42,
        )
        store.record_feedback(
            message_id="msg_2",
            suggestion_id="sug_2",
            action=FeedbackAction.REJECTED,
            contact_id=42,
        )
        store.record_feedback(
            message_id="msg_3",
            suggestion_id="sug_3",
            action=FeedbackAction.EDITED,
            contact_id=99,
        )

        results = store.get_feedback_by_contact(42)

        assert len(results) == 2
        for f in results:
            assert f.contact_id == 42

    def test_get_feedback_by_contact_with_pagination(self, store: FeedbackStore) -> None:
        """Test get_feedback_by_contact with limit and offset."""
        for i in range(10):
            store.record_feedback(
                message_id=f"msg_{i}",
                suggestion_id=f"sug_{i}",
                action=FeedbackAction.ACCEPTED,
                contact_id=42,
            )

        results = store.get_feedback_by_contact(42, limit=3, offset=2)

        assert len(results) == 3


class TestListFeedback:
    """Tests for listing feedback."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> FeedbackStore:
        """Create a fresh feedback store with test data."""
        db_path = tmp_path / "jarvis.db"
        store = FeedbackStore(db_path)
        store.init_schema()

        # Add test data with contact associations
        for i in range(10):
            action = [FeedbackAction.ACCEPTED, FeedbackAction.REJECTED, FeedbackAction.EDITED][
                i % 3
            ]
            store.record_feedback(
                message_id=f"msg_{i}",
                suggestion_id=f"sug_{i}",
                action=action,
                contact_id=i % 2 + 1,  # Alternates between contact 1 and 2
            )

        return store

    def test_list_feedback_all(self, store: FeedbackStore) -> None:
        """Test listing all feedback."""
        results = store.list_feedback()

        assert len(results) == 10

    def test_list_feedback_with_limit(self, store: FeedbackStore) -> None:
        """Test listing feedback with limit."""
        results = store.list_feedback(limit=5)

        assert len(results) == 5

    def test_list_feedback_with_offset(self, store: FeedbackStore) -> None:
        """Test listing feedback with offset."""
        results = store.list_feedback(limit=3, offset=3)

        assert len(results) == 3

    def test_list_feedback_by_action(self, store: FeedbackStore) -> None:
        """Test filtering feedback by action."""
        results = store.list_feedback(action=FeedbackAction.ACCEPTED)

        # With 10 items cycling through 3 actions: 4 accepted, 3 rejected, 3 edited
        assert len(results) == 4
        for f in results:
            assert f.action == FeedbackAction.ACCEPTED

    def test_list_feedback_by_action_string(self, store: FeedbackStore) -> None:
        """Test filtering feedback by action as string."""
        results = store.list_feedback(action="rejected")

        assert len(results) == 3
        for f in results:
            assert f.action == FeedbackAction.REJECTED

    def test_list_feedback_by_contact_id(self, store: FeedbackStore) -> None:
        """Test filtering feedback by contact ID."""
        results = store.list_feedback(contact_id=1)

        assert len(results) == 5
        for f in results:
            assert f.contact_id == 1

    def test_list_feedback_by_action_and_contact(self, store: FeedbackStore) -> None:
        """Test filtering by both action and contact."""
        results = store.list_feedback(action=FeedbackAction.ACCEPTED, contact_id=1)

        for f in results:
            assert f.action == FeedbackAction.ACCEPTED
            assert f.contact_id == 1

    def test_list_feedback_ordered_by_timestamp(self, store: FeedbackStore) -> None:
        """Test that feedback is ordered by timestamp descending."""
        results = store.list_feedback()

        # Most recent should be first
        for i in range(len(results) - 1):
            assert results[i].timestamp >= results[i + 1].timestamp


class TestListFeedbackInRange:
    """Tests for time-range queries."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> FeedbackStore:
        """Create a fresh feedback store for each test."""
        db_path = tmp_path / "jarvis.db"
        store = FeedbackStore(db_path)
        store.init_schema()
        return store

    def test_list_feedback_in_range_basic(self, store: FeedbackStore) -> None:
        """Test listing feedback within a time range."""
        now = datetime.now()
        two_days_ago = now - timedelta(days=2)

        # Add feedback at different times
        store.record_feedback(
            message_id="msg_old",
            suggestion_id="sug_old",
            action=FeedbackAction.ACCEPTED,
            timestamp=two_days_ago,
        )
        store.record_feedback(
            message_id="msg_recent",
            suggestion_id="sug_recent",
            action=FeedbackAction.REJECTED,
            timestamp=now,
        )

        # Query for last 36 hours
        start = now - timedelta(hours=36)
        results = store.list_feedback_in_range(start, now)

        assert len(results) == 1
        assert results[0].message_id == "msg_recent"

    def test_list_feedback_in_range_with_action_filter(self, store: FeedbackStore) -> None:
        """Test time range query with action filter."""
        now = datetime.now()

        store.record_feedback(
            message_id="msg_1",
            suggestion_id="sug_1",
            action=FeedbackAction.ACCEPTED,
            timestamp=now,
        )
        store.record_feedback(
            message_id="msg_2",
            suggestion_id="sug_2",
            action=FeedbackAction.REJECTED,
            timestamp=now,
        )

        start = now - timedelta(hours=1)
        results = store.list_feedback_in_range(start, now, action=FeedbackAction.ACCEPTED)

        assert len(results) == 1
        assert results[0].action == FeedbackAction.ACCEPTED

    def test_list_feedback_in_range_with_contact_filter(self, store: FeedbackStore) -> None:
        """Test time range query with contact filter."""
        now = datetime.now()

        store.record_feedback(
            message_id="msg_1",
            suggestion_id="sug_1",
            action=FeedbackAction.ACCEPTED,
            timestamp=now,
            contact_id=1,
        )
        store.record_feedback(
            message_id="msg_2",
            suggestion_id="sug_2",
            action=FeedbackAction.ACCEPTED,
            timestamp=now,
            contact_id=2,
        )

        start = now - timedelta(hours=1)
        results = store.list_feedback_in_range(start, now, contact_id=1)

        assert len(results) == 1
        assert results[0].contact_id == 1


class TestCountFeedback:
    """Tests for counting feedback."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> FeedbackStore:
        """Create a fresh feedback store for each test."""
        db_path = tmp_path / "jarvis.db"
        store = FeedbackStore(db_path)
        store.init_schema()
        return store

    def test_count_feedback_empty(self, store: FeedbackStore) -> None:
        """Test counting feedback on empty database."""
        assert store.count_feedback() == 0

    def test_count_feedback_all(self, store: FeedbackStore) -> None:
        """Test counting all feedback."""
        for i in range(5):
            store.record_feedback(
                message_id=f"msg_{i}",
                suggestion_id=f"sug_{i}",
                action=FeedbackAction.ACCEPTED,
            )

        assert store.count_feedback() == 5

    def test_count_feedback_by_action(self, store: FeedbackStore) -> None:
        """Test counting feedback by action."""
        store.record_feedback(
            message_id="msg_1", suggestion_id="sug_1", action=FeedbackAction.ACCEPTED
        )
        store.record_feedback(
            message_id="msg_2", suggestion_id="sug_2", action=FeedbackAction.ACCEPTED
        )
        store.record_feedback(
            message_id="msg_3", suggestion_id="sug_3", action=FeedbackAction.REJECTED
        )

        assert store.count_feedback(action=FeedbackAction.ACCEPTED) == 2
        assert store.count_feedback(action=FeedbackAction.REJECTED) == 1
        assert store.count_feedback(action=FeedbackAction.EDITED) == 0

    def test_count_feedback_by_contact(self, store: FeedbackStore) -> None:
        """Test counting feedback by contact."""
        store.record_feedback(
            message_id="msg_1",
            suggestion_id="sug_1",
            action=FeedbackAction.ACCEPTED,
            contact_id=1,
        )
        store.record_feedback(
            message_id="msg_2",
            suggestion_id="sug_2",
            action=FeedbackAction.ACCEPTED,
            contact_id=1,
        )
        store.record_feedback(
            message_id="msg_3",
            suggestion_id="sug_3",
            action=FeedbackAction.REJECTED,
            contact_id=2,
        )

        assert store.count_feedback(contact_id=1) == 2
        assert store.count_feedback(contact_id=2) == 1


class TestDeleteFeedback:
    """Tests for deleting feedback."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> FeedbackStore:
        """Create a fresh feedback store for each test."""
        db_path = tmp_path / "jarvis.db"
        store = FeedbackStore(db_path)
        store.init_schema()
        return store

    def test_delete_feedback(self, store: FeedbackStore) -> None:
        """Test deleting a feedback record."""
        feedback = store.record_feedback(
            message_id="msg_123",
            suggestion_id="sug_456",
            action=FeedbackAction.ACCEPTED,
        )

        result = store.delete_feedback(feedback.id)  # type: ignore[arg-type]

        assert result is True
        assert store.get_feedback(feedback.id) is None  # type: ignore[arg-type]

    def test_delete_feedback_returns_false_for_missing(self, store: FeedbackStore) -> None:
        """Test that delete_feedback returns False for non-existent ID."""
        result = store.delete_feedback(99999)
        assert result is False

    def test_clear_feedback(self, store: FeedbackStore) -> None:
        """Test clearing all feedback."""
        for i in range(5):
            store.record_feedback(
                message_id=f"msg_{i}",
                suggestion_id=f"sug_{i}",
                action=FeedbackAction.ACCEPTED,
            )

        assert store.count_feedback() == 5

        deleted = store.clear_feedback()

        assert deleted == 5
        assert store.count_feedback() == 0


class TestFeedbackStats:
    """Tests for feedback statistics."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> FeedbackStore:
        """Create a fresh feedback store for each test."""
        db_path = tmp_path / "jarvis.db"
        store = FeedbackStore(db_path)
        store.init_schema()
        return store

    def test_get_stats_empty(self, store: FeedbackStore) -> None:
        """Test statistics on empty database."""
        stats = store.get_stats()

        assert stats["total"] == 0
        assert stats["accepted"] == 0
        assert stats["rejected"] == 0
        assert stats["edited"] == 0
        assert stats["acceptance_rate"] == 0.0

    def test_get_stats_with_data(self, store: FeedbackStore) -> None:
        """Test statistics with populated database."""
        # Add 4 accepted, 3 rejected, 2 edited
        for i in range(4):
            store.record_feedback(
                message_id=f"msg_a{i}",
                suggestion_id=f"sug_a{i}",
                action=FeedbackAction.ACCEPTED,
            )
        for i in range(3):
            store.record_feedback(
                message_id=f"msg_r{i}",
                suggestion_id=f"sug_r{i}",
                action=FeedbackAction.REJECTED,
            )
        for i in range(2):
            store.record_feedback(
                message_id=f"msg_e{i}",
                suggestion_id=f"sug_e{i}",
                action=FeedbackAction.EDITED,
            )

        stats = store.get_stats()

        assert stats["total"] == 9
        assert stats["accepted"] == 4
        assert stats["rejected"] == 3
        assert stats["edited"] == 2
        assert abs(stats["acceptance_rate"] - 4 / 9) < 0.001


class TestStatsByDay:
    """Tests for daily statistics."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> FeedbackStore:
        """Create a fresh feedback store for each test."""
        db_path = tmp_path / "jarvis.db"
        store = FeedbackStore(db_path)
        store.init_schema()
        return store

    def test_get_stats_by_day_empty(self, store: FeedbackStore) -> None:
        """Test daily stats on empty database."""
        stats = store.get_stats_by_day()
        assert stats == []

    def test_get_stats_by_day_with_data(self, store: FeedbackStore) -> None:
        """Test daily stats with data."""
        now = datetime.now()
        yesterday = now - timedelta(days=1)

        # Add feedback for today
        store.record_feedback(
            message_id="msg_today_1",
            suggestion_id="sug_today_1",
            action=FeedbackAction.ACCEPTED,
            timestamp=now,
        )
        store.record_feedback(
            message_id="msg_today_2",
            suggestion_id="sug_today_2",
            action=FeedbackAction.REJECTED,
            timestamp=now,
        )

        # Add feedback for yesterday
        store.record_feedback(
            message_id="msg_yesterday",
            suggestion_id="sug_yesterday",
            action=FeedbackAction.ACCEPTED,
            timestamp=yesterday,
        )

        stats = store.get_stats_by_day(days=7)

        assert len(stats) == 2
        # Results are ordered by date descending
        today_stats = stats[0]
        yesterday_stats = stats[1]

        assert today_stats.total == 2
        assert today_stats.accepted == 1
        assert today_stats.rejected == 1
        assert today_stats.acceptance_rate == 0.5

        assert yesterday_stats.total == 1
        assert yesterday_stats.accepted == 1
        assert yesterday_stats.acceptance_rate == 1.0

    def test_get_stats_by_day_respects_days_param(self, store: FeedbackStore) -> None:
        """Test that days parameter limits results."""
        now = datetime.now()
        old = now - timedelta(days=60)

        store.record_feedback(
            message_id="msg_now",
            suggestion_id="sug_now",
            action=FeedbackAction.ACCEPTED,
            timestamp=now,
        )
        store.record_feedback(
            message_id="msg_old",
            suggestion_id="sug_old",
            action=FeedbackAction.ACCEPTED,
            timestamp=old,
        )

        stats = store.get_stats_by_day(days=30)

        # Should only include recent feedback
        assert len(stats) == 1


class TestStatsByContact:
    """Tests for per-contact statistics."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> FeedbackStore:
        """Create a fresh feedback store for each test."""
        db_path = tmp_path / "jarvis.db"
        store = FeedbackStore(db_path)
        store.init_schema()
        return store

    def test_get_stats_by_contact_empty(self, store: FeedbackStore) -> None:
        """Test contact stats on empty database."""
        stats = store.get_stats_by_contact()
        assert stats == []

    def test_get_stats_by_contact_excludes_null_contacts(self, store: FeedbackStore) -> None:
        """Test that feedback without contact_id is excluded."""
        store.record_feedback(
            message_id="msg_1",
            suggestion_id="sug_1",
            action=FeedbackAction.ACCEPTED,
            contact_id=None,
        )

        stats = store.get_stats_by_contact()
        assert stats == []

    def test_get_stats_by_contact_with_data(self, store: FeedbackStore) -> None:
        """Test contact stats with data."""
        # Contact 1: 3 accepted, 1 rejected
        for i in range(3):
            store.record_feedback(
                message_id=f"msg_c1_a{i}",
                suggestion_id=f"sug_c1_a{i}",
                action=FeedbackAction.ACCEPTED,
                contact_id=1,
            )
        store.record_feedback(
            message_id="msg_c1_r",
            suggestion_id="sug_c1_r",
            action=FeedbackAction.REJECTED,
            contact_id=1,
        )

        # Contact 2: 1 accepted
        store.record_feedback(
            message_id="msg_c2_a",
            suggestion_id="sug_c2_a",
            action=FeedbackAction.ACCEPTED,
            contact_id=2,
        )

        stats = store.get_stats_by_contact()

        assert len(stats) == 2
        # Results ordered by total descending
        contact_1_stats = stats[0]
        contact_2_stats = stats[1]

        assert contact_1_stats.contact_id == 1
        assert contact_1_stats.total == 4
        assert contact_1_stats.accepted == 3
        assert contact_1_stats.rejected == 1
        assert contact_1_stats.acceptance_rate == 0.75

        assert contact_2_stats.contact_id == 2
        assert contact_2_stats.total == 1
        assert contact_2_stats.acceptance_rate == 1.0


class TestSingleton:
    """Tests for singleton pattern."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_feedback_store()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_feedback_store()

    def test_get_feedback_store_returns_same_instance(self) -> None:
        """Test that get_feedback_store returns the same instance."""
        store1 = get_feedback_store()
        store2 = get_feedback_store()
        assert store1 is store2

    def test_reset_feedback_store_clears_singleton(self) -> None:
        """Test that reset_feedback_store clears the singleton."""
        store1 = get_feedback_store()
        reset_feedback_store()
        store2 = get_feedback_store()
        assert store1 is not store2


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> FeedbackStore:
        """Create a fresh feedback store for each test."""
        db_path = tmp_path / "jarvis.db"
        store = FeedbackStore(db_path)
        store.init_schema()
        return store

    def test_empty_database_operations(self, store: FeedbackStore) -> None:
        """Test operations on empty database don't crash."""
        assert store.list_feedback() == []
        assert store.get_feedback(1) is None
        assert store.get_feedback_by_suggestion("nonexistent") is None
        assert store.get_feedback_by_message("nonexistent") == []
        assert store.get_feedback_by_contact(1) == []
        assert store.count_feedback() == 0
        assert store.get_stats_by_day() == []
        assert store.get_stats_by_contact() == []

    def test_unicode_in_ids(self, store: FeedbackStore) -> None:
        """Test handling of unicode characters in IDs."""
        feedback = store.record_feedback(
            message_id="msg_unicode_test",
            suggestion_id="sug_unicode_test",
            action=FeedbackAction.ACCEPTED,
        )

        assert feedback is not None
        retrieved = store.get_feedback(feedback.id)  # type: ignore[arg-type]
        assert retrieved is not None

    def test_long_ids(self, store: FeedbackStore) -> None:
        """Test handling of very long ID strings."""
        long_id = "x" * 1000
        feedback = store.record_feedback(
            message_id=long_id,
            suggestion_id=long_id,
            action=FeedbackAction.ACCEPTED,
        )

        assert feedback is not None
        assert feedback.message_id == long_id
        assert feedback.suggestion_id == long_id

    def test_special_characters_in_ids(self, store: FeedbackStore) -> None:
        """Test handling of special characters in IDs."""
        special_ids = [
            "msg-with-dashes",
            "msg_with_underscores",
            "msg.with.dots",
            "msg/with/slashes",
            "msg:with:colons",
        ]

        for msg_id in special_ids:
            feedback = store.record_feedback(
                message_id=msg_id,
                suggestion_id=f"sug_{msg_id}",
                action=FeedbackAction.ACCEPTED,
            )
            assert feedback.message_id == msg_id

    def test_long_suggestion_and_edited_text(self, store: FeedbackStore) -> None:
        """Test handling of very long suggestion text."""
        long_text = "x" * 10000
        feedback = store.record_feedback(
            message_id="msg_long",
            suggestion_id="sug_long",
            action=FeedbackAction.EDITED,
            original_suggestion=long_text,
            edited_text=long_text + "!",
        )

        assert feedback.original_suggestion == long_text
        assert feedback.edited_text == long_text + "!"

        retrieved = store.get_feedback(feedback.id)  # type: ignore[arg-type]
        assert retrieved is not None
        assert retrieved.original_suggestion == long_text

    def test_connection_close_and_reopen(self, tmp_path: Path) -> None:
        """Test that data persists after connection close and reopen."""
        db_path = tmp_path / "jarvis.db"

        # Create store and add data
        store1 = FeedbackStore(db_path)
        store1.init_schema()
        store1.record_feedback(
            message_id="msg_persist",
            suggestion_id="sug_persist",
            action=FeedbackAction.ACCEPTED,
            contact_id=42,
            original_suggestion="Test",
        )
        store1.close()

        # Create new store and verify data
        store2 = FeedbackStore(db_path)
        feedback = store2.get_feedback_by_suggestion("sug_persist")
        assert feedback is not None
        assert feedback.message_id == "msg_persist"
        assert feedback.contact_id == 42
        assert feedback.original_suggestion == "Test"
        store2.close()
