"""Tests for the chat.db file watcher."""

from unittest.mock import patch

import pytest

from jarvis.watcher import APPLE_EPOCH_OFFSET, ChatDBWatcher


class MockBroadcastHandler:
    """Mock broadcast handler for testing."""

    def __init__(self) -> None:
        self.broadcasts: list[tuple[str, dict]] = []

    async def broadcast(self, method: str, params: dict) -> None:
        self.broadcasts.append((method, params))


class TestChatDBWatcher:
    """Tests for ChatDBWatcher."""

    @pytest.fixture
    def handler(self) -> MockBroadcastHandler:
        """Create a mock broadcast handler."""
        return MockBroadcastHandler()

    @pytest.fixture
    def watcher(self, handler: MockBroadcastHandler) -> ChatDBWatcher:
        """Create a watcher instance."""
        return ChatDBWatcher(handler, poll_interval=0.1)

    def test_init(self, watcher: ChatDBWatcher) -> None:
        """Watcher initializes with correct state."""
        assert watcher._running is False
        assert watcher._last_rowid is None
        assert watcher._task is None

    @pytest.mark.asyncio
    async def test_start_stop(self, watcher: ChatDBWatcher) -> None:
        """Watcher can start and stop."""
        with patch.object(watcher, "_validate_schema", return_value=True):
            with patch.object(watcher, "_get_last_rowid", return_value=100):
                with patch("jarvis.watcher.CHAT_DB_PATH") as mock_path:
                    mock_path.exists.return_value = True
                    mock_path.stat.return_value.st_mtime = 1000.0

                    await watcher.start()
                    assert watcher._running is True
                    assert watcher._last_rowid == 100

                    await watcher.stop()
                    assert watcher._running is False

    @pytest.mark.asyncio
    async def test_query_last_rowid(self, watcher: ChatDBWatcher) -> None:
        """Can query the last message ROWID."""
        with patch("jarvis.watcher.CHAT_DB_PATH") as mock_path:
            mock_path.exists.return_value = False

            result = watcher._query_last_rowid()
            assert result is None

    @pytest.mark.asyncio
    async def test_query_new_messages_no_db(self, watcher: ChatDBWatcher) -> None:
        """Returns empty list when database doesn't exist."""
        with patch("jarvis.watcher.CHAT_DB_PATH") as mock_path:
            mock_path.exists.return_value = False

            result = watcher._query_new_messages(100)
            assert result == []

    @pytest.mark.asyncio
    async def test_check_new_messages(
        self, watcher: ChatDBWatcher, handler: MockBroadcastHandler
    ) -> None:
        """New messages trigger broadcast notifications."""
        watcher._last_rowid = 100

        mock_messages = [
            {
                "id": 101,
                "chat_id": "chat123",
                "sender": "+15551234567",
                "text": "Hello!",
                "date": "2024-01-15T10:00:00",
                "is_from_me": False,
            }
        ]

        with patch.object(watcher, "_get_new_messages", return_value=mock_messages):
            await watcher._check_new_messages()

            # Should have broadcast the new message
            assert len(handler.broadcasts) == 1
            method, params = handler.broadcasts[0]
            assert method == "new_message"
            assert params["message_id"] == 101
            assert params["chat_id"] == "chat123"

            # Last rowid should be updated
            assert watcher._last_rowid == 101

    @pytest.mark.asyncio
    async def test_check_new_messages_updates_max_rowid(
        self, watcher: ChatDBWatcher, handler: MockBroadcastHandler
    ) -> None:
        """Last rowid is updated to max of new messages."""
        watcher._last_rowid = 100

        mock_messages = [
            {
                "id": 101,
                "chat_id": "c1",
                "sender": "s",
                "text": "a",
                "date": "d",
                "is_from_me": False,
            },
            {
                "id": 105,
                "chat_id": "c2",
                "sender": "s",
                "text": "b",
                "date": "d",
                "is_from_me": False,
            },
            {
                "id": 103,
                "chat_id": "c3",
                "sender": "s",
                "text": "c",
                "date": "d",
                "is_from_me": False,
            },
        ]

        with patch.object(watcher, "_get_new_messages", return_value=mock_messages):
            await watcher._check_new_messages()

            # Should update to max rowid (105)
            assert watcher._last_rowid == 105


class TestAppleTimestamp:
    """Tests for Apple timestamp handling."""

    def test_apple_epoch_offset(self) -> None:
        """Apple epoch offset is correct (2001-01-01 00:00:00 UTC)."""
        # Unix timestamp for 2001-01-01 00:00:00 UTC
        assert APPLE_EPOCH_OFFSET == 978307200

    def test_timestamp_conversion(self) -> None:
        """Apple timestamp converts to correct datetime."""
        from datetime import datetime

        # Apple timestamp for 2024-01-15 10:00:00 UTC
        # = (Unix timestamp - Apple epoch) * 1e9
        unix_ts = datetime(2024, 1, 15, 10, 0, 0).timestamp()
        apple_ts = (unix_ts - APPLE_EPOCH_OFFSET) * 1_000_000_000

        # Convert back
        recovered_unix = (apple_ts / 1_000_000_000) + APPLE_EPOCH_OFFSET
        recovered_dt = datetime.fromtimestamp(recovered_unix)

        assert recovered_dt.year == 2024
        assert recovered_dt.month == 1
        assert recovered_dt.day == 15
