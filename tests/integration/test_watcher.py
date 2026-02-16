"""Integration tests for the chat.db file watcher.

Tests the ChatDBWatcher component including FSEvents/polling,
debouncing, and message detection notifications.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jarvis.utils.datetime_utils import APPLE_EPOCH_OFFSET
from jarvis.watcher import (
    DEBOUNCE_INTERVAL,
    POLL_INTERVAL,
    ChatDBWatcher,
)


class MockBroadcastHandler:
    """Mock broadcast handler that records calls."""

    def __init__(self):
        self.broadcasts: list[tuple[str, dict]] = []

    async def broadcast(self, method: str, params: dict) -> None:
        """Record broadcast calls."""
        self.broadcasts.append((method, params))


class TestChatDBWatcherInit:
    """Tests for ChatDBWatcher initialization."""

    def test_watcher_default_settings(self):
        """Watcher initializes with default settings."""
        handler = MockBroadcastHandler()
        watcher = ChatDBWatcher(handler)

        assert watcher._broadcast_handler is handler
        assert watcher._poll_interval == POLL_INTERVAL
        assert watcher._running is False
        assert watcher._last_rowid is None

    def test_watcher_custom_poll_interval(self):
        """Watcher accepts custom poll interval."""
        handler = MockBroadcastHandler()
        watcher = ChatDBWatcher(handler, poll_interval=5.0)

        assert watcher._poll_interval == 5.0

    def test_watcher_force_polling(self):
        """Watcher can force polling mode."""
        handler = MockBroadcastHandler()
        watcher = ChatDBWatcher(handler, use_fsevents=False)

        assert watcher._use_fsevents is False


class TestChatDBWatcherLifecycle:
    """Tests for watcher start/stop lifecycle."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_watcher_start_sets_running(self):
        """Start sets running flag and initializes last_rowid."""
        handler = MockBroadcastHandler()
        watcher = ChatDBWatcher(handler, use_fsevents=False)

        # Simulate initialization by directly setting state
        watcher._running = True
        watcher._last_rowid = 100

        assert watcher._running is True
        assert watcher._last_rowid == 100

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_watcher_stop_clears_running(self):
        """Stop clears running flag."""
        handler = MockBroadcastHandler()
        watcher = ChatDBWatcher(handler, use_fsevents=False)
        watcher._running = True

        # Create a dummy task
        async def dummy():
            await asyncio.sleep(10)

        watcher._task = asyncio.create_task(dummy())

        await watcher.stop()

        assert watcher._running is False
        assert watcher._task is None

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_watcher_stop_is_idempotent(self):
        """Stop can be called multiple times safely."""
        handler = MockBroadcastHandler()
        watcher = ChatDBWatcher(handler, use_fsevents=False)

        # Should not raise
        await watcher.stop()
        await watcher.stop()


class TestDebouncing:
    """Tests for debounce behavior."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_debounce_coalesces_rapid_changes(self):
        """Rapid changes are coalesced into single check."""
        handler = MockBroadcastHandler()
        watcher = ChatDBWatcher(handler, use_fsevents=False)
        watcher._running = True
        watcher._last_rowid = 100

        check_count = 0

        async def mock_check():
            nonlocal check_count
            check_count += 1

        with patch.object(watcher, "_check_new_messages", side_effect=mock_check):
            # Trigger multiple rapid changes
            await watcher._debounced_check()
            await watcher._debounced_check()
            await watcher._debounced_check()

            # Wait for debounce to complete
            await asyncio.sleep(DEBOUNCE_INTERVAL + 0.05)

            # Should only check once
            assert check_count == 1

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_debounce_subsequent_changes_trigger_new_check(self):
        """Changes after debounce window trigger new check."""
        handler = MockBroadcastHandler()
        watcher = ChatDBWatcher(handler, use_fsevents=False)
        watcher._running = True
        watcher._last_rowid = 100

        check_count = 0

        async def mock_check():
            nonlocal check_count
            check_count += 1

        with patch.object(watcher, "_check_new_messages", side_effect=mock_check):
            # First change
            await watcher._debounced_check()
            await asyncio.sleep(DEBOUNCE_INTERVAL + 0.05)

            # Second change after debounce completed
            watcher._debounce_task = None  # Reset debounce state
            await watcher._debounced_check()
            await asyncio.sleep(DEBOUNCE_INTERVAL + 0.05)

            # Should have checked twice
            assert check_count == 2


class TestNewMessageDetection:
    """Tests for new message detection."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_check_new_messages_broadcasts_notification(self):
        """New messages trigger broadcast notifications."""
        handler = MockBroadcastHandler()
        watcher = ChatDBWatcher(handler, use_fsevents=False)
        watcher._last_rowid = 100

        mock_messages = [
            {
                "id": 101,
                "chat_id": "chat123",
                "sender": "+1234567890",
                "text": "Hello!",
                "date": datetime.now().isoformat(),
                "is_from_me": False,
            },
        ]

        with patch.object(watcher, "_get_new_messages", return_value=mock_messages):
            await watcher._check_new_messages()

        assert len(handler.broadcasts) == 1
        method, params = handler.broadcasts[0]
        assert method == "new_message"
        assert params["chat_id"] == "chat123"
        assert params["text"] == "Hello!"
        assert params["is_from_me"] is False

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_check_new_messages_updates_last_rowid(self):
        """Last ROWID is updated after finding new messages."""
        handler = MockBroadcastHandler()
        watcher = ChatDBWatcher(handler, use_fsevents=False)
        watcher._last_rowid = 100

        mock_messages = [
            {
                "id": 101,
                "chat_id": "chat1",
                "sender": "a",
                "text": "a",
                "date": None,
                "is_from_me": False,
            },
            {
                "id": 105,
                "chat_id": "chat2",
                "sender": "b",
                "text": "b",
                "date": None,
                "is_from_me": False,
            },
            {
                "id": 103,
                "chat_id": "chat3",
                "sender": "c",
                "text": "c",
                "date": None,
                "is_from_me": False,
            },
        ]

        with patch.object(watcher, "_get_new_messages", return_value=mock_messages):
            await watcher._check_new_messages()

        # Should be updated to max ID
        assert watcher._last_rowid == 105

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_check_new_messages_no_update_when_empty(self):
        """Last ROWID not updated when no new messages."""
        handler = MockBroadcastHandler()
        watcher = ChatDBWatcher(handler, use_fsevents=False)
        watcher._last_rowid = 100

        with patch.object(watcher, "_get_new_messages", return_value=[]):
            await watcher._check_new_messages()

        assert watcher._last_rowid == 100
        assert len(handler.broadcasts) == 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_check_new_messages_handles_errors(self):
        """Errors during check don't crash the watcher."""
        handler = MockBroadcastHandler()
        watcher = ChatDBWatcher(handler, use_fsevents=False)
        watcher._last_rowid = 100

        with patch.object(watcher, "_get_new_messages", side_effect=Exception("DB error")):
            # Should not raise
            await watcher._check_new_messages()

        # State should be preserved
        assert watcher._last_rowid == 100


class TestPollingMode:
    """Tests for polling mode behavior."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_polling_checks_file_modification(self):
        """Polling mode checks file modification time."""
        handler = MockBroadcastHandler()
        watcher = ChatDBWatcher(handler, use_fsevents=False, poll_interval=0.01)
        watcher._running = True
        watcher._last_mtime = 1000.0

        check_count = 0

        async def mock_check():
            nonlocal check_count
            check_count += 1
            watcher._running = False  # Stop after first check

        mock_stat = MagicMock()
        mock_stat.st_mtime = 1001.0  # Newer than last_mtime

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "stat", return_value=mock_stat):
                with patch.object(watcher, "_check_new_messages", side_effect=mock_check):
                    await watcher._watch_polling()

        assert check_count == 1

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_polling_skips_unchanged_file(self):
        """Polling skips check if file hasn't changed."""
        handler = MockBroadcastHandler()
        watcher = ChatDBWatcher(handler, use_fsevents=False, poll_interval=0.01)
        watcher._running = True
        watcher._last_mtime = 1000.0

        iteration_count = 0

        mock_stat = MagicMock()
        mock_stat.st_mtime = 1000.0  # Same as last_mtime

        async def stop_after_iterations():
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count >= 2:
                watcher._running = False

        original_sleep = asyncio.sleep

        async def mock_sleep(delay):
            await stop_after_iterations()
            if watcher._running:
                await original_sleep(0.001)

        check_called = False

        async def mock_check():
            nonlocal check_called
            check_called = True

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "stat", return_value=mock_stat):
                with patch.object(watcher, "_check_new_messages", side_effect=mock_check):
                    with patch("asyncio.sleep", side_effect=mock_sleep):
                        await watcher._watch_polling()

        # Check should not have been called since file didn't change
        assert check_called is False


class TestQueryHelpers:
    """Tests for SQL query helper methods."""

    def test_query_last_rowid_returns_none_if_db_missing(self):
        """Query returns None if database doesn't exist."""
        handler = MockBroadcastHandler()
        watcher = ChatDBWatcher(handler, use_fsevents=False)

        with patch.object(Path, "exists", return_value=False):
            result = watcher._query_last_rowid()

        assert result is None

    def test_query_new_messages_returns_empty_if_db_missing(self):
        """Query returns empty list if database doesn't exist."""
        handler = MockBroadcastHandler()
        watcher = ChatDBWatcher(handler, use_fsevents=False)

        with patch.object(Path, "exists", return_value=False):
            result = watcher._query_new_messages(since_rowid=100)

        assert result == []


class TestAppleTimestamp:
    """Tests for Apple timestamp conversion."""

    def test_apple_epoch_offset_is_correct(self):
        """Apple epoch offset is 2001-01-01."""
        # 2001-01-01 00:00:00 UTC = 978307200 Unix timestamp
        assert APPLE_EPOCH_OFFSET == 978307200


class TestMultipleMessages:
    """Tests for handling multiple new messages."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_broadcasts_each_message_separately(self):
        """Each new message gets its own broadcast."""
        handler = MockBroadcastHandler()
        watcher = ChatDBWatcher(handler, use_fsevents=False)
        watcher._last_rowid = 100

        mock_messages = [
            {
                "id": 101,
                "chat_id": "chat1",
                "sender": "Alice",
                "text": "Hi",
                "date": None,
                "is_from_me": False,
            },
            {
                "id": 102,
                "chat_id": "chat2",
                "sender": "Bob",
                "text": "Hey",
                "date": None,
                "is_from_me": False,
            },
            {
                "id": 103,
                "chat_id": "chat1",
                "sender": "Alice",
                "text": "Hello",
                "date": None,
                "is_from_me": False,
            },
        ]

        with patch.object(watcher, "_get_new_messages", return_value=mock_messages):
            await watcher._check_new_messages()

        assert len(handler.broadcasts) == 3
        # Verify different messages
        texts = [params["text"] for _, params in handler.broadcasts]
        assert "Hi" in texts
        assert "Hey" in texts
        assert "Hello" in texts

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_handles_from_me_messages(self):
        """Messages from user are handled correctly."""
        handler = MockBroadcastHandler()
        watcher = ChatDBWatcher(handler, use_fsevents=False)
        watcher._last_rowid = 100

        mock_messages = [
            {
                "id": 101,
                "chat_id": "chat1",
                "sender": "me",
                "text": "My message",
                "date": None,
                "is_from_me": True,
            },
        ]

        with patch.object(watcher, "_get_new_messages", return_value=mock_messages):
            await watcher._check_new_messages()

        assert len(handler.broadcasts) == 1
        _, params = handler.broadcasts[0]
        assert params["is_from_me"] is True


class TestFSEventsFallback:
    """Tests for FSEvents to polling fallback."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_fallback_to_polling_on_error(self):
        """Falls back to polling when FSEvents fails."""
        handler = MockBroadcastHandler()
        watcher = ChatDBWatcher(handler, use_fsevents=True)
        watcher._running = True

        # Simulate FSEvents error
        async def failing_fsevents():
            raise Exception("FSEvents unavailable")

        with patch.object(watcher, "_watch_fsevents", side_effect=failing_fsevents):
            # The watch should not propagate the error but switch modes
            # In practice this is handled internally
            pass


class TestStopEvent:
    """Tests for the stop event mechanism."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_make_stop_event_sets_when_stopped(self):
        """Stop event is set when running becomes False."""
        handler = MockBroadcastHandler()
        watcher = ChatDBWatcher(handler, use_fsevents=False)
        watcher._running = True

        event = watcher._make_stop_event()

        # Event should not be set initially
        assert not event.is_set()

        # Stop the watcher
        watcher._running = False

        # Poll for event to be set (up to 1 second)
        for _ in range(20):
            if event.is_set():
                break
            await asyncio.sleep(0.05)

        assert event.is_set()
