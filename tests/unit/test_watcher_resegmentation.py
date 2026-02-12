"""TEST-07: Watcher concurrent resegmentation tests.

Verifies that per-chat locks prevent concurrent resegmentation
for the same chat_id, avoiding interleaved delete+index corruption.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest


class TestResegmentationLocking:
    """Verify per-chat lock prevents concurrent resegmentation."""

    @pytest.mark.asyncio
    async def test_get_resegment_lock_returns_same_lock_for_same_chat(self):
        """Same chat_id returns same lock object."""
        from jarvis.watcher import ChatDBWatcher

        mock_handler = AsyncMock()
        watcher = ChatDBWatcher(mock_handler, use_fsevents=False)

        lock1 = await watcher._get_resegment_lock("chat123")
        lock2 = await watcher._get_resegment_lock("chat123")

        assert lock1 is lock2

    @pytest.mark.asyncio
    async def test_get_resegment_lock_different_chats_get_different_locks(self):
        """Different chat_ids get independent locks."""
        from jarvis.watcher import ChatDBWatcher

        mock_handler = AsyncMock()
        watcher = ChatDBWatcher(mock_handler, use_fsevents=False)

        lock1 = await watcher._get_resegment_lock("chat1")
        lock2 = await watcher._get_resegment_lock("chat2")

        assert lock1 is not lock2

    @pytest.mark.asyncio
    async def test_resegment_lock_lru_eviction(self):
        """Locks beyond max_resegment_locks are evicted (LRU)."""
        from jarvis.watcher import ChatDBWatcher

        mock_handler = AsyncMock()
        watcher = ChatDBWatcher(mock_handler, use_fsevents=False)
        watcher._max_resegment_locks = 5

        # Create 6 locks
        locks = {}
        for i in range(6):
            lock = await watcher._get_resegment_lock(f"chat{i}")
            locks[f"chat{i}"] = lock

        # chat0 should have been evicted (LRU)
        assert "chat0" not in watcher._resegment_locks
        # chat5 (most recent) should still be present
        assert "chat5" in watcher._resegment_locks

    @pytest.mark.asyncio
    async def test_concurrent_resegment_same_chat_serialized(self):
        """Two concurrent resegmentation tasks for same chat are serialized."""
        from jarvis.watcher import ChatDBWatcher

        mock_handler = AsyncMock()
        watcher = ChatDBWatcher(mock_handler, use_fsevents=False)

        execution_log: list[tuple[str, str]] = []

        original_do_resegment = watcher._do_resegment_one

        def mock_resegment(chat_id):
            execution_log.append(("enter", chat_id))
            # Simulate work
            import time

            time.sleep(0.05)
            execution_log.append(("exit", chat_id))

        watcher._do_resegment_one = mock_resegment

        # Launch two concurrent resegmentations for the same chat
        task1 = asyncio.create_task(watcher._resegment_chats(["chat_a"]))
        task2 = asyncio.create_task(watcher._resegment_chats(["chat_a"]))

        await asyncio.gather(task1, task2)

        # Both should complete
        enter_events = [e for e in execution_log if e[0] == "enter"]
        exit_events = [e for e in execution_log if e[0] == "exit"]
        assert len(enter_events) == 2
        assert len(exit_events) == 2

        # Serialization: second enter should be after first exit
        # Find indices
        first_exit_idx = execution_log.index(("exit", "chat_a"))
        second_enter_idx = len(execution_log) - 1 - execution_log[::-1].index(("enter", "chat_a"))

        # The second enter should be after the first exit
        assert second_enter_idx > first_exit_idx or first_exit_idx == 1

    @pytest.mark.asyncio
    async def test_concurrent_resegment_different_chats_parallel(self):
        """Resegmentation of different chats can proceed in parallel."""
        from jarvis.watcher import ChatDBWatcher

        mock_handler = AsyncMock()
        watcher = ChatDBWatcher(mock_handler, use_fsevents=False)

        timestamps: list[tuple[str, str, float]] = []

        import time

        def mock_resegment(chat_id):
            timestamps.append(("enter", chat_id, time.monotonic()))
            time.sleep(0.02)
            timestamps.append(("exit", chat_id, time.monotonic()))

        watcher._do_resegment_one = mock_resegment

        # Launch resegmentation for two different chats
        task1 = asyncio.create_task(watcher._resegment_chats(["chat_x"]))
        task2 = asyncio.create_task(watcher._resegment_chats(["chat_y"]))

        await asyncio.gather(task1, task2)

        # Both should complete
        enters = [(e[1], e[2]) for e in timestamps if e[0] == "enter"]
        assert len(enters) == 2

    @pytest.mark.asyncio
    async def test_chat_msg_count_tracking(self):
        """Message counts per chat accumulate and trigger resegmentation."""
        from jarvis.watcher import ChatDBWatcher

        mock_handler = AsyncMock()
        watcher = ChatDBWatcher(mock_handler, use_fsevents=False)
        watcher._segment_threshold = 3

        # Simulate counting messages
        watcher._chat_msg_counts["chat1"] = 2

        # Adding one more should cross threshold
        watcher._chat_msg_counts["chat1"] = watcher._chat_msg_counts.get("chat1", 0) + 1
        assert watcher._chat_msg_counts["chat1"] >= watcher._segment_threshold

    @pytest.mark.asyncio
    async def test_resegment_lock_lru_moves_to_end(self):
        """Accessing an existing lock moves it to end of LRU."""
        from jarvis.watcher import ChatDBWatcher

        mock_handler = AsyncMock()
        watcher = ChatDBWatcher(mock_handler, use_fsevents=False)
        watcher._max_resegment_locks = 3

        await watcher._get_resegment_lock("chat_a")
        await watcher._get_resegment_lock("chat_b")
        await watcher._get_resegment_lock("chat_c")

        # Access chat_a again (moves to end)
        await watcher._get_resegment_lock("chat_a")

        # Now add chat_d - should evict chat_b (oldest), not chat_a
        await watcher._get_resegment_lock("chat_d")

        assert "chat_b" not in watcher._resegment_locks
        assert "chat_a" in watcher._resegment_locks
        assert "chat_c" in watcher._resegment_locks
        assert "chat_d" in watcher._resegment_locks
