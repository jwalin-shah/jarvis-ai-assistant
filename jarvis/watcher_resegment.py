"""Resegmentation helpers extracted from chat watcher."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


async def resegment_chats(watcher: Any, chat_ids: list[str]) -> None:
    """Re-segment recent messages for chats that hit the threshold."""
    for chat_id in chat_ids:
        lock = await get_resegment_lock(watcher, chat_id)
        async with lock:
            try:
                await asyncio.to_thread(watcher._do_resegment_one, chat_id)
            except Exception as e:
                logger.warning("Error re-segmenting %s: %s", chat_id, e)


async def get_resegment_lock(watcher: Any, chat_id: str) -> asyncio.Lock:
    """Return per-chat async lock with LRU eviction for bounded growth."""
    async with watcher._resegment_locks_guard:
        lock = watcher._resegment_locks.get(chat_id)
        if lock is not None:
            typed_lock: asyncio.Lock = lock
            watcher._resegment_locks.move_to_end(chat_id)
            return typed_lock

        lock = asyncio.Lock()
        watcher._resegment_locks[chat_id] = lock

        if len(watcher._resegment_locks) > watcher._max_resegment_locks:
            watcher._resegment_locks.popitem(last=False)

        return lock


def do_resegment_one(watcher: Any, chat_id: str) -> None:
    """Sync worker: incrementally segment new messages for a single chat."""
    from integrations.imessage import ChatDBReader
    from jarvis.search.vec_search import get_vec_searcher
    from jarvis.topics.segment_pipeline import process_segments
    from jarvis.topics.topic_segmenter import segment_conversation

    try:
        get_vec_searcher()
    except Exception as e:
        logger.debug("Cannot get vec_searcher for re-segmentation: %s", e)
        return

    db = None
    try:
        from jarvis.db import get_db

        db = get_db()
    except Exception as e:
        logger.debug("Cannot get db for segment persistence: %s", e)
        return

    with ChatDBReader() as reader:
        last_segmented_time = None
        if db is not None:
            with db.connection() as conn:
                row = conn.execute(
                    """SELECT MAX(end_time) as last_time
                       FROM conversation_segments
                       WHERE chat_id = ?""",
                    (chat_id,),
                ).fetchone()
                if row and row["last_time"]:
                    try:
                        last_segmented_time = datetime.fromisoformat(row["last_time"])
                    except (ValueError, TypeError):
                        pass

        if last_segmented_time:
            messages = reader.get_messages(chat_id, limit=watcher._segment_window * 2)
            messages.reverse()

            new_messages = [m for m in messages if m.date and m.date > last_segmented_time]
            if not new_messages:
                logger.debug("No new messages to segment for %s", chat_id[:20])
                return

            context_messages = [m for m in messages if m.date and m.date <= last_segmented_time][
                -5:
            ]
            messages_to_segment = context_messages + new_messages
            logger.debug(
                "Segmenting %d new messages (+ %d context) for %s",
                len(new_messages),
                len(context_messages),
                chat_id[:20],
            )
        else:
            messages = reader.get_messages(chat_id, limit=watcher._segment_window)
            messages.reverse()
            messages_to_segment = messages
            logger.debug(
                "First-time segmentation for %s: %d messages",
                chat_id[:20],
                len(messages_to_segment),
            )

        if not messages_to_segment:
            return

        segments = segment_conversation(messages_to_segment, contact_id=chat_id)
        if not segments:
            logger.debug("No segments created for %s", chat_id[:20])
            return

        if last_segmented_time and len(segments) > 0:
            with db.connection() as conn:
                last_seg_row = conn.execute(
                    """SELECT id, segment_id, message_count, preview, end_time
                       FROM conversation_segments
                       WHERE chat_id = ?
                       ORDER BY end_time DESC LIMIT 1""",
                    (chat_id,),
                ).fetchone()

                if last_seg_row:
                    first_new_seg = segments[0]
                    if len(first_new_seg.messages) <= 3:
                        pass

        stats = process_segments(segments, chat_id, contact_id=chat_id, extract_facts=False)

        logger.info(
            "Incremental segment %s: persisted=%d, indexed=%d (new: %s)",
            chat_id[:20],
            stats["persisted"],
            stats["indexed"],
            "yes" if last_segmented_time else "first-run",
        )
