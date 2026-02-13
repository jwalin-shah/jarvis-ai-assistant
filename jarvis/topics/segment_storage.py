"""Persist and retrieve topic segments from the database.

CRUD operations for conversation_segments and segment_messages tables.
All write operations use batch inserts (executemany) to avoid N+1 patterns.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import sqlite3

    from jarvis.topics.topic_segmenter import TopicSegment

logger = logging.getLogger(__name__)


def persist_segments(
    conn: sqlite3.Connection,
    segments: list[TopicSegment],
    chat_id: str,
    contact_id: str | None = None,
) -> list[int]:
    """Insert segments and their message memberships into the database.

    Uses batch inserts (executemany) for optimal performance.
    SQLite 3.35+ RETURNING clause used to get inserted row IDs.

    Args:
        conn: Active database connection (caller manages transaction).
        segments: TopicSegment objects from the segmenter.
        chat_id: iMessage chat identifier.
        contact_id: Optional contact identifier.

    Returns:
        List of database IDs for the inserted conversation_segments rows.
    """
    if not segments:
        return []

    # Prepare batch data for segments
    segment_rows = []
    for segment in segments:
        entities_json = json.dumps(segment.entities) if segment.entities else None
        keywords_json = json.dumps(segment.keywords) if segment.keywords else None
        segment_rows.append(
            (
                segment.segment_id,
                chat_id,
                contact_id,
                segment.start_time.isoformat(),
                segment.end_time.isoformat(),
                segment.topic_label,
                keywords_json,
                entities_json,
                segment.message_count,
                segment.confidence,
            )
        )

    # Batch insert segments with RETURNING to get IDs
    # SQLite 3.35+ supports RETURNING; fallback for older versions
    try:
        cursor = conn.executemany(
            """
            INSERT INTO conversation_segments
            (segment_id, chat_id, contact_id, start_time, end_time,
             topic_label, keywords_json, entities_json, message_count,
             confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            segment_rows,
        )
        db_ids = [row[0] for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        # Fallback: insert individually if RETURNING not supported (SQLite < 3.35)
        # This maintains compatibility with older SQLite versions
        db_ids = []
        for row_data in segment_rows:
            cursor = conn.execute(
                """
                INSERT INTO conversation_segments
                (segment_id, chat_id, contact_id, start_time, end_time,
                 topic_label, keywords_json, entities_json, message_count,
                 confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row_data,
            )
            db_ids.append(cursor.lastrowid)

    # Build message membership rows with the retrieved segment IDs
    msg_rows = []
    for seg_db_id, segment in zip(db_ids, segments):
        for pos, msg in enumerate(segment.messages):
            msg_rows.append(
                (
                    seg_db_id,
                    msg.message_id or 0,
                    pos,
                    msg.is_from_me,
                )
            )

    # Batch insert all message memberships at once
    if msg_rows:
        conn.executemany(
            """
            INSERT OR IGNORE INTO segment_messages
            (segment_id, message_rowid, position, is_from_me)
            VALUES (?, ?, ?, ?)
            """,
            msg_rows,
        )

    logger.debug(
        "Persisted %d segments (%d messages) for chat %s",
        len(db_ids),
        sum(s.message_count for s in segments),
        chat_id[:20],
    )
    return db_ids


def link_vec_chunk_rowids(
    conn: sqlite3.Connection,
    segment_db_ids: list[int],
    vec_chunk_rowids: list[int],
) -> None:
    """Link persisted segments to their vec_chunks rowids.

    Args:
        conn: Active database connection.
        segment_db_ids: IDs from conversation_segments.
        vec_chunk_rowids: Corresponding rowids from vec_chunks.
    """
    if not segment_db_ids or not vec_chunk_rowids:
        return

    batch = [
        (rowid, db_id)
        for db_id, rowid in zip(segment_db_ids, vec_chunk_rowids)
    ]
    conn.executemany(
        "UPDATE conversation_segments SET vec_chunk_rowid = ? WHERE id = ?",
        batch,
    )


def mark_facts_extracted(
    conn: sqlite3.Connection,
    segment_db_ids: list[int],
) -> None:
    """Mark segments as having had facts extracted.

    Args:
        conn: Active database connection.
        segment_db_ids: IDs of segments to mark.
    """
    if not segment_db_ids:
        return

    batch = [(db_id,) for db_id in segment_db_ids]
    conn.executemany(
        "UPDATE conversation_segments SET facts_extracted = TRUE WHERE id = ?",
        batch,
    )


def delete_segments_for_chat(conn: sqlite3.Connection, chat_id: str) -> int:
    """Delete all segments and their message memberships for a chat.

    Deletes segment_messages first (FK child), then conversation_segments.

    Args:
        conn: Active database connection.
        chat_id: Chat to delete segments for.

    Returns:
        Number of segments deleted.
    """
    # Get segment IDs for this chat
    rows = conn.execute(
        "SELECT id FROM conversation_segments WHERE chat_id = ?",
        (chat_id,),
    ).fetchall()

    if not rows:
        return 0

    seg_ids = [r["id"] for r in rows]
    placeholders = ",".join("?" * len(seg_ids))

    # Delete child rows first
    conn.execute(
        f"DELETE FROM segment_messages WHERE segment_id IN ({placeholders})",  # noqa: S608
        seg_ids,
    )

    # Delete parent rows
    cursor = conn.execute(
        "DELETE FROM conversation_segments WHERE chat_id = ?",
        (chat_id,),
    )
    deleted = cursor.rowcount
    logger.debug("Deleted %d segments for chat %s", deleted, chat_id[:20])
    return deleted


def get_segments_for_chat(
    conn: sqlite3.Connection,
    chat_id: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Retrieve recent segments with their message ROWIDs.

    Args:
        conn: Active database connection.
        chat_id: Chat to retrieve segments for.
        limit: Max segments to return.

    Returns:
        List of segment dicts with 'message_rowids' list included.
    """
    seg_rows = conn.execute(
        """
        SELECT id, segment_id, chat_id, contact_id, start_time, end_time,
               topic_label, keywords_json, entities_json, message_count,
               confidence, vec_chunk_rowid, facts_extracted
        FROM conversation_segments
        WHERE chat_id = ?
        ORDER BY start_time DESC
        LIMIT ?
        """,
        (chat_id, limit),
    ).fetchall()

    if not seg_rows:
        return []

    # Batch fetch all message memberships
    seg_ids = [r["id"] for r in seg_rows]
    placeholders = ",".join("?" * len(seg_ids))
    msg_rows = conn.execute(
        f"""
        SELECT segment_id, message_rowid, position, is_from_me
        FROM segment_messages
        WHERE segment_id IN ({placeholders})
        ORDER BY segment_id, position
        """,  # noqa: S608
        seg_ids,
    ).fetchall()

    # Group messages by segment_id
    msgs_by_seg: dict[int, list[dict[str, Any]]] = {}
    for mr in msg_rows:
        sid = mr["segment_id"]
        msgs_by_seg.setdefault(sid, []).append(
            {
                "message_rowid": mr["message_rowid"],
                "position": mr["position"],
                "is_from_me": bool(mr["is_from_me"]),
            }
        )

    results = []
    for sr in seg_rows:
        results.append(
            {
                "id": sr["id"],
                "segment_id": sr["segment_id"],
                "chat_id": sr["chat_id"],
                "contact_id": sr["contact_id"],
                "start_time": sr["start_time"],
                "end_time": sr["end_time"],
                "topic_label": sr["topic_label"],
                "keywords_json": sr["keywords_json"],
                "entities_json": sr["entities_json"],
                "message_count": sr["message_count"],
                "confidence": sr["confidence"],
                "vec_chunk_rowid": sr["vec_chunk_rowid"],
                "facts_extracted": bool(sr["facts_extracted"]),
                "messages": msgs_by_seg.get(sr["id"], []),
            }
        )

    return results


def get_unextracted_segments(
    conn: sqlite3.Connection,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Get segments where facts have not been extracted yet.

    Args:
        conn: Active database connection.
        limit: Max segments to return.

    Returns:
        List of segment dicts (without message details for efficiency).
    """
    rows = conn.execute(
        """
        SELECT id, segment_id, chat_id, contact_id, start_time, end_time,
               topic_label, message_count
        FROM conversation_segments
        WHERE facts_extracted = FALSE
        ORDER BY start_time ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    return [dict(r) for r in rows]
