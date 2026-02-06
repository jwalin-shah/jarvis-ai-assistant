"""Vector Search - High-performance semantic search using sqlite-vec.

This module provides the `VecSearcher` class, which handles:
1.  Indexing messages and chunks into `sqlite-vec` virtual tables.
2.  Semantic search for messages (`vec_messages`).
3.  Semantic search for conversation pairs/chunks (`vec_chunks`).

It replaces the legacy `jarvis/embeddings.py` (numpy-based) and
`jarvis/semantic_search.py` (blob-based) with a unified, efficient backend.

Usage:
    from jarvis.search.vec_search import get_vec_searcher

    searcher = get_vec_searcher()

    # Search for messages
    results = searcher.search("dinner plans", limit=5)

    # Search for similar conversation pairs (RAG)
    pairs = searcher.search_with_pairs("Want to grab lunch?", limit=5)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from jarvis.db import JarvisDB, get_db
from jarvis.embedding_adapter import get_embedder

if TYPE_CHECKING:
    from contracts.imessage import Message
    from jarvis.topics.topic_segmenter import TopicSegment

logger = logging.getLogger(__name__)


@dataclass
class VecSearchResult:
    """Result from vector search."""

    rowid: int
    distance: float
    score: float  # Converted from distance (0..1)

    # Message fields (if searching messages)
    chat_id: str | None = None
    text: str | None = None
    sender: str | None = None
    timestamp: float | None = None
    is_from_me: bool | None = None

    # Chunk fields (if searching pairs)
    trigger_text: str | None = None
    response_text: str | None = None
    response_type: str | None = None
    topic: str | None = None
    quality_score: float | None = None


class VecSearcher:
    """High-performance vector search using sqlite-vec."""

    def __init__(self, db: JarvisDB) -> None:
        self.db = db
        self._embedder = get_embedder()
        self._lock = threading.Lock()

    def _vec_tables_exist(self) -> bool:
        """Check if sqlite-vec tables exist."""
        try:
            with self.db.connection() as conn:
                # Check for vec_messages
                res = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_messages'"
                ).fetchone()
                return res is not None
        except Exception:
            return False

    def index_message(self, message: Message) -> bool:
        """Index a single message into vec_messages.

        Args:
            message: Message to index

        Returns:
            True if indexed, False if skipped
        """
        if not message.text or len(message.text.strip()) < 3:
            return False

        try:
            # Compute embedding
            embedding = self._embedder.encode(message.text, normalize=True)

            with self.db.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO vec_messages(
                        rowid,
                        embedding,
                        chat_id,
                        text_preview,
                        sender,
                        timestamp,
                        is_from_me
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message.id,
                        self._quantize_embedding(embedding),
                        message.chat_id,
                        message.text[:200],  # Preview
                        message.sender,
                        message.date.timestamp(),
                        1 if message.is_from_me else 0,
                    ),
                )
            return True
        except Exception as e:
            logger.error("Failed to index message %s: %s", message.id, e)
            return False

    def index_messages(self, messages: list[Message]) -> int:
        """Index multiple messages efficiently.

        Args:
            messages: List of messages to index

        Returns:
            Number of messages successfully indexed
        """
        valid_messages = [m for m in messages if m.text and len(m.text.strip()) >= 3]
        if not valid_messages:
            return 0

        count = 0
        try:
            # Batch compute embeddings
            texts = [m.text for m in valid_messages]
            # Use cached embedder which handles batching internally
            embeddings = self._embedder.encode(texts, normalize=True)

            with self.db.connection() as conn:
                # Prepare batch data
                batch_data = []
                for msg, emb in zip(valid_messages, embeddings):
                    batch_data.append(
                        (
                            msg.id,
                            self._quantize_embedding(emb),
                            msg.chat_id,
                            msg.text[:200],
                            msg.sender,
                            msg.date.timestamp(),
                            1 if msg.is_from_me else 0,
                        )
                    )

                conn.executemany(
                    """
                    INSERT INTO vec_messages(
                        rowid,
                        embedding,
                        chat_id,
                        text_preview,
                        sender,
                        timestamp,
                        is_from_me
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch_data,
                )
                count = len(batch_data)
        except Exception as e:
            logger.error("Failed to batch index messages: %s", e)

        return count

    def _quantize_embedding(self, embedding: np.ndarray) -> bytes:
        """Quantize float32 embedding to int8 for sqlite-vec storage.

        Args:
            embedding: float32 numpy array

        Returns:
            bytes of int8 array
        """
        # Simple scalar quantization to int8 (-127 to 127)
        # Assumes normalized embedding in [-1, 1]
        # Map [-1, 1] -> [-127, 127]
        quantized = (embedding * 127).astype(np.int8)
        return quantized.tobytes()

    def index_segment(
        self,
        segment: TopicSegment,
        contact_id: int | None = None,
        chat_id: str | None = None,
    ) -> bool:
        """Index a topic segment into vec_chunks.

        Args:
            segment: TopicSegment with computed centroid and metadata.
            contact_id: Contact ID for partition key.
            chat_id: Chat ID for the conversation.

        Returns:
            True if indexed, False if skipped/failed.
        """
        if segment.centroid is None or not segment.messages:
            return False

        # Collect trigger (them) and response (me) text from the segment
        trigger_parts = [m.text for m in segment.messages if not m.is_from_me and m.text]
        response_parts = [m.text for m in segment.messages if m.is_from_me and m.text]

        trigger_text = "\n".join(trigger_parts) if trigger_parts else None
        response_text = "\n".join(response_parts) if response_parts else None

        # Skip segments with no response text (nothing to learn from)
        if not response_text:
            return False

        import json

        try:
            with self.db.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO vec_chunks(
                        embedding,
                        contact_id,
                        chat_id,
                        source_timestamp,
                        quality_score,
                        topic_label,
                        trigger_text,
                        response_text,
                        formatted_text,
                        keywords_json,
                        message_count,
                        source_type,
                        source_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self._quantize_embedding(segment.centroid),
                        contact_id or 0,
                        chat_id,
                        segment.start_time.timestamp(),
                        segment.confidence,
                        segment.topic_label,
                        trigger_text,
                        response_text,
                        segment.text[:500],  # formatted_text preview
                        json.dumps(segment.keywords) if segment.keywords else None,
                        segment.message_count,
                        "chunk",
                        segment.segment_id,
                    ),
                )
            return True
        except Exception as e:
            logger.error("Failed to index segment %s: %s", segment.segment_id, e)
            return False

    def index_segments(
        self,
        segments: list[TopicSegment],
        contact_id: int | None = None,
        chat_id: str | None = None,
    ) -> int:
        """Index multiple topic segments into vec_chunks.

        Args:
            segments: List of TopicSegment objects.
            contact_id: Contact ID for partition key.
            chat_id: Chat ID for the conversation.

        Returns:
            Number of segments successfully indexed.
        """
        count = 0
        for segment in segments:
            if self.index_segment(segment, contact_id, chat_id):
                count += 1
        return count

    def delete_chunks_for_chat(self, chat_id: str) -> int:
        """Delete all chunks for a chat_id. Returns count deleted."""
        try:
            with self.db.connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM vec_chunks WHERE chat_id = ?", (chat_id,)
                )
                return cursor.rowcount
        except Exception as e:
            logger.error("Failed to delete chunks for chat %s: %s", chat_id, e)
            return 0

    def search(
        self, query: str, chat_id: str | None = None, limit: int = 10
    ) -> list[VecSearchResult]:
        """Search for messages semantically.

        Args:
            query: Search query
            chat_id: Optional chat_id to filter by
            limit: Max results

        Returns:
            List of VecSearchResult
        """
        embedding = self._embedder.encode(query, normalize=True)
        query_blob = self._quantize_embedding(embedding)

        with self.db.connection() as conn:
            # Build SQL
            sql = """
                SELECT
                    rowid,
                    distance,
                    chat_id,
                    text_preview,
                    sender,
                    timestamp,
                    is_from_me
                FROM vec_messages
                WHERE embedding MATCH ?
                AND k = ?
            """
            params: list[Any] = [query_blob, limit]

            if chat_id:
                sql += " AND chat_id = ?"
                params.append(chat_id)

            rows = conn.execute(sql, params).fetchall()

        results = []
        for row in rows:
            # Convert L2 distance to similarity score
            # 1 / (1 + distance) is a common heuristic
            dist = row["distance"]
            score = 1.0 / (1.0 + dist)

            results.append(
                VecSearchResult(
                    rowid=row["rowid"],
                    distance=dist,
                    score=score,
                    chat_id=row["chat_id"],
                    text=row["text_preview"],
                    sender=row["sender"],
                    timestamp=row["timestamp"],
                    is_from_me=bool(row["is_from_me"]),
                )
            )

        return results

    def search_with_pairs(
        self, query: str, limit: int = 5, response_type: str | None = None
    ) -> list[VecSearchResult]:
        """Search for conversation pairs (chunks) in vec_chunks.

        Args:
            query: Input trigger text
            limit: Max results
            response_type: Optional filter for response_da_type

        Returns:
            List of results with trigger/response details
        """
        embedding = self._embedder.encode(query, normalize=True)
        query_blob = self._quantize_embedding(embedding)

        with self.db.connection() as conn:
            sql = """
                SELECT
                    rowid,
                    distance,
                    chat_id,
                    trigger_text,
                    response_text,
                    response_da_type,
                    topic_label,
                    quality_score
                FROM vec_chunks
                WHERE embedding MATCH ?
                AND k = ?
            """
            params: list[Any] = [query_blob, limit]

            if response_type:
                sql += " AND response_da_type = ?"
                params.append(response_type)

            rows = conn.execute(sql, params).fetchall()

        results = []
        for row in rows:
            dist = row["distance"]
            score = 1.0 / (1.0 + dist)

            results.append(
                VecSearchResult(
                    rowid=row["rowid"],
                    distance=dist,
                    score=score,
                    chat_id=row["chat_id"],
                    trigger_text=row["trigger_text"],
                    response_text=row["response_text"],
                    response_type=row["response_da_type"],
                    topic=row["topic_label"],
                    quality_score=row["quality_score"],
                )
            )

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get index statistics.

        Returns:
            Dict with index stats
        """
        try:
            with self.db.connection() as conn:
                # Count total messages
                total = conn.execute("SELECT count(*) FROM vec_messages").fetchone()[0]

                # Count unique chats
                chats = conn.execute("SELECT count(DISTINCT chat_id) FROM vec_messages").fetchone()[
                    0
                ]

                return {
                    "total_embeddings": total,
                    "unique_chats": chats,
                    "db_path": str(self.db.db_path),
                }
        except Exception as e:
            logger.error("Failed to get stats: %s", e)
            return {
                "total_embeddings": 0,
                "unique_chats": 0,
                "db_path": str(self.db.db_path),
                "error": str(e),
            }


# Singleton
_vec_searcher: VecSearcher | None = None
_lock = threading.Lock()


def get_vec_searcher(db: JarvisDB | None = None) -> VecSearcher:
    """Get singleton VecSearcher."""
    global _vec_searcher
    if _vec_searcher is None:
        with _lock:
            if _vec_searcher is None:
                if db is None:
                    db = get_db()
                _vec_searcher = VecSearcher(db)
    return _vec_searcher
