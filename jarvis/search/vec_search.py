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

    # Search for similar conversation chunks (RAG)
    chunks = searcher.search_with_chunks("Want to grab lunch?", limit=5)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import orjson

from jarvis.db import JarvisDB, get_db
from jarvis.embedding_adapter import get_embedder

if TYPE_CHECKING:
    from contracts.imessage import Message
    from jarvis.topics.topic_segmenter import TopicSegment

logger = logging.getLogger(__name__)


def _validate_placeholders(placeholders: str) -> None:
    """Validate SQL placeholder string contains only safe characters.

    SECURITY: Ensures placeholder strings like "?,?,?" don't contain SQL injection.
    Raises ValueError if placeholders contain anything other than '?' and ','.

    Args:
        placeholders: The placeholder string to validate (e.g., "?,?,?")

    Raises:
        ValueError: If placeholders contain invalid characters
    """
    if not placeholders:
        return
    allowed_chars = set("?,")
    if not set(placeholders).issubset(allowed_chars):
        raise ValueError(f"Invalid characters in SQL placeholders: {placeholders}")
    # SQLite has a hard limit of 999 bound parameters per query (SQLITE_MAX_VARIABLE_NUMBER).
    # Enforce a safe ceiling to prevent runtime errors deep in the query engine.
    param_count = placeholders.count("?")
    if param_count > 900:
        raise ValueError(
            f"Too many SQL parameters ({param_count}). "
            f"SQLite supports at most 999; limit to 900 for safety."
        )


@dataclass
class VecSearchResult:
    """Result from vector search (simplified schema)."""

    rowid: int
    distance: float
    score: float  # Converted from distance (0..1)

    # Message fields (if searching messages)
    chat_id: str | None = None
    text: str | None = None
    sender: str | None = None
    timestamp: float | None = None
    is_from_me: bool | None = None

    # Chunk fields (if searching chunks)
    context_text: str | None = None
    reply_text: str | None = None
    topic: str | None = None


class VecSearcher:
    """High-performance vector search using sqlite-vec."""

    # Scale factor used during int8 quantization (embedding * SCALE -> int8)
    _INT8_SCALE = 127.0

    def __init__(self, db: JarvisDB) -> None:
        self.db = db
        self._embedder = get_embedder()
        self._lock = threading.Lock()

    @staticmethod
    def _distance_to_similarity(
        distance: float, query_vec: np.ndarray | None = None, doc_vec: np.ndarray | None = None
    ) -> float:
        """Convert distance to approximate cosine similarity.

        If query_vec and doc_vec (dequantized float32) are provided, computes
        exact cosine similarity via dot product. Otherwise uses the
        L2-in-int8-space approximation.

        For normalized embeddings quantized to int8 via (emb * 127):
            L2_int8 = 127 * L2_float
            L2_float^2 = 2 * (1 - cos_sim)
            cos_sim = 1 - (L2_int8 / 127)^2 / 2
        """
        if query_vec is not None and doc_vec is not None:
            # Exact cosine similarity for normalized vectors is just dot product
            return float(np.dot(query_vec, doc_vec))

        cos_sim = 1.0 - (distance / VecSearcher._INT8_SCALE) ** 2 / 2.0
        return max(0.0, min(1.0, cos_sim))

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

        For batch indexing multiple messages, use index_messages() instead
        to benefit from batch embedding computation and bulk inserts.

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
                    INSERT OR IGNORE INTO vec_messages(
                        rowid,
                        embedding,
                        chat_id,
                        text_preview,
                        sender,
                        timestamp,
                        is_from_me
                    ) VALUES (?, vec_int8(?), ?, ?, ?, ?, ?)
                    """,
                    (
                        message.id,
                        self._quantize_embedding(embedding),
                        message.chat_id,
                        message.text[:200],  # Preview
                        message.sender,
                        int(message.date.timestamp()),
                        1 if message.is_from_me else 0,
                    ),
                )
            return True
        except Exception as e:
            logger.error("Failed to index message %s: %s", message.id, e)
            return False

    def index_messages(self, messages: list[Message], dtype: Any = np.float32) -> int:
        """Index multiple messages efficiently.

        Args:
            messages: List of messages to index
            dtype: Output dtype for embeddings (use np.float16 for GPU memory savings)

        Returns:
            Number of messages successfully indexed
        """
        valid_messages = [m for m in messages if m.text and len(m.text.strip()) >= 3]
        if not valid_messages:
            return 0

        count = 0
        try:
            # Filter out already indexed messages to avoid UNIQUE constraint violations
            # on virtual tables that might not fully support OR IGNORE
            all_ids = [m.id for m in valid_messages]
            existing_ids = set()
            with self.db.connection() as conn:
                for i in range(0, len(all_ids), 900):
                    chunk = all_ids[i : i + 900]
                    placeholders = ",".join(["?"] * len(chunk))
                    rows = conn.execute(
                        f"SELECT rowid FROM vec_messages WHERE rowid IN ({placeholders})", chunk
                    ).fetchall()
                    for r in rows:
                        existing_ids.add(r[0])

            to_index = [m for m in valid_messages if m.id not in existing_ids]
            if not to_index:
                return 0

            # Batch compute embeddings
            texts = [m.text for m in to_index]
            # Use cached embedder which handles batching internally
            embeddings = self._embedder.encode(texts, normalize=True, dtype=dtype)

            # Vectorized quantization for performance (avoids loop overhead)
            quantized_embeddings = (embeddings * 127).astype(np.int8)

            with self.db.connection() as conn:
                # Prepare batch data
                batch_data = []
                for msg, q_emb in zip(to_index, quantized_embeddings):
                    batch_data.append(
                        (
                            msg.id,
                            q_emb.tobytes(),
                            msg.chat_id,
                            msg.text[:200],
                            msg.sender,
                            int(msg.date.timestamp()),
                            1 if msg.is_from_me else 0,
                        )
                    )

                conn.executemany(
                    """
                    INSERT OR IGNORE INTO vec_messages(
                        rowid,
                        embedding,
                        chat_id,
                        text_preview,
                        sender,
                        timestamp,
                        is_from_me
                    ) VALUES (?, vec_int8(?), ?, ?, ?, ?, ?)
                    """,
                    batch_data,
                )
                count = len(batch_data)
        except Exception as e:
            logger.error("Failed to batch index messages: %s", e)

        return count

    @staticmethod
    def _quantize_embedding(embedding: np.ndarray) -> bytes:
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

    @staticmethod
    def _binarize_embedding(embedding: np.ndarray) -> bytes:
        """Binarize float embedding to bit-packed bytes for hamming search.

        Each dimension becomes 1 if positive, 0 if negative/zero.
        384 dims -> 48 bytes.

        Args:
            embedding: float32 numpy array (any length divisible by 8)

        Returns:
            bytes of bit-packed array
        """
        return np.packbits((embedding > 0).astype(np.uint8)).tobytes()

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

        # Collect context/reply text
        context_parts = [m.text for m in segment.messages if not m.is_from_me and m.text]
        reply_parts = [m.text for m in segment.messages if m.is_from_me and m.text]

        context_text = "\n".join(context_parts) if context_parts else None
        reply_text = "\n".join(reply_parts) if reply_parts else ""

        # Skip segments with no context text (need at least something to search)
        if not context_text:
            return False

        try:
            int8_blob = self._quantize_embedding(segment.centroid)
            binary_blob = self._binarize_embedding(segment.centroid)

            with self.db.connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO vec_chunks(
                        embedding, contact_id, chat_id, source_timestamp,
                        context_text, reply_text, topic_label, message_count
                    ) VALUES (vec_int8(?), ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int8_blob,
                        contact_id or 0,
                        chat_id or "",
                        segment.start_time.timestamp(),
                        context_text[:1000],
                        reply_text[:1000],
                        segment.topic_label,
                        segment.message_count,
                    ),
                )
                chunk_rowid = cursor.lastrowid

                # Dual-insert into vec_binary for fast hamming pre-filter
                try:
                    conn.execute(
                        """
                        INSERT INTO vec_binary(embedding, chunk_rowid, embedding_int8)
                        VALUES (vec_bit(?), ?, ?)
                        """,
                        (binary_blob, chunk_rowid, int8_blob),
                    )
                except Exception as e:
                    # Non-fatal: vec_binary may not exist on older schemas
                    logger.debug("vec_binary insert skipped: %s", e)

            return True
        except Exception as e:
            logger.error("Failed to index segment %s: %s", segment.segment_id, e)
            return False

    def index_segments(
        self,
        segments: list[TopicSegment],
        contact_id: int | None = None,
        chat_id: str | None = None,
    ) -> list[int]:
        """Index multiple topic segments into vec_chunks.

        Args:
            segments: List of TopicSegment objects.
            contact_id: Contact ID for partition key.
            chat_id: Chat ID for the conversation.

        Returns:
            List of vec_chunks rowids for successfully indexed segments.
        """
        if not segments:
            return []

        # Prepare batch data
        vec_chunks_batch = []
        skipped_no_centroid = 0
        has_reply_count = 0
        no_reply_count = 0

        for idx, segment in enumerate(segments):
            if segment.centroid is None or not segment.messages:
                skipped_no_centroid += 1
                continue

            # Collect context/reply text
            context_parts = [m.text for m in segment.messages if not m.is_from_me and m.text]
            reply_parts = [m.text for m in segment.messages if m.is_from_me and m.text]

            context_text = "\n".join(context_parts) if context_parts else None
            reply_text = "\n".join(reply_parts) if reply_parts else ""

            # Track whether segment has user reply (for logging)
            if reply_text:
                has_reply_count += 1
            else:
                no_reply_count += 1

            int8_blob = self._quantize_embedding(segment.centroid)
            binary_blob = self._binarize_embedding(segment.centroid)

            # Handle None start_time gracefully
            start_ts = segment.start_time.timestamp() if segment.start_time else 0.0

            # Use segment index as unique identifier to avoid timestamp collisions
            vec_chunks_batch.append(
                (
                    int8_blob,
                    contact_id or 0,
                    chat_id or "",
                    start_ts,
                    (context_text or "")[:1000],
                    reply_text[:1000],
                    segment.topic_label,
                    segment.message_count,
                    binary_blob,  # Store for vec_binary insert
                    idx,  # segment index for unique lookup key
                )
            )

        if skipped_no_centroid:
            logger.debug("Skipped %d segments with no centroid/messages", skipped_no_centroid)
        if no_reply_count > 0:
            logger.info(
                "Indexed %d segments with reply, %d without reply (context-only)",
                has_reply_count,
                no_reply_count,
            )

        if not vec_chunks_batch:
            return []

        try:
            with self.db.connection() as conn:
                chunk_rowids: list[int] = []

                try:
                    insert_sql = """
                        INSERT INTO vec_chunks(
                            embedding, contact_id, chat_id, source_timestamp,
                            context_text, reply_text, topic_label, message_count
                        ) VALUES (vec_int8(?), ?, ?, ?, ?, ?, ?, ?)
                    """

                    # Bulk insert all chunks at once
                    conn.executemany(insert_sql, [row[:8] for row in vec_chunks_batch])

                    # Fetch rowids using unique composite key to avoid collisions
                    # Use (contact_id, chat_id, source_timestamp, topic_label, message_count)
                    rowid_map = {}
                    chunk_size = 100
                    for i in range(0, len(vec_chunks_batch), chunk_size):
                        batch_chunk = vec_chunks_batch[i : i + chunk_size]
                        rows_payload: list[dict[str, Any]] = []
                        for row in batch_chunk:
                            rows_payload.append(
                                {
                                    "contact_id": row[1],
                                    "chat_id": row[2],
                                    "source_timestamp": row[3],
                                    "topic_label": row[6],
                                    "message_count": row[7],
                                }
                            )
                        cursor = conn.execute(
                            """
                            WITH input_rows AS (
                                SELECT
                                    json_extract(value, '$.contact_id') AS contact_id,
                                    json_extract(value, '$.chat_id') AS chat_id,
                                    json_extract(value, '$.source_timestamp') AS source_timestamp,
                                    json_extract(value, '$.topic_label') AS topic_label,
                                    json_extract(value, '$.message_count') AS message_count
                                FROM json_each(?)
                            )
                            SELECT vc.rowid, vc.contact_id, vc.chat_id,
                                   vc.source_timestamp, vc.topic_label, vc.message_count
                            FROM vec_chunks vc
                            JOIN input_rows ir
                              ON vc.contact_id = ir.contact_id
                             AND vc.chat_id = ir.chat_id
                             AND vc.source_timestamp = ir.source_timestamp
                             AND vc.topic_label = ir.topic_label
                             AND vc.message_count = ir.message_count
                            ORDER BY vc.rowid DESC
                            """,
                            (orjson.dumps(rows_payload).decode("utf-8"),),
                        )

                        for row in cursor:
                            key = (
                                row["contact_id"],
                                row["chat_id"],
                                row["source_timestamp"],
                                row["topic_label"],
                                row["message_count"],
                            )
                            # Only store first occurrence (highest rowid due to DESC order)
                            if key not in rowid_map:
                                rowid_map[key] = row["rowid"]

                    # Get rowids in the same order as input
                    missed = 0
                    for row in vec_chunks_batch:
                        key = (row[1], row[2], row[3], row[6], row[7])
                        if key in rowid_map:
                            chunk_rowids.append(rowid_map[key])
                        else:
                            missed += 1

                    if missed:
                        logger.warning("Failed to retrieve %d rowids after insert", missed)

                    # Commit once after all operations
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise

                # Batch insert into vec_binary (if it exists)
                if chunk_rowids and len(chunk_rowids) == len(vec_chunks_batch):
                    try:
                        binary_batch = [
                            (row[8], chunk_rowid, row[0])  # binary_blob, chunk_rowid, int8_blob
                            for row, chunk_rowid in zip(vec_chunks_batch, chunk_rowids)
                        ]
                        conn.executemany(
                            """
                            INSERT INTO vec_binary(embedding, chunk_rowid, embedding_int8)
                            VALUES (vec_bit(?), ?, ?)
                            """,
                            binary_batch,
                        )
                    except Exception as e:
                        logger.debug("vec_binary batch insert skipped: %s", e)

                return chunk_rowids

        except Exception as e:
            logger.error("Failed to batch index segments: %s", e)
            return []

    def index_feedback_pair(
        self,
        context_text: str,
        reply_text: str,
        chat_id: str,
    ) -> int | None:
        """Index a high-quality (user-edited or confirmed) feedback pair.

        This allows the system to learn from manual edits by making them
        available for future RAG searches as 'gold' examples.

        Returns:
            The rowid of the new chunk, or None if failed.
        """
        if not context_text or not reply_text:
            return None

        try:
            # Combine context and reply for the centroid embedding
            # This follows the same logic as segment centroids
            full_text = f"{context_text}\n{reply_text}"
            embedding = self._embedder.encode(full_text, normalize=True)
            int8_blob = self._quantize_embedding(embedding)
            binary_blob = self._binarize_embedding(embedding)

            with self.db.connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO vec_chunks(
                        embedding, chat_id, source_timestamp,
                        context_text, reply_text, topic_label, message_count
                    ) VALUES (vec_int8(?), ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int8_blob,
                        chat_id,
                        datetime.now().timestamp(),
                        context_text[:1000],
                        reply_text[:1000],
                        "gold_feedback",  # Special label to prioritize or identify
                        1,
                    ),
                )
                chunk_rowid = cursor.lastrowid

                # Insert into vec_binary for fast global search
                try:
                    conn.execute(
                        """
                        INSERT INTO vec_binary(embedding, chunk_rowid, embedding_int8)
                        VALUES (vec_bit(?), ?, ?)
                        """,
                        (binary_blob, chunk_rowid, int8_blob),
                    )
                except Exception as e:
                    logger.debug("vec_binary insert failed: %s", e)

                return chunk_rowid
        except Exception as e:
            logger.error("Failed to index feedback pair: %s", e)
            return None

    def delete_chunks_for_chat(self, chat_id: str) -> int:
        """Delete all chunks for a chat_id. Returns count deleted."""
        try:
            with self.db.connection() as conn:
                # Fetch chunk rowids first so we can clean up vec_binary
                rows = conn.execute(
                    "SELECT rowid FROM vec_chunks WHERE chat_id = ?", (chat_id,)
                ).fetchall()
                rowids = [r["rowid"] for r in rows]

                if rowids:
                    # Delete from vec_binary by chunk_rowid
                    # Use chunking to avoid SQLite parameter limits
                    try:
                        chunk_size = 500
                        for i in range(0, len(rowids), chunk_size):
                            batch_rowids = rowids[i : i + chunk_size]
                            placeholders = ",".join("?" * len(batch_rowids))
                            # SECURITY: Validate placeholders before SQL interpolation
                            _validate_placeholders(placeholders)
                            conn.execute(
                                f"DELETE FROM vec_binary WHERE chunk_rowid IN ({placeholders})",  # nosec B608
                                batch_rowids,
                            )
                    except Exception as e:
                        logger.debug("vec_binary delete skipped: %s", e)

                cursor = conn.execute("DELETE FROM vec_chunks WHERE chat_id = ?", (chat_id,))
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
                WHERE embedding MATCH vec_int8(?)
                AND k = ?
            """
            params: list[Any] = [query_blob, limit]

            if chat_id:
                sql += " AND chat_id = ?"
                params.append(chat_id)

            rows = conn.execute(sql, params).fetchall()

        results = []
        for row in rows:
            dist = row["distance"]
            score = self._distance_to_similarity(dist)

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

    def search_with_chunks(
        self,
        query: str,
        limit: int = 5,
        response_type: str | None = None,
        contact_id: int | None = None,
        embedder: Any | None = None,
    ) -> list[VecSearchResult]:
        """Search for conversation chunks in vec_chunks.

        Args:
            query: Input trigger text
            limit: Max results
            response_type: Optional filter (ignored in simplified schema)
            contact_id: Optional partition key to search only one contact's chunks
            embedder: Optional embedder override (e.g. CachedEmbedder)

        Returns:
            List of results with context/reply details
        """
        enc = embedder or self._embedder
        embedding = enc.encode(query, normalize=True)
        query_blob = self._quantize_embedding(embedding)

        with self.db.connection() as conn:
            sql = """
                SELECT
                    rowid,
                    distance,
                    chat_id,
                    context_text,
                    reply_text,
                    topic_label
                FROM vec_chunks
                WHERE embedding MATCH vec_int8(?)
                AND k = ?
            """
            params: list[Any] = [query_blob, limit]

            if contact_id is not None:
                sql += " AND contact_id = ?"
                params.append(contact_id)

            rows = conn.execute(sql, params).fetchall()

        results = []
        for row in rows:
            dist = row["distance"]
            score = self._distance_to_similarity(dist)

            results.append(
                VecSearchResult(
                    rowid=row["rowid"],
                    distance=dist,
                    score=score,
                    chat_id=row["chat_id"],
                    context_text=row["context_text"],
                    reply_text=row["reply_text"],
                    topic=row["topic_label"],
                )
            )

        return results

    def search_with_chunks_global(
        self,
        query: str,
        limit: int = 5,
        embedder: Any | None = None,
        rerank: bool = True,
    ) -> list[VecSearchResult]:
        """Two-phase global search: fast hamming pre-filter then int8 re-rank.

        Optional Phase 4: Rerank top candidates with a cross-encoder for max precision.

        Phase 1: Hamming search on vec_binary for limit*10 candidates
        Phase 2: Re-rank candidates by L2 distance on stored int8 embeddings
        Phase 3: Fetch metadata from vec_chunks for top results
        Phase 4: (Optional) Cross-encoder rerank top results

        Falls back to standard search_with_chunks() if vec_binary is empty/missing.

        Args:
            query: Input trigger text
            limit: Max results
            embedder: Optional embedder override
            rerank: Whether to rerank top candidates with cross-encoder

        Returns:
            List of results with trigger/response details, sorted by relevance
        """
        enc = embedder or self._embedder
        embedding = enc.encode(query, normalize=True)
        binary_blob = self._binarize_embedding(embedding)
        int8_query = (embedding * self._INT8_SCALE).astype(np.int8)

        # Increase candidate pool if reranking is enabled
        candidates_k = limit * (20 if rerank else 10)

        try:
            with self.db.connection() as conn:
                # Phase 1: Fast hamming pre-filter
                rows = conn.execute(
                    """
                    SELECT rowid, chunk_rowid, embedding_int8
                    FROM vec_binary
                    WHERE embedding MATCH vec_bit(?)
                    AND k = ?
                    """,
                    (binary_blob, candidates_k),
                ).fetchall()

                if not rows:
                    # Fall back to standard search
                    return self.search_with_chunks(query, limit=limit, embedder=embedder)

                # Phase 2: Re-rank by L2 distance on int8 embeddings
                scored: list[tuple[int, float]] = []
                for row in rows:
                    chunk_rowid = row["chunk_rowid"]
                    stored_int8 = np.frombuffer(row["embedding_int8"], dtype=np.int8)
                    # L2 distance in int8 space
                    diff = int8_query.astype(np.int16) - stored_int8.astype(np.int16)
                    dist = float(np.sqrt(np.sum(diff * diff)))
                    scored.append((chunk_rowid, dist))

                scored.sort(key=lambda x: x[1])
                # Take slightly more for the cross-encoder to refine
                top_candidates_count = limit * 3 if rerank else limit
                top_indices = scored[:top_candidates_count]

                # Phase 3: Fetch metadata from vec_chunks
                chunk_rowids = [r[0] for r in top_indices]
                dist_by_rowid = {r[0]: r[1] for r in top_indices}

                meta_rows = conn.execute(
                    """
                    SELECT rowid, chat_id, context_text, reply_text, topic_label
                    FROM vec_chunks
                    WHERE rowid IN (SELECT value FROM json_each(?))
                    """,
                    (orjson.dumps(chunk_rowids).decode("utf-8"),),
                ).fetchall()

                results = []
                for mrow in meta_rows:
                    rid = mrow["rowid"]
                    dist = dist_by_rowid.get(rid, 0.0)
                    score = self._distance_to_similarity(dist)
                    results.append(
                        VecSearchResult(
                            rowid=rid,
                            distance=dist,
                            score=score,
                            chat_id=mrow["chat_id"],
                            context_text=mrow["context_text"],
                            reply_text=mrow["reply_text"],
                            topic=mrow["topic_label"],
                        )
                    )

                # Phase 4: Cross-encoder rerank
                if rerank and len(results) > 1:
                    try:
                        from models.cross_encoder import get_reranker

                        reranker = get_reranker()
                        # Convert to dict format expected by reranker
                        candidates_dicts = [
                            {
                                "index": i,
                                "context_text": r.context_text or "",
                            }
                            for i, r in enumerate(results)
                        ]
                        # Rerank based on context_text
                        reranked = reranker.rerank(
                            query=query,
                            candidates=candidates_dicts,
                            text_key="context_text",
                            top_k=limit,
                        )
                        # Map back to VecSearchResult objects
                        final_results = []
                        for item in reranked:
                            res = results[item["index"]]
                            res.score = item["rerank_score"]  # Update score with rerank score
                            final_results.append(res)
                        return final_results
                    except Exception as e:
                        logger.warning("Cross-encoder rerank failed: %s", e)

                # Standard sort by distance ascending if no rerank or rerank failed
                results.sort(key=lambda r: r.distance)
                return results[:limit]

        except Exception as e:
            logger.warning("Two-phase search failed, falling back: %s", e)
            return self.search_with_chunks(query, limit=limit, embedder=embedder)

    def backfill_vec_binary(self) -> int:
        """Populate vec_binary from existing vec_chunks embeddings.

        Reads int8 embeddings from vec_chunks, binarizes them, and inserts
        into vec_binary for fast hamming pre-filter on global searches.

        Returns:
            Number of rows inserted.
        """
        with self.db.connection() as conn:
            # Check if vec_binary exists
            res = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_binary'"
            ).fetchone()
            if res is None:
                logger.warning("vec_binary table does not exist, run init_schema first")
                return 0

            # Check existing count to avoid duplicates
            existing = conn.execute("SELECT COUNT(*) FROM vec_binary").fetchone()[0]
            if existing > 0:
                logger.info("vec_binary already has %d rows, skipping backfill", existing)
                return 0

            # Read chunk embeddings in batches, vectorize binarization per batch
            cursor = conn.execute("SELECT rowid, embedding FROM vec_chunks")

            count = 0
            batch_size = 1000
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break

                rowids = [r["rowid"] for r in rows]
                int8_blobs = [r["embedding"] for r in rows]

                # Vectorize: stack into matrix, packbits across all rows at once
                int8_matrix = np.vstack([np.frombuffer(b, dtype=np.int8) for b in int8_blobs])
                binary_matrix = np.packbits((int8_matrix > 0).astype(np.uint8), axis=1)

                batch = [
                    (binary_matrix[i].tobytes(), rowids[i], int8_blobs[i]) for i in range(len(rows))
                ]
                conn.executemany(
                    "INSERT INTO vec_binary(embedding, chunk_rowid, embedding_int8) "
                    "VALUES (vec_bit(?), ?, ?)",
                    batch,
                )
                count += len(batch)
            logger.info("Backfilled %d rows into vec_binary", count)
            return count

    def get_embeddings_by_ids(self, message_ids: list[int]) -> dict[int, np.ndarray]:
        """Retrieve cached embeddings from vec_messages by message ID.

        Returns dequantized float32 embeddings (int8 / 127.0).

        Args:
            message_ids: List of message ROWIDs to look up.

        Returns:
            Dict mapping message_id → (384,) float32 embedding array.
        """
        if not message_ids:
            return {}

        result: dict[int, np.ndarray] = {}
        with self.db.connection() as conn:
            # Chunk to stay within SQLite parameter limits
            for chunk_start in range(0, len(message_ids), 900):
                chunk = message_ids[chunk_start : chunk_start + 900]
                placeholders = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT rowid, embedding FROM vec_messages WHERE rowid IN ({placeholders})",  # nosec B608
                    chunk,
                ).fetchall()
                for row in rows:
                    int8_arr = np.frombuffer(row["embedding"], dtype=np.int8)
                    result[row["rowid"]] = int8_arr.astype(np.float32) / self._INT8_SCALE
        return result

    def search_with_full_segments(
        self,
        query: str,
        limit: int = 5,
        contact_id: int | None = None,
        embedder: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Search vec_chunks and return full segment context with all messages.

        Joins vec_chunks hits to conversation_segments and segment_messages
        to retrieve full topic blocks instead of just trigger/response text.
        Falls back to search_with_chunks() if segment tables don't exist.

        Args:
            query: Search query text.
            limit: Max results.
            contact_id: Optional partition key filter.
            embedder: Optional embedder override.

        Returns:
            List of dicts with segment metadata and message_rowids.
        """
        # First, do the standard search
        if contact_id is not None:
            hits = self.search_with_chunks(
                query=query, limit=limit, contact_id=contact_id, embedder=embedder
            )
        else:
            hits = self.search_with_chunks_global(query=query, limit=limit, embedder=embedder)

        if not hits:
            return []

        # Try to enrich with segment data
        try:
            chunk_rowids = [h.rowid for h in hits]
            with self.db.connection() as conn:
                # Check if conversation_segments table exists
                table_check = conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name='conversation_segments'"
                ).fetchone()
                if table_check is None:
                    # Fall back to basic results
                    return self._hits_to_dicts(hits)

                # Join vec_chunks rowids → conversation_segments
                seg_rows = conn.execute(
                    """
                    SELECT cs.id, cs.segment_id, cs.chat_id, cs.contact_id,
                           cs.start_time, cs.end_time, cs.preview,
                           cs.message_count, cs.vec_chunk_rowid
                    FROM conversation_segments cs
                    WHERE cs.vec_chunk_rowid IN (SELECT value FROM json_each(?))
                    """,
                    (orjson.dumps(chunk_rowids).decode("utf-8"),),
                ).fetchall()

                if not seg_rows:
                    return self._hits_to_dicts(hits)

                # Batch fetch message memberships
                seg_ids = [r["id"] for r in seg_rows]
                msg_rows = conn.execute(
                    """
                    SELECT segment_id, message_rowid, position, is_from_me
                    FROM segment_messages
                    WHERE segment_id IN (SELECT value FROM json_each(?))
                    ORDER BY segment_id, position
                    """,
                    (orjson.dumps(seg_ids).decode("utf-8"),),
                ).fetchall()

            # Group messages by segment
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

            # Build enriched results, keyed by vec_chunk_rowid
            seg_by_chunk: dict[int, dict[str, Any]] = {}
            for sr in seg_rows:
                seg_by_chunk[sr["vec_chunk_rowid"]] = {
                    "segment_id": sr["segment_id"],
                    "chat_id": sr["chat_id"],
                    "contact_id": sr["contact_id"],
                    "start_time": sr["start_time"],
                    "end_time": sr["end_time"],
                    "message_count": sr["message_count"],
                    "messages": msgs_by_seg.get(sr["id"], []),
                }

            # Merge search hits with segment data
            results = []
            for hit in hits:
                entry: dict[str, Any] = {
                    "rowid": hit.rowid,
                    "score": hit.score,
                    "context_text": hit.context_text,
                    "reply_text": hit.reply_text,
                    "topic": hit.topic,
                }
                seg_data = seg_by_chunk.get(hit.rowid)
                if seg_data:
                    entry["segment"] = seg_data
                results.append(entry)

            return results

        except Exception as e:
            logger.debug("Full segment search failed, falling back: %s", e)
            return self._hits_to_dicts(hits)

    @staticmethod
    def _hits_to_dicts(hits: list[VecSearchResult]) -> list[dict[str, Any]]:
        """Convert VecSearchResult list to basic dicts."""
        return [
            {
                "rowid": h.rowid,
                "score": h.score,
                "context_text": h.context_text,
            }
            for h in hits
        ]

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


from jarvis.utils.singleton import thread_safe_singleton  # noqa: E402


@thread_safe_singleton
def get_vec_searcher(db: JarvisDB | None = None) -> VecSearcher:
    """Get singleton VecSearcher."""
    if db is None:
        db = get_db()
    return VecSearcher(db)
