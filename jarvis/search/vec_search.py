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
    response_da_conf: float | None = None
    topic: str | None = None
    quality_score: float | None = None


class VecSearcher:
    """High-performance vector search using sqlite-vec."""

    # Scale factor used during int8 quantization (embedding * SCALE -> int8)
    _INT8_SCALE = 127.0

    def __init__(self, db: JarvisDB) -> None:
        self.db = db
        self._embedder = get_embedder()
        self._lock = threading.Lock()

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        """Convert int8-quantized L2 distance to approximate cosine similarity.

        For normalized embeddings quantized to int8 via (emb * 127):
            L2_int8 = 127 * L2_float
            L2_float^2 = 2 * (1 - cos_sim)
            cos_sim = 1 - (L2_int8 / 127)^2 / 2
        """
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
                    ) VALUES (?, vec_int8(?), ?, ?, ?, ?, ?)
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
                    ) VALUES (?, vec_int8(?), ?, ?, ?, ?, ?)
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
            int8_blob = self._quantize_embedding(segment.centroid)
            binary_blob = self._binarize_embedding(segment.centroid)

            with self.db.connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO vec_chunks(
                        embedding,
                        contact_id,
                        chat_id,
                        response_da_type,
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
                    ) VALUES (vec_int8(?), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int8_blob,
                        contact_id or 0,
                        chat_id or "",
                        "",  # response_da_type: not classified during segment ingestion
                        segment.start_time.timestamp(),
                        segment.confidence or 0.0,
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
    ) -> int:
        """Index multiple topic segments into vec_chunks.

        Args:
            segments: List of TopicSegment objects.
            contact_id: Contact ID for partition key.
            chat_id: Chat ID for the conversation.

        Returns:
            Number of segments successfully indexed.
        """
        if not segments:
            return 0

        import json

        # Prepare batch data
        vec_chunks_batch = []

        for segment in segments:
            if segment.centroid is None or not segment.messages:
                continue

            # Collect trigger/response text
            trigger_parts = [m.text for m in segment.messages if not m.is_from_me and m.text]
            response_parts = [m.text for m in segment.messages if m.is_from_me and m.text]

            trigger_text = "\n".join(trigger_parts) if trigger_parts else None
            response_text = "\n".join(response_parts) if response_parts else None

            # Skip segments with no response text
            if not response_text:
                continue

            int8_blob = self._quantize_embedding(segment.centroid)
            binary_blob = self._binarize_embedding(segment.centroid)

            vec_chunks_batch.append(
                (
                    int8_blob,
                    contact_id or 0,
                    chat_id or "",
                    "",  # response_da_type
                    segment.start_time.timestamp(),
                    segment.confidence or 0.0,
                    segment.topic_label,
                    trigger_text,
                    response_text,
                    segment.text[:500],  # formatted_text preview
                    json.dumps(segment.keywords) if segment.keywords else None,
                    segment.message_count,
                    "chunk",
                    segment.segment_id,
                    binary_blob,  # Store for vec_binary insert
                )
            )

        if not vec_chunks_batch:
            return 0

        try:
            with self.db.connection() as conn:
                # Insert chunks individually to get reliable rowids.
                # executemany + lastrowid is unreliable for virtual tables (sqlite-vec).
                insert_sql = """
                    INSERT INTO vec_chunks(
                        embedding, contact_id, chat_id, response_da_type,
                        source_timestamp, quality_score, topic_label,
                        trigger_text, response_text, formatted_text,
                        keywords_json, message_count, source_type, source_id
                    ) VALUES (vec_int8(?), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                chunk_rowids = []
                for row in vec_chunks_batch:
                    cursor = conn.execute(insert_sql, row[:14])
                    chunk_rowids.append(cursor.lastrowid)

                # Batch insert into vec_binary
                try:
                    binary_batch = [
                        (row[14], chunk_rowid, row[0])
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

                return len(vec_chunks_batch)

        except Exception as e:
            logger.error("Failed to batch index segments: %s", e)
            return 0

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
                    try:
                        placeholders = ",".join("?" * len(rowids))
                        # SECURITY: Validate placeholders only contain "?" and "," before SQL interpolation
                        _validate_placeholders(placeholders)
                        conn.execute(
                            f"DELETE FROM vec_binary WHERE chunk_rowid IN ({placeholders})",
                            rowids,
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

    def search_with_pairs(
        self,
        query: str,
        limit: int = 5,
        response_type: str | None = None,
        contact_id: int | None = None,
        embedder: Any | None = None,
    ) -> list[VecSearchResult]:
        """Search for conversation pairs (chunks) in vec_chunks.

        Args:
            query: Input trigger text
            limit: Max results
            response_type: Optional filter for response_da_type
            contact_id: Optional partition key to search only one contact's chunks
            embedder: Optional embedder override (e.g. CachedEmbedder)

        Returns:
            List of results with trigger/response details
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
                    trigger_text,
                    response_text,
                    response_da_type,
                    response_da_conf,
                    topic_label,
                    quality_score
                FROM vec_chunks
                WHERE embedding MATCH vec_int8(?)
                AND k = ?
            """
            params: list[Any] = [query_blob, limit]

            if contact_id is not None:
                sql += " AND contact_id = ?"
                params.append(contact_id)

            if response_type:
                sql += " AND response_da_type = ?"
                params.append(response_type)

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
                    trigger_text=row["trigger_text"],
                    response_text=row["response_text"],
                    response_type=row["response_da_type"],
                    response_da_conf=row["response_da_conf"],
                    topic=row["topic_label"],
                    quality_score=row["quality_score"],
                )
            )

        return results

    def search_with_pairs_global(
        self,
        query: str,
        limit: int = 5,
        embedder: Any | None = None,
    ) -> list[VecSearchResult]:
        """Two-phase global search: fast hamming pre-filter then int8 re-rank.

        Phase 1: Hamming search on vec_binary for limit*10 candidates
        Phase 2: Re-rank candidates by L2 distance on stored int8 embeddings
        Phase 3: Fetch metadata from vec_chunks for top results

        Falls back to standard search_with_pairs() if vec_binary is empty/missing.

        Args:
            query: Input trigger text
            limit: Max results
            embedder: Optional embedder override

        Returns:
            List of results with trigger/response details, sorted by int8 L2 distance
        """
        enc = embedder or self._embedder
        embedding = enc.encode(query, normalize=True)
        binary_blob = self._binarize_embedding(embedding)
        int8_query = (embedding * self._INT8_SCALE).astype(np.int8)

        candidates_k = limit * 10

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
                    return self.search_with_pairs(query, limit=limit, embedder=embedder)

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
                top = scored[:limit]

                # Phase 3: Fetch metadata from vec_chunks
                chunk_rowids = [r[0] for r in top]
                dist_by_rowid = {r[0]: r[1] for r in top}

                placeholders = ",".join("?" * len(chunk_rowids))
                # SECURITY: Validate placeholders only contain "?" and "," before SQL interpolation
                _validate_placeholders(placeholders)
                meta_rows = conn.execute(
                    f"""
                    SELECT rowid, chat_id, trigger_text, response_text,
                           response_da_type, response_da_conf, topic_label,
                           quality_score
                    FROM vec_chunks
                    WHERE rowid IN ({placeholders})
                    """,
                    chunk_rowids,
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
                            trigger_text=mrow["trigger_text"],
                            response_text=mrow["response_text"],
                            response_type=mrow["response_da_type"],
                            response_da_conf=mrow["response_da_conf"],
                            topic=mrow["topic_label"],
                            quality_score=mrow["quality_score"],
                        )
                    )

                # Sort by distance ascending (best first)
                results.sort(key=lambda r: r.distance)
                return results

        except Exception as e:
            logger.warning("Two-phase search failed, falling back: %s", e)
            return self.search_with_pairs(query, limit=limit, embedder=embedder)

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
                    f"SELECT rowid, embedding FROM vec_messages WHERE rowid IN ({placeholders})",
                    chunk,
                ).fetchall()
                for row in rows:
                    int8_arr = np.frombuffer(row["embedding"], dtype=np.int8)
                    result[row["rowid"]] = int8_arr.astype(np.float32) / self._INT8_SCALE
        return result

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


from jarvis.utils.singleton import thread_safe_singleton


@thread_safe_singleton
def get_vec_searcher(db: JarvisDB | None = None) -> VecSearcher:
    """Get singleton VecSearcher."""
    if db is None:
        db = get_db()
    return VecSearcher(db)
