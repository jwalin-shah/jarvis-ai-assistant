"""Integration tests for jarvis.search.vec_search with real sqlite-vec.

Tests the VecSearcher against an actual JarvisDB with sqlite-vec loaded,
verifying indexing, search ranking, filtering, quantization round-trip,
and performance characteristics.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from jarvis.db import JarvisDB
from jarvis.search.vec_search import VecSearcher, _validate_placeholders

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_normalized(seed: int, dim: int = 384) -> np.ndarray:
    """Generate a deterministic normalized float32 vector."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _make_cluster(center_seed: int, n: int, noise: float = 0.05) -> list[np.ndarray]:
    """Generate *n* vectors clustered around a center (normalized)."""
    center = _make_normalized(center_seed)
    rng = np.random.RandomState(center_seed + 1000)
    vecs = []
    for _ in range(n):
        v = center + rng.randn(384).astype(np.float32) * noise
        v = v / np.linalg.norm(v)
        vecs.append(v)
    return vecs


class FakeEmbedder:
    """Embedder that returns a pre-set embedding so we control search queries."""

    def __init__(self, embedding: np.ndarray) -> None:
        self._emb = embedding

    def encode(self, text, normalize: bool = True) -> np.ndarray:
        if isinstance(text, list):
            return np.stack([self._emb] * len(text))
        return self._emb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    """Create a real JarvisDB with sqlite-vec and full schema."""
    db_path = tmp_path / "test.db"
    database = JarvisDB(db_path=db_path)
    database.init_schema()
    yield database
    database.close()


@pytest.fixture
def searcher(db):
    """VecSearcher wired to the test database."""
    vs = VecSearcher(db)
    vs._embedder = FakeEmbedder(_make_normalized(0))
    return vs


# ---------------------------------------------------------------------------
# _validate_placeholders
# ---------------------------------------------------------------------------


class TestValidatePlaceholders:
    def test_valid_placeholders(self):
        _validate_placeholders("?,?,?")
        _validate_placeholders("?")
        _validate_placeholders("")

    def test_rejects_sql_injection(self):
        with pytest.raises(ValueError, match="Invalid characters"):
            _validate_placeholders("?; DROP TABLE vec_messages --")

    def test_rejects_spaces(self):
        with pytest.raises(ValueError, match="Invalid characters"):
            _validate_placeholders("?, ?, ?")

    def test_rejects_parens(self):
        with pytest.raises(ValueError, match="Invalid characters"):
            _validate_placeholders("(?)")

    def test_rejects_letters(self):
        with pytest.raises(ValueError, match="Invalid characters"):
            _validate_placeholders("abc")


# ---------------------------------------------------------------------------
# Indexing into vec_messages
# ---------------------------------------------------------------------------


class TestIndexMessages:
    """Insert embeddings directly into vec_messages and verify retrieval."""

    def _insert_message(self, db, rowid, embedding, chat_id, text, sender, ts, is_from_me):
        """Low-level insert bypassing the embedder."""
        int8_blob = (embedding * 127).astype(np.int8).tobytes()
        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO vec_messages(
                    rowid, embedding, chat_id, text_preview,
                    sender, timestamp, is_from_me
                ) VALUES (?, vec_int8(?), ?, ?, ?, ?, ?)
                """,
                (
                    rowid,
                    int8_blob,
                    chat_id,
                    text[:200],
                    sender,
                    ts,
                    1 if is_from_me else 0,
                ),
            )

    def test_index_and_search_basic(self, db, searcher):
        """Index a few messages, search returns them ranked by similarity."""
        food_embs = _make_cluster(center_seed=10, n=3, noise=0.02)
        work_embs = _make_cluster(center_seed=99, n=3, noise=0.02)

        food_texts = [
            "Let's get sushi tonight",
            "Pizza for dinner?",
            "I love tacos",
        ]
        work_texts = [
            "Q3 report is due",
            "Meeting at 3pm",
            "Sent the PR for review",
        ]

        for i, (emb, txt) in enumerate(zip(food_embs, food_texts)):
            self._insert_message(
                db,
                i + 1,
                emb,
                "chat_food",
                txt,
                "Alice",
                1000 + i,
                False,
            )

        for i, (emb, txt) in enumerate(zip(work_embs, work_texts)):
            self._insert_message(
                db,
                i + 10,
                emb,
                "chat_work",
                txt,
                "Bob",
                2000 + i,
                True,
            )

        # Search with a query near the food cluster
        food_query = _make_cluster(center_seed=10, n=1, noise=0.01)[0]
        searcher._embedder = FakeEmbedder(food_query)

        results = searcher.search("food query", limit=6)
        assert len(results) == 6

        # Top 3 should be the food messages (closer to food cluster center)
        top_texts = [r.text for r in results[:3]]
        for txt in food_texts:
            assert txt in top_texts, f"Expected '{txt}' in top 3, got {top_texts}"

        # Scores should be monotonically non-increasing
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_search_with_chat_id_filter(self, db, searcher):
        """chat_id partition key filters results correctly."""
        emb_a = _make_normalized(1)
        emb_b = _make_normalized(2)

        self._insert_message(
            db,
            1,
            emb_a,
            "chat_A",
            "msg in A",
            "Alice",
            100,
            False,
        )
        self._insert_message(
            db,
            2,
            emb_b,
            "chat_B",
            "msg in B",
            "Bob",
            200,
            True,
        )

        searcher._embedder = FakeEmbedder(emb_a)
        results = searcher.search("query", chat_id="chat_A", limit=10)
        assert len(results) == 1
        assert results[0].chat_id == "chat_A"
        assert results[0].text == "msg in A"

    def test_search_empty_index(self, db, searcher):
        """Search on empty vec_messages returns no results."""
        results = searcher.search("anything", limit=5)
        assert results == []

    def test_result_fields_populated(self, db, searcher):
        """VecSearchResult contains all expected message metadata."""
        emb = _make_normalized(42)
        self._insert_message(
            db,
            77,
            emb,
            "chat_X",
            "hello world",
            "Eve",
            1234,
            True,
        )

        searcher._embedder = FakeEmbedder(emb)
        results = searcher.search("query", limit=1)
        assert len(results) == 1
        r = results[0]
        assert r.rowid == 77
        assert r.chat_id == "chat_X"
        assert r.text == "hello world"
        assert r.sender == "Eve"
        assert r.timestamp == 1234
        assert r.is_from_me is True
        assert 0.0 <= r.score <= 1.0
        assert r.distance >= 0.0


# ---------------------------------------------------------------------------
# Indexing into vec_chunks and search_with_pairs
# ---------------------------------------------------------------------------


class TestSearchWithPairs:
    """Insert into vec_chunks and test search_with_pairs."""

    def _insert_chunk(
        self,
        db,
        embedding,
        contact_id,
        chat_id,
        trigger_text,
        response_text,
        topic_label="",
        quality_score=0.5,
        source_timestamp=1000.0,
    ):
        """Low-level insert into vec_chunks."""
        int8_blob = (embedding * 127).astype(np.int8).tobytes()
        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO vec_chunks(
                    embedding, contact_id, chat_id,
                    response_da_type, source_timestamp, quality_score,
                    topic_label, trigger_text, response_text,
                    formatted_text, keywords_json, message_count,
                    source_type, source_id
                ) VALUES (
                    vec_int8(?), ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?
                )
                """,
                (
                    int8_blob,
                    contact_id,
                    chat_id,
                    "",
                    source_timestamp,
                    quality_score,
                    topic_label,
                    trigger_text,
                    response_text,
                    (trigger_text or "")[:500],
                    None,
                    2,
                    "chunk",
                    f"seg_{hash(trigger_text or '') % 10000}",
                ),
            )

    def test_returns_trigger_and_response(self, db, searcher):
        """search_with_pairs returns joined trigger/response text."""
        emb = _make_normalized(50)
        self._insert_chunk(
            db,
            emb,
            contact_id=1,
            chat_id="chat_1",
            trigger_text="Want to grab lunch?",
            response_text="Sure, how about noon?",
            topic_label="food",
            quality_score=0.9,
        )

        searcher._embedder = FakeEmbedder(emb)
        results = searcher.search_with_pairs("lunch query", limit=5)
        assert len(results) == 1
        r = results[0]
        assert r.trigger_text == "Want to grab lunch?"
        assert r.response_text == "Sure, how about noon?"
        assert r.topic == "food"
        assert r.quality_score == pytest.approx(0.9, abs=0.01)
        assert r.chat_id == "chat_1"

    def test_contact_id_filter(self, db, searcher):
        """contact_id partition key filters chunks by contact."""
        emb = _make_normalized(60)
        self._insert_chunk(
            db,
            emb,
            contact_id=1,
            chat_id="c1",
            trigger_text="hi",
            response_text="hey",
        )
        self._insert_chunk(
            db,
            emb,
            contact_id=2,
            chat_id="c2",
            trigger_text="yo",
            response_text="sup",
        )

        searcher._embedder = FakeEmbedder(emb)

        results_c1 = searcher.search_with_pairs("q", limit=10, contact_id=1)
        assert len(results_c1) == 1
        assert results_c1[0].trigger_text == "hi"

        results_c2 = searcher.search_with_pairs("q", limit=10, contact_id=2)
        assert len(results_c2) == 1
        assert results_c2[0].trigger_text == "yo"

    def test_ranking(self, db, searcher):
        """Closer chunks rank higher in search_with_pairs."""
        close_emb = _make_normalized(70)
        far_emb = _make_normalized(999)

        self._insert_chunk(
            db,
            close_emb,
            contact_id=0,
            chat_id="c",
            trigger_text="close",
            response_text="near resp",
        )
        self._insert_chunk(
            db,
            far_emb,
            contact_id=0,
            chat_id="c",
            trigger_text="far",
            response_text="far resp",
        )

        searcher._embedder = FakeEmbedder(close_emb)
        results = searcher.search_with_pairs("q", limit=10)
        assert len(results) == 2
        assert results[0].trigger_text == "close"
        assert results[0].score >= results[1].score

    def test_empty(self, db, searcher):
        """search_with_pairs on empty vec_chunks returns []."""
        results = searcher.search_with_pairs("anything", limit=5)
        assert results == []

    def test_custom_embedder_override(self, db, searcher):
        """The embedder= kwarg is used instead of the default."""
        emb = _make_normalized(80)
        self._insert_chunk(
            db,
            emb,
            contact_id=0,
            chat_id="c",
            trigger_text="test",
            response_text="resp",
        )

        # Default embedder points elsewhere
        searcher._embedder = FakeEmbedder(_make_normalized(999))

        # Override with an embedder that returns the correct vector
        custom = FakeEmbedder(emb)
        results = searcher.search_with_pairs("q", limit=5, embedder=custom)
        assert len(results) == 1
        assert results[0].trigger_text == "test"


# ---------------------------------------------------------------------------
# Quantization round-trip
# ---------------------------------------------------------------------------


class TestQuantizationRoundTrip:
    """Verify int8 quantize -> dequantize preserves enough fidelity."""

    def test_cosine_similarity_preserved(self):
        """Round-tripped embedding has >0.99 cosine similarity."""
        original = _make_normalized(42)
        int8_bytes = (original * 127).astype(np.int8).tobytes()
        recovered = np.frombuffer(int8_bytes, dtype=np.int8).astype(np.float32) / 127.0

        cos_sim = float(
            np.dot(original, recovered) / (np.linalg.norm(original) * np.linalg.norm(recovered))
        )
        assert cos_sim > 0.99, f"Cosine similarity too low: {cos_sim}"

    def test_ranking_preserved(self, db, searcher):
        """Despite quantization, nearest neighbor ordering is preserved."""
        base = _make_normalized(100)
        near = base + np.random.RandomState(101).randn(384).astype(np.float32) * 0.05
        near = near / np.linalg.norm(near)
        far = _make_normalized(200)

        int8_near = (near * 127).astype(np.int8).tobytes()
        int8_far = (far * 127).astype(np.int8).tobytes()

        with db.connection() as conn:
            conn.execute(
                "INSERT INTO vec_messages(rowid, embedding, chat_id, "
                "text_preview, sender, timestamp, is_from_me) "
                "VALUES (?, vec_int8(?), ?, ?, ?, ?, ?)",
                (1, int8_near, "c", "near msg", "A", 100, 0),
            )
            conn.execute(
                "INSERT INTO vec_messages(rowid, embedding, chat_id, "
                "text_preview, sender, timestamp, is_from_me) "
                "VALUES (?, vec_int8(?), ?, ?, ?, ?, ?)",
                (2, int8_far, "c", "far msg", "B", 200, 0),
            )

        searcher._embedder = FakeEmbedder(base)
        results = searcher.search("q", limit=2)
        assert len(results) == 2
        assert results[0].text == "near msg"
        assert results[1].text == "far msg"


# ---------------------------------------------------------------------------
# get_embeddings_by_ids
# ---------------------------------------------------------------------------


class TestGetEmbeddingsByIds:
    def test_retrieves_indexed_embeddings(self, db, searcher):
        """get_embeddings_by_ids returns dequantized embeddings."""
        emb = _make_normalized(300)
        int8_blob = (emb * 127).astype(np.int8).tobytes()

        with db.connection() as conn:
            conn.execute(
                "INSERT INTO vec_messages(rowid, embedding, chat_id, "
                "text_preview, sender, timestamp, is_from_me) "
                "VALUES (?, vec_int8(?), ?, ?, ?, ?, ?)",
                (42, int8_blob, "c", "text", "s", 100, 0),
            )

        result = searcher.get_embeddings_by_ids([42])
        assert 42 in result
        recovered = result[42]
        assert recovered.shape == (384,)
        cos_sim = float(np.dot(emb, recovered) / (np.linalg.norm(emb) * np.linalg.norm(recovered)))
        assert cos_sim > 0.99

    def test_empty_ids(self, db, searcher):
        assert searcher.get_embeddings_by_ids([]) == {}

    def test_missing_ids_returns_empty(self, db, searcher):
        result = searcher.get_embeddings_by_ids([9999])
        assert result == {}


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_stats_on_empty_db(self, db, searcher):
        stats = searcher.get_stats()
        assert stats["total_embeddings"] == 0
        assert stats["unique_chats"] == 0

    def test_stats_counts(self, db, searcher):
        emb = _make_normalized(1)
        int8_blob = (emb * 127).astype(np.int8).tobytes()
        with db.connection() as conn:
            for i in range(5):
                chat = "chat_A" if i < 3 else "chat_B"
                conn.execute(
                    "INSERT INTO vec_messages(rowid, embedding, chat_id, "
                    "text_preview, sender, timestamp, is_from_me) "
                    "VALUES (?, vec_int8(?), ?, ?, ?, ?, ?)",
                    (i + 1, int8_blob, chat, f"msg {i}", "s", i, 0),
                )

        stats = searcher.get_stats()
        assert stats["total_embeddings"] == 5
        assert stats["unique_chats"] == 2


# ---------------------------------------------------------------------------
# delete_chunks_for_chat
# ---------------------------------------------------------------------------


class TestDeleteChunksForChat:
    def _insert_chunk(self, db, embedding, chat_id, trigger="t", response="r"):
        int8_blob = (embedding * 127).astype(np.int8).tobytes()
        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO vec_chunks(
                    embedding, contact_id, chat_id,
                    response_da_type, source_timestamp, quality_score,
                    topic_label, trigger_text, response_text,
                    formatted_text, keywords_json, message_count,
                    source_type, source_id
                ) VALUES (
                    vec_int8(?), 0, ?,
                    '', 1000.0, 0.5,
                    '', ?, ?,
                    '', NULL, 1,
                    'chunk', 'seg_1'
                )
                """,
                (int8_blob, chat_id, trigger, response),
            )

    def test_delete_specific_chat(self, db, searcher):
        emb = _make_normalized(1)
        self._insert_chunk(db, emb, "chat_A", trigger="a1", response="a1r")
        self._insert_chunk(db, emb, "chat_A", trigger="a2", response="a2r")
        self._insert_chunk(db, emb, "chat_B", trigger="b1", response="b1r")

        deleted = searcher.delete_chunks_for_chat("chat_A")
        assert deleted == 2

        # chat_B still exists
        searcher._embedder = FakeEmbedder(emb)
        results = searcher.search_with_pairs("q", limit=10)
        assert len(results) == 1
        assert results[0].trigger_text == "b1"

    def test_delete_nonexistent_chat(self, db, searcher):
        deleted = searcher.delete_chunks_for_chat("no_such_chat")
        assert deleted == 0


# ---------------------------------------------------------------------------
# _vec_tables_exist
# ---------------------------------------------------------------------------


class TestVecTablesExist:
    def test_tables_exist_after_init(self, db):
        searcher = VecSearcher(db)
        searcher._embedder = FakeEmbedder(_make_normalized(0))
        assert searcher._vec_tables_exist() is True

    def test_tables_missing_on_bare_db(self, tmp_path):
        """A bare database without init_schema has no vec tables."""
        import sqlite3

        bare_path = tmp_path / "bare.db"
        conn = sqlite3.connect(str(bare_path))
        conn.execute("CREATE TABLE dummy (id INTEGER)")
        conn.commit()
        conn.close()

        bare_db = JarvisDB(db_path=bare_path)
        searcher = VecSearcher(bare_db)
        searcher._embedder = FakeEmbedder(_make_normalized(0))
        assert searcher._vec_tables_exist() is False
        bare_db.close()


# ---------------------------------------------------------------------------
# _distance_to_similarity edge cases
# ---------------------------------------------------------------------------


class TestDistanceToSimilarity:
    def test_exact_match(self):
        assert VecSearcher._distance_to_similarity(0.0) == 1.0

    def test_clamps_negative_to_zero(self):
        """Very large distance gives cos_sim < 0, clamped to 0."""
        assert VecSearcher._distance_to_similarity(500.0) == 0.0

    def test_moderate(self):
        s = VecSearcher._distance_to_similarity(50.0)
        assert 0.0 < s < 1.0


# ---------------------------------------------------------------------------
# Performance test
# ---------------------------------------------------------------------------


class TestPerformance:
    def test_index_and_search_100_vectors_under_50ms(self, db, searcher):
        """Index 120 vectors, then search in <50ms (wall clock)."""
        n = 120
        rng = np.random.RandomState(777)
        int8_blobs = []
        for i in range(n):
            v = rng.randn(384).astype(np.float32)
            v = v / np.linalg.norm(v)
            int8_blobs.append((v * 127).astype(np.int8).tobytes())

        with db.connection() as conn:
            conn.executemany(
                "INSERT INTO vec_messages(rowid, embedding, chat_id, "
                "text_preview, sender, timestamp, is_from_me) "
                "VALUES (?, vec_int8(?), ?, ?, ?, ?, ?)",
                [
                    (
                        i + 1,
                        int8_blobs[i],
                        "perf_chat",
                        f"msg {i}",
                        "s",
                        i,
                        0,
                    )
                    for i in range(n)
                ],
            )

        query_emb = _make_normalized(888)
        searcher._embedder = FakeEmbedder(query_emb)

        # Warm-up
        searcher.search("warmup", limit=10)

        start = time.perf_counter()
        results = searcher.search("perf query", limit=10)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(results) == 10
        assert elapsed_ms < 50, f"Search took {elapsed_ms:.1f}ms (should be <50ms)"
