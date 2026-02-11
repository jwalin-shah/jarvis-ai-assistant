"""Tests for semantic fact retrieval (jarvis/contacts/fact_index.py).

Tests the core question: does search_relevant_facts("want food?") return
food-related facts and NOT work facts?

Uses a real sqlite-vec DB with a deterministic fake embedder that maps
semantically similar words to nearby vectors, so we can test the full
search pipeline without loading MLX models.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from jarvis.contacts.contact_profile import Fact
from jarvis.contacts.fact_index import (
    _distance_to_similarity,
    _fact_to_text,
    _quantize,
    index_facts,
    reindex_all_facts,
    search_relevant_facts,
)

# ---------------------------------------------------------------------------
# Deterministic fake embedder
# ---------------------------------------------------------------------------

# Semantic clusters: words that should be "near" each other in embedding space.
# We assign each cluster a random but fixed direction in 384-d space.
_CLUSTER_SEEDS = {
    "food": 1,
    "work": 2,
    "location": 3,
    "family": 4,
    "health": 5,
    "activity": 6,
    "pet": 7,
}

# Map keywords to clusters
_WORD_CLUSTERS = {
    "food": "food",
    "sushi": "food",
    "pizza": "food",
    "dinner": "food",
    "lunch": "food",
    "eat": "food",
    "grab": "food",
    "restaurant": "food",
    "peanuts": "food",
    "allergic": "food",
    "work": "work",
    "google": "work",
    "engineer": "work",
    "job": "work",
    "office": "work",
    "company": "work",
    "works_at": "work",
    "job_title": "work",
    "live": "location",
    "austin": "location",
    "city": "location",
    "move": "location",
    "lives_in": "location",
    "where": "location",
    "sister": "family",
    "sarah": "family",
    "family": "family",
    "is_family_of": "family",
    "mother": "family",
    "brother": "family",
    "allergy": "health",
    "allergic_to": "health",
    "health": "health",
    "hiking": "activity",
    "running": "activity",
    "enjoys": "activity",
    "outdoor": "activity",
    "trail": "activity",
    "pet": "pet",
    "dog": "pet",
    "max": "pet",
    "has_pet": "pet",
}


def _cluster_embedding(cluster: str, dim: int = 384) -> np.ndarray:
    """Get a deterministic unit vector for a semantic cluster."""
    seed = _CLUSTER_SEEDS.get(cluster, 99)
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _text_to_embedding(text: str, dim: int = 384) -> np.ndarray:
    """Map text to an embedding based on keyword matching.

    Averages cluster vectors for all recognized keywords in the text.
    Falls back to a hash-based random vector for unrecognized text.
    """
    words = text.lower().replace("(", " ").replace(")", " ").split()
    matched_clusters: list[str] = []
    for w in words:
        w_clean = w.strip(".,!?:;")
        if w_clean in _WORD_CLUSTERS:
            matched_clusters.append(_WORD_CLUSTERS[w_clean])

    if matched_clusters:
        vecs = [_cluster_embedding(c, dim) for c in matched_clusters]
        avg = np.mean(vecs, axis=0).astype(np.float32)
        return avg / np.linalg.norm(avg)

    # Hash-based fallback for unknown text
    h = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
    rng = np.random.RandomState(h)
    vec = rng.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


class FakeEmbedder:
    """Deterministic embedder that maps words to semantic cluster vectors."""

    model_name = "fake-384d"

    def encode(self, texts, normalize=True):
        if isinstance(texts, str):
            texts = [texts]
        return np.array([_text_to_embedding(t) for t in texts], dtype=np.float32)


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestFactToText:
    def test_predicate_and_subject(self):
        fact = Fact(category="preference", subject="sushi", predicate="likes_food")
        assert _fact_to_text(fact) == "likes_food: sushi"

    def test_value_appended_in_parens(self):
        fact = Fact(
            category="relationship", subject="Sarah", predicate="is_family_of", value="sister"
        )
        assert _fact_to_text(fact) == "is_family_of: Sarah (sister)"

    def test_empty_value_not_appended(self):
        fact = Fact(category="work", subject="Google", predicate="works_at", value="")
        assert "(" not in _fact_to_text(fact)


class TestQuantize:
    def test_round_trip_preserves_direction(self):
        emb = np.array([1.0, -1.0, 0.5, -0.5, 0.0], dtype=np.float32)
        blob = _quantize(emb)
        recovered = np.frombuffer(blob, dtype=np.int8).astype(np.float32) / 127.0
        assert recovered[0] > 0
        assert recovered[1] < 0
        assert recovered[4] == 0.0


class TestDistanceToSimilarity:
    def test_identical_vectors_give_similarity_1(self):
        assert _distance_to_similarity(0.0) == 1.0

    def test_monotonically_decreasing(self):
        sims = [_distance_to_similarity(d) for d in [0, 50, 100, 150, 200]]
        for i in range(len(sims) - 1):
            assert sims[i] >= sims[i + 1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_db(tmp_path: Path):
    from jarvis.db import JarvisDB

    db_path = tmp_path / "test_facts.db"
    db = JarvisDB(db_path=db_path)
    db.init_schema()
    return db


def _has_sqlite_vec(db) -> bool:
    try:
        with db.connection() as conn:
            conn.execute("SELECT 1 FROM vec_facts LIMIT 0")
        return True
    except Exception:
        return False


def _seed_facts(db, contact_id: str, facts: list[Fact]) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with db.connection() as conn:
        for f in facts:
            conn.execute(
                """
                INSERT OR IGNORE INTO contact_facts
                (contact_id, category, subject, predicate, value, confidence,
                 source_text, extracted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    contact_id,
                    f.category,
                    f.subject,
                    f.predicate,
                    f.value or "",
                    f.confidence,
                    f.source_text or "",
                    now,
                ),
            )


SAMPLE_FACTS = [
    Fact(category="preference", subject="sushi", predicate="likes_food", confidence=0.9),
    Fact(category="preference", subject="pizza", predicate="likes_food", confidence=0.8),
    Fact(category="preference", subject="hiking", predicate="enjoys", confidence=0.85),
    Fact(category="work", subject="Google", predicate="works_at", confidence=0.95),
    Fact(category="work", subject="software engineer", predicate="job_title", confidence=0.9),
    Fact(category="location", subject="Austin", predicate="lives_in", confidence=0.9),
    Fact(
        category="relationship",
        subject="Sarah",
        predicate="is_family_of",
        value="sister",
        confidence=0.95,
    ),
    Fact(category="health", subject="peanuts", predicate="allergic_to", confidence=0.85),
    Fact(category="personal", subject="Max", predicate="has_pet", value="dog", confidence=0.8),
    Fact(category="preference", subject="running", predicate="enjoys", confidence=0.7),
]

CONTACT_ID = "iMessage;-;+15551234567"


# ---------------------------------------------------------------------------
# Integration tests with real sqlite-vec + fake embedder
# ---------------------------------------------------------------------------


class TestSemanticFactRetrieval:
    """Does search_relevant_facts return the RIGHT facts for a given query?"""

    @pytest.fixture(autouse=True)
    def setup_db(self, tmp_path):
        self.db = _make_test_db(tmp_path)
        if not _has_sqlite_vec(self.db):
            pytest.skip("sqlite-vec not available")

        _seed_facts(self.db, CONTACT_ID, SAMPLE_FACTS)
        self.fake_embedder = FakeEmbedder()

        # Patch both get_db and get_embedder so the real pipeline runs
        # with our test DB and deterministic embedder
        self._patches = [
            patch("jarvis.db.get_db", return_value=self.db),
            patch("jarvis.embedding_adapter.get_embedder", return_value=self.fake_embedder),
        ]
        for p in self._patches:
            p.start()

        # Index all facts using the fake embedder
        index_facts(SAMPLE_FACTS, CONTACT_ID)

        yield
        for p in self._patches:
            p.stop()
        self.db.close()

    def test_food_query_returns_food_facts_not_work_facts(self):
        """'want to grab food?' should return food-related facts."""
        results = search_relevant_facts("want to grab food for dinner?", CONTACT_ID, limit=3)
        predicates = {f.predicate for f in results}
        subjects = {f.subject.lower() for f in results}

        # Must include food facts
        assert subjects & {"sushi", "pizza", "peanuts"}, (
            f"Expected food facts, got: {[(f.predicate, f.subject) for f in results]}"
        )
        # Work facts should NOT dominate
        work_count = sum(1 for f in results if f.predicate in ("works_at", "job_title"))
        food_count = sum(1 for f in results if f.predicate in ("likes_food", "allergic_to"))
        assert food_count > work_count, (
            f"Food query returned more work ({work_count}) than food ({food_count}) facts"
        )

    def test_work_query_returns_work_facts(self):
        """'where do you work?' should return work-related facts."""
        results = search_relevant_facts("where do you work at your job?", CONTACT_ID, limit=3)
        predicates = {f.predicate for f in results}

        assert predicates & {"works_at", "job_title"}, (
            f"Expected work facts, got: {[(f.predicate, f.subject) for f in results]}"
        )

    def test_location_query_returns_location_facts(self):
        """'where do you live?' should return location facts."""
        results = search_relevant_facts("where do you live in what city?", CONTACT_ID, limit=3)
        predicates = {f.predicate for f in results}

        assert "lives_in" in predicates, (
            f"Expected lives_in, got: {[(f.predicate, f.subject) for f in results]}"
        )

    def test_family_query_returns_family_facts(self):
        """'how is your sister?' should return family facts."""
        results = search_relevant_facts("how is your sister and family?", CONTACT_ID, limit=3)
        predicates = {f.predicate for f in results}

        assert "is_family_of" in predicates, (
            f"Expected family facts, got: {[(f.predicate, f.subject) for f in results]}"
        )

    def test_respects_limit(self):
        results = search_relevant_facts("tell me everything", CONTACT_ID, limit=2)
        assert len(results) <= 2

    def test_unknown_contact_returns_empty(self):
        results = search_relevant_facts("hello", "nonexistent_chat", limit=5)
        # Nonexistent contact has no facts, should return empty list
        assert results == [], f"Expected empty list for unknown contact, got {results}"


class TestFallbackBehavior:
    def test_falls_back_when_vec_facts_missing(self, tmp_path):
        """When vec_facts table is dropped, should fall back to get_facts_for_contact."""
        db = _make_test_db(tmp_path)
        _seed_facts(db, CONTACT_ID, SAMPLE_FACTS)

        with patch("jarvis.db.get_db", return_value=db):
            # Drop vec_facts
            try:
                with db.connection() as conn:
                    conn.execute("DROP TABLE IF EXISTS vec_facts")
            except Exception:
                pass

            results = search_relevant_facts("anything", CONTACT_ID, limit=5)

            # Should return facts from fallback, not crash
            assert len(results) > 0
            assert all(isinstance(f, Fact) for f in results)

        db.close()


class TestReindexAllFacts:
    def test_reindex_populates_vec_facts(self, tmp_path):
        db = _make_test_db(tmp_path)
        if not _has_sqlite_vec(db):
            db.close()
            pytest.skip("sqlite-vec not available")

        _seed_facts(db, CONTACT_ID, SAMPLE_FACTS)
        fake = FakeEmbedder()

        with (
            patch("jarvis.db.get_db", return_value=db),
            patch("jarvis.embedding_adapter.get_embedder", return_value=fake),
        ):
            count = reindex_all_facts()

        assert count == len(SAMPLE_FACTS)

        with db.connection() as conn:
            row = conn.execute("SELECT COUNT(*) FROM vec_facts").fetchone()
            assert row[0] == len(SAMPLE_FACTS)

        db.close()
