"""Integration test: semantic fact retrieval with real MLX embedder.

This test loads the actual BERT embedder, embeds real facts into a
sqlite-vec DB, and verifies that queries return semantically relevant
results. This is the ground truth test for the context enrichment pipeline.

Requires: MLX, sqlite-vec, ~384MB model weights.
Runtime: ~5-10 seconds (model load + inference).

Run:
    uv run python -m pytest tests/integration/test_fact_retrieval.py -v
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from jarvis.contacts.contact_profile import Fact
from jarvis.contacts.fact_index import (
    index_facts,
    reindex_all_facts,
    search_relevant_facts,
)

CONTACT_ID = "iMessage;-;test_integration"


def _make_test_db(tmp_path: Path):
    from jarvis.db import JarvisDB

    db_path = tmp_path / "integration_facts.db"
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


def _can_load_embedder() -> bool:
    try:
        from jarvis.embedding_adapter import get_embedder

        embedder = get_embedder()
        result = embedder.encode(["test"])
        return result.shape[1] == 384
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


# Diverse facts spanning multiple categories
FACTS = [
    Fact(category="preference", subject="sushi", predicate="likes_food", confidence=0.9),
    Fact(category="preference", subject="pizza", predicate="likes_food", confidence=0.8),
    Fact(category="preference", subject="Thai food", predicate="likes_food", confidence=0.85),
    Fact(category="preference", subject="hiking", predicate="enjoys", confidence=0.85),
    Fact(category="preference", subject="running", predicate="enjoys", confidence=0.7),
    Fact(category="work", subject="Google", predicate="works_at", confidence=0.95),
    Fact(category="work", subject="software engineer", predicate="job_title", confidence=0.9),
    Fact(category="location", subject="Austin, Texas", predicate="lives_in", confidence=0.9),
    Fact(
        category="relationship",
        subject="Sarah",
        predicate="is_family_of",
        value="sister",
        confidence=0.95,
    ),
    Fact(category="health", subject="peanuts", predicate="allergic_to", confidence=0.85),
    Fact(
        category="personal",
        subject="Max",
        predicate="has_pet",
        value="golden retriever",
        confidence=0.8,
    ),
    Fact(category="personal", subject="UT Austin", predicate="attends", confidence=0.75),
]


@pytest.fixture(scope="module")
def embedder_available():
    """Check if MLX embedder can be loaded (once per module)."""
    if not _can_load_embedder():
        pytest.skip("MLX embedder not available (model not downloaded or no GPU)")


@pytest.fixture
def fact_db(tmp_path, embedder_available):
    """Create a DB with indexed facts using the real embedder."""
    db = _make_test_db(tmp_path)
    if not _has_sqlite_vec(db):
        db.close()
        pytest.skip("sqlite-vec not available")

    _seed_facts(db, CONTACT_ID, FACTS)

    with patch("jarvis.db.get_db", return_value=db):
        t0 = time.time()
        count = index_facts(FACTS, CONTACT_ID)
        elapsed = time.time() - t0
        print(f"\nIndexed {count} facts in {elapsed:.2f}s", flush=True)
        assert count == len(FACTS), f"Expected {len(FACTS)} indexed, got {count}"

    yield db
    db.close()


class TestRealSemanticSearch:
    """Integration tests using the actual MLX BERT embedder."""

    def test_food_query_returns_food_facts(self, fact_db):
        """The core test: 'want to grab food?' must return food facts."""
        with patch("jarvis.db.get_db", return_value=fact_db):
            results = search_relevant_facts(
                "want to grab food? maybe sushi or something", CONTACT_ID, limit=3
            )

        predicates = {f.predicate for f in results}
        {f.subject.lower() for f in results}
        print(f"  Food query results: {[(f.predicate, f.subject) for f in results]}")

        # At least one food fact must be in top 3
        food_predicates = {"likes_food", "allergic_to"}
        assert predicates & food_predicates, (
            f"No food facts in top 3. Got: {[(f.predicate, f.subject) for f in results]}"
        )

    def test_work_query_returns_work_facts(self, fact_db):
        with patch("jarvis.db.get_db", return_value=fact_db):
            results = search_relevant_facts(
                "how's work going? what company are you at?", CONTACT_ID, limit=3
            )

        predicates = {f.predicate for f in results}
        print(f"  Work query results: {[(f.predicate, f.subject) for f in results]}")

        assert predicates & {"works_at", "job_title"}, (
            f"No work facts in top 3. Got: {[(f.predicate, f.subject) for f in results]}"
        )

    def test_location_query_returns_location_facts(self, fact_db):
        with patch("jarvis.db.get_db", return_value=fact_db):
            results = search_relevant_facts("where do you live? which city?", CONTACT_ID, limit=3)

        predicates = {f.predicate for f in results}
        print(f"  Location query results: {[(f.predicate, f.subject) for f in results]}")

        assert "lives_in" in predicates, (
            f"No location facts. Got: {[(f.predicate, f.subject) for f in results]}"
        )

    def test_family_query_returns_family_facts(self, fact_db):
        with patch("jarvis.db.get_db", return_value=fact_db):
            results = search_relevant_facts("how is your sister doing?", CONTACT_ID, limit=3)

        predicates = {f.predicate for f in results}
        print(f"  Family query results: {[(f.predicate, f.subject) for f in results]}")

        assert "is_family_of" in predicates, (
            f"No family facts. Got: {[(f.predicate, f.subject) for f in results]}"
        )

    def test_pet_query_returns_pet_facts(self, fact_db):
        with patch("jarvis.db.get_db", return_value=fact_db):
            results = search_relevant_facts(
                "how's your dog? is Max doing well?", CONTACT_ID, limit=3
            )

        predicates = {f.predicate for f in results}
        print(f"  Pet query results: {[(f.predicate, f.subject) for f in results]}")

        assert "has_pet" in predicates, (
            f"No pet facts. Got: {[(f.predicate, f.subject) for f in results]}"
        )

    def test_search_latency_under_100ms(self, fact_db):
        """Semantic fact search must complete in <100ms (excluding model load)."""
        with patch("jarvis.db.get_db", return_value=fact_db):
            # Warm up (embedder already loaded from indexing)
            search_relevant_facts("warm up", CONTACT_ID, limit=3)

            t0 = time.time()
            for _ in range(10):
                search_relevant_facts("want food?", CONTACT_ID, limit=5)
            elapsed_ms = (time.time() - t0) * 1000 / 10

        print(f"  Average search latency: {elapsed_ms:.1f}ms")
        assert elapsed_ms < 100, f"Search too slow: {elapsed_ms:.1f}ms (target <100ms)"

    def test_reindex_then_search(self, tmp_path, embedder_available):
        """Full pipeline: seed facts -> reindex -> search -> correct results."""
        db = _make_test_db(tmp_path)
        if not _has_sqlite_vec(db):
            db.close()
            pytest.skip("sqlite-vec not available")

        _seed_facts(db, CONTACT_ID, FACTS)

        with patch("jarvis.db.get_db", return_value=db):
            count = reindex_all_facts()
            assert count == len(FACTS)

            # Now search
            results = search_relevant_facts("allergic to anything?", CONTACT_ID, limit=3)
            predicates = {f.predicate for f in results}
            print(f"  Allergy query results: {[(f.predicate, f.subject) for f in results]}")

            assert "allergic_to" in predicates, (
                f"No allergy facts. Got: {[(f.predicate, f.subject) for f in results]}"
            )

        db.close()
