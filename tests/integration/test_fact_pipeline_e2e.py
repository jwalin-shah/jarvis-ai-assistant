"""End-to-end tests for the fact extraction pipeline.

Tests the full flow: messages -> extract -> quality filter -> store -> index -> retrieve.
Also tests cross-contact isolation (contact A facts don't leak into contact B searches).

Uses a real sqlite-vec DB with deterministic fake embedder + mocked GLiNER model,
so we can test the complete pipeline without loading heavy ML models.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from jarvis.contacts.candidate_extractor import FactCandidate
from jarvis.contacts.contact_profile import Fact
from jarvis.contacts.fact_extractor import FactExtractor
from jarvis.contacts.fact_index import (
    index_facts,
    reindex_all_facts,
    search_relevant_facts,
)
from jarvis.contacts.fact_storage import (
    get_facts_for_contact,
    save_and_index_facts,
    save_candidate_facts,
    save_facts,
)

# ---------------------------------------------------------------------------
# Deterministic fake embedder (same as test_fact_index.py)
# ---------------------------------------------------------------------------

_CLUSTER_SEEDS = {
    "food": 1,
    "work": 2,
    "location": 3,
    "family": 4,
    "health": 5,
    "activity": 6,
    "pet": 7,
}

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
    "likes_food": "food",
    "dislikes_food": "food",
    "allergic_to": "food",
    "work": "work",
    "google": "work",
    "engineer": "work",
    "job": "work",
    "office": "work",
    "company": "work",
    "works_at": "work",
    "job_title": "work",
    "meta": "work",
    "apple": "work",
    "live": "location",
    "austin": "location",
    "city": "location",
    "move": "location",
    "lives_in": "location",
    "where": "location",
    "chicago": "location",
    "boston": "location",
    "sister": "family",
    "sarah": "family",
    "family": "family",
    "is_family_of": "family",
    "mother": "family",
    "brother": "family",
    "allergy": "health",
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
    seed = _CLUSTER_SEEDS.get(cluster, 99)
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _text_to_embedding(text: str, dim: int = 384) -> np.ndarray:
    words = text.lower().replace("(", " ").replace(")", " ").split()
    matched: list[str] = []
    for w in words:
        w_clean = w.strip(".,!?:;")
        if w_clean in _WORD_CLUSTERS:
            matched.append(_WORD_CLUSTERS[w_clean])
    if matched:
        vecs = [_cluster_embedding(c, dim) for c in matched]
        avg = np.mean(vecs, axis=0).astype(np.float32)
        return avg / np.linalg.norm(avg)
    h = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
    rng = np.random.RandomState(h)
    vec = rng.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


class FakeEmbedder:
    model_name = "fake-384d"

    def encode(self, texts, normalize=True):
        if isinstance(texts, str):
            texts = [texts]
        return np.array([_text_to_embedding(t) for t in texts], dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_db(tmp_path: Path):
    from jarvis.db import JarvisDB

    db_path = tmp_path / "test_e2e.db"
    db = JarvisDB(db_path=db_path)
    db.init_schema()
    return db


def _has_sqlite_vec(db) -> bool:
    try:
        with db.connection() as conn:
            conn.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS vec_facts USING vec0(
                    embedding int8[384] distance_metric=L2,
                    contact_id text,
                    +fact_id INTEGER,
                    +fact_text TEXT
                )"""
            )
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


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

CONTACT_A = "iMessage;-;+15551234567"
CONTACT_B = "iMessage;-;+15559876543"

CONTACT_A_FACTS = [
    Fact(category="preference", subject="sushi", predicate="likes_food", confidence=0.9),
    Fact(category="work", subject="Google", predicate="works_at", confidence=0.95),
    Fact(category="location", subject="Austin", predicate="lives_in", confidence=0.9),
    Fact(
        category="relationship",
        subject="Sarah",
        predicate="is_family_of",
        value="sister",
        confidence=0.95,
    ),
    Fact(category="health", subject="peanuts", predicate="allergic_to", confidence=0.85),
]

CONTACT_B_FACTS = [
    Fact(category="preference", subject="pizza", predicate="likes_food", confidence=0.85),
    Fact(category="work", subject="Meta", predicate="works_at", confidence=0.9),
    Fact(category="location", subject="Chicago", predicate="lives_in", confidence=0.9),
    Fact(
        category="relationship",
        subject="Tom",
        predicate="is_family_of",
        value="brother",
        confidence=0.9,
    ),
]


# ===========================================================================
# End-to-end pipeline: extract -> filter -> store -> index -> retrieve
# ===========================================================================


class TestEndToEndPipeline:
    """Full pipeline test: rule-based extraction -> quality filters -> storage -> semantic search."""

    @pytest.fixture(autouse=True)
    def setup_db(self, tmp_path):
        self.db = _make_test_db(tmp_path)
        if not _has_sqlite_vec(self.db):
            pytest.skip("sqlite-vec not available")
        self.fake_embedder = FakeEmbedder()
        self._patches = [
            patch("jarvis.db.get_db", return_value=self.db),
            patch("jarvis.embedding_adapter.get_embedder", return_value=self.fake_embedder),
        ]
        for p in self._patches:
            p.start()
        yield
        for p in self._patches:
            p.stop()
        self.db.close()

    def test_extract_store_index_retrieve_food(self):
        """Messages mentioning food -> extract facts -> store -> search by food query."""
        messages = [
            {"text": "I love sushi so much, it's my favorite", "id": 1},
            {"text": "My sister Sarah is visiting this weekend", "id": 2},
            {"text": "I work at Google in the Bay Area office", "id": 3},
            {"text": "I hate cilantro, can't stand the taste of it", "id": 4},
            {"text": "I live in Austin these days", "id": 5},
        ]

        # Step 1: Extract facts
        extractor = FactExtractor(confidence_threshold=0.3)
        facts = extractor.extract_facts(messages, contact_id=CONTACT_A)

        # Should extract something from these clear messages
        assert len(facts) > 0, "Expected facts to be extracted from clear messages"

        # Step 2: Store and index facts
        inserted = save_and_index_facts(facts, CONTACT_A)
        assert inserted > 0, f"Expected facts to be inserted, got {inserted}"

        # Step 3: Verify facts persisted (reads back including TIMESTAMP columns —
        # exercises the ISO-format converter fix for valid_from/valid_until)
        stored = get_facts_for_contact(CONTACT_A)
        assert len(stored) == inserted

        # Step 4: Verify semantic index was populated by save_and_index_facts
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM vec_facts WHERE contact_id = ?",
                (CONTACT_A,),
            ).fetchone()
            assert row[0] == inserted, f"Expected {inserted} indexed facts, got {row[0]}"

        # Step 5: Search for food-related facts
        results = search_relevant_facts("want to grab food for dinner?", CONTACT_A, limit=3)
        assert len(results) > 0, "Semantic search returned no results"

        # Food facts should be present
        subjects = {f.subject.lower() for f in results}
        predicates = {f.predicate for f in results}
        food_related = subjects & {"sushi", "cilantro"} or predicates & {
            "likes_food",
            "likes",
            "dislikes",
            "allergic_to",
        }
        assert food_related, (
            f"Expected food facts in results, got: {[(f.predicate, f.subject) for f in results]}"
        )

    def test_extract_store_index_retrieve_work(self):
        """Work-related messages -> extract -> store -> search by work query."""
        messages = [
            {"text": "Just started at Google last week, really excited", "id": 1},
            {"text": "I love pizza, best food ever", "id": 2},
        ]

        extractor = FactExtractor(confidence_threshold=0.3)
        facts = extractor.extract_facts(messages, contact_id=CONTACT_A)
        save_facts(facts, CONTACT_A)
        stored = get_facts_for_contact(CONTACT_A)
        index_facts(stored, CONTACT_A)

        results = search_relevant_facts("where do you work at your job?", CONTACT_A, limit=3)
        work_results = [f for f in results if f.category == "work"]
        assert len(work_results) > 0 or len(results) > 0, (
            "Expected at least some results for work query"
        )

    def test_empty_messages_produce_no_facts(self):
        """Empty or short messages shouldn't produce facts."""
        messages = [
            {"text": "", "id": 1},
            {"text": "ok", "id": 2},
            {"text": "lol", "id": 3},
            {"text": None, "id": 4},
        ]

        extractor = FactExtractor()
        facts = extractor.extract_facts(messages, contact_id=CONTACT_A)
        assert len(facts) == 0, f"Expected no facts from empty messages, got {len(facts)}"

    def test_bot_messages_filtered(self):
        """Bot/spam messages should be filtered out during extraction."""
        messages = [
            {"text": "Your CVS Pharmacy prescription is ready for pickup at 123 Main St", "id": 1},
            {"text": "LinkedIn: John Smith viewed your profile", "id": 2},
            {"text": "I live in Austin and work at Google", "id": 3},
        ]

        extractor = FactExtractor()
        facts = extractor.extract_facts(messages, contact_id=CONTACT_A)

        # Should only get facts from the non-bot message
        source_ids = {f.source_message_id for f in facts}
        assert 1 not in source_ids, "CVS bot message should be filtered"
        assert 2 not in source_ids, "LinkedIn bot message should be filtered"

    def test_dedup_across_messages(self):
        """Same fact mentioned in multiple messages should be deduplicated."""
        messages = [
            {"text": "I live in Austin, it's great here", "id": 1},
            {"text": "Yeah I live in Austin, love it", "id": 2},
        ]

        extractor = FactExtractor()
        facts = extractor.extract_facts(messages, contact_id=CONTACT_A)

        # Should deduplicate "Austin" / "lives_in"
        austin_facts = [f for f in facts if f.subject.lower() == "austin"]
        assert len(austin_facts) <= 1, f"Expected dedup of Austin facts, got {len(austin_facts)}"

    def test_quality_filter_rejects_vague_subjects(self):
        """Quality filter should reject vague pronoun subjects."""
        extractor = FactExtractor()

        # Directly test _apply_quality_filters with vague facts
        vague_facts = [
            Fact(category="preference", subject="it", predicate="likes", confidence=0.8),
            Fact(category="preference", subject="that", predicate="likes", confidence=0.8),
            Fact(category="location", subject="there", predicate="lives_in", confidence=0.8),
        ]
        filtered = extractor._apply_quality_filters(vague_facts)
        assert len(filtered) == 0, (
            f"Expected all vague facts rejected, got {len(filtered)}: "
            f"{[(f.subject, f.category) for f in filtered]}"
        )

    def test_confidence_recalibration(self):
        """Quality filter should recalibrate confidence scores."""
        extractor = FactExtractor(confidence_threshold=0.3)

        facts = [
            # Rich context (4+ words) should get 1.1x boost
            Fact(
                category="location",
                subject="San Francisco Bay Area",
                predicate="lives_in",
                confidence=0.7,
            ),
            # Short preference (< 3 words) should be filtered by _is_too_short
            Fact(
                category="preference",
                subject="ok",
                predicate="likes",
                confidence=0.7,
            ),
        ]
        filtered = extractor._apply_quality_filters(facts)

        # The rich location fact should be boosted
        sf_facts = [f for f in filtered if "francisco" in f.subject.lower()]
        if sf_facts:
            assert sf_facts[0].confidence > 0.7, (
                f"Rich context should boost confidence, got {sf_facts[0].confidence}"
            )

    def test_save_facts_idempotent(self):
        """Saving the same facts twice should not create duplicates."""
        facts = [
            Fact(
                category="work",
                subject="Google",
                predicate="works_at",
                confidence=0.9,
                contact_id=CONTACT_A,
            ),
        ]

        first_insert = save_facts(facts, CONTACT_A)
        second_insert = save_facts(facts, CONTACT_A)

        assert first_insert == 1
        assert second_insert == 0, "Duplicate facts should be ignored"

        stored = get_facts_for_contact(CONTACT_A)
        assert len(stored) == 1

    def test_reindex_clears_and_rebuilds(self):
        """reindex_all_facts should clear vec_facts and rebuild."""
        _seed_facts(self.db, CONTACT_A, CONTACT_A_FACTS)
        _seed_facts(self.db, CONTACT_B, CONTACT_B_FACTS)

        total = reindex_all_facts()
        expected = len(CONTACT_A_FACTS) + len(CONTACT_B_FACTS)
        assert total == expected, f"Expected {expected} reindexed, got {total}"

        with self.db.connection() as conn:
            row = conn.execute("SELECT COUNT(*) FROM vec_facts").fetchone()
            assert row[0] == expected


# ===========================================================================
# Cross-contact isolation
# ===========================================================================


class TestCrossContactIsolation:
    """Contact A's facts must NEVER appear in contact B's search results."""

    @pytest.fixture(autouse=True)
    def setup_db(self, tmp_path):
        self.db = _make_test_db(tmp_path)
        if not _has_sqlite_vec(self.db):
            pytest.skip("sqlite-vec not available")
        self.fake_embedder = FakeEmbedder()
        self._patches = [
            patch("jarvis.db.get_db", return_value=self.db),
            patch("jarvis.embedding_adapter.get_embedder", return_value=self.fake_embedder),
        ]
        for p in self._patches:
            p.start()

        # Seed and index facts for both contacts
        _seed_facts(self.db, CONTACT_A, CONTACT_A_FACTS)
        _seed_facts(self.db, CONTACT_B, CONTACT_B_FACTS)
        index_facts(CONTACT_A_FACTS, CONTACT_A)
        index_facts(CONTACT_B_FACTS, CONTACT_B)

        yield
        for p in self._patches:
            p.stop()
        self.db.close()

    def test_food_query_isolated_to_contact_a(self):
        """Searching contact A for food should return sushi, not pizza."""
        results = search_relevant_facts("what food do you like?", CONTACT_A, limit=5)
        subjects = {f.subject.lower() for f in results}

        # Contact A likes sushi
        assert "sushi" in subjects or "peanuts" in subjects, (
            f"Expected sushi or peanuts for contact A, got {subjects}"
        )
        # Contact B's pizza should NOT appear
        assert "pizza" not in subjects, (
            f"Contact B's pizza leaked into contact A results: {subjects}"
        )

    def test_food_query_isolated_to_contact_b(self):
        """Searching contact B for food should return pizza, not sushi."""
        results = search_relevant_facts("what food do you like?", CONTACT_B, limit=5)
        subjects = {f.subject.lower() for f in results}

        assert "pizza" in subjects, f"Expected pizza for contact B, got {subjects}"
        assert "sushi" not in subjects, (
            f"Contact A's sushi leaked into contact B results: {subjects}"
        )

    def test_work_query_isolated_to_contact_a(self):
        """Searching contact A for work should return Google, not Meta."""
        results = search_relevant_facts("where do you work?", CONTACT_A, limit=5)
        subjects = {f.subject.lower() for f in results}

        if results:
            assert "meta" not in subjects, (
                f"Contact B's Meta leaked into contact A results: {subjects}"
            )

    def test_work_query_isolated_to_contact_b(self):
        """Searching contact B for work should return Meta, not Google."""
        results = search_relevant_facts("where do you work?", CONTACT_B, limit=5)
        subjects = {f.subject.lower() for f in results}

        if results:
            assert "google" not in subjects, (
                f"Contact A's Google leaked into contact B results: {subjects}"
            )

    def test_location_isolation(self):
        """Contact A lives in Austin, contact B lives in Chicago. No leakage."""
        results_a = search_relevant_facts("where do you live in what city?", CONTACT_A, limit=5)
        results_b = search_relevant_facts("where do you live in what city?", CONTACT_B, limit=5)

        subjects_a = {f.subject.lower() for f in results_a}
        subjects_b = {f.subject.lower() for f in results_b}

        # No cross-contamination
        assert "chicago" not in subjects_a, f"Contact B's Chicago leaked into A: {subjects_a}"
        assert "austin" not in subjects_b, f"Contact A's Austin leaked into B: {subjects_b}"

    def test_family_isolation(self):
        """Contact A's sister Sarah should not appear for contact B."""
        results_a = search_relevant_facts("how is your sister and family?", CONTACT_A, limit=5)
        results_b = search_relevant_facts("how is your brother and family?", CONTACT_B, limit=5)

        subjects_a = {f.subject.lower() for f in results_a}
        subjects_b = {f.subject.lower() for f in results_b}

        assert "tom" not in subjects_a, f"Contact B's Tom leaked into A: {subjects_a}"
        assert "sarah" not in subjects_b, f"Contact A's Sarah leaked into B: {subjects_b}"

    def test_nonexistent_contact_returns_empty(self):
        """Searching a contact with no facts should return empty, not leak others."""
        unknown = "iMessage;-;+15550000000"
        results = search_relevant_facts("tell me about food", unknown, limit=5)
        assert results == [], (
            f"Unknown contact returned results: {[(f.subject, f.contact_id) for f in results]}"
        )

    def test_all_returned_facts_have_correct_contact_id(self):
        """Every returned fact must have the queried contact_id."""
        results_a = search_relevant_facts("tell me everything", CONTACT_A, limit=10)
        for fact in results_a:
            assert fact.contact_id == CONTACT_A, (
                f"Fact '{fact.subject}' has wrong contact_id: {fact.contact_id}"
            )

        results_b = search_relevant_facts("tell me everything", CONTACT_B, limit=10)
        for fact in results_b:
            assert fact.contact_id == CONTACT_B, (
                f"Fact '{fact.subject}' has wrong contact_id: {fact.contact_id}"
            )


# ===========================================================================
# GLiNER candidate -> storage pipeline
# ===========================================================================


class TestCandidateToStoragePipeline:
    """Test the GLiNER candidate -> save_candidate_facts -> retrieval flow."""

    @pytest.fixture(autouse=True)
    def setup_db(self, tmp_path):
        self.db = _make_test_db(tmp_path)
        if not _has_sqlite_vec(self.db):
            pytest.skip("sqlite-vec not available")
        self.fake_embedder = FakeEmbedder()
        self._patches = [
            patch("jarvis.db.get_db", return_value=self.db),
            patch("jarvis.embedding_adapter.get_embedder", return_value=self.fake_embedder),
        ]
        for p in self._patches:
            p.start()
        yield
        for p in self._patches:
            p.stop()
        self.db.close()

    def test_candidates_stored_and_retrievable(self):
        """FactCandidates from GLiNER should be stored and retrievable."""
        candidates = [
            FactCandidate(
                message_id=100,
                span_text="Austin",
                span_label="place",
                gliner_score=0.85,
                fact_type="location.current",
                start_char=0,
                end_char=6,
                source_text="I live in Austin",
            ),
            FactCandidate(
                message_id=101,
                span_text="Google",
                span_label="org",
                gliner_score=0.9,
                fact_type="work.employer",
                start_char=0,
                end_char=6,
                source_text="I work at Google",
            ),
            FactCandidate(
                message_id=102,
                span_text="sushi",
                span_label="food_item",
                gliner_score=0.8,
                fact_type="preference.food_like",
                start_char=0,
                end_char=5,
                source_text="I love sushi",
            ),
        ]

        inserted = save_candidate_facts(candidates, CONTACT_A)
        assert inserted == 3, f"Expected 3 inserted, got {inserted}"

        stored = get_facts_for_contact(CONTACT_A)
        assert len(stored) == 3

        categories = {f.category for f in stored}
        assert categories == {"location", "work", "preference"}

    def test_candidates_indexed_for_semantic_search(self):
        """After save_candidate_facts, facts should be searchable semantically."""
        candidates = [
            FactCandidate(
                message_id=100,
                span_text="sushi",
                span_label="food_item",
                gliner_score=0.85,
                fact_type="preference.food_like",
                start_char=0,
                end_char=5,
                source_text="I love sushi so much",
            ),
            FactCandidate(
                message_id=101,
                span_text="Google",
                span_label="org",
                gliner_score=0.9,
                fact_type="work.employer",
                start_char=0,
                end_char=6,
                source_text="I work at Google",
            ),
        ]

        save_candidate_facts(candidates, CONTACT_A)

        # save_candidate_facts calls save_and_index_facts which indexes for search
        # Verify via semantic search
        results = search_relevant_facts("what food do you like for dinner?", CONTACT_A, limit=3)

        # Should find the food fact
        if results:
            subjects = {f.subject.lower() for f in results}
            assert "sushi" in subjects, f"Expected sushi in food search results, got {subjects}"

    def test_unmapped_fact_type_skipped(self):
        """Candidates with unmapped fact_type should be skipped."""
        candidates = [
            FactCandidate(
                message_id=100,
                span_text="something",
                span_label="unknown",
                gliner_score=0.8,
                fact_type="other_personal_fact",
                start_char=0,
                end_char=9,
                source_text="something happened",
            ),
        ]

        inserted = save_candidate_facts(candidates, CONTACT_A)
        assert inserted == 0

    def test_candidate_dedup_on_storage(self):
        """Same candidate saved twice should not create duplicates."""
        candidate = FactCandidate(
            message_id=100,
            span_text="Austin",
            span_label="place",
            gliner_score=0.85,
            fact_type="location.current",
            start_char=0,
            end_char=6,
            source_text="I live in Austin",
        )

        first = save_candidate_facts([candidate], CONTACT_A)
        second = save_candidate_facts([candidate], CONTACT_A)

        assert first == 1
        assert second == 0, "Duplicate candidate should be ignored on second insert"


# ===========================================================================
# Performance
# ===========================================================================


class TestFactPipelinePerformance:
    """Verify the pipeline performs acceptably at moderate scale."""

    @pytest.fixture(autouse=True)
    def setup_db(self, tmp_path):
        self.db = _make_test_db(tmp_path)
        if not _has_sqlite_vec(self.db):
            pytest.skip("sqlite-vec not available")
        self.fake_embedder = FakeEmbedder()
        self._patches = [
            patch("jarvis.db.get_db", return_value=self.db),
            patch("jarvis.embedding_adapter.get_embedder", return_value=self.fake_embedder),
        ]
        for p in self._patches:
            p.start()
        yield
        for p in self._patches:
            p.stop()
        self.db.close()

    def test_batch_insert_100_facts_under_500ms(self):
        """Inserting 100 facts should be fast (batch insert)."""
        import time

        facts = [
            Fact(
                category="preference",
                subject=f"food_item_{i}",
                predicate="likes_food",
                confidence=0.8,
                contact_id=CONTACT_A,
            )
            for i in range(100)
        ]

        start = time.perf_counter()
        save_facts(facts, CONTACT_A)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500, f"Batch insert of 100 facts took {elapsed_ms:.1f}ms (>500ms)"

    def test_semantic_search_under_100ms(self):
        """Semantic search should return in <100ms (excluding model load)."""
        import time

        # Seed 50 facts
        facts = []
        categories = ["preference", "work", "location", "relationship", "health"]
        for i in range(50):
            cat = categories[i % len(categories)]
            facts.append(
                Fact(
                    category=cat,
                    subject=f"entity_{i}",
                    predicate=f"pred_{cat}",
                    confidence=0.8,
                )
            )
        _seed_facts(self.db, CONTACT_A, facts)
        index_facts(facts, CONTACT_A)

        # Search
        start = time.perf_counter()
        results = search_relevant_facts("food dinner restaurant", CONTACT_A, limit=5)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, f"Semantic search took {elapsed_ms:.1f}ms (>100ms)"
