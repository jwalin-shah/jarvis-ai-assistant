"""Advanced tests for jarvis.contacts.fact_extractor.

Covers NER person linking (Jaccard), temporal validity, preference filler
detection, subject cleaning, coherence checks, and end-to-end extraction
with quality filters.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from jarvis.contacts.contact_profile import Fact
from jarvis.contacts.fact_extractor import (
    FactExtractor,
    LOCATION_FUTURE_PATTERN,
    LOCATION_PAST_PATTERN,
    LOCATION_PRESENT_PATTERN,
    PREFERENCE_PATTERN,
)


def _make_fact(
    subject: str = "sushi",
    category: str = "preference",
    predicate: str = "likes",
    confidence: float = 0.8,
    contact_id: str = "c1",
    **kwargs,
) -> Fact:
    return Fact(
        category=category,
        subject=subject,
        predicate=predicate,
        confidence=confidence,
        contact_id=contact_id,
        extracted_at="2025-01-01T00:00:00",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# NER Person Linking — Jaccard similarity
# ---------------------------------------------------------------------------


class TestResolvePersonToContact:
    """_resolve_person_to_contact fuzzy matching via Jaccard similarity."""

    def _make_extractor_with_contacts(
        self, contacts: list[tuple[str, str]]
    ) -> FactExtractor:
        """Create extractor with pre-populated contacts cache."""
        ext = FactExtractor()
        cache = [(cid, name, set(name.lower().split())) for cid, name in contacts]
        ext._get_contacts_for_resolution = lambda: cache
        return ext

    def test_exact_match_returns_contact_id(self):
        ext = self._make_extractor_with_contacts([("c1", "Sarah Johnson")])
        assert ext._resolve_person_to_contact("Sarah Johnson") == "c1"

    def test_partial_match_below_threshold(self):
        """Single shared token out of 3+ gives Jaccard < 0.7 => no match."""
        ext = self._make_extractor_with_contacts([("c1", "Sarah Marie Johnson")])
        # "Sarah" vs "Sarah Marie Johnson": Jaccard = 1/3 ≈ 0.33
        assert ext._resolve_person_to_contact("Sarah") is None

    def test_first_name_match_two_token_contact(self):
        """'Sarah' vs 'Sarah Johnson': Jaccard = 1/2 = 0.5 < 0.7 => no match."""
        ext = self._make_extractor_with_contacts([("c1", "Sarah Johnson")])
        assert ext._resolve_person_to_contact("Sarah") is None

    def test_exact_single_name_match(self):
        """'Sarah' vs 'Sarah': Jaccard = 1/1 = 1.0 => match."""
        ext = self._make_extractor_with_contacts([("c1", "Sarah")])
        assert ext._resolve_person_to_contact("Sarah") == "c1"

    def test_ambiguous_match_rejected(self):
        """Two contacts with similar scores (gap < 0.2) => None."""
        ext = self._make_extractor_with_contacts([
            ("c1", "John Smith"),
            ("c2", "John Davis"),
        ])
        # "John" vs "John Smith": 1/2 = 0.5; "John" vs "John Davis": 1/2 = 0.5
        # Both < 0.7 anyway, but even if high, gap = 0 => ambiguous
        assert ext._resolve_person_to_contact("John") is None

    def test_clear_winner_accepted(self):
        """One contact scores high, others low => accepted with 0.2+ gap."""
        ext = self._make_extractor_with_contacts([
            ("c1", "Sarah Johnson"),
            ("c2", "Mike Thompson"),
        ])
        # "Sarah Johnson" vs "Sarah Johnson": 1.0; vs "Mike Thompson": 0.0
        # Gap = 1.0 - 0.0 = 1.0 >= 0.2 => match
        assert ext._resolve_person_to_contact("Sarah Johnson") == "c1"

    def test_empty_contacts_returns_none(self):
        ext = FactExtractor()
        ext._get_contacts_for_resolution = lambda: []
        assert ext._resolve_person_to_contact("Anyone") is None

    def test_case_insensitive_matching(self):
        """Jaccard uses lowercased tokens."""
        ext = self._make_extractor_with_contacts([("c1", "Sarah")])
        assert ext._resolve_person_to_contact("sarah") == "c1"
        assert ext._resolve_person_to_contact("SARAH") == "c1"

    def test_contacts_cache_is_lazy(self):
        """_get_contacts_for_resolution uses module-level cache."""
        from unittest.mock import patch

        cache = [("c1", "Test", {"test"})]
        ext = FactExtractor()
        with patch(
            "jarvis.contacts.fact_extractor._get_cached_contacts", return_value=cache
        ):
            result = ext._get_contacts_for_resolution()
        assert result == [("c1", "Test", {"test"})]


# ---------------------------------------------------------------------------
# Temporal Validity — location patterns set valid_from/valid_until
# ---------------------------------------------------------------------------


class TestTemporalValidity:
    """Location patterns assign valid_from or valid_until timestamps."""

    def test_present_location_sets_valid_from(self):
        ext = FactExtractor()
        facts = ext._extract_rule_based("I live in Austin", "c1", "2025-01-15T10:00:00")
        location_facts = [f for f in facts if f.category == "location"]
        assert len(location_facts) >= 1
        present = [f for f in location_facts if f.predicate == "lives_in" and f.valid_from]
        assert len(present) >= 1
        assert present[0].valid_from == "2025-01-15T10:00:00"
        assert present[0].valid_until is None

    def test_past_location_sets_valid_until(self):
        ext = FactExtractor()
        facts = ext._extract_rule_based("I grew up in Texas", "c1", "2025-01-15T10:00:00")
        location_facts = [f for f in facts if f.predicate == "lived_in"]
        assert len(location_facts) >= 1
        assert location_facts[0].valid_until == "2025-01-15T10:00:00"
        assert location_facts[0].valid_from is None

    def test_future_location_sets_valid_from(self):
        ext = FactExtractor()
        facts = ext._extract_rule_based("I'm moving to Portland", "c1", "2025-01-15T10:00:00")
        future = [f for f in facts if f.predicate == "moving_to"]
        assert len(future) >= 1
        assert future[0].valid_from == "2025-01-15T10:00:00"

    def test_work_has_no_temporal_fields(self):
        ext = FactExtractor()
        facts = ext._extract_rule_based("I work at Google", "c1", "2025-01-15T10:00:00")
        work_facts = [f for f in facts if f.category == "work"]
        assert len(work_facts) >= 1
        assert work_facts[0].valid_from is None
        assert work_facts[0].valid_until is None


# ---------------------------------------------------------------------------
# Preference Filler Detection
# ---------------------------------------------------------------------------


class TestPreferenceFillerDetection:
    """_is_like_filler_word correctly identifies filler vs preference 'like'."""

    def setup_method(self):
        self.ext = FactExtractor()

    def test_like_no_longer_triggers_preference(self):
        """'like' was removed from PREFERENCE_PATTERN (too many false positives)."""
        text = "I like sushi"
        match = PREFERENCE_PATTERN.search(text)
        assert match is None  # "like" is no longer a preference trigger

    def test_love_triggers_preference(self):
        """'love' remains a valid preference trigger."""
        text = "I love sushi"
        match = PREFERENCE_PATTERN.search(text)
        assert match is not None
        assert match.group(1).strip() == "sushi"

    def test_filler_its_like(self):
        text = "it's like a dream"
        # Find any "like" position for the filler check
        idx = text.find("like")
        assert self.ext._is_like_filler_word(text, idx, idx + 4)

    def test_filler_i_was_like(self):
        text = "i was like omg"
        idx = text.find("like")
        assert self.ext._is_like_filler_word(text, idx, idx + 4)

    def test_filler_like_number(self):
        text = "there were like 5 people"
        idx = text.find("like")
        assert self.ext._is_like_filler_word(text, idx, idx + 4)

    def test_filler_not_like(self):
        text = "it's not like that matters"
        idx = text.find("not like")
        assert self.ext._is_like_filler_word(text, idx, idx + 8)


class TestPreferenceSubjectValidation:
    """_is_valid_preference_subject rejects clause fragments."""

    def setup_method(self):
        self.ext = FactExtractor()

    def test_valid_noun_phrase(self):
        assert self.ext._is_valid_preference_subject("spicy ramen") is True

    def test_reject_starts_with_interrogative(self):
        assert self.ext._is_valid_preference_subject("how it feels") is False
        assert self.ext._is_valid_preference_subject("when you think") is False

    def test_reject_starts_with_pronoun(self):
        assert self.ext._is_valid_preference_subject("i was gonna") is False
        assert self.ext._is_valid_preference_subject("you know better") is False

    def test_reject_contains_verb_pattern(self):
        assert self.ext._is_valid_preference_subject("the thing was great") is False
        assert self.ext._is_valid_preference_subject("stuff i realized later") is False


# ---------------------------------------------------------------------------
# Subject Cleaning
# ---------------------------------------------------------------------------


class TestCleanSubject:
    """_clean_subject removes trailing conjunctions/prepositions."""

    def setup_method(self):
        self.ext = FactExtractor()

    def test_strips_trailing_conjunction(self):
        assert self.ext._clean_subject("sushi and") == "sushi"
        assert self.ext._clean_subject("pizza but") == "pizza"
        assert self.ext._clean_subject("tacos or") == "tacos"

    def test_strips_trailing_verb(self):
        assert self.ext._clean_subject("the thing is") == "the thing"
        assert self.ext._clean_subject("something was") == "something"

    def test_preserves_valid_subject(self):
        assert self.ext._clean_subject("spicy ramen") == "spicy ramen"
        assert self.ext._clean_subject("Google") == "Google"

    def test_strips_whitespace(self):
        assert self.ext._clean_subject("  sushi  ") == "sushi"


# ---------------------------------------------------------------------------
# Coherence Checks
# ---------------------------------------------------------------------------


class TestCoherenceChecks:
    """_is_coherent_subject rejects fragments and malformed text."""

    def setup_method(self):
        self.ext = FactExtractor()

    @pytest.mark.parametrize("word", ["it", "that", "this", "them", "there", "what", "how"])
    def test_single_vague_word_rejected(self, word):
        assert self.ext._is_coherent_subject(word) is False

    def test_pronoun_preposition_fragment_rejected(self):
        assert self.ext._is_coherent_subject("it in August") is False
        assert self.ext._is_coherent_subject("that in the") is False

    def test_valid_noun_phrase_accepted(self):
        assert self.ext._is_coherent_subject("spicy ramen") is True
        assert self.ext._is_coherent_subject("Google") is True
        assert self.ext._is_coherent_subject("Austin Texas") is True

    def test_single_char_rejected(self):
        assert self.ext._is_coherent_subject("x") is False

    def test_no_letters_rejected(self):
        assert self.ext._is_coherent_subject("123") is False

    def test_incomplete_infinitive_rejected(self):
        assert self.ext._is_coherent_subject("to call this") is False

    def test_abbreviation_overload_rejected(self):
        assert self.ext._is_coherent_subject("sm rn tb") is False

    def test_camelcase_malformed_rejected(self):
        assert self.ext._is_coherent_subject("ofMetal") is False


# ---------------------------------------------------------------------------
# End-to-end Extraction with Quality Filters
# ---------------------------------------------------------------------------


class TestEndToEndExtraction:
    """Full extract_facts pipeline including quality filters."""

    def test_relationship_extraction(self):
        ext = FactExtractor()
        facts = ext.extract_facts([{"text": "My sister Sarah is visiting"}], "c1")
        rels = [f for f in facts if f.category == "relationship"]
        assert len(rels) >= 1
        assert rels[0].subject == "Sarah"
        assert rels[0].predicate == "is_family_of"
        assert rels[0].value == "sister"

    def test_work_extraction(self):
        ext = FactExtractor()
        facts = ext.extract_facts([{"text": "I work at Google and love it"}], "c1")
        work = [f for f in facts if f.category == "work"]
        assert len(work) >= 1
        assert work[0].subject == "Google"

    def test_preference_sentiment_positive(self):
        ext = FactExtractor()
        facts = ext.extract_facts(
            [{"text": "I love spicy ramen so much"}], "c1"
        )
        prefs = [f for f in facts if f.category == "preference"]
        assert len(prefs) >= 1
        assert prefs[0].predicate == "likes"

    def test_preference_sentiment_negative(self):
        ext = FactExtractor()
        facts = ext.extract_facts(
            [{"text": "I hate cold weather every year"}], "c1"
        )
        prefs = [f for f in facts if f.category == "preference"]
        assert len(prefs) >= 1
        assert prefs[0].predicate == "dislikes"

    def test_deduplication_across_messages(self):
        ext = FactExtractor()
        facts = ext.extract_facts(
            [
                {"text": "I live in Austin"},
                {"text": "I live in Austin"},
            ],
            "c1",
        )
        # Should deduplicate identical facts
        locations = [f for f in facts if f.category == "location" and f.subject.lower() == "austin"]
        assert len(locations) == 1

    def test_short_messages_skipped(self):
        ext = FactExtractor()
        facts = ext.extract_facts([{"text": "ok"}, {"text": "hey"}], "c1")
        assert len(facts) == 0

    def test_confidence_threshold_filtering(self):
        """High threshold filters out lower-confidence facts."""
        ext = FactExtractor(confidence_threshold=0.9)
        facts = ext.extract_facts(
            [{"text": "I like playing basketball with my friends after school"}], "c1"
        )
        # Preferences have base confidence 0.6, should be filtered at 0.9
        prefs = [f for f in facts if f.category == "preference"]
        assert len(prefs) == 0
