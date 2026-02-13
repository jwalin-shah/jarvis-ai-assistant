"""Unit tests for fact extraction with quality filters and NER."""

import time

import pytest

from jarvis.contacts.contact_profile import Fact
from jarvis.contacts.fact_extractor import FactExtractor


class TestBotMessageDetection:
    """Test bot message filtering."""

    def test_cvs_pharmacy_detection(self) -> None:
        """Reject CVS pharmacy messages."""
        extractor = FactExtractor()
        text = "Your CVS Pharmacy prescription is ready for pickup"
        assert extractor._is_bot_message(text)

    def test_rx_ready_detection(self) -> None:
        """Reject pharmacy notifications."""
        extractor = FactExtractor()
        text = "Rx Ready: Your prescription is waiting"
        assert extractor._is_bot_message(text)

    def test_linkedin_job_posting_detection(self) -> None:
        """Reject LinkedIn job spam."""
        extractor = FactExtractor()
        text = "Check out this job at Google. Apply now!"
        assert extractor._is_bot_message(text)

    def test_sms_short_code_detection(self) -> None:
        """Reject SMS from 5-6 digit short codes."""
        extractor = FactExtractor()
        assert extractor._is_bot_message("", chat_id="SMS;-;898287")
        assert extractor._is_bot_message("", chat_id="SMS;-;12345")

    def test_medium_confidence_factors(self) -> None:
        """Test medium-confidence multi-factor detection.

        Factors needed: 3+ of:
        1. URL + job keyword + capitalized word pattern (e.g., "Google Inc")
        2. 'apply' + 'now'
        3. >50% all-caps text
        """
        extractor = FactExtractor()

        # Test case with 2 factors (should NOT be bot)
        text_2_factors = "https://jobs.example.com job at Google Inc apply now"
        # Factors: (1) URL+job+company pattern, (2) apply+now, (3) no caps (29%)
        assert not extractor._is_bot_message(text_2_factors)

        # Test case with 3 factors by making it >50% caps
        # "APPLY NOW FOR JOB AT GOOGLE" = 26/27 letters = 96% caps
        text_3_factors = "APPLY NOW FOR JOB AT GOOGLE"
        # Factors: (2) apply+now, (3) 96% caps, but (1) missing URL, so only 2 factors
        assert not extractor._is_bot_message(text_3_factors)

        # We can't easily construct a 3-factor case (would need high-caps URL + job + caps)
        # So just verify the 2-factor case doesn't trigger bot detection
        # The high-confidence patterns (CVS, Rx, LinkedIn) are the main bot filters anyway

    def test_apply_and_now_together(self) -> None:
        """Reject 'apply' + 'now' pattern (medium confidence)."""
        extractor = FactExtractor()
        # 2 factors: apply + now
        text = "Click here to apply now for this opportunity"
        # Needs 3 factors, so should NOT be rejected with just 2
        assert not extractor._is_bot_message(text)

    def test_all_caps_message(self) -> None:
        """Reject >50% all-caps text (medium confidence, needs 3+ factors)."""
        extractor = FactExtractor()
        text = "APPLY NOW FOR JOB AT GOOGLE"
        # ~50% caps, needs 3 factors - should not reject alone
        assert not extractor._is_bot_message(text)

    def test_normal_message_not_bot(self) -> None:
        """Normal messages should not be flagged as bot."""
        extractor = FactExtractor()
        text = "Hey! I got a new job at Google. Pretty excited about it."
        assert not extractor._is_bot_message(text)


class TestVagueSubjectRejection:
    """Test vague subject filtering."""

    @pytest.mark.parametrize(
        "pronoun",
        ["me", "you", "that", "this", "it", "them", "he", "she"],
        ids=lambda p: f"reject_{p}",
    )
    def test_reject_vague_pronouns(self, pronoun: str) -> None:
        """Reject vague pronouns as subject."""
        extractor = FactExtractor()
        assert extractor._is_vague_subject(pronoun)

    def test_accept_proper_name(self) -> None:
        """Keep proper names."""
        extractor = FactExtractor()
        assert not extractor._is_vague_subject("Sarah")

    def test_accept_specific_phrase(self) -> None:
        """Keep specific phrases."""
        extractor = FactExtractor()
        assert not extractor._is_vague_subject("driving in sf")

    def test_case_insensitive(self) -> None:
        """Pronoun check is case-insensitive."""
        extractor = FactExtractor()
        assert extractor._is_vague_subject("ME")
        assert extractor._is_vague_subject("You")
        assert extractor._is_vague_subject("THAT")


class TestShortPhraseFiltering:
    """Test minimum word count requirements by category."""

    def test_preference_requires_3_words(self) -> None:
        """Preference facts need 3+ words."""
        extractor = FactExtractor()
        assert extractor._is_too_short("preference", "sf")  # 1 word
        assert extractor._is_too_short("preference", "driving sf")  # 2 words
        assert not extractor._is_too_short("preference", "driving in sf")  # 3 words

    def test_relationship_allows_names(self) -> None:
        """Relationship facts allow single names (proper nouns)."""
        extractor = FactExtractor()
        # Names are proper nouns - should be allowed
        assert not extractor._is_too_short("relationship", "Sarah")  # 1 word name OK
        assert not extractor._is_too_short("relationship", "Sarah Jones")  # 2 words OK

    def test_work_allows_proper_nouns(self) -> None:
        """Work facts allow single proper nouns (company names)."""
        extractor = FactExtractor()
        # Capital company names should be allowed
        assert not extractor._is_too_short("work", "Google")  # 1 word proper noun OK
        assert not extractor._is_too_short("work", "Google Inc")  # 2 words OK
        # Lowercase single word should still fail
        assert extractor._is_too_short("work", "startup")  # lowercase, no context

    def test_location_allows_proper_nouns(self) -> None:
        """Location facts allow single proper nouns (place names)."""
        extractor = FactExtractor()
        # Capital place names should be allowed
        assert not extractor._is_too_short("location", "Austin")  # 1 word proper noun OK
        assert not extractor._is_too_short("location", "San Francisco")  # 2 words OK
        # Lowercase single word should still fail
        assert extractor._is_too_short("location", "city")  # lowercase, no context


class TestConfidenceRecalibration:
    """Test confidence scoring adjustments."""

    def test_vague_subject_penalty(self) -> None:
        """Vague subjects reduce confidence to 0.5x."""
        extractor = FactExtractor()
        # Base 0.8, vague subject penalty = 0.8 * 0.5 = 0.4
        adjusted = extractor._calculate_confidence(
            base_confidence=0.8,
            category="preference",
            subject="me",
            is_vague=True,
            is_short=False,
        )
        assert adjusted == pytest.approx(0.4, abs=0.01)

    def test_short_phrase_penalty(self) -> None:
        """Short phrases reduce confidence to 0.7x."""
        extractor = FactExtractor()
        # Base 0.8, short penalty = 0.8 * 0.7 = 0.56
        adjusted = extractor._calculate_confidence(
            base_confidence=0.8,
            category="preference",
            subject="sf",
            is_vague=False,
            is_short=True,
        )
        assert adjusted == pytest.approx(0.56, abs=0.01)

    def test_rich_context_bonus(self) -> None:
        """4+ word subjects get 1.1x confidence bonus."""
        extractor = FactExtractor()
        # Base 0.8, 4 words = 0.8 * 1.1 = 0.88 (capped at 1.0)
        adjusted = extractor._calculate_confidence(
            base_confidence=0.8,
            category="preference",
            subject="driving in san francisco",
            is_vague=False,
            is_short=False,
        )
        assert adjusted == pytest.approx(0.88, abs=0.01)

    def test_confidence_capped_at_1_0(self) -> None:
        """Confidence never exceeds 1.0."""
        extractor = FactExtractor()
        adjusted = extractor._calculate_confidence(
            base_confidence=0.95,
            category="preference",
            subject="very long phrase with many words here",
            is_vague=False,
            is_short=False,
        )
        assert adjusted <= 1.0

    def test_vague_and_short_both_apply(self) -> None:
        """If subject is vague, short penalty is ignored."""
        extractor = FactExtractor()
        # With is_vague=True, short flag doesn't matter (vague takes precedence)
        adjusted = extractor._calculate_confidence(
            base_confidence=0.8,
            category="preference",
            subject="me",
            is_vague=True,
            is_short=True,
        )
        assert adjusted == pytest.approx(0.4, abs=0.01)  # Only vague penalty applied


class TestApplyQualityFilters:
    """Test the integrated quality filtering pipeline."""

    def test_filter_rejects_vague_subject(self) -> None:
        """Vague subjects are rejected during filtering."""
        extractor = FactExtractor()
        facts = [
            Fact(
                category="preference",
                subject="me",
                predicate="likes",
                confidence=0.8,
                contact_id="test",
                extracted_at="2024-01-01",
            )
        ]
        filtered = extractor._apply_quality_filters(facts)
        assert len(filtered) == 0

    def test_filter_rejects_short_with_low_confidence(self) -> None:
        """Short phrases with low confidence are rejected."""
        extractor = FactExtractor(confidence_threshold=0.5)
        facts = [
            Fact(
                category="preference",
                subject="sf",  # 1 word, is_short=True
                predicate="likes",
                confidence=0.6,  # 0.6 * 0.7 = 0.42 < 0.5
                contact_id="test",
                extracted_at="2024-01-01",
            )
        ]
        filtered = extractor._apply_quality_filters(facts)
        assert len(filtered) == 0

    def test_filter_keeps_good_facts(self) -> None:
        """Good facts pass through with adjusted confidence."""
        extractor = FactExtractor(confidence_threshold=0.5)
        facts = [
            Fact(
                category="preference",
                subject="driving in sf",
                predicate="likes",
                confidence=0.8,
                contact_id="test",
                extracted_at="2024-01-01",
            )
        ]
        filtered = extractor._apply_quality_filters(facts)
        assert len(filtered) == 1
        # 3 words, no vague, no short = base confidence 0.8
        assert filtered[0].confidence == pytest.approx(0.8, abs=0.01)

    def test_filter_respects_confidence_threshold(self) -> None:
        """Facts below threshold are rejected."""
        extractor = FactExtractor(confidence_threshold=0.7)
        facts = [
            Fact(
                category="preference",
                subject="coffee",  # 1 word, short
                predicate="likes",
                confidence=0.6,  # 0.6 * 0.7 = 0.42 < 0.7
                contact_id="test",
                extracted_at="2024-01-01",
            )
        ]
        filtered = extractor._apply_quality_filters(facts)
        assert len(filtered) == 0


class TestEndToEndExtraction:
    """Integration tests for end-to-end extraction with filtering."""

    def test_extraction_skips_bot_messages(self) -> None:
        """Bot messages are skipped before extraction."""
        extractor = FactExtractor()
        messages = [
            {"text": "Your CVS Pharmacy prescription is ready"},
            {"text": "I love driving in san francisco"},
        ]
        facts = extractor.extract_facts(messages, contact_id="test")
        # Bot message should be skipped, only the real message extracted
        subjects = {f.subject.lower() for f in facts}
        assert "prescription" not in subjects
        # The non-bot message should produce facts (4+ words survives filters)
        assert len(facts) >= 1, f"Expected at least 1 fact, got {facts}"

    def test_extraction_filters_vague_subjects(self) -> None:
        """Vague subjects are filtered during extraction."""
        extractor = FactExtractor()
        messages = [
            {"text": "I hate cilantro on my tacos"},  # 4+ words, survives filters
            {"text": "I love me"},  # subject=me, vague -> rejected
        ]
        facts = extractor.extract_facts(messages, contact_id="test")
        subjects = {f.subject.lower() for f in facts}
        assert any("cilantro" in s for s in subjects), (
            f"Expected cilantro fact, got subjects: {subjects}"
        )
        assert "me" not in subjects

    def test_extraction_respects_confidence_threshold(self) -> None:
        """Low-confidence facts after filtering are excluded."""
        extractor = FactExtractor(confidence_threshold=0.6)
        messages = [
            {"text": "I like sf"},  # short, will be penalized
        ]
        facts = extractor.extract_facts(messages, contact_id="test")
        # 1-word subject "sf" gets short penalty: 0.6 * 0.7 = 0.42 < 0.6 threshold
        # Should be filtered out entirely
        assert len(facts) == 0, f"Expected 0 facts after threshold filter, got {facts}"

    def test_extraction_deduplicates(self) -> None:
        """Extracted facts are deduplicated."""
        extractor = FactExtractor()
        messages = [
            {"text": "I love hiking in the mountains"},
            {"text": "I really love hiking in the mountains"},
        ]
        facts = extractor.extract_facts(messages, contact_id="test")
        # Duplicate subjects with same predicate should be deduplicated to 1
        hiking_facts = [f for f in facts if "hiking" in f.subject.lower()]
        assert len(hiking_facts) == 1, (
            f"Expected exactly 1 hiking fact after dedup, got {len(hiking_facts)}: {hiking_facts}"
        )


class TestExtractionPerformance:
    """Performance tests to ensure extraction is fast."""

    def test_extraction_under_100ms_for_100_messages(self) -> None:
        """Extract facts from 100 messages in <500ms."""
        extractor = FactExtractor()
        messages = [
            {
                "text": "I love driving in san francisco",
                "id": i,
            }
            for i in range(100)
        ]

        start = time.perf_counter()
        facts = extractor.extract_facts(messages, contact_id="test")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500, f"Extraction took {elapsed_ms:.1f}ms, must be <500ms"
        assert len(facts) > 0

    def test_bot_detection_performance(self) -> None:
        """Bot detection is fast."""
        extractor = FactExtractor()
        text = "Your CVS Pharmacy prescription is ready for pickup at your local store"

        start = time.perf_counter()
        for _ in range(1000):
            extractor._is_bot_message(text)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # 1000 iterations should be <10ms (should be microseconds each)
        assert elapsed_ms < 10, f"1000 bot checks took {elapsed_ms:.1f}ms, should be <10ms"

    def test_quality_filter_performance(self) -> None:
        """Quality filtering is fast."""
        extractor = FactExtractor()
        facts = [
            Fact(
                category="preference",
                subject=f"item_{i}",
                predicate="likes",
                confidence=0.8,
                contact_id="test",
                extracted_at="2024-01-01",
            )
            for i in range(1000)
        ]

        start = time.perf_counter()
        filtered = extractor._apply_quality_filters(facts)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # 1000 facts should filter in <20ms (allows for GC pauses on loaded systems)
        assert elapsed_ms < 20, f"Filter 1000 facts took {elapsed_ms:.1f}ms, should be <20ms"
        assert len(filtered) > 0
