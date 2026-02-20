"""Behavior tests for fact extraction.

Tests verify: given input messages, expect specific facts extracted.
No testing of private methods - only public API behavior.
"""

from __future__ import annotations

import time

from jarvis.contacts.fact_extractor import FactExtractor


class TestBotMessageFiltering:
    """Bot messages should not produce facts."""

    def test_cvs_pharmacy_messages_filtered(self) -> None:
        """CVS pharmacy notifications produce no facts."""
        extractor = FactExtractor()
        messages = [{"text": "Your CVS Pharmacy prescription is ready for pickup"}]
        facts = extractor.extract_facts(messages, contact_id="test")
        assert len(facts) == 0

    def test_rx_ready_messages_filtered(self) -> None:
        """Rx Ready notifications produce no facts."""
        extractor = FactExtractor()
        messages = [{"text": "Rx Ready: Your prescription is waiting"}]
        facts = extractor.extract_facts(messages, contact_id="test")
        assert len(facts) == 0

    def test_linkedin_job_messages_filtered(self) -> None:
        """LinkedIn job spam produces no facts."""
        extractor = FactExtractor()
        messages = [{"text": "Check out this job at Google. Apply now!"}]
        facts = extractor.extract_facts(messages, contact_id="test")
        assert len(facts) == 0

    def test_short_code_sms_filtered(self) -> None:
        """SMS from short codes (5-6 digits) produce no facts."""
        extractor = FactExtractor()
        messages = [{"text": "Your code is 12345", "chat_id": "SMS;-;898287"}]
        facts = extractor.extract_facts(messages, contact_id="test")
        assert len(facts) == 0

    def test_normal_messages_extract_facts(self) -> None:
        """Normal conversation messages produce facts."""
        extractor = FactExtractor()
        messages = [{"text": "Hey! I got a new job at Google. Pretty excited about it."}]
        facts = extractor.extract_facts(messages, contact_id="test")
        # Should extract at least one fact (job at Google)
        assert len(facts) >= 1


class TestQualityFiltering:
    """Low-quality facts are filtered out."""

    def test_vague_subjects_filtered(self) -> None:
        """Facts with vague pronoun subjects are filtered."""
        extractor = FactExtractor()
        messages = [
            {"text": "I love driving in san francisco"},  # good: specific subject
            {"text": "I love me"},  # bad: vague subject
        ]
        facts = extractor.extract_facts(messages, contact_id="test")
        subjects = {f.subject.lower() for f in facts}
        # "driving in san francisco" should survive, "me" should not
        assert any("driving" in s for s in subjects)
        assert "me" not in subjects

    def test_short_phrases_filtered(self) -> None:
        """Very short facts are filtered based on confidence threshold."""
        extractor = FactExtractor(confidence_threshold=0.6)
        messages = [{"text": "I like sf"}]  # short subject gets penalized
        facts = extractor.extract_facts(messages, contact_id="test")
        # With high threshold, short low-confidence facts are filtered
        assert len(facts) == 0

    def test_good_facts_survive_filtering(self) -> None:
        """High-quality facts pass through filtering."""
        extractor = FactExtractor(confidence_threshold=0.5)
        messages = [{"text": "I love hiking in the mountains on weekends"}]
        facts = extractor.extract_facts(messages, contact_id="test")
        # Rich context (4+ words) should survive
        assert len(facts) >= 1
        assert any("hiking" in f.subject.lower() for f in facts)


class TestFactDeduplication:
    """Duplicate facts are deduplicated."""

    def test_similar_facts_deduplicated(self) -> None:
        """Facts with same subject and predicate are deduplicated."""
        extractor = FactExtractor()
        messages = [
            {"text": "I love hiking in the mountains"},
            {"text": "I really love hiking in the mountains"},
        ]
        facts = extractor.extract_facts(messages, contact_id="test")
        hiking_facts = [f for f in facts if "hiking" in f.subject.lower()]
        # Should be deduplicated to single fact
        assert len(hiking_facts) == 1


class TestEndToEndExtraction:
    """End-to-end extraction scenarios."""

    def test_mixed_messages_processed_correctly(self) -> None:
        """Mix of bot and normal messages - only normal produce facts."""
        extractor = FactExtractor()
        messages = [
            {"text": "Your CVS Pharmacy prescription is ready"},  # bot
            {"text": "I love driving in san francisco"},  # normal
            {"text": "Check out this job at Google. Apply now!"},  # bot
        ]
        facts = extractor.extract_facts(messages, contact_id="test")
        subjects = {f.subject.lower() for f in facts}
        # Only the normal message should produce facts
        assert "prescription" not in subjects
        assert any("driving" in s for s in subjects)

    def test_extraction_respects_confidence_threshold(self) -> None:
        """Higher threshold filters more aggressively."""
        messages = [{"text": "I like coffee"}]

        low_threshold = FactExtractor(confidence_threshold=0.3)
        high_threshold = FactExtractor(confidence_threshold=0.8)

        facts_low = low_threshold.extract_facts(messages, contact_id="test")
        facts_high = high_threshold.extract_facts(messages, contact_id="test")

        # Higher threshold should produce equal or fewer facts
        assert len(facts_high) <= len(facts_low)


class TestExtractionPerformance:
    """Performance requirements."""

    def test_extraction_100_messages_under_500ms(self) -> None:
        """Extract facts from 100 messages in <500ms."""
        extractor = FactExtractor()
        messages = [{"text": "I love driving in san francisco", "id": i} for i in range(100)]

        start = time.perf_counter()
        facts = extractor.extract_facts(messages, contact_id="test")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500, f"Extraction took {elapsed_ms:.1f}ms, must be <500ms"
        assert len(facts) > 0

    def test_extraction_1000_facts_filter_under_20ms(self) -> None:
        """Quality filtering 1000 facts should be fast."""
        extractor = FactExtractor()
        from jarvis.contacts.contact_profile import Fact

        facts = [
            Fact(
                category="preference",
                subject=f"driving in san francisco route {i}",
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

        assert elapsed_ms < 20, f"Filter 1000 facts took {elapsed_ms:.1f}ms, should be <20ms"
        assert len(filtered) > 0
