"""Tests for fact attribution resolver.

Tests the AttributionResolver and its integration into FactExtractor.
"""

from jarvis.contacts.attribution import AttributionResolver
from jarvis.contacts.contact_profile import Fact


class TestAttributionResolver:
    """Unit tests for AttributionResolver.resolve()."""

    def setup_method(self):
        self.resolver = AttributionResolver()

    def test_incoming_first_person(self):
        """Incoming message with first person -> contact."""
        result = self.resolver.resolve(
            source_text="I live in Austin",
            subject="Austin",
            is_from_me=False,
            category="location",
        )
        assert result == "contact"

    def test_outgoing_first_person(self):
        """Outgoing message with first person -> user."""
        result = self.resolver.resolve(
            source_text="I live in Austin",
            subject="Austin",
            is_from_me=True,
            category="location",
        )
        assert result == "user"

    def test_relationship_pattern(self):
        """Possessive relationship pattern -> third_party."""
        result = self.resolver.resolve(
            source_text="My sister Sarah lives in Boston",
            subject="Sarah",
            is_from_me=False,
            category="relationship",
        )
        assert result == "third_party"

    def test_relationship_pattern_outgoing(self):
        """Possessive relationship pattern from user -> still third_party."""
        result = self.resolver.resolve(
            source_text="My friend John works at Google",
            subject="John",
            is_from_me=True,
            category="relationship",
        )
        assert result == "third_party"

    def test_third_person_pronoun(self):
        """Third-person pronoun in relationship category -> third_party."""
        result = self.resolver.resolve(
            source_text="She works at Google",
            subject="Google",
            is_from_me=False,
            category="relationship",
        )
        assert result == "third_party"

    def test_third_person_non_relationship(self):
        """Third-person pronoun in non-relationship category -> normal rules."""
        result = self.resolver.resolve(
            source_text="She loves sushi",
            subject="sushi",
            is_from_me=False,
            category="preference",
        )
        # Not relationship category, so third-person pronoun rule doesn't apply
        assert result == "contact"

    def test_outgoing_no_pronoun(self):
        """Outgoing message without pronouns -> user."""
        result = self.resolver.resolve(
            source_text="Austin is great",
            subject="Austin",
            is_from_me=True,
            category="location",
        )
        assert result == "user"

    def test_incoming_no_pronoun(self):
        """Incoming message without pronouns -> contact."""
        result = self.resolver.resolve(
            source_text="Austin is great",
            subject="Austin",
            is_from_me=False,
            category="location",
        )
        assert result == "contact"

    def test_empty_text(self):
        """Empty text -> contact (default)."""
        result = self.resolver.resolve(
            source_text="",
            subject="Austin",
            is_from_me=False,
        )
        assert result == "contact"

    def test_grandma_pattern(self):
        """Grandma relationship pattern -> third_party."""
        result = self.resolver.resolve(
            source_text="My grandma Betty makes great pie",
            subject="Betty",
            is_from_me=True,
            category="relationship",
        )
        assert result == "third_party"

    def test_work_outgoing(self):
        """User talking about their own work -> user."""
        result = self.resolver.resolve(
            source_text="I work at Google now",
            subject="Google",
            is_from_me=True,
            category="work",
        )
        assert result == "user"

    def test_work_incoming(self):
        """Contact talking about their own work -> contact."""
        result = self.resolver.resolve(
            source_text="I just started at Meta",
            subject="Meta",
            is_from_me=False,
            category="work",
        )
        assert result == "contact"


class TestFactDataclassAttribution:
    """Tests for the attribution field on the Fact dataclass."""

    def test_default_attribution(self):
        """Fact() with no attribution defaults to 'contact'."""
        fact = Fact(category="location", subject="Austin", predicate="lives_in")
        assert fact.attribution == "contact"

    def test_explicit_attribution(self):
        """Fact with explicit attribution value."""
        fact = Fact(
            category="location",
            subject="Austin",
            predicate="lives_in",
            attribution="user",
        )
        assert fact.attribution == "user"


class TestFactExtractorAttribution:
    """Integration tests for attribution in FactExtractor pipeline."""

    def test_extract_with_is_from_me_true(self):
        """extract_facts sets attribution=user for outgoing messages."""
        from jarvis.contacts.fact_extractor import FactExtractor

        extractor = FactExtractor()
        messages = [
            {"text": "I live in Austin", "id": 1, "is_from_me": True},
        ]
        facts = extractor.extract_facts(messages, "test_contact")
        location_facts = [f for f in facts if f.category == "location"]
        assert len(location_facts) >= 1
        assert location_facts[0].attribution == "user"

    def test_extract_with_is_from_me_false(self):
        """extract_facts sets attribution=contact for incoming messages."""
        from jarvis.contacts.fact_extractor import FactExtractor

        extractor = FactExtractor()
        messages = [
            {"text": "I live in Austin", "id": 2, "is_from_me": False},
        ]
        facts = extractor.extract_facts(messages, "test_contact")
        location_facts = [f for f in facts if f.category == "location"]
        assert len(location_facts) >= 1
        assert location_facts[0].attribution == "contact"

    def test_extract_relationship_third_party(self):
        """extract_facts sets attribution=third_party for relationship patterns."""
        from jarvis.contacts.fact_extractor import FactExtractor

        extractor = FactExtractor()
        messages = [
            {"text": "My sister Sarah lives in Boston", "id": 3, "is_from_me": False},
        ]
        facts = extractor.extract_facts(messages, "test_contact")
        rel_facts = [f for f in facts if f.category == "relationship"]
        assert len(rel_facts) >= 1
        assert rel_facts[0].attribution == "third_party"

    def test_batch_mixed_attribution(self):
        """extract_facts handles mixed is_from_me in same batch."""
        from jarvis.contacts.fact_extractor import FactExtractor

        extractor = FactExtractor()
        messages = [
            {"text": "I work at Google", "id": 10, "is_from_me": True},
            {"text": "I work at Meta", "id": 11, "is_from_me": False},
        ]
        facts = extractor.extract_facts(messages, "test_contact")
        work_facts = [f for f in facts if f.category == "work"]
        assert len(work_facts) == 2

        by_subject = {f.subject: f for f in work_facts}
        assert by_subject["Google"].attribution == "user"
        assert by_subject["Meta"].attribution == "contact"

    def test_no_id_defaults_to_contact(self):
        """Messages without id field default attribution to contact."""
        from jarvis.contacts.fact_extractor import FactExtractor

        extractor = FactExtractor()
        messages = [
            {"text": "I live in Austin"},
        ]
        facts = extractor.extract_facts(messages, "test_contact")
        location_facts = [f for f in facts if f.category == "location"]
        if location_facts:
            # Without id, can't look up is_from_me, defaults to False -> contact
            assert location_facts[0].attribution == "contact"
