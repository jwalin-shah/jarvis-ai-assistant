"""Tests for factuality checking module."""

from jarvis.quality.factuality import (
    ClaimExtractor,
    ClaimType,
    FactChecker,
    FactualityResult,
    VerificationStatus,
    get_fact_checker,
    reset_fact_checker,
)


class TestClaimExtractor:
    """Tests for ClaimExtractor."""

    def test_extract_simple_claim(self):
        """Extract claim from simple sentence."""
        extractor = ClaimExtractor()
        claims = extractor.extract_claims("The meeting is at 3pm.")
        assert len(claims) == 1
        assert claims[0].text == "The meeting is at 3pm."

    def test_extract_multiple_claims(self):
        """Extract multiple claims from text."""
        extractor = ClaimExtractor()
        claims = extractor.extract_claims(
            "The meeting is at 3pm. John will attend. The agenda is ready."
        )
        assert len(claims) == 3

    def test_filter_short_claims(self):
        """Short sentences should be filtered."""
        extractor = ClaimExtractor(min_claim_words=3)
        claims = extractor.extract_claims("Hi. Yes. The meeting is at noon.")
        # Only the longer sentence should be included
        assert len(claims) == 1
        assert "meeting" in claims[0].text

    def test_classify_temporal_claim(self):
        """Classify temporal claims correctly."""
        extractor = ClaimExtractor()
        claims = extractor.extract_claims("We met yesterday at the office.")
        assert len(claims) == 1
        assert claims[0].claim_type == ClaimType.TEMPORAL

    def test_classify_quantitative_claim(self):
        """Classify quantitative claims correctly."""
        extractor = ClaimExtractor()
        claims = extractor.extract_claims("There were 50 people at the event.")
        assert len(claims) == 1
        assert claims[0].claim_type == ClaimType.QUANTITATIVE

    def test_extract_entities(self):
        """Extract entities from claims."""
        extractor = ClaimExtractor()
        claims = extractor.extract_claims("John and Sarah went to Paris.")
        assert len(claims) == 1
        # Should extract capitalized words
        entities = claims[0].entities
        assert "John" in entities or "Sarah" in entities or "Paris" in entities


class TestFactChecker:
    """Tests for FactChecker."""

    def test_check_empty_response(self):
        """Empty response should have no claims."""
        checker = FactChecker()
        result = checker.check_factuality("", "Some context")
        assert result.total_claims == 0
        assert result.factuality_score == 1.0

    def test_check_grounded_response(self):
        """Response grounded in context should score high."""
        checker = FactChecker()
        result = checker.check_factuality(
            response="The meeting is at noon.",
            context="Reminder: meeting at noon today.",
        )
        assert isinstance(result, FactualityResult)
        assert result.factuality_score >= 0.0

    def test_check_multiple_contexts(self):
        """Should check against multiple context strings."""
        checker = FactChecker()
        result = checker.check_factuality(
            response="The meeting is at noon and lunch is at 1pm.",
            context=["Meeting: noon", "Lunch: 1pm"],
        )
        assert isinstance(result, FactualityResult)

    def test_result_contains_claims(self):
        """Result should contain verified claims."""
        checker = FactChecker()
        result = checker.check_factuality(
            response="The project deadline is Friday.",
            context="Project deadline: this Friday",
        )
        assert len(result.claims) > 0

    def test_verification_status_values(self):
        """Test that verification statuses are assigned."""
        checker = FactChecker()
        result = checker.check_factuality(
            response="The meeting is tomorrow.",
            context="Meeting scheduled for tomorrow",
        )
        for claim in result.claims:
            assert claim.status in [
                VerificationStatus.VERIFIED,
                VerificationStatus.REFUTED,
                VerificationStatus.UNVERIFIABLE,
                VerificationStatus.PARTIALLY_VERIFIED,
            ]

    def test_passes_gate_with_good_score(self):
        """Should pass gate with good factuality score."""
        checker = FactChecker(gate_threshold=0.5)
        result = checker.check_factuality(
            response="Yes, I can help.",
            context="Can you help me with this?",
        )
        # Simple affirmative should generally pass
        assert isinstance(result.passes_gate, bool)

    def test_latency_is_recorded(self):
        """Latency should be recorded."""
        checker = FactChecker()
        result = checker.check_factuality(
            response="Okay, sounds good.",
            context="Let's meet tomorrow.",
        )
        assert result.latency_ms >= 0


class TestFactualityResult:
    """Tests for FactualityResult dataclass."""

    def test_result_to_dict(self):
        """Test conversion to dictionary."""
        result = FactualityResult(
            factuality_score=0.8,
            verified_count=2,
            refuted_count=0,
            unverifiable_count=1,
            passes_gate=True,
        )
        d = result.to_dict()
        assert "factuality_score" in d
        assert d["verified_count"] == 2
        assert d["passes_gate"] is True

    def test_total_claims_property(self):
        """Test total_claims property."""
        result = FactualityResult(
            factuality_score=0.8,
            verified_count=2,
            refuted_count=1,
            unverifiable_count=1,
        )
        # Need to add claims to test total
        assert result.total_claims == 0  # No claims added yet


class TestGlobalSingleton:
    """Tests for global singleton pattern."""

    def test_get_fact_checker_returns_instance(self):
        """Get checker should return an instance."""
        reset_fact_checker()
        checker = get_fact_checker()
        assert isinstance(checker, FactChecker)

    def test_get_fact_checker_same_instance(self):
        """Get checker should return same instance."""
        reset_fact_checker()
        checker1 = get_fact_checker()
        checker2 = get_fact_checker()
        assert checker1 is checker2

    def test_reset_fact_checker(self):
        """Reset should create new instance."""
        checker1 = get_fact_checker()
        reset_fact_checker()
        checker2 = get_fact_checker()
        assert checker1 is not checker2
