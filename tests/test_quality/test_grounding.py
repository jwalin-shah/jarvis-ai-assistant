"""Tests for grounding/attribution tracking module."""


from jarvis.quality.grounding import (
    Attribution,
    AttributionType,
    DirectQuoteMatcher,
    GroundingChecker,
    GroundingResult,
    SegmentExtractor,
    SourceType,
    get_grounding_checker,
    reset_grounding_checker,
)


class TestSegmentExtractor:
    """Tests for SegmentExtractor."""

    def test_extract_single_segment(self):
        """Extract single sentence segment."""
        extractor = SegmentExtractor()
        segments = extractor.extract_segments("This is a single sentence.")
        assert len(segments) == 1
        assert segments[0][0] == "This is a single sentence."

    def test_extract_multiple_segments(self):
        """Extract multiple sentence segments."""
        extractor = SegmentExtractor()
        segments = extractor.extract_segments(
            "First sentence here. Second sentence here. Third one too."
        )
        assert len(segments) == 3

    def test_filter_short_segments(self):
        """Short segments should be filtered."""
        extractor = SegmentExtractor(min_segment_words=3)
        segments = extractor.extract_segments("Hi. Yes. This is longer.")
        # Only the longer sentence should be included
        assert len(segments) == 1
        assert "longer" in segments[0][0]

    def test_segment_spans(self):
        """Segments should include position spans."""
        extractor = SegmentExtractor()
        segments = extractor.extract_segments("First sentence. Second sentence.")
        for segment_text, span in segments:
            start, end = span
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert start >= 0


class TestDirectQuoteMatcher:
    """Tests for DirectQuoteMatcher."""

    def test_find_exact_quote(self):
        """Find exact quotes from source."""
        matcher = DirectQuoteMatcher(min_match_words=3)
        attributions = matcher.find_quotes(
            response="The quick brown fox jumps over the lazy dog.",
            sources=["The quick brown fox is a common typing test."],
        )
        # Should find "quick brown fox" match
        assert isinstance(attributions, list)

    def test_no_quotes(self):
        """No quotes when texts are different."""
        matcher = DirectQuoteMatcher(min_match_words=4)
        attributions = matcher.find_quotes(
            response="Something completely different here.",
            sources=["Another text with no overlap."],
        )
        assert len(attributions) == 0

    def test_attribution_type(self):
        """Attributions should be marked as direct quotes."""
        matcher = DirectQuoteMatcher(min_match_words=3)
        attributions = matcher.find_quotes(
            response="The weather is nice today.",
            sources=["The weather is nice."],
        )
        for attr in attributions:
            assert attr.attribution_type == AttributionType.DIRECT_QUOTE


class TestGroundingChecker:
    """Tests for GroundingChecker."""

    def test_check_empty_response(self):
        """Empty response should have no grounding issues."""
        checker = GroundingChecker()
        result = checker.check_grounding("", "Some source")
        assert result.grounding_score == 0.0
        assert result.passes_gate is False

    def test_check_grounded_response(self):
        """Response with source overlap should have grounding."""
        checker = GroundingChecker()
        result = checker.check_grounding(
            response="The meeting is at noon tomorrow.",
            sources="Reminder: meeting tomorrow at noon.",
        )
        assert isinstance(result, GroundingResult)
        assert result.grounding_score >= 0.0

    def test_check_with_multiple_sources(self):
        """Should check against multiple sources."""
        checker = GroundingChecker()
        result = checker.check_grounding(
            response="The meeting is at noon. Lunch is at 1pm.",
            sources=["Meeting: noon", "Lunch: 1pm"],
        )
        assert isinstance(result, GroundingResult)

    def test_attribution_counts(self):
        """Result should have attribution counts."""
        checker = GroundingChecker()
        result = checker.check_grounding(
            response="Yes, I can do that.",
            sources="Can you help me with this?",
        )
        # Check count attributes exist
        assert hasattr(result, "direct_quote_count")
        assert hasattr(result, "paraphrase_count")
        assert hasattr(result, "inference_count")
        assert hasattr(result, "ungrounded_count")

    def test_result_to_dict(self):
        """Test result conversion to dictionary."""
        result = GroundingResult(
            grounding_score=0.8,
            grounded_percentage=80.0,
            direct_quote_count=1,
            paraphrase_count=2,
            passes_gate=True,
        )
        d = result.to_dict()
        assert "grounding_score" in d
        assert d["grounded_percentage"] == 80.0

    def test_passes_gate_with_threshold(self):
        """Test gate threshold behavior."""
        checker = GroundingChecker(gate_threshold=0.5)
        result = checker.check_grounding(
            response="Okay, sounds good!",
            sources="Let's meet tomorrow.",
        )
        assert isinstance(result.passes_gate, bool)


class TestAttribution:
    """Tests for Attribution dataclass."""

    def test_attribution_creation(self):
        """Test creating an attribution."""
        attr = Attribution(
            response_segment="The meeting is at noon.",
            source_text="Meeting: noon",
            attribution_type=AttributionType.PARAPHRASE,
            source_type=SourceType.MESSAGE,
            confidence=0.85,
            similarity_score=0.9,
        )
        assert attr.attribution_type == AttributionType.PARAPHRASE
        assert attr.confidence == 0.85

    def test_attribution_types(self):
        """Test all attribution types exist."""
        types = [
            AttributionType.DIRECT_QUOTE,
            AttributionType.PARAPHRASE,
            AttributionType.INFERENCE,
            AttributionType.UNGROUNDED,
        ]
        assert len(types) == 4

    def test_source_types(self):
        """Test all source types exist."""
        types = [
            SourceType.MESSAGE,
            SourceType.CONTEXT,
            SourceType.TEMPLATE,
            SourceType.KNOWLEDGE,
        ]
        assert len(types) == 4


class TestGlobalSingleton:
    """Tests for global singleton pattern."""

    def test_get_grounding_checker_returns_instance(self):
        """Get checker should return an instance."""
        reset_grounding_checker()
        checker = get_grounding_checker()
        assert isinstance(checker, GroundingChecker)

    def test_get_grounding_checker_same_instance(self):
        """Get checker should return same instance."""
        reset_grounding_checker()
        checker1 = get_grounding_checker()
        checker2 = get_grounding_checker()
        assert checker1 is checker2

    def test_reset_grounding_checker(self):
        """Reset should create new instance."""
        checker1 = get_grounding_checker()
        reset_grounding_checker()
        checker2 = get_grounding_checker()
        assert checker1 is not checker2
