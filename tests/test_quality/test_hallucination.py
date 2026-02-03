"""Tests for hallucination detection module."""


from jarvis.quality.hallucination import (
    EnsembleHallucinationDetector,
    HallucinationResult,
    HallucinationSeverity,
    TokenOverlapAnalyzer,
    get_hallucination_detector,
    reset_hallucination_detector,
)


class TestTokenOverlapAnalyzer:
    """Tests for TokenOverlapAnalyzer."""

    def test_identical_texts_full_overlap(self):
        """Identical texts should have full overlap."""
        analyzer = TokenOverlapAnalyzer()
        score = analyzer.compute_overlap(
            "The quick brown fox jumps over the lazy dog.",
            "The quick brown fox jumps over the lazy dog.",
        )
        assert score == 1.0

    def test_no_overlap(self):
        """Completely different texts should have no overlap."""
        analyzer = TokenOverlapAnalyzer()
        score = analyzer.compute_overlap(
            "The quick brown fox",
            "Completely different words here",
        )
        assert score == 0.0

    def test_partial_overlap(self):
        """Partial overlap should give score between 0 and 1."""
        analyzer = TokenOverlapAnalyzer()
        score = analyzer.compute_overlap(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox runs fast",
        )
        assert 0.0 < score < 1.0

    def test_empty_response(self):
        """Empty response should return 1.0 (not hallucinating nothing)."""
        analyzer = TokenOverlapAnalyzer()
        score = analyzer.compute_overlap("Some source text", "")
        assert score == 1.0

    def test_empty_source(self):
        """Empty source should return 0.0 (no grounding)."""
        analyzer = TokenOverlapAnalyzer()
        score = analyzer.compute_overlap("", "Some response text")
        assert score == 0.0

    def test_min_token_length_filtering(self):
        """Short tokens should be filtered out."""
        analyzer = TokenOverlapAnalyzer(min_token_length=4)
        # "The" and "a" should be filtered
        score = analyzer.compute_overlap(
            "The quick fox",
            "A quick fox",
        )
        # "quick" and "fox" should match
        assert score > 0.5


class TestHallucinationResult:
    """Tests for HallucinationResult dataclass."""

    def test_result_initialization(self):
        """Test basic result initialization."""
        result = HallucinationResult(
            hallucination_score=0.3,
            hhem_score=0.7,
            severity=HallucinationSeverity.LOW,
            passes_gate=True,
            latency_ms=50.0,
        )
        assert result.hallucination_score == 0.3
        assert result.hhem_score == 0.7
        assert result.severity == HallucinationSeverity.LOW
        assert result.passes_gate is True

    def test_result_to_dict(self):
        """Test conversion to dictionary."""
        result = HallucinationResult(
            hallucination_score=0.5,
            severity=HallucinationSeverity.MEDIUM,
            passes_gate=True,
        )
        d = result.to_dict()
        assert "hallucination_score" in d
        assert "severity" in d
        assert d["severity"] == "medium"


class TestEnsembleHallucinationDetector:
    """Tests for EnsembleHallucinationDetector."""

    def test_detector_initialization(self):
        """Test detector initialization with defaults."""
        detector = EnsembleHallucinationDetector()
        assert detector._gate_threshold == 0.5

    def test_detector_custom_threshold(self):
        """Test detector with custom threshold."""
        detector = EnsembleHallucinationDetector(gate_threshold=0.7)
        assert detector._gate_threshold == 0.7

    def test_detect_grounded_response(self):
        """Test detection of grounded response."""
        detector = EnsembleHallucinationDetector(
            enable_hhem=False,  # Disable slow models for fast test
            enable_nli=False,
            enable_similarity=False,
            enable_overlap=True,
        )
        result = detector.detect(
            source="What time is the meeting tomorrow?",
            response="The meeting is tomorrow, I'll check the time.",
        )
        assert isinstance(result, HallucinationResult)
        assert result.latency_ms >= 0

    def test_detect_hallucinated_response(self):
        """Test detection of potentially hallucinated response."""
        detector = EnsembleHallucinationDetector(
            enable_hhem=False,
            enable_nli=False,
            enable_similarity=False,
            enable_overlap=True,
        )
        result = detector.detect(
            source="What time is the meeting?",
            response="John said to meet at the new restaurant downtown at 3pm.",
        )
        assert isinstance(result, HallucinationResult)
        # Response introduces new entities not in source
        assert result.overlap_score is not None

    def test_detect_batch(self):
        """Test batch detection."""
        detector = EnsembleHallucinationDetector(
            enable_hhem=False,
            enable_nli=False,
            enable_similarity=False,
            enable_overlap=True,
        )
        pairs = [
            ("Hello, how are you?", "I'm doing well, thanks!"),
            ("What's the weather?", "It's sunny today."),
        ]
        results = detector.detect_batch(pairs)
        assert len(results) == 2
        assert all(isinstance(r, HallucinationResult) for r in results)

    def test_detect_batch_empty(self):
        """Test batch detection with empty list."""
        detector = EnsembleHallucinationDetector()
        results = detector.detect_batch([])
        assert results == []

    def test_severity_classification(self):
        """Test severity classification from scores."""
        detector = EnsembleHallucinationDetector()

        # Test different severity levels
        assert detector._classify_severity(0.1) == HallucinationSeverity.NONE
        assert detector._classify_severity(0.3) == HallucinationSeverity.LOW
        assert detector._classify_severity(0.5) == HallucinationSeverity.MEDIUM
        assert detector._classify_severity(0.7) == HallucinationSeverity.HIGH
        assert detector._classify_severity(0.9) == HallucinationSeverity.CRITICAL


class TestGlobalSingleton:
    """Tests for global singleton pattern."""

    def test_get_hallucination_detector_returns_instance(self):
        """Get detector should return an instance."""
        reset_hallucination_detector()
        detector = get_hallucination_detector()
        assert isinstance(detector, EnsembleHallucinationDetector)

    def test_get_hallucination_detector_same_instance(self):
        """Get detector should return same instance."""
        reset_hallucination_detector()
        detector1 = get_hallucination_detector()
        detector2 = get_hallucination_detector()
        assert detector1 is detector2

    def test_reset_hallucination_detector(self):
        """Reset should create new instance."""
        detector1 = get_hallucination_detector()
        reset_hallucination_detector()
        detector2 = get_hallucination_detector()
        assert detector1 is not detector2
