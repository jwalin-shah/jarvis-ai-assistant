"""Integration tests for the quality module.

Tests end-to-end quality flow including gates, dashboard, and feedback.
"""


from jarvis.quality.dashboard import (
    get_quality_dashboard,
    reset_quality_dashboard,
)
from jarvis.quality.dimensions import (
    CoherenceScorer,
    LengthScorer,
    MultiDimensionScorer,
    QualityDimension,
    RelevanceScorer,
    ToneScorer,
)
from jarvis.quality.feedback import (
    EditAnalyzer,
    EditType,
    get_feedback_collector,
    reset_feedback_collector,
)
from jarvis.quality.gates import (
    GateDecision,
    QualityGate,
    QualityGateConfig,
    get_quality_gate,
    reset_quality_gate,
)


class TestQualityGateIntegration:
    """Integration tests for QualityGate."""

    def test_gate_with_default_config(self):
        """Test gate with default configuration."""
        gate = QualityGate()
        result = gate.check(
            response="Sure, I can help with that!",
            source="Can you help me with this task?",
        )
        valid_decisions = [GateDecision.PASS, GateDecision.SOFT_FAIL, GateDecision.HARD_FAIL]
        assert result.decision in valid_decisions
        assert result.quality_score >= 0.0
        assert result.quality_score <= 1.0

    def test_gate_with_strict_config(self):
        """Test gate with strict configuration."""
        config = QualityGateConfig.strict()
        gate = QualityGate(config)
        result = gate.check(
            response="Okay.",
            source="Can you explain the project details?",
        )
        assert isinstance(result.decision, GateDecision)

    def test_gate_with_lenient_config(self):
        """Test gate with lenient configuration."""
        config = QualityGateConfig.lenient()
        gate = QualityGate(config)
        result = gate.check(
            response="I'll look into it.",
            source="Any updates?",
        )
        assert isinstance(result.decision, GateDecision)

    def test_gate_fast_check(self):
        """Test fast quality check."""
        gate = QualityGate()
        result = gate.check_fast(
            response="Yes, I can do that.",
            source="Can you help?",
        )
        # Fast check should complete quickly
        assert result.latency_ms < 1000  # Should be under 1 second
        assert isinstance(result.quality_indicator, str)

    def test_gate_result_to_dict(self):
        """Test gate result serialization."""
        gate = QualityGate()
        result = gate.check(
            response="Thanks for letting me know!",
            source="I'll be there at 3pm.",
        )
        d = result.to_dict()
        assert "decision" in d
        assert "quality_score" in d
        assert "gates" in d


class TestDimensionScorersIntegration:
    """Integration tests for quality dimension scorers."""

    def test_coherence_scorer(self):
        """Test coherence scorer."""
        scorer = CoherenceScorer()
        result = scorer.score(
            "This is a well-structured response. It has multiple sentences. They flow logically.",
        )
        assert 0.0 <= result.score <= 1.0
        assert result.dimension == QualityDimension.COHERENCE

    def test_relevance_scorer(self):
        """Test relevance scorer."""
        scorer = RelevanceScorer()
        result = scorer.score(
            "The meeting is at noon.",
            context="What time is the meeting?",
        )
        assert 0.0 <= result.score <= 1.0
        assert result.dimension == QualityDimension.RELEVANCE

    def test_tone_scorer(self):
        """Test tone scorer."""
        scorer = ToneScorer()
        result = scorer.score(
            "Thank you for your inquiry. We appreciate your interest.",
            expected_tone="formal",
        )
        assert 0.0 <= result.score <= 1.0
        assert result.dimension == QualityDimension.TONE

    def test_length_scorer(self):
        """Test length scorer."""
        scorer = LengthScorer()
        result = scorer.score(
            "Yes.",
            context="Yes or no?",
        )
        assert 0.0 <= result.score <= 1.0
        assert result.dimension == QualityDimension.LENGTH

    def test_multi_dimension_scorer(self):
        """Test multi-dimension scorer."""
        scorer = MultiDimensionScorer()
        result = scorer.score_all(
            "Sure, I can help you with that. Let me know what you need.",
            context="Can you help me?",
        )
        assert len(result.results) > 0
        assert 0.0 <= result.overall_score <= 1.0


class TestQualityDashboardIntegration:
    """Integration tests for QualityDashboard."""

    def test_record_quality_check(self):
        """Test recording quality checks."""
        reset_quality_dashboard()
        dashboard = get_quality_dashboard()

        dashboard.record_quality_check(
            dimension_scores={"coherence": 0.8, "relevance": 0.9},
            overall_score=0.85,
            model_name="test_model",
            latency_ms=50.0,
        )

        summary = dashboard.get_summary()
        assert summary["total_checks"] == 1

    def test_get_trends(self):
        """Test getting quality trends."""
        reset_quality_dashboard()
        dashboard = get_quality_dashboard()

        # Record some data
        for i in range(5):
            dashboard.record_quality_check(
                dimension_scores={"coherence": 0.7 + i * 0.02},
                overall_score=0.7 + i * 0.02,
            )

        trends = dashboard.get_trends(days=7)
        assert isinstance(trends, list)

    def test_model_comparison(self):
        """Test model comparison."""
        reset_quality_dashboard()
        dashboard = get_quality_dashboard()

        # Record for multiple models
        dashboard.record_quality_check(
            dimension_scores={"coherence": 0.8},
            overall_score=0.8,
            model_name="model_a",
        )
        dashboard.record_quality_check(
            dimension_scores={"coherence": 0.7},
            overall_score=0.7,
            model_name="model_b",
        )

        comparison = dashboard.get_model_comparison()
        assert len(comparison) == 2

    def test_alerts(self):
        """Test alert generation."""
        reset_quality_dashboard()
        dashboard = get_quality_dashboard()

        # Record a very low quality check to trigger alert
        dashboard.record_quality_check(
            dimension_scores={"coherence": 0.2},
            overall_score=0.2,  # Below critical threshold
        )

        alerts = dashboard.get_alerts()
        # May or may not have alerts depending on thresholds
        assert isinstance(alerts, list)


class TestFeedbackCollectorIntegration:
    """Integration tests for FeedbackCollector."""

    def test_record_acceptance(self):
        """Test recording acceptance."""
        reset_feedback_collector()
        collector = get_feedback_collector()

        collector.record_acceptance(
            original_text="Sure, I can help!",
            quality_scores={"coherence": 0.9},
            contact_id="test_contact",
            model_name="test_model",
        )

        stats = collector.get_stats()
        assert stats.accepted_count == 1

    def test_record_edit(self):
        """Test recording edits."""
        reset_feedback_collector()
        collector = get_feedback_collector()

        entry = collector.record_edit(
            original_text="Sure, I can help with that!",
            edited_text="I can help you.",
            quality_scores={"coherence": 0.9},
            contact_id="test_contact",
        )

        valid_edit_types = [EditType.MINOR, EditType.MODERATE, EditType.MAJOR, EditType.COMPLETE]
        assert entry.edit_type in valid_edit_types
        assert entry.edit_distance > 0

    def test_record_rejection(self):
        """Test recording rejections."""
        reset_feedback_collector()
        collector = get_feedback_collector()

        collector.record_rejection(
            original_text="Some response that was rejected.",
            contact_id="test_contact",
        )

        stats = collector.get_stats()
        assert stats.rejected_count == 1

    def test_record_rating(self):
        """Test recording ratings."""
        reset_feedback_collector()
        collector = get_feedback_collector()

        collector.record_rating(
            original_text="Great response!",
            rating=5,
        )

        stats = collector.get_stats()
        assert stats.avg_rating == 5.0

    def test_get_contact_preferences(self):
        """Test getting contact preferences."""
        reset_feedback_collector()
        collector = get_feedback_collector()

        # Record some edits
        collector.record_edit(
            original_text="This is a longer response with more detail.",
            edited_text="Short reply.",
            contact_id="prefers_short",
        )

        preferences = collector.get_contact_preferences("prefers_short")
        # May have preferences based on patterns
        assert isinstance(preferences, dict)


class TestEditAnalyzer:
    """Tests for EditAnalyzer."""

    def test_analyze_minor_edit(self):
        """Test analyzing minor edit."""
        analyzer = EditAnalyzer()
        edit_type, distance, patterns = analyzer.analyze_edit(
            "Hello there!",
            "Hello there.",
        )
        assert edit_type == EditType.MINOR
        assert distance <= 10

    def test_analyze_major_edit(self):
        """Test analyzing major edit."""
        analyzer = EditAnalyzer()
        edit_type, distance, patterns = analyzer.analyze_edit(
            "This is the original response with lots of content.",
            "Completely different text here.",
        )
        # Significant changes should be at least MODERATE or higher
        assert edit_type in [EditType.MODERATE, EditType.MAJOR, EditType.COMPLETE]

    def test_identify_patterns(self):
        """Test pattern identification."""
        analyzer = EditAnalyzer()
        _, _, patterns = analyzer.analyze_edit(
            "This is a long response with many words.",
            "Short.",
        )
        assert "shortened" in patterns


class TestEndToEndQualityFlow:
    """End-to-end tests for complete quality flow."""

    def test_full_quality_check_flow(self):
        """Test complete quality check flow."""
        # Setup
        reset_quality_gate()
        reset_quality_dashboard()
        reset_feedback_collector()

        gate = get_quality_gate()
        dashboard = get_quality_dashboard()
        collector = get_feedback_collector()

        # Simulate quality check
        result = gate.check(
            response="I'll meet you there at noon.",
            source="Want to meet for lunch?",
        )

        # Record to dashboard
        dashboard.record_quality_check(
            dimension_scores={
                gr.gate_name: gr.score for gr in result.gate_results
            },
            overall_score=result.quality_score,
        )

        # Always record feedback for testing the flow
        # (In production, this would be conditional on user action)
        collector.record_acceptance(
            original_text="I'll meet you there at noon.",
            quality_scores={"overall": result.quality_score},
        )

        # Verify flow
        summary = dashboard.get_summary()
        assert summary["total_checks"] >= 1

        stats = collector.get_stats()
        assert stats.total_feedback >= 1

    def test_quality_improvement_cycle(self):
        """Test quality improvement cycle with feedback."""
        reset_feedback_collector()
        collector = get_feedback_collector()

        # Simulate multiple interactions
        collector.record_acceptance("Good response 1.", quality_scores={"overall": 0.9})
        collector.record_edit(
            "Original response 2.",
            "Better version.",
            quality_scores={"overall": 0.7},
        )
        collector.record_rejection("Bad response 3.", quality_scores={"overall": 0.3})
        collector.record_rating("Rated response.", rating=4)

        stats = collector.get_stats()

        # Check stats are accurate
        assert stats.accepted_count == 1
        assert stats.edited_count == 1
        assert stats.rejected_count == 1
        assert stats.rating_count == 1
        assert stats.acceptance_rate > 0  # Some were accepted/edited
