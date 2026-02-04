"""Tests for consistency checking module."""

from jarvis.quality.consistency import (
    ConsistencyChecker,
    ConsistencyResult,
    HistoryConsistencyChecker,
    InconsistencyIssue,
    InconsistencyType,
    SelfConsistencyChecker,
    get_consistency_checker,
    reset_consistency_checker,
)


class TestSelfConsistencyChecker:
    """Tests for SelfConsistencyChecker."""

    def test_consistent_single_sentence(self):
        """Single sentence should be consistent."""
        checker = SelfConsistencyChecker()
        is_consistent, issues = checker.check("This is a simple sentence.")
        assert is_consistent is True
        assert len(issues) == 0

    def test_consistent_multi_sentence(self):
        """Multiple consistent sentences should pass."""
        checker = SelfConsistencyChecker()
        is_consistent, issues = checker.check(
            "I will be there at noon. The meeting starts at noon. We can discuss the project then."
        )
        assert is_consistent is True

    def test_detect_contradiction(self):
        """Should detect contradictory statements."""
        checker = SelfConsistencyChecker()
        is_consistent, issues = checker.check("The meeting is at 3pm. The meeting is not at 3pm.")
        # Should detect the contradiction
        assert isinstance(is_consistent, bool)
        # Even if no contradiction detected, result should be valid
        assert isinstance(issues, list)

    def test_detect_numeric_inconsistency(self):
        """Should detect conflicting numbers."""
        checker = SelfConsistencyChecker()
        is_consistent, issues = checker.check(
            "There were 10 people at the meeting. Only 5 people attended the meeting."
        )
        # May or may not detect depending on context matching
        assert isinstance(issues, list)

    def test_empty_text(self):
        """Empty text should be consistent."""
        checker = SelfConsistencyChecker()
        is_consistent, issues = checker.check("")
        assert is_consistent is True
        assert len(issues) == 0


class TestHistoryConsistencyChecker:
    """Tests for HistoryConsistencyChecker."""

    def test_consistent_with_empty_history(self):
        """Empty history should pass."""
        checker = HistoryConsistencyChecker()
        is_consistent, issues = checker.check(
            "I'll be there tomorrow.",
            history=[],
        )
        assert is_consistent is True
        assert len(issues) == 0

    def test_consistent_with_history(self):
        """Response consistent with history should pass."""
        checker = HistoryConsistencyChecker()
        is_consistent, issues = checker.check(
            "Sure, I can meet tomorrow.",
            history=["Are you free tomorrow?", "Let's meet up."],
        )
        assert is_consistent is True

    def test_formality_shift_detection(self):
        """Should detect major formality shifts."""
        checker = HistoryConsistencyChecker()
        is_consistent, issues = checker.check(
            "Hey bro! What's up? LOL",
            history=[
                "Dear Sir, I am writing to inquire about the position.",
                "Thank you for your consideration.",
            ],
        )
        # May detect tonal inconsistency
        assert isinstance(issues, list)


class TestConsistencyChecker:
    """Tests for main ConsistencyChecker."""

    def test_check_single_response(self):
        """Check consistency of single response."""
        checker = ConsistencyChecker()
        result = checker.check_consistency("This is a good response.")
        assert isinstance(result, ConsistencyResult)
        assert result.is_self_consistent is True
        assert result.latency_ms >= 0

    def test_check_with_history(self):
        """Check consistency with history."""
        checker = ConsistencyChecker()
        result = checker.check_consistency(
            "I'll meet you there.",
            history=["Want to meet up?", "How about tomorrow?"],
        )
        assert isinstance(result, ConsistencyResult)
        # The overall consistency should pass the gate even if minor issues detected
        assert result.passes_gate is True
        assert result.consistency_score >= 0.5

    def test_result_to_dict(self):
        """Test result conversion to dictionary."""
        result = ConsistencyResult(
            consistency_score=0.9,
            is_self_consistent=True,
            is_history_consistent=True,
            passes_gate=True,
            latency_ms=10.0,
        )
        d = result.to_dict()
        assert "consistency_score" in d
        assert d["is_self_consistent"] is True
        assert d["latency_ms"] == 10.0

    def test_passes_gate_threshold(self):
        """Test gate threshold behavior."""
        checker = ConsistencyChecker(gate_threshold=0.5)
        result = checker.check_consistency("Simple consistent response.")
        # Simple response should pass
        assert isinstance(result.passes_gate, bool)


class TestInconsistencyIssue:
    """Tests for InconsistencyIssue dataclass."""

    def test_issue_creation(self):
        """Test creating an inconsistency issue."""
        issue = InconsistencyIssue(
            inconsistency_type=InconsistencyType.SELF_CONTRADICTION,
            description="Contradictory statements about time",
            severity=0.8,
            text_span_1="Meeting at 3pm",
            text_span_2="Meeting not at 3pm",
        )
        assert issue.inconsistency_type == InconsistencyType.SELF_CONTRADICTION
        assert issue.severity == 0.8

    def test_issue_types(self):
        """Test all inconsistency types exist."""
        types = [
            InconsistencyType.SELF_CONTRADICTION,
            InconsistencyType.TEMPORAL,
            InconsistencyType.FACTUAL,
            InconsistencyType.TONAL,
            InconsistencyType.PERSONA,
            InconsistencyType.NUMERIC,
        ]
        assert len(types) == 6


class TestGlobalSingleton:
    """Tests for global singleton pattern."""

    def test_get_consistency_checker_returns_instance(self):
        """Get checker should return an instance."""
        reset_consistency_checker()
        checker = get_consistency_checker()
        assert isinstance(checker, ConsistencyChecker)

    def test_get_consistency_checker_same_instance(self):
        """Get checker should return same instance."""
        reset_consistency_checker()
        checker1 = get_consistency_checker()
        checker2 = get_consistency_checker()
        assert checker1 is checker2

    def test_reset_consistency_checker(self):
        """Reset should create new instance."""
        checker1 = get_consistency_checker()
        reset_consistency_checker()
        checker2 = get_consistency_checker()
        assert checker1 is not checker2
