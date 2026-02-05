"""Tests for contract dataclass validation."""

from datetime import datetime, timezone

import pytest

from contracts import (
    CreateEventResult,
    DetectedEvent,
    GenerationRequest,
    GenerationResponse,
    HHEMBenchmarkResult,
    HHEMResult,
    LatencyBenchmarkResult,
    LatencyResult,
    MemoryMode,
    MemoryProfile,
    MemoryState,
)
from contracts.health import DegradationPolicy
from contracts.imessage import Attachment, AttachmentSummary, Conversation, Message, Reaction


class TestDetectedEventValidation:
    """Test DetectedEvent validation."""

    def test_valid_detected_event(self) -> None:
        """Test creating a valid DetectedEvent."""
        event = DetectedEvent(
            title="Team Meeting",
            start=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
            confidence=0.8,
        )
        assert event.title == "Team Meeting"
        assert event.confidence == 0.8

    def test_confidence_out_of_range(self) -> None:
        """Test that confidence must be 0.0-1.0."""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            DetectedEvent(
                title="Meeting",
                start=datetime.now(tz=timezone.utc),
                confidence=1.5,
            )

    def test_empty_title(self) -> None:
        """Test that title cannot be empty."""
        with pytest.raises(ValueError, match="Event title cannot be empty"):
            DetectedEvent(
                title="   ",
                start=datetime.now(tz=timezone.utc),
            )


class TestCreateEventResultValidation:
    """Test CreateEventResult validation."""

    def test_success_without_event_id(self) -> None:
        """Test that success=True requires event_id."""
        with pytest.raises(ValueError, match="success=True requires event_id"):
            CreateEventResult(success=True, event_id=None)

    def test_failure_without_error(self) -> None:
        """Test that success=False requires error message."""
        with pytest.raises(ValueError, match="success=False requires error message"):
            CreateEventResult(success=False, error=None)

    def test_valid_success(self) -> None:
        """Test valid success result."""
        result = CreateEventResult(success=True, event_id="evt-123")
        assert result.success
        assert result.event_id == "evt-123"

    def test_valid_failure(self) -> None:
        """Test valid failure result."""
        result = CreateEventResult(success=False, error="Permission denied")
        assert not result.success
        assert result.error == "Permission denied"


class TestHHEMResultValidation:
    """Test HHEMResult validation."""

    def test_valid_hhem_result(self) -> None:
        """Test creating a valid HHEMResult."""
        result = HHEMResult(
            model_name="test-model",
            prompt_template="template",
            source_text="source",
            generated_summary="summary",
            hhem_score=0.75,
            timestamp="2024-01-15T10:00:00Z",
        )
        assert result.hhem_score == 0.75

    def test_hhem_score_out_of_range(self) -> None:
        """Test that HHEM score must be 0.0-1.0."""
        with pytest.raises(ValueError, match="HHEM score must be between 0.0 and 1.0"):
            HHEMResult(
                model_name="test",
                prompt_template="template",
                source_text="source",
                generated_summary="summary",
                hhem_score=1.5,
                timestamp="2024-01-15T10:00:00Z",
            )


class TestHHEMBenchmarkResultValidation:
    """Test HHEMBenchmarkResult validation."""

    def test_num_samples_mismatch(self) -> None:
        """Test that num_samples must match len(results)."""
        result = HHEMResult(
            model_name="test",
            prompt_template="template",
            source_text="source",
            generated_summary="summary",
            hhem_score=0.75,
            timestamp="2024-01-15T10:00:00Z",
        )
        with pytest.raises(ValueError, match="num_samples.*!= len\\(results\\)"):
            HHEMBenchmarkResult(
                model_name="test",
                num_samples=10,
                mean_score=0.75,
                median_score=0.75,
                std_score=0.1,
                pass_rate_at_05=80.0,
                pass_rate_at_07=60.0,
                results=[result],
                timestamp="2024-01-15T10:00:00Z",
            )

    def test_pass_rate_out_of_range(self) -> None:
        """Test that pass rates must be 0-100."""
        result = HHEMResult(
            model_name="test",
            prompt_template="template",
            source_text="source",
            generated_summary="summary",
            hhem_score=0.75,
            timestamp="2024-01-15T10:00:00Z",
        )
        with pytest.raises(ValueError, match="pass_rate_at_05 must be 0-100"):
            HHEMBenchmarkResult(
                model_name="test",
                num_samples=1,
                mean_score=0.75,
                median_score=0.75,
                std_score=0.1,
                pass_rate_at_05=150.0,
                pass_rate_at_07=60.0,
                results=[result],
                timestamp="2024-01-15T10:00:00Z",
            )


class TestLatencyResultValidation:
    """Test LatencyResult validation."""

    def test_negative_context_length(self) -> None:
        """Test that context_length must be non-negative."""
        with pytest.raises(ValueError, match="context_length must be >= 0"):
            LatencyResult(
                scenario="cold",
                model_name="test",
                context_length=-1,
                output_tokens=100,
                load_time_ms=100.0,
                prefill_time_ms=50.0,
                generation_time_ms=200.0,
                total_time_ms=350.0,
                tokens_per_second=10.0,
                timestamp="2024-01-15T10:00:00Z",
            )


class TestMemoryProfileValidation:
    """Test MemoryProfile validation."""

    def test_negative_memory(self) -> None:
        """Test that memory values must be non-negative."""
        with pytest.raises(ValueError, match="rss_mb must be >= 0"):
            MemoryProfile(
                model_name="test",
                quantization="4bit",
                context_length=1024,
                rss_mb=-100.0,
                virtual_mb=1000.0,
                metal_mb=500.0,
                load_time_seconds=2.5,
                timestamp="2024-01-15T10:00:00Z",
            )


class TestMemoryStateValidation:
    """Test MemoryState validation."""

    def test_invalid_pressure_level(self) -> None:
        """Test that pressure_level must be valid."""
        with pytest.raises(ValueError, match="pressure_level must be one of"):
            MemoryState(
                available_mb=4096.0,
                used_mb=2048.0,
                model_loaded=True,
                current_mode=MemoryMode.LITE,
                pressure_level="invalid",
            )

    def test_valid_pressure_levels(self) -> None:
        """Test all valid pressure levels."""
        for level in ["green", "yellow", "red", "critical"]:
            state = MemoryState(
                available_mb=4096.0,
                used_mb=2048.0,
                model_loaded=True,
                current_mode=MemoryMode.LITE,
                pressure_level=level,
            )
            assert state.pressure_level == level


class TestGenerationRequestValidation:
    """Test GenerationRequest validation."""

    def test_empty_prompt(self) -> None:
        """Test that prompt cannot be empty."""
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            GenerationRequest(prompt="   ")

    def test_invalid_temperature(self) -> None:
        """Test that temperature must be in valid range."""
        with pytest.raises(ValueError, match="temperature must be 0.0-2.0"):
            GenerationRequest(prompt="test", temperature=3.0)

    def test_invalid_top_p(self) -> None:
        """Test that top_p must be 0.0-1.0."""
        with pytest.raises(ValueError, match="top_p must be 0.0-1.0"):
            GenerationRequest(prompt="test", top_p=1.5)

    def test_invalid_repetition_penalty(self) -> None:
        """Test that repetition_penalty must be >= 1.0."""
        with pytest.raises(ValueError, match="repetition_penalty must be >= 1.0"):
            GenerationRequest(prompt="test", repetition_penalty=0.5)


class TestGenerationResponseValidation:
    """Test GenerationResponse validation."""

    def test_invalid_finish_reason(self) -> None:
        """Test that finish_reason must be valid."""
        with pytest.raises(ValueError, match="finish_reason must be one of"):
            GenerationResponse(
                text="output",
                tokens_used=10,
                generation_time_ms=100.0,
                model_name="test",
                used_template=False,
                template_name=None,
                finish_reason="invalid",
            )

    def test_error_without_message(self) -> None:
        """Test that error finish_reason requires error message."""
        with pytest.raises(ValueError, match='finish_reason "error" requires error message'):
            GenerationResponse(
                text="",
                tokens_used=0,
                generation_time_ms=100.0,
                model_name="test",
                used_template=False,
                template_name=None,
                finish_reason="error",
                error=None,
            )

    def test_template_without_name(self) -> None:
        """Test that used_template=True requires template_name."""
        with pytest.raises(ValueError, match="used_template=True requires template_name"):
            GenerationResponse(
                text="output",
                tokens_used=10,
                generation_time_ms=100.0,
                model_name="test",
                used_template=True,
                template_name=None,
                finish_reason="template",
            )


class TestDegradationPolicyValidation:
    """Test DegradationPolicy validation."""

    def test_empty_feature_name(self) -> None:
        """Test that feature_name cannot be empty."""
        with pytest.raises(ValueError, match="feature_name cannot be empty"):
            DegradationPolicy(
                feature_name="   ",
                health_check=lambda: True,
                degraded_behavior=lambda: None,
                fallback_behavior=lambda: None,
                recovery_check=lambda: True,
            )

    def test_max_failures_too_low(self) -> None:
        """Test that max_failures must be >= 1."""
        with pytest.raises(ValueError, match="max_failures must be >= 1"):
            DegradationPolicy(
                feature_name="test",
                health_check=lambda: True,
                degraded_behavior=lambda: None,
                fallback_behavior=lambda: None,
                recovery_check=lambda: True,
                max_failures=0,
            )


class TestAttachmentValidation:
    """Test Attachment validation."""

    def test_negative_file_size(self) -> None:
        """Test that file_size must be non-negative."""
        with pytest.raises(ValueError, match="file_size must be >= 0"):
            Attachment(
                filename="test.jpg",
                file_path="/path/to/test.jpg",
                mime_type="image/jpeg",
                file_size=-100,
            )

    def test_negative_dimensions(self) -> None:
        """Test that width/height must be non-negative."""
        with pytest.raises(ValueError, match="width must be >= 0"):
            Attachment(
                filename="test.jpg",
                file_path="/path/to/test.jpg",
                mime_type="image/jpeg",
                file_size=1000,
                width=-100,
            )


class TestReactionValidation:
    """Test Reaction validation."""

    def test_invalid_reaction_type(self) -> None:
        """Test that reaction type must be valid."""
        with pytest.raises(ValueError, match="type must be one of"):
            Reaction(
                type="invalid",
                sender="+1234567890",
                sender_name="John",
                date=datetime.now(tz=timezone.utc),
            )

    def test_valid_reaction_types(self) -> None:
        """Test all valid reaction types."""
        valid_types = ["love", "like", "dislike", "laugh", "emphasize", "question"]
        for reaction_type in valid_types:
            reaction = Reaction(
                type=reaction_type,
                sender="+1234567890",
                sender_name="John",
                date=datetime.now(tz=timezone.utc),
            )
            assert reaction.type == reaction_type

    def test_removed_reactions(self) -> None:
        """Test removed_ prefix for reactions."""
        reaction = Reaction(
            type="removed_love",
            sender="+1234567890",
            sender_name="John",
            date=datetime.now(tz=timezone.utc),
        )
        assert reaction.type == "removed_love"


class TestMessageValidation:
    """Test Message validation."""

    def test_negative_message_id(self) -> None:
        """Test that message ID must be non-negative."""
        with pytest.raises(ValueError, match="id must be >= 0"):
            Message(
                id=-1,
                chat_id="chat-123",
                sender="+1234567890",
                sender_name="John",
                text="Hello",
                date=datetime.now(tz=timezone.utc),
                is_from_me=False,
            )


class TestConversationValidation:
    """Test Conversation validation."""

    def test_no_participants(self) -> None:
        """Test that conversation must have participants."""
        with pytest.raises(ValueError, match="Conversation must have at least one participant"):
            Conversation(
                chat_id="chat-123",
                participants=[],
                display_name="Empty Chat",
                last_message_date=datetime.now(tz=timezone.utc),
                message_count=0,
                is_group=False,
            )

    def test_group_with_one_participant(self) -> None:
        """Test that group conversation needs 2+ participants."""
        with pytest.raises(ValueError, match="Group conversation must have >= 2 participants"):
            Conversation(
                chat_id="chat-123",
                participants=["+1234567890"],
                display_name="Invalid Group",
                last_message_date=datetime.now(tz=timezone.utc),
                message_count=10,
                is_group=True,
            )

    def test_valid_group(self) -> None:
        """Test valid group conversation."""
        conv = Conversation(
            chat_id="chat-123",
            participants=["+1234567890", "+0987654321"],
            display_name="Team Chat",
            last_message_date=datetime.now(tz=timezone.utc),
            message_count=100,
            is_group=True,
        )
        assert conv.is_group
        assert len(conv.participants) == 2
