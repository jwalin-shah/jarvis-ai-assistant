"""Tests for contract dataclass validation and serialization."""

from dataclasses import asdict, fields
from datetime import UTC, datetime

import pytest

from contracts import (
    Calendar,
    CalendarEvent,
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
    Permission,
    PermissionStatus,
    SchemaInfo,
)
from contracts.health import DegradationPolicy, FeatureState
from contracts.imessage import Attachment, AttachmentSummary, Conversation, Message, Reaction
from contracts.memory import MemoryProfile
from jarvis.contracts.pipeline import (
    CategoryType,
    ClassificationResult,
    Entity,
    ExtractionResult,
    Fact,
    IntentType,
    MessageContext,
    RAGDocument,
    Relationship,
    UrgencyLevel,
)
from jarvis.contracts.pipeline import (
    GenerationRequest as PipelineGenerationRequest,
)
from jarvis.contracts.pipeline import (
    GenerationResponse as PipelineGenerationResponse,
)


class TestDetectedEventValidation:
    """Test DetectedEvent validation."""

    def test_valid_detected_event(self) -> None:
        """Test creating a valid DetectedEvent."""
        event = DetectedEvent(
            title="Team Meeting",
            start=datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
            confidence=0.8,
        )
        assert event.title == "Team Meeting"
        assert event.confidence == 0.8

    def test_confidence_out_of_range(self) -> None:
        """Test that confidence must be 0.0-1.0."""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            DetectedEvent(
                title="Meeting",
                start=datetime.now(tz=UTC),
                confidence=1.5,
            )

    def test_empty_title(self) -> None:
        """Test that title cannot be empty."""
        with pytest.raises(ValueError, match="Event title cannot be empty"):
            DetectedEvent(
                title="   ",
                start=datetime.now(tz=UTC),
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
                date=datetime.now(tz=UTC),
            )

    def test_valid_reaction_types(self) -> None:
        """Test all valid reaction types."""
        valid_types = ["love", "like", "dislike", "laugh", "emphasize", "question"]
        for reaction_type in valid_types:
            reaction = Reaction(
                type=reaction_type,
                sender="+1234567890",
                sender_name="John",
                date=datetime.now(tz=UTC),
            )
            assert reaction.type == reaction_type

    def test_removed_reactions(self) -> None:
        """Test removed_ prefix for reactions."""
        reaction = Reaction(
            type="removed_love",
            sender="+1234567890",
            sender_name="John",
            date=datetime.now(tz=UTC),
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
                date=datetime.now(tz=UTC),
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
                last_message_date=datetime.now(tz=UTC),
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
                last_message_date=datetime.now(tz=UTC),
                message_count=10,
                is_group=True,
            )

    def test_valid_group(self) -> None:
        """Test valid group conversation."""
        conv = Conversation(
            chat_id="chat-123",
            participants=["+1234567890", "+0987654321"],
            display_name="Team Chat",
            last_message_date=datetime.now(tz=UTC),
            message_count=100,
            is_group=True,
        )
        assert conv.is_group
        assert len(conv.participants) == 2


# ============================================================================
# CONTRACT SERIALIZATION TESTS
# ============================================================================


class TestContractSerialization:
    """Test that all contract dataclasses can be serialized and round-tripped."""

    def test_generation_request_serialization(self) -> None:
        """Test GenerationRequest serialization and round-trip."""
        original = GenerationRequest(
            prompt="Test prompt",
            context_documents=["doc1", "doc2"],
            few_shot_examples=[("input1", "output1")],
            max_tokens=50,
            temperature=0.5,
            top_p=0.9,
            top_k=25,
            repetition_penalty=1.1,
            stop_sequences=["END"],
        )

        # Serialize to dict
        data = asdict(original)
        assert data["prompt"] == "Test prompt"
        assert data["max_tokens"] == 50
        assert data["temperature"] == 0.5
        assert data["stop_sequences"] == ["END"]

        # Round-trip
        restored = GenerationRequest(**data)
        assert restored.prompt == original.prompt
        assert restored.max_tokens == original.max_tokens
        assert restored.stop_sequences == original.stop_sequences

    def test_generation_request_defaults(self) -> None:
        """Test GenerationRequest with default values."""
        req = GenerationRequest(prompt="Simple")
        data = asdict(req)

        # Check defaults are present
        assert data["max_tokens"] == 100
        assert data["temperature"] == 0.1
        assert data["context_documents"] == []
        assert data["stop_sequences"] is None

        # Round-trip
        restored = GenerationRequest(**data)
        assert restored.max_tokens == 100

    def test_generation_response_serialization(self) -> None:
        """Test GenerationResponse serialization and round-trip."""
        original = GenerationResponse(
            text="Generated output",
            tokens_used=25,
            generation_time_ms=150.5,
            model_name="test-model",
            used_template=False,
            template_name=None,
            finish_reason="stop",
        )

        data = asdict(original)
        restored = GenerationResponse(**data)
        assert restored.text == original.text
        assert restored.finish_reason == "stop"

    def test_generation_response_with_error(self) -> None:
        """Test GenerationResponse with error fields."""
        original = GenerationResponse(
            text="",
            tokens_used=0,
            generation_time_ms=100.0,
            model_name="test",
            used_template=False,
            template_name=None,
            finish_reason="error",
            error="Model failed to load",
        )

        data = asdict(original)
        restored = GenerationResponse(**data)
        assert restored.error == "Model failed to load"

    def test_detected_event_serialization(self) -> None:
        """Test DetectedEvent serialization and round-trip."""
        now = datetime.now(tz=UTC)
        original = DetectedEvent(
            title="Meeting",
            start=now,
            end=now,
            location="Conference Room",
            description="Team sync",
            all_day=False,
            confidence=0.85,
            source_text="Let's meet tomorrow",
            message_id=123,
        )

        data = asdict(original)
        restored = DetectedEvent(**data)
        assert restored.title == "Meeting"
        assert restored.confidence == 0.85
        assert restored.location == "Conference Room"

    def test_create_event_result_serialization(self) -> None:
        """Test CreateEventResult serialization."""
        # Success case
        success = CreateEventResult(success=True, event_id="evt-123")
        data = asdict(success)
        restored = CreateEventResult(**data)
        assert restored.success is True
        assert restored.event_id == "evt-123"

        # Failure case
        failure = CreateEventResult(success=False, error="Permission denied")
        data = asdict(failure)
        restored = CreateEventResult(**data)
        assert restored.success is False
        assert restored.error == "Permission denied"

    def test_calendar_event_serialization(self) -> None:
        """Test CalendarEvent serialization."""
        now = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)
        later = datetime(2024, 1, 15, 11, 0, tzinfo=UTC)

        original = CalendarEvent(
            id="evt-1",
            calendar_id="cal-1",
            calendar_name="Work",
            title="Team Meeting",
            start=now,
            end=later,
            all_day=False,
            location="Office",
            notes="Weekly sync",
            url="https://example.com",
            attendees=["a@example.com", "b@example.com"],
            status="confirmed",
        )

        data = asdict(original)
        restored = CalendarEvent(**data)
        assert restored.title == "Team Meeting"
        assert restored.status == "confirmed"
        assert len(restored.attendees) == 2

    def test_calendar_serialization(self) -> None:
        """Test Calendar serialization."""
        original = Calendar(
            id="cal-1",
            name="Personal",
            color="#FF0000",
            is_editable=True,
        )

        data = asdict(original)
        restored = Calendar(**data)
        assert restored.name == "Personal"
        assert restored.color == "#FF0000"

    def test_attachment_summary_serialization(self) -> None:
        """Test AttachmentSummary serialization."""
        original = AttachmentSummary(
            total_count=10,
            total_size_bytes=1024000,
            by_type={"images": 5, "videos": 2},
            size_by_type={"images": 512000, "videos": 512000},
        )

        data = asdict(original)
        restored = AttachmentSummary(**data)
        assert restored.total_count == 10
        assert restored.by_type["images"] == 5

    def test_hhem_result_serialization(self) -> None:
        """Test HHEMResult serialization."""
        original = HHEMResult(
            model_name="test-model",
            prompt_template="template1",
            source_text="Original text",
            generated_summary="Summary",
            hhem_score=0.75,
            timestamp="2024-01-15T10:00:00Z",
        )

        data = asdict(original)
        restored = HHEMResult(**data)
        assert restored.hhem_score == 0.75
        assert restored.model_name == "test-model"

    def test_hhem_benchmark_result_serialization(self) -> None:
        """Test HHEMBenchmarkResult serialization."""
        result = HHEMResult(
            model_name="test",
            prompt_template="template",
            source_text="source",
            generated_summary="summary",
            hhem_score=0.8,
            timestamp="2024-01-15T10:00:00Z",
        )

        original = HHEMBenchmarkResult(
            model_name="test-model",
            num_samples=1,
            mean_score=0.8,
            median_score=0.8,
            std_score=0.0,
            pass_rate_at_05=100.0,
            pass_rate_at_07=100.0,
            results=[result],
            timestamp="2024-01-15T10:00:00Z",
        )

        data = asdict(original)
        restored = HHEMBenchmarkResult(**data)
        assert restored.num_samples == 1
        assert restored.mean_score == 0.8

    def test_latency_result_serialization(self) -> None:
        """Test LatencyResult serialization."""
        original = LatencyResult(
            scenario="warm",
            model_name="test-model",
            context_length=1024,
            output_tokens=100,
            load_time_ms=50.0,
            prefill_time_ms=25.0,
            generation_time_ms=200.0,
            total_time_ms=275.0,
            tokens_per_second=50.0,
            timestamp="2024-01-15T10:00:00Z",
        )

        data = asdict(original)
        restored = LatencyResult(**data)
        assert restored.scenario == "warm"
        assert restored.tokens_per_second == 50.0

    def test_latency_benchmark_result_serialization(self) -> None:
        """Test LatencyBenchmarkResult serialization."""
        result = LatencyResult(
            scenario="cold",
            model_name="test",
            context_length=512,
            output_tokens=50,
            load_time_ms=100.0,
            prefill_time_ms=20.0,
            generation_time_ms=150.0,
            total_time_ms=270.0,
            tokens_per_second=30.0,
            timestamp="2024-01-15T10:00:00Z",
        )

        original = LatencyBenchmarkResult(
            scenario="cold",
            model_name="test-model",
            num_runs=1,
            p50_ms=270.0,
            p95_ms=270.0,
            p99_ms=270.0,
            mean_ms=270.0,
            std_ms=0.0,
            results=[result],
            timestamp="2024-01-15T10:00:00Z",
        )

        data = asdict(original)
        restored = LatencyBenchmarkResult(**data)
        assert restored.num_runs == 1
        assert restored.p50_ms == 270.0

    def test_memory_profile_serialization(self) -> None:
        """Test MemoryProfile serialization."""
        original = MemoryProfile(
            model_name="test-model",
            quantization="4bit",
            context_length=2048,
            rss_mb=2048.5,
            virtual_mb=4096.0,
            metal_mb=1024.0,
            load_time_seconds=3.5,
            timestamp="2024-01-15T10:00:00Z",
        )

        data = asdict(original)
        restored = MemoryProfile(**data)
        assert restored.rss_mb == 2048.5
        assert restored.quantization == "4bit"

    def test_memory_state_serialization(self) -> None:
        """Test MemoryState serialization."""
        original = MemoryState(
            available_mb=4096.0,
            used_mb=2048.0,
            model_loaded=True,
            current_mode=MemoryMode.LITE,
            pressure_level="green",
        )

        data = asdict(original)
        restored = MemoryState(**data)
        assert restored.current_mode == MemoryMode.LITE
        assert restored.pressure_level == "green"

    def test_permission_status_serialization(self) -> None:
        """Test PermissionStatus serialization."""
        original = PermissionStatus(
            permission=Permission.CALENDAR,
            granted=True,
            last_checked="2024-01-15T10:00:00Z",
            fix_instructions="Enable in System Settings",
        )

        data = asdict(original)
        restored = PermissionStatus(**data)
        assert restored.permission == Permission.CALENDAR
        assert restored.granted is True

    def test_schema_info_serialization(self) -> None:
        """Test SchemaInfo serialization."""
        original = SchemaInfo(
            version="v1.0",
            tables=["message", "chat", "attachment"],
            compatible=True,
            migration_needed=False,
            known_schema=True,
        )

        data = asdict(original)
        restored = SchemaInfo(**data)
        assert restored.version == "v1.0"
        assert len(restored.tables) == 3

    def test_attachment_serialization(self) -> None:
        """Test Attachment serialization."""
        original = Attachment(
            filename="image.jpg",
            file_path="/path/to/image.jpg",
            mime_type="image/jpeg",
            file_size=1024000,
            width=1920,
            height=1080,
            duration_seconds=None,
            created_date=datetime.now(tz=UTC),
            is_sticker=False,
            uti="public.jpeg",
        )

        data = asdict(original)
        restored = Attachment(**data)
        assert restored.filename == "image.jpg"
        assert restored.width == 1920

    def test_reaction_serialization(self) -> None:
        """Test Reaction serialization."""
        original = Reaction(
            type="love",
            sender="+1234567890",
            sender_name="John",
            date=datetime.now(tz=UTC),
        )

        data = asdict(original)
        restored = Reaction(**data)
        assert restored.type == "love"
        assert restored.sender == "+1234567890"

    def test_message_serialization(self) -> None:
        """Test Message serialization."""
        attachment = Attachment(
            filename="photo.jpg",
            file_path="/path/to/photo.jpg",
            mime_type="image/jpeg",
            file_size=2048000,
        )
        reaction = Reaction(
            type="like",
            sender="+0987654321",
            sender_name="Jane",
            date=datetime.now(tz=UTC),
        )

        original = Message(
            id=12345,
            chat_id="chat-abc",
            sender="+1234567890",
            sender_name="John",
            text="Hello!",
            date=datetime.now(tz=UTC),
            is_from_me=False,
            attachments=[attachment],
            reply_to_id=None,
            reactions=[reaction],
        )

        data = asdict(original)
        restored = Message(**data)
        assert restored.text == "Hello!"
        assert len(restored.attachments) == 1
        assert len(restored.reactions) == 1

    def test_conversation_serialization(self) -> None:
        """Test Conversation serialization."""
        original = Conversation(
            chat_id="chat-123",
            participants=["+1234567890", "+0987654321"],
            display_name="Team Chat",
            last_message_date=datetime.now(tz=UTC),
            message_count=150,
            is_group=True,
            last_message_text="See you tomorrow!",
        )

        data = asdict(original)
        restored = Conversation(**data)
        assert restored.chat_id == "chat-123"
        assert restored.is_group is True
        assert restored.message_count == 150


class TestPipelineContractSerialization:
    """Test serialization for pipeline contracts (jarvis/contracts/pipeline.py)."""

    def test_intent_type_enum_values(self) -> None:
        """Test IntentType enum values match expected strings."""
        assert IntentType.QUESTION == "question"
        assert IntentType.STATEMENT == "statement"
        assert IntentType.REQUEST == "request"
        assert IntentType.CLARIFICATION == "clarification"
        assert IntentType.GREETING == "greeting"
        assert IntentType.UNKNOWN == "unknown"

    def test_category_type_enum_values(self) -> None:
        """Test CategoryType enum values match expected strings."""
        assert CategoryType.ACKNOWLEDGE == "acknowledge"
        assert CategoryType.CLOSING == "closing"
        assert CategoryType.DEFER == "defer"
        assert CategoryType.FULL_RESPONSE == "full_response"
        assert CategoryType.OFF_TOPIC == "off_topic"

    def test_urgency_level_enum_values(self) -> None:
        """Test UrgencyLevel enum values match expected strings."""
        assert UrgencyLevel.LOW == "low"
        assert UrgencyLevel.MEDIUM == "medium"
        assert UrgencyLevel.HIGH == "high"

    def test_message_context_serialization(self) -> None:
        """Test MessageContext serialization."""
        now = datetime.now(tz=UTC)
        original = MessageContext(
            chat_id="chat-123",
            message_text="Hello, how are you?",
            is_from_me=False,
            timestamp=now,
            sender_id="+1234567890",
            thread_id="thread-456",
            metadata={"source": "imessage", "priority": "high"},
        )

        data = asdict(original)
        restored = MessageContext(**data)
        assert restored.chat_id == "chat-123"
        assert restored.message_text == "Hello, how are you?"
        assert restored.metadata["source"] == "imessage"

    def test_message_context_defaults(self) -> None:
        """Test MessageContext with default values."""
        now = datetime.now(tz=UTC)
        original = MessageContext(
            chat_id="chat-123",
            message_text="Hello",
            is_from_me=True,
            timestamp=now,
        )

        data = asdict(original)
        assert data["sender_id"] is None
        assert data["thread_id"] is None
        assert data["metadata"] == {}

        restored = MessageContext(**data)
        assert restored.sender_id is None

    def test_classification_result_serialization(self) -> None:
        """Test ClassificationResult serialization."""
        original = ClassificationResult(
            intent=IntentType.QUESTION,
            category=CategoryType.FULL_RESPONSE,
            urgency=UrgencyLevel.MEDIUM,
            confidence=0.85,
            requires_knowledge=True,
            metadata={"model": "test", "latency_ms": 50},
        )

        data = asdict(original)
        restored = ClassificationResult(**data)
        assert restored.intent == IntentType.QUESTION
        assert restored.category == CategoryType.FULL_RESPONSE
        assert restored.confidence == 0.85
        assert restored.requires_knowledge is True

    def test_classification_result_defaults(self) -> None:
        """Test ClassificationResult with default values."""
        original = ClassificationResult(
            intent=IntentType.STATEMENT,
            category=CategoryType.ACKNOWLEDGE,
            urgency=UrgencyLevel.LOW,
            confidence=0.95,
            requires_knowledge=False,
        )

        data = asdict(original)
        assert data["metadata"] == {}

        restored = ClassificationResult(**data)
        assert restored.metadata == {}

    def test_entity_serialization(self) -> None:
        """Test Entity serialization."""
        original = Entity(
            name="Apple Inc",
            label="ORG",
            text="Apple",
            start_char=10,
            end_char=15,
        )

        data = asdict(original)
        restored = Entity(**data)
        assert restored.name == "Apple Inc"
        assert restored.label == "ORG"
        assert restored.start_char == 10
        assert restored.end_char == 15

    def test_fact_serialization(self) -> None:
        """Test Fact serialization."""
        original = Fact(
            subject="John",
            predicate="works_at",
            object="Google",
            confidence=0.9,
            source_text="John works at Google",
        )

        data = asdict(original)
        restored = Fact(**data)
        assert restored.subject == "John"
        assert restored.predicate == "works_at"
        assert restored.object == "Google"
        assert restored.confidence == 0.9

    def test_relationship_serialization(self) -> None:
        """Test Relationship serialization."""
        original = Relationship(
            source_entity="John",
            target_entity="Mary",
            relation_type="colleague",
            confidence=0.8,
        )

        data = asdict(original)
        restored = Relationship(**data)
        assert restored.source_entity == "John"
        assert restored.target_entity == "Mary"
        assert restored.relation_type == "colleague"

    def test_extraction_result_serialization(self) -> None:
        """Test ExtractionResult serialization."""
        entity = Entity(name="John", label="PERSON", text="John", start_char=0, end_char=4)
        fact = Fact(
            subject="John",
            predicate="likes",
            object="pizza",
            confidence=0.95,
            source_text="John likes pizza",
        )
        relationship = Relationship(
            source_entity="John",
            target_entity="Mary",
            relation_type="friend",
            confidence=0.85,
        )

        original = ExtractionResult(
            entities=[entity],
            facts=[fact],
            relationships=[relationship],
            topics=["food", "preferences"],
        )

        data = asdict(original)
        restored = ExtractionResult(**data)
        assert len(restored.entities) == 1
        assert len(restored.facts) == 1
        assert len(restored.relationships) == 1
        assert restored.topics == ["food", "preferences"]

    def test_extraction_result_defaults(self) -> None:
        """Test ExtractionResult with default empty lists."""
        original = ExtractionResult()

        data = asdict(original)
        assert data["entities"] == []
        assert data["facts"] == []
        assert data["relationships"] == []
        assert data["topics"] == []

        restored = ExtractionResult(**data)
        assert restored.entities == []

    def test_rag_document_serialization(self) -> None:
        """Test RAGDocument serialization."""
        original = RAGDocument(
            content="This is relevant content",
            source="document.txt",
            score=0.92,
            metadata={"chunk_id": 1, "page": 5},
        )

        data = asdict(original)
        restored = RAGDocument(**data)
        assert restored.content == "This is relevant content"
        assert restored.score == 0.92
        assert restored.metadata["page"] == 5

    def test_rag_document_defaults(self) -> None:
        """Test RAGDocument with default metadata."""
        original = RAGDocument(
            content="Content",
            source="source.txt",
            score=0.8,
        )

        data = asdict(original)
        assert data["metadata"] == {}

        restored = RAGDocument(**data)
        assert restored.metadata == {}

    def test_pipeline_generation_request_structure(self) -> None:
        """Test GenerationRequest structure (not full serialization due to nesting)."""
        now = datetime.now(tz=UTC)
        context = MessageContext(
            chat_id="chat-123",
            message_text="What is the weather?",
            is_from_me=False,
            timestamp=now,
        )
        classification = ClassificationResult(
            intent=IntentType.QUESTION,
            category=CategoryType.FULL_RESPONSE,
            urgency=UrgencyLevel.LOW,
            confidence=0.9,
            requires_knowledge=True,
        )
        extraction = ExtractionResult(
            entities=[Entity(name="User", label="PERSON", text="User", start_char=0, end_char=4)],
            topics=["weather"],
        )
        rag_doc = RAGDocument(content="Weather is sunny", source="weather.com", score=0.95)

        original = PipelineGenerationRequest(
            context=context,
            classification=classification,
            extraction=extraction,
            retrieved_docs=[rag_doc],
            few_shot_examples=[{"input": "Hi", "output": "Hello!"}],
        )

        # Test structure by accessing fields directly
        assert original.context.message_text == "What is the weather?"
        assert original.classification.intent == IntentType.QUESTION
        assert len(original.retrieved_docs) == 1
        assert original.extraction is not None

        # Verify the object can be converted to dict (serialization for API)
        data = asdict(original)
        assert data["context"]["message_text"] == "What is the weather?"
        assert data["classification"]["intent"] == IntentType.QUESTION
        assert len(data["retrieved_docs"]) == 1

    def test_pipeline_generation_response_serialization(self) -> None:
        """Test GenerationResponse from pipeline module."""
        original = PipelineGenerationResponse(
            response="The weather is sunny today.",
            confidence=0.92,
            used_kg_facts=["fact_1", "fact_2"],
            streaming=False,
            metadata={"model": "lfm-1.2b"},
        )

        data = asdict(original)
        restored = PipelineGenerationResponse(**data)
        assert restored.response == "The weather is sunny today."
        assert restored.confidence == 0.92
        assert restored.streaming is False

    def test_pipeline_generation_response_defaults(self) -> None:
        """Test GenerationResponse with default values."""
        original = PipelineGenerationResponse(
            response="Hello!",
            confidence=0.95,
        )

        data = asdict(original)
        assert data["used_kg_facts"] == []
        assert data["streaming"] is False
        assert data["metadata"] == {}

        restored = PipelineGenerationResponse(**data)
        assert restored.used_kg_facts == []


class TestEnumStringValues:
    """Test that enum values match expected string representations."""

    def test_feature_state_enum_values(self) -> None:
        """Test FeatureState enum values."""
        assert FeatureState.HEALTHY.value == "healthy"
        assert FeatureState.DEGRADED.value == "degraded"
        assert FeatureState.FAILED.value == "failed"

    def test_permission_enum_values(self) -> None:
        """Test Permission enum values."""
        assert Permission.FULL_DISK_ACCESS.value == "full_disk_access"
        assert Permission.CONTACTS.value == "contacts"
        assert Permission.CALENDAR.value == "calendar"
        assert Permission.AUTOMATION.value == "automation"

    def test_memory_mode_enum_values(self) -> None:
        """Test MemoryMode enum values."""
        assert MemoryMode.FULL.value == "full"
        assert MemoryMode.LITE.value == "lite"
        assert MemoryMode.MINIMAL.value == "minimal"

    def test_scenario_literal_values(self) -> None:
        """Test Scenario type allows expected values."""
        # Scenario is a Literal type, not an Enum
        valid_scenarios = ["cold", "warm", "hot"]
        for scenario in valid_scenarios:
            # Create a LatencyResult to verify the scenario is valid
            result = LatencyResult(
                scenario=scenario,  # type: ignore[arg-type]
                model_name="test",
                context_length=512,
                output_tokens=50,
                load_time_ms=100.0,
                prefill_time_ms=20.0,
                generation_time_ms=150.0,
                total_time_ms=270.0,
                tokens_per_second=30.0,
                timestamp="2024-01-15T10:00:00Z",
            )
            assert result.scenario == scenario


class TestContractFieldCoverage:
    """Test that all contract dataclass fields are covered by tests."""

    def test_all_pipeline_contracts_have_fields_tested(self) -> None:
        """Verify we have tests for all pipeline contract fields."""
        # This test documents the fields of each pipeline contract
        message_context_fields = {f.name for f in fields(MessageContext)}
        expected_mc_fields = {
            "chat_id",
            "message_text",
            "is_from_me",
            "timestamp",
            "sender_id",
            "thread_id",
            "metadata",
        }
        assert message_context_fields == expected_mc_fields

        classification_fields = {f.name for f in fields(ClassificationResult)}
        expected_cl_fields = {
            "intent",
            "category",
            "urgency",
            "confidence",
            "requires_knowledge",
            "metadata",
        }
        assert classification_fields == expected_cl_fields

        entity_fields = {f.name for f in fields(Entity)}
        expected_entity_fields = {"name", "label", "text", "start_char", "end_char"}
        assert entity_fields == expected_entity_fields

        fact_fields = {f.name for f in fields(Fact)}
        expected_fact_fields = {"subject", "predicate", "object", "confidence", "source_text"}
        assert fact_fields == expected_fact_fields

        relationship_fields = {f.name for f in fields(Relationship)}
        expected_rel_fields = {"source_entity", "target_entity", "relation_type", "confidence"}
        assert relationship_fields == expected_rel_fields

        extraction_fields = {f.name for f in fields(ExtractionResult)}
        expected_extraction_fields = {"entities", "facts", "relationships", "topics"}
        assert extraction_fields == expected_extraction_fields

        rag_fields = {f.name for f in fields(RAGDocument)}
        expected_rag_fields = {"content", "source", "score", "metadata"}
        assert rag_fields == expected_rag_fields

        gen_req_fields = {f.name for f in fields(PipelineGenerationRequest)}
        expected_gen_req_fields = {
            "context",
            "classification",
            "extraction",
            "retrieved_docs",
            "few_shot_examples",
        }
        assert gen_req_fields == expected_gen_req_fields

        gen_res_fields = {f.name for f in fields(PipelineGenerationResponse)}
        expected_gen_res_fields = {
            "response",
            "confidence",
            "used_kg_facts",
            "streaming",
            "metadata",
        }
        assert gen_res_fields == expected_gen_res_fields
