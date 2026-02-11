"""Integration boundary tests between Lane B (ML) and Lane A (App).

These tests verify that the outputs of Lane B's functions (classification,
extraction) match what Lane A expects as inputs.

This is a contract test - it ensures the interface between lanes remains stable.
"""

from datetime import UTC, datetime

from jarvis.contracts.pipeline import (
    CategoryType,
    ClassificationResult,
    Entity,
    ExtractionResult,
    Fact,
    GenerationRequest,
    GenerationResponse,
    IntentType,
    MessageContext,
    RAGDocument,
    Relationship,
    UrgencyLevel,
)


class TestClassificationBoundary:
    """Test classification output boundary between Lane B and Lane A."""

    def test_classification_result_has_required_fields(self) -> None:
        """Verify ClassificationResult has all fields Lane A expects."""
        result = ClassificationResult(
            intent=IntentType.QUESTION,
            category=CategoryType.FULL_RESPONSE,
            urgency=UrgencyLevel.HIGH,
            confidence=0.92,
            requires_knowledge=True,
            metadata={"model": "test-model"},
        )

        # All these fields are required by Lane A
        assert result.intent is not None
        assert result.category is not None
        assert result.urgency is not None
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.requires_knowledge, bool)
        assert isinstance(result.metadata, dict)

    def test_all_intent_types_are_strings(self) -> None:
        """Verify all IntentType values are valid strings."""
        for intent in IntentType:
            result = ClassificationResult(
                intent=intent,
                category=CategoryType.FULL_RESPONSE,
                urgency=UrgencyLevel.MEDIUM,
                confidence=0.8,
                requires_knowledge=False,
            )
            # Lane A expects string values
            assert isinstance(result.intent.value, str)
            assert result.intent.value in [
                "question",
                "statement",
                "request",
                "clarification",
                "greeting",
                "unknown",
            ]

    def test_all_category_types_are_strings(self) -> None:
        """Verify all CategoryType values are valid strings."""
        for category in CategoryType:
            result = ClassificationResult(
                intent=IntentType.STATEMENT,
                category=category,
                urgency=UrgencyLevel.LOW,
                confidence=0.85,
                requires_knowledge=True,
            )
            assert isinstance(result.category.value, str)
            assert result.category.value in [
                "acknowledge",
                "closing",
                "defer",
                "full_response",
                "off_topic",
            ]

    def test_all_urgency_levels_are_strings(self) -> None:
        """Verify all UrgencyLevel values are valid strings."""
        for urgency in UrgencyLevel:
            result = ClassificationResult(
                intent=IntentType.REQUEST,
                category=CategoryType.DEFER,
                urgency=urgency,
                confidence=0.75,
                requires_knowledge=True,
            )
            assert isinstance(result.urgency.value, str)
            assert result.urgency.value in ["low", "medium", "high"]

    def test_classification_result_serialization_for_api(self) -> None:
        """Test that ClassificationResult can be serialized for API transfer."""
        from dataclasses import asdict

        result = ClassificationResult(
            intent=IntentType.QUESTION,
            category=CategoryType.FULL_RESPONSE,
            urgency=UrgencyLevel.HIGH,
            confidence=0.95,
            requires_knowledge=True,
            metadata={"latency_ms": 45, "cache_hit": True},
        )

        # Serialize as would happen in API
        data = asdict(result)

        # Verify structure matches what Lane A expects
        assert data["intent"] == IntentType.QUESTION
        assert data["category"] == CategoryType.FULL_RESPONSE
        assert data["urgency"] == UrgencyLevel.HIGH
        assert data["confidence"] == 0.95
        assert data["requires_knowledge"] is True
        assert data["metadata"]["latency_ms"] == 45


class TestExtractionBoundary:
    """Test extraction output boundary between Lane B and Lane A."""

    def test_extraction_result_has_required_fields(self) -> None:
        """Verify ExtractionResult has all fields Lane A expects."""
        result = ExtractionResult()

        # All these lists are required by Lane A
        assert isinstance(result.entities, list)
        assert isinstance(result.facts, list)
        assert isinstance(result.relationships, list)
        assert isinstance(result.topics, list)

    def test_entity_structure(self) -> None:
        """Verify Entity has the structure Lane A expects."""
        entity = Entity(
            name="John Smith",
            label="PERSON",
            text="John",
            start_char=10,
            end_char=14,
        )

        # Lane A expects these exact fields
        assert entity.name == "John Smith"
        assert entity.label == "PERSON"
        assert entity.text == "John"
        assert entity.start_char == 10
        assert entity.end_char == 14

    def test_fact_structure(self) -> None:
        """Verify Fact has the structure Lane A expects."""
        fact = Fact(
            subject="John",
            predicate="works_at",
            object="Google",
            confidence=0.88,
            source_text="John works at Google",
        )

        # Lane A expects these exact fields
        assert fact.subject == "John"
        assert fact.predicate == "works_at"
        assert fact.object == "Google"
        assert 0.0 <= fact.confidence <= 1.0
        assert fact.source_text == "John works at Google"

    def test_relationship_structure(self) -> None:
        """Verify Relationship has the structure Lane A expects."""
        relationship = Relationship(
            source_entity="John",
            target_entity="Mary",
            relation_type="colleague",
            confidence=0.75,
        )

        # Lane A expects these exact fields
        assert relationship.source_entity == "John"
        assert relationship.target_entity == "Mary"
        assert relationship.relation_type == "colleague"
        assert 0.0 <= relationship.confidence <= 1.0

    def test_extraction_with_sample_data(self) -> None:
        """Test extraction result with realistic sample data."""
        result = ExtractionResult(
            entities=[
                Entity(name="Alice", label="PERSON", text="Alice", start_char=0, end_char=5),
                Entity(name="New York", label="GPE", text="New York", start_char=20, end_char=28),
            ],
            facts=[
                Fact(
                    subject="Alice",
                    predicate="lives_in",
                    object="New York",
                    confidence=0.9,
                    source_text="Alice lives in New York",
                ),
            ],
            relationships=[
                Relationship(
                    source_entity="Alice",
                    target_entity="Bob",
                    relation_type="friend",
                    confidence=0.85,
                ),
            ],
            topics=["location", "personal"],
        )

        # Verify Lane A can access all the data it needs
        assert len(result.entities) == 2
        assert len(result.facts) == 1
        assert len(result.relationships) == 1
        assert "location" in result.topics

    def test_empty_extraction_is_valid(self) -> None:
        """Test that empty extraction result is valid (no entities/facts found)."""
        result = ExtractionResult()

        # Lane A should handle empty results gracefully
        assert result.entities == []
        assert result.facts == []
        assert result.relationships == []
        assert result.topics == []


class TestMessageContextBoundary:
    """Test MessageContext input boundary from Lane A to Lane B."""

    def test_message_context_has_required_fields(self) -> None:
        """Verify MessageContext has all fields Lane B expects."""
        now = datetime.now(tz=UTC)
        context = MessageContext(
            chat_id="chat-123",
            message_text="What time is the meeting?",
            is_from_me=False,
            timestamp=now,
            sender_id="+1234567890",
            thread_id="thread-456",
            metadata={"source": "imessage"},
        )

        # Lane B expects these fields
        assert context.chat_id == "chat-123"
        assert context.message_text == "What time is the meeting?"
        assert context.is_from_me is False
        assert context.timestamp == now

    def test_message_context_optional_fields(self) -> None:
        """Test MessageContext with minimal required fields."""
        now = datetime.now(tz=UTC)
        context = MessageContext(
            chat_id="chat-123",
            message_text="Hello",
            is_from_me=True,
            timestamp=now,
        )

        # Optional fields should default appropriately
        assert context.sender_id is None
        assert context.thread_id is None
        assert context.metadata == {}


class TestRAGDocumentBoundary:
    """Test RAGDocument boundary for retrieval results."""

    def test_rag_document_structure(self) -> None:
        """Verify RAGDocument has the structure Lane A expects."""
        doc = RAGDocument(
            content="This is the retrieved content",
            source="knowledge_base.txt",
            score=0.95,
            metadata={"chunk_id": 42, "relevance": "high"},
        )

        # Lane A expects these exact fields
        assert doc.content == "This is the retrieved content"
        assert doc.source == "knowledge_base.txt"
        assert doc.score == 0.95
        assert doc.metadata["chunk_id"] == 42

    def test_rag_document_score_range(self) -> None:
        """Test that RAG document scores are in expected range."""
        for score in [0.0, 0.5, 1.0]:
            doc = RAGDocument(
                content="Content",
                source="source.txt",
                score=score,
            )
            assert 0.0 <= doc.score <= 1.0


class TestGenerationRequestBoundary:
    """Test GenerationRequest boundary for the complete pipeline."""

    def test_generation_request_structure(self) -> None:
        """Verify GenerationRequest has the structure generator expects."""
        now = datetime.now(tz=UTC)

        request = GenerationRequest(
            context=MessageContext(
                chat_id="chat-123",
                message_text="What is AI?",
                is_from_me=False,
                timestamp=now,
            ),
            classification=ClassificationResult(
                intent=IntentType.QUESTION,
                category=CategoryType.FULL_RESPONSE,
                urgency=UrgencyLevel.MEDIUM,
                confidence=0.9,
                requires_knowledge=True,
            ),
            extraction=ExtractionResult(
                entities=[Entity(name="AI", label="TECH", text="AI", start_char=8, end_char=10)],
                topics=["technology"],
            ),
            retrieved_docs=[
                RAGDocument(
                    content="AI is artificial intelligence",
                    source="wiki.txt",
                    score=0.95,
                ),
            ],
            few_shot_examples=[
                {"input": "What is ML?", "output": "ML is machine learning"},
            ],
        )

        # Verify all components are accessible
        assert request.context.message_text == "What is AI?"
        assert request.classification.intent == IntentType.QUESTION
        assert len(request.extraction.entities) == 1
        assert len(request.retrieved_docs) == 1
        assert len(request.few_shot_examples) == 1

    def test_generation_request_optional_extraction(self) -> None:
        """Test GenerationRequest with optional extraction as None."""
        now = datetime.now(tz=UTC)

        request = GenerationRequest(
            context=MessageContext(
                chat_id="chat-123",
                message_text="Hi",
                is_from_me=False,
                timestamp=now,
            ),
            classification=ClassificationResult(
                intent=IntentType.GREETING,
                category=CategoryType.ACKNOWLEDGE,
                urgency=UrgencyLevel.LOW,
                confidence=0.95,
                requires_knowledge=False,
            ),
            extraction=None,  # Optional field
        )

        assert request.extraction is None


class TestGenerationResponseBoundary:
    """Test GenerationResponse boundary from generator to Lane A."""

    def test_generation_response_structure(self) -> None:
        """Verify GenerationResponse has the structure Lane A expects."""
        response = GenerationResponse(
            response="This is the generated response",
            confidence=0.92,
            used_kg_facts=["fact_1", "fact_2"],
            streaming=False,
            metadata={"model": "lfm-1.2b", "tokens": 25},
        )

        # Lane A expects these exact fields
        assert response.response == "This is the generated response"
        assert response.confidence == 0.92
        assert "fact_1" in response.used_kg_facts
        assert response.streaming is False
        assert response.metadata["model"] == "lfm-1.2b"

    def test_generation_response_defaults(self) -> None:
        """Test GenerationResponse with default values."""
        response = GenerationResponse(
            response="Hello!",
            confidence=0.95,
        )

        # Defaults
        assert response.used_kg_facts == []
        assert response.streaming is False
        assert response.metadata == {}


class TestMockClassificationFlow:
    """Test mock classification flow simulating Lane B outputs."""

    def test_mock_classify_returns_valid_result(self) -> None:
        """Mock classify() returning a valid ClassificationResult.

        This simulates what Lane B's classify() function should produce
        for Lane A to consume.
        """

        def mock_classify(message_context: MessageContext) -> ClassificationResult:
            """Mock classifier that returns a valid result."""
            return ClassificationResult(
                intent=IntentType.QUESTION,
                category=CategoryType.FULL_RESPONSE,
                urgency=UrgencyLevel.MEDIUM,
                confidence=0.87,
                requires_knowledge=True,
                metadata={"classifier": "mock", "latency_ms": 25},
            )

        # Create a mock MessageContext
        now = datetime.now(tz=UTC)
        context = MessageContext(
            chat_id="chat-123",
            message_text="What is the weather?",
            is_from_me=False,
            timestamp=now,
        )

        # Pass through mock classify()
        result = mock_classify(context)

        # Verify we get a valid ClassificationResult
        assert isinstance(result, ClassificationResult)
        assert result.intent == IntentType.QUESTION
        assert 0.0 <= result.confidence <= 1.0

    def test_mock_extraction_returns_valid_result(self) -> None:
        """Mock extract() returning a valid ExtractionResult.

        This simulates what Lane B's extraction should produce
        for Lane A to consume.
        """

        def mock_extract(message_context: MessageContext) -> ExtractionResult:
            """Mock extractor that returns a valid result."""
            return ExtractionResult(
                entities=[
                    Entity(
                        name="Weather", label="TOPIC", text="weather", start_char=12, end_char=19
                    ),
                ],
                facts=[],
                relationships=[],
                topics=["weather", "query"],
            )

        now = datetime.now(tz=UTC)
        context = MessageContext(
            chat_id="chat-123",
            message_text="What is the weather?",
            is_from_me=False,
            timestamp=now,
        )

        # Pass through mock extract()
        result = mock_extract(context)

        # Verify we get a valid ExtractionResult
        assert isinstance(result, ExtractionResult)
        assert len(result.entities) == 1
        assert result.entities[0].label == "TOPIC"


class TestContractTypeValidation:
    """Test type validation for contract fields."""

    def test_confidence_must_be_float(self) -> None:
        """Verify confidence fields accept float values."""
        result = ClassificationResult(
            intent=IntentType.STATEMENT,
            category=CategoryType.ACKNOWLEDGE,
            urgency=UrgencyLevel.LOW,
            confidence=0.95,  # float
            requires_knowledge=False,
        )
        assert isinstance(result.confidence, float)

    def test_timestamp_must_be_datetime(self) -> None:
        """Verify timestamp fields accept datetime objects."""
        now = datetime.now(tz=UTC)
        context = MessageContext(
            chat_id="chat-123",
            message_text="Hello",
            is_from_me=False,
            timestamp=now,
        )
        assert isinstance(context.timestamp, datetime)

    def test_metadata_must_be_dict(self) -> None:
        """Verify metadata fields accept dict values."""
        result = ClassificationResult(
            intent=IntentType.UNKNOWN,
            category=CategoryType.OFF_TOPIC,
            urgency=UrgencyLevel.LOW,
            confidence=0.5,
            requires_knowledge=False,
            metadata={"key": "value", "number": 42, "flag": True},
        )
        assert isinstance(result.metadata, dict)
        assert result.metadata["key"] == "value"
