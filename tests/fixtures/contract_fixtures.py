"""Comprehensive test fixtures for pipeline contract dataclasses.

This module provides factory functions, edge case fixtures, and serialization
fixtures for testing all dataclasses and enums in jarvis/contracts/pipeline.py.

Usage:
    from tests.fixtures.contract_fixtures import (
        make_message_context,
        make_classification_result,
        MESSAGE_CONTEXT_EMPTY,
        MESSAGE_CONTEXT_MAXIMAL,
        INTENT_TYPE_VALUES,
    )
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

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

# =============================================================================
# ENUM FIXTURES
# =============================================================================

INTENT_TYPE_VALUES: list[IntentType] = [
    IntentType.QUESTION,
    IntentType.STATEMENT,
    IntentType.REQUEST,
    IntentType.CLARIFICATION,
    IntentType.GREETING,
    IntentType.UNKNOWN,
]
"""List of all IntentType enum values for testing."""

CATEGORY_TYPE_VALUES: list[CategoryType] = [
    CategoryType.ACKNOWLEDGE,
    CategoryType.CLOSING,
    CategoryType.DEFER,
    CategoryType.FULL_RESPONSE,
    CategoryType.OFF_TOPIC,
]
"""List of all CategoryType enum values for testing."""

URGENCY_LEVEL_VALUES: list[UrgencyLevel] = [
    UrgencyLevel.LOW,
    UrgencyLevel.MEDIUM,
    UrgencyLevel.HIGH,
]
"""List of all UrgencyLevel enum values for testing."""

# Parametrize-friendly lists (strings for pytest.mark.parametrize)
INTENT_TYPE_PARAMETRIZE: list[str] = [e.value for e in IntentType]
"""Parametrize-friendly list of IntentType string values."""

CATEGORY_TYPE_PARAMETRIZE: list[str] = [e.value for e in CategoryType]
"""Parametrize-friendly list of CategoryType string values."""

URGENCY_LEVEL_PARAMETRIZE: list[str] = [e.value for e in UrgencyLevel]
"""Parametrize-friendly list of UrgencyLevel string values."""


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def make_message_context(
    chat_id: str = "chat_001",
    message_text: str = "Hello, can you help me with something?",
    is_from_me: bool = False,
    timestamp: datetime | None = None,
    sender_id: str | None = "user_123",
    thread_id: str | None = "thread_abc",
    metadata: dict[str, Any] | None = None,
    **overrides: Any,
) -> MessageContext:
    """Factory for MessageContext with realistic defaults.

    Args:
        chat_id: Unique identifier for the chat
        message_text: The actual message content
        is_from_me: Whether the message is from the assistant
        timestamp: When the message was sent (defaults to now)
        sender_id: Optional sender identifier
        thread_id: Optional thread identifier
        metadata: Optional metadata dict
        **overrides: Additional field overrides

    Returns:
        A MessageContext instance
    """
    result = MessageContext(
        chat_id=chat_id,
        message_text=message_text,
        is_from_me=is_from_me,
        timestamp=timestamp or datetime(2024, 1, 15, 10, 30, 0),
        sender_id=sender_id,
        thread_id=thread_id,
        metadata=metadata or {"platform": "imessage", "version": "1.0"},
    )
    for key, value in overrides.items():
        object.__setattr__(result, key, value)
    return result


def make_classification_result(
    intent: IntentType = IntentType.QUESTION,
    category: CategoryType = CategoryType.FULL_RESPONSE,
    urgency: UrgencyLevel = UrgencyLevel.MEDIUM,
    confidence: float = 0.85,
    requires_knowledge: bool = True,
    metadata: dict[str, Any] | None = None,
    **overrides: Any,
) -> ClassificationResult:
    """Factory for ClassificationResult with realistic defaults.

    Args:
        intent: The detected intent type
        category: The response category
        urgency: The urgency level
        confidence: Classification confidence (0.0-1.0)
        requires_knowledge: Whether knowledge retrieval is needed
        metadata: Optional metadata dict
        **overrides: Additional field overrides

    Returns:
        A ClassificationResult instance
    """
    result = ClassificationResult(
        intent=intent,
        category=category,
        urgency=urgency,
        confidence=confidence,
        requires_knowledge=requires_knowledge,
        metadata=metadata or {"model": "test_classifier", "version": "1.0"},
    )
    for key, value in overrides.items():
        object.__setattr__(result, key, value)
    return result


def make_entity(
    name: str = "John Doe",
    label: str = "PERSON",
    text: str = "John",
    start_char: int = 0,
    end_char: int = 4,
    **overrides: Any,
) -> Entity:
    """Factory for Entity with realistic defaults.

    Args:
        name: Canonical entity name
        label: Entity type label (e.g., PERSON, ORG, DATE)
        text: The text span that matched
        start_char: Start character index in source text
        end_char: End character index in source text
        **overrides: Additional field overrides

    Returns:
        An Entity instance
    """
    result = Entity(
        name=name,
        label=label,
        text=text,
        start_char=start_char,
        end_char=end_char,
    )
    for key, value in overrides.items():
        object.__setattr__(result, key, value)
    return result


def make_fact(
    subject: str = "John Doe",
    predicate: str = "works_at",
    object: str = "Acme Corp",
    confidence: float = 0.92,
    source_text: str = "John works at Acme Corp",
    **overrides: Any,
) -> Fact:
    """Factory for Fact with realistic defaults.

    Args:
        subject: The subject of the fact
        predicate: The relationship/predicate
        object: The object of the fact
        confidence: Extraction confidence (0.0-1.0)
        source_text: The original text this fact was extracted from
        **overrides: Additional field overrides

    Returns:
        A Fact instance
    """
    result = Fact(
        subject=subject,
        predicate=predicate,
        object=object,
        confidence=confidence,
        source_text=source_text,
    )
    for key, value in overrides.items():
        object.__setattr__(result, key, value)
    return result


def make_relationship(
    source_entity: str = "John Doe",
    target_entity: str = "Jane Smith",
    relation_type: str = "colleague_of",
    confidence: float = 0.78,
    **overrides: Any,
) -> Relationship:
    """Factory for Relationship with realistic defaults.

    Args:
        source_entity: Source entity name
        target_entity: Target entity name
        relation_type: Type of relationship
        confidence: Extraction confidence (0.0-1.0)
        **overrides: Additional field overrides

    Returns:
        A Relationship instance
    """
    result = Relationship(
        source_entity=source_entity,
        target_entity=target_entity,
        relation_type=relation_type,
        confidence=confidence,
    )
    for key, value in overrides.items():
        object.__setattr__(result, key, value)
    return result


def make_extraction_result(
    entities: list[Entity] | None = None,
    facts: list[Fact] | None = None,
    relationships: list[Relationship] | None = None,
    topics: list[str] | None = None,
    **overrides: Any,
) -> ExtractionResult:
    """Factory for ExtractionResult with realistic defaults.

    Args:
        entities: List of extracted entities
        facts: List of extracted facts
        relationships: List of extracted relationships
        topics: List of detected topics
        **overrides: Additional field overrides

    Returns:
        An ExtractionResult instance
    """
    result = ExtractionResult(
        entities=entities or [make_entity()],
        facts=facts or [make_fact()],
        relationships=relationships or [make_relationship()],
        topics=topics or ["work", "people"],
    )
    for key, value in overrides.items():
        object.__setattr__(result, key, value)
    return result


def make_rag_document(
    content: str = "This is a retrieved document about machine learning.",
    source: str = "docs/ml_overview.md",
    score: float = 0.89,
    metadata: dict[str, Any] | None = None,
    **overrides: Any,
) -> RAGDocument:
    """Factory for RAGDocument with realistic defaults.

    Args:
        content: The document content/chunk
        source: Source identifier (file path, URL, etc.)
        score: Retrieval similarity score
        metadata: Optional metadata dict
        **overrides: Additional field overrides

    Returns:
        A RAGDocument instance
    """
    result = RAGDocument(
        content=content,
        source=source,
        score=score,
        metadata=metadata or {"chunk_index": 0, "total_chunks": 5},
    )
    for key, value in overrides.items():
        object.__setattr__(result, key, value)
    return result


def make_generation_request(
    context: MessageContext | None = None,
    classification: ClassificationResult | None = None,
    extraction: ExtractionResult | None = None,
    retrieved_docs: list[RAGDocument] | None = None,
    few_shot_examples: list[dict[str, str]] | None = None,
    **overrides: Any,
) -> GenerationRequest:
    """Factory for GenerationRequest with realistic defaults.

    Args:
        context: The message context
        classification: Classification result
        extraction: Extraction result (optional)
        retrieved_docs: List of retrieved RAG documents
        few_shot_examples: Few-shot examples for generation
        **overrides: Additional field overrides

    Returns:
        A GenerationRequest instance
    """
    result = GenerationRequest(
        context=context or make_message_context(),
        classification=classification or make_classification_result(),
        extraction=extraction,
        retrieved_docs=retrieved_docs or [make_rag_document()],
        few_shot_examples=few_shot_examples or [{"input": "Hi", "output": "Hello!"}],
    )
    for key, value in overrides.items():
        object.__setattr__(result, key, value)
    return result


def make_generation_response(
    response: str = "I'd be happy to help you with that!",
    confidence: float = 0.91,
    used_kg_facts: list[str] | None = None,
    streaming: bool = False,
    metadata: dict[str, Any] | None = None,
    **overrides: Any,
) -> GenerationResponse:
    """Factory for GenerationResponse with realistic defaults.

    Args:
        response: The generated response text
        confidence: Generation confidence (0.0-1.0)
        used_kg_facts: IDs of knowledge graph facts used
        streaming: Whether this is a streaming response chunk
        metadata: Optional metadata dict
        **overrides: Additional field overrides

    Returns:
        A GenerationResponse instance
    """
    result = GenerationResponse(
        response=response,
        confidence=confidence,
        used_kg_facts=used_kg_facts or ["fact_001", "fact_002"],
        streaming=streaming,
        metadata=metadata or {"model": "test_llm", "tokens": 42},
    )
    for key, value in overrides.items():
        object.__setattr__(result, key, value)
    return result


# =============================================================================
# EDGE CASE FIXTURES - EMPTY/MINIMAL
# =============================================================================

MESSAGE_CONTEXT_EMPTY = make_message_context(
    chat_id="",
    message_text="",
    is_from_me=False,
    timestamp=datetime(2024, 1, 1, 0, 0, 0),
    sender_id=None,
    thread_id=None,
    metadata={},
)
"""MessageContext with only required fields, all optional fields empty/None."""

CLASSIFICATION_RESULT_EMPTY = make_classification_result(
    intent=IntentType.UNKNOWN,
    category=CategoryType.OFF_TOPIC,
    urgency=UrgencyLevel.LOW,
    confidence=0.0,
    requires_knowledge=False,
    metadata={},
)
"""ClassificationResult with minimal/empty values."""

ENTITY_EMPTY = make_entity(
    name="",
    label="",
    text="",
    start_char=0,
    end_char=0,
)
"""Entity with empty string values."""

FACT_EMPTY = make_fact(
    subject="",
    predicate="",
    object="",
    confidence=0.0,
    source_text="",
)
"""Fact with empty string values and zero confidence."""

RELATIONSHIP_EMPTY = make_relationship(
    source_entity="",
    target_entity="",
    relation_type="",
    confidence=0.0,
)
"""Relationship with empty string values and zero confidence."""

EXTRACTION_RESULT_EMPTY = make_extraction_result(
    entities=[],
    facts=[],
    relationships=[],
    topics=[],
)
"""ExtractionResult with all lists empty."""

RAG_DOCUMENT_EMPTY = make_rag_document(
    content="",
    source="",
    score=0.0,
    metadata={},
)
"""RAGDocument with empty content and zero score."""

GENERATION_REQUEST_EMPTY = make_generation_request(
    context=MESSAGE_CONTEXT_EMPTY,
    classification=CLASSIFICATION_RESULT_EMPTY,
    extraction=None,
    retrieved_docs=[],
    few_shot_examples=[],
)
"""GenerationRequest with minimal nested objects and empty lists."""

GENERATION_RESPONSE_EMPTY = make_generation_response(
    response="",
    confidence=0.0,
    used_kg_facts=[],
    streaming=False,
    metadata={},
)
"""GenerationResponse with empty response and zero confidence."""


# =============================================================================
# EDGE CASE FIXTURES - MAXIMAL (ALL OPTIONAL FIELDS POPULATED)
# =============================================================================

MESSAGE_CONTEXT_MAXIMAL = make_message_context(
    chat_id="chat_max_999",
    message_text="This is a detailed message with multiple sentences. "
    "It contains references to various topics and entities.",
    is_from_me=True,
    timestamp=datetime(2024, 12, 31, 23, 59, 59),
    sender_id="sender_max_12345",
    thread_id="thread_max_abcde",
    metadata={
        "platform": "imessage",
        "version": "2.0",
        "attachments": ["image1.jpg", "image2.png"],
        "reactions": ["👍", "❤️"],
        "edited": True,
        "deleted": False,
    },
)
"""MessageContext with all optional fields populated with rich data."""

CLASSIFICATION_RESULT_MAXIMAL = make_classification_result(
    intent=IntentType.REQUEST,
    category=CategoryType.FULL_RESPONSE,
    urgency=UrgencyLevel.HIGH,
    confidence=0.9999,
    requires_knowledge=True,
    metadata={
        "model": "advanced_classifier_v3",
        "version": "3.0.1",
        "features": ["tfidf", "embeddings", "ner"],
        "processing_time_ms": 45.5,
        "cached": False,
        "fallback_used": False,
    },
)
"""ClassificationResult with rich metadata."""

ENTITY_MAXIMAL = make_entity(
    name="Dr. Jane Elizabeth Smith Jr.",
    label="PERSON",
    text="Dr. Jane Smith",
    start_char=42,
    end_char=58,
)
"""Entity with a complex full name."""

FACT_MAXIMAL = make_fact(
    subject="Dr. Jane Smith",
    predicate="is_employed_by",
    object="International Business Machines Corporation",
    confidence=0.9876,
    source_text="Dr. Jane Smith has been employed by "
    "International Business Machines Corporation since 2020.",
)
"""Fact with complex subject and object values."""

RELATIONSHIP_MAXIMAL = make_relationship(
    source_entity="Dr. Jane Smith",
    target_entity="International Business Machines Corporation",
    relation_type="senior_vice_president_of",
    confidence=0.9234,
)
"""Relationship with complex entity names and specific relation type."""

EXTRACTION_RESULT_MAXIMAL = make_extraction_result(
    entities=[
        make_entity(name="Alice", label="PERSON", text="Alice", start_char=0, end_char=5),
        make_entity(name="Bob", label="PERSON", text="Bob", start_char=10, end_char=13),
        make_entity(name="Acme Corp", label="ORG", text="Acme", start_char=20, end_char=24),
    ],
    facts=[
        make_fact(
            subject="Alice",
            predicate="knows",
            object="Bob",
            confidence=0.95,
            source_text="Alice knows Bob",
        ),
        make_fact(
            subject="Bob",
            predicate="works_at",
            object="Acme Corp",
            confidence=0.88,
            source_text="Bob works at Acme",
        ),
    ],
    relationships=[
        make_relationship(
            source_entity="Alice", target_entity="Bob", relation_type="friend_of", confidence=0.82
        ),
    ],
    topics=["friendship", "employment", "business", "technology"],
)
"""ExtractionResult with multiple entities, facts, and relationships."""

RAG_DOCUMENT_MAXIMAL = make_rag_document(
    content="Machine learning is a subset of artificial intelligence "
    "that enables systems to learn from data. "
    "Deep learning uses neural networks with many layers. "
    "These techniques are used in computer vision, NLP, and more.",
    source="knowledge_base/ai_ml/comprehensive_guide.md#section-3.2",
    score=0.9567,
    metadata={
        "chunk_index": 3,
        "total_chunks": 12,
        "document_id": "doc_ml_001",
        "title": "Comprehensive Guide to Machine Learning",
        "author": "AI Research Team",
        "last_updated": "2024-06-15",
        "tags": ["machine_learning", "ai", "deep_learning"],
    },
)
"""RAGDocument with rich content and extensive metadata."""

GENERATION_REQUEST_MAXIMAL = make_generation_request(
    context=MESSAGE_CONTEXT_MAXIMAL,
    classification=CLASSIFICATION_RESULT_MAXIMAL,
    extraction=EXTRACTION_RESULT_MAXIMAL,
    retrieved_docs=[
        RAG_DOCUMENT_MAXIMAL,
        make_rag_document(
            content="Another relevant document about the topic.",
            source="docs/additional_info.md",
            score=0.8234,
            metadata={"priority": "secondary"},
        ),
    ],
    few_shot_examples=[
        {"input": "What is AI?", "output": "AI stands for Artificial Intelligence..."},
        {"input": "Explain ML", "output": "Machine Learning is..."},
        {"input": "How does DL work?", "output": "Deep Learning works by..."},
    ],
)
"""GenerationRequest with all nested objects fully populated."""

GENERATION_RESPONSE_MAXIMAL = make_generation_response(
    response="Based on the information provided, I can explain that "
    "machine learning is indeed a fascinating field. "
    "It involves training algorithms on data to make predictions. "
    "Would you like to know more about specific techniques?",
    confidence=0.9432,
    used_kg_facts=["fact_ml_001", "fact_ml_002", "fact_ai_045", "fact_dl_012"],
    streaming=False,
    metadata={
        "model": "llama-3.1-8b-instruct",
        "tokens_input": 256,
        "tokens_output": 64,
        "generation_time_ms": 1250.5,
        "finish_reason": "stop",
        "stop_sequence": None,
    },
)
"""GenerationResponse with detailed response and rich metadata."""


# =============================================================================
# EDGE CASE FIXTURES - UNICODE CONTENT
# =============================================================================

MESSAGE_CONTEXT_UNICODE = make_message_context(
    chat_id="chat_unicode_🎉",
    message_text="Hello! 👋 你好世界 🌍 مرحبا! עִבְרִית 日本語",
    is_from_me=False,
    timestamp=datetime(2024, 6, 15, 12, 0, 0),
    sender_id="user_🦄_123",
    thread_id="thread_🔥_abc",
    metadata={"emoji": "🚀", "language": "multi"},
)
"""MessageContext with emoji, CJK characters, and RTL text."""

CLASSIFICATION_RESULT_UNICODE = make_classification_result(
    intent=IntentType.GREETING,
    category=CategoryType.ACKNOWLEDGE,
    urgency=UrgencyLevel.MEDIUM,
    confidence=0.88,
    requires_knowledge=False,
    metadata={"note": "Contains 🎉 emoji and 中文"},
)
"""ClassificationResult with unicode metadata."""

ENTITY_UNICODE = make_entity(
    name="北京科技有限公司",
    label="ORG",
    text="北京科技",
    start_char=0,
    end_char=4,
)
"""Entity with Chinese characters."""

FACT_UNICODE = make_fact(
    subject="🎬 Movie Studios",
    predicate="produces",
    object="🎭 Entertainment Content",
    confidence=0.92,
    source_text="🎬 Movie Studios produces 🎭 Entertainment Content for audiences worldwide",
)
"""Fact with emoji in subject and object."""

RELATIONSHIP_UNICODE = make_relationship(
    source_entity="👨‍💼 Manager",
    target_entity="👩‍💻 Developer",
    relation_type="supervises",
    confidence=0.85,
)
"""Relationship with emoji in entity names."""

RAG_DOCUMENT_UNICODE = make_rag_document(
    content="Documentation in multiple languages: English, 中文, العربية, עברית, 日本語 🌐",
    source="docs/🌍/international.md",
    score=0.90,
    metadata={"languages": ["en", "zh", "ar", "he", "ja"], "🌟": "multilingual"},
)
"""RAGDocument with multilingual content and emoji paths."""

GENERATION_RESPONSE_UNICODE = make_generation_response(
    response="مرحباً! 👋 你好! ¡Hola! Bonjour! 🎉",
    confidence=0.95,
    used_kg_facts=["fact_🌍_001"],
    metadata={"response_type": "greeting", "languages": 5},
)
"""GenerationResponse with multilingual greeting and emoji."""


# =============================================================================
# EDGE CASE FIXTURES - LONG STRINGS (>1000 chars)
# =============================================================================

# Generate a long string (>1000 chars)
_LONG_TEXT = "This is a very long message. " * 50  # ~1450 chars
_LONG_CONTENT = "Document content that goes on and on. " * 50  # ~1850 chars

MESSAGE_CONTEXT_LONG = make_message_context(
    chat_id="chat_" + "x" * 100,
    message_text=_LONG_TEXT,
    is_from_me=False,
    timestamp=datetime(2024, 3, 15, 10, 0, 0),
    sender_id="user_" + "y" * 100,
    thread_id="thread_" + "z" * 100,
    metadata={"note": "a" * 500},
)
"""MessageContext with very long strings (>1000 chars)."""

RAG_DOCUMENT_LONG = make_rag_document(
    content=_LONG_CONTENT,
    source="/very/long/path/" + "a" * 200 + "/document.md",
    score=0.75,
    metadata={"description": "b" * 500, "tags": ["t" * 100 for _ in range(10)]},
)
"""RAGDocument with very long content string."""

GENERATION_RESPONSE_LONG = make_generation_response(
    response=_LONG_TEXT,
    confidence=0.82,
    used_kg_facts=["fact_" + "f" * 100 for _ in range(20)],
    metadata={"long_key": "v" * 500},
)
"""GenerationResponse with very long response text."""


# =============================================================================
# EDGE CASE FIXTURES - BOUNDARY VALUES
# =============================================================================

# Confidence = 0.0
CLASSIFICATION_RESULT_CONFIDENCE_0 = make_classification_result(confidence=0.0)
"""ClassificationResult with minimum confidence (0.0)."""

CLASSIFICATION_RESULT_CONFIDENCE_1 = make_classification_result(confidence=1.0)
"""ClassificationResult with maximum confidence (1.0)."""

FACT_CONFIDENCE_0 = make_fact(confidence=0.0)
"""Fact with minimum confidence (0.0)."""

FACT_CONFIDENCE_1 = make_fact(confidence=1.0)
"""Fact with maximum confidence (1.0)."""

RELATIONSHIP_CONFIDENCE_0 = make_relationship(confidence=0.0)
"""Relationship with minimum confidence (0.0)."""

RELATIONSHIP_CONFIDENCE_1 = make_relationship(confidence=1.0)
"""Relationship with maximum confidence (1.0)."""

RAG_DOCUMENT_SCORE_0 = make_rag_document(score=0.0)
"""RAGDocument with minimum score (0.0)."""

RAG_DOCUMENT_SCORE_1 = make_rag_document(score=1.0)
"""RAGDocument with maximum score (1.0)."""

GENERATION_RESPONSE_CONFIDENCE_0 = make_generation_response(confidence=0.0)
"""GenerationResponse with minimum confidence (0.0)."""

GENERATION_RESPONSE_CONFIDENCE_1 = make_generation_response(confidence=1.0)
"""GenerationResponse with maximum confidence (1.0)."""

# Zero character positions
ENTITY_ZERO_POSITIONS = make_entity(start_char=0, end_char=0)
"""Entity with zero start/end positions."""

# Large character positions
ENTITY_LARGE_POSITIONS = make_entity(start_char=99999, end_char=100000)
"""Entity with large start/end positions."""


# =============================================================================
# SERIALIZATION FIXTURES - (instance, expected_dict) pairs for asdict() testing
# =============================================================================

# Simple MessageContext serialization pair
MESSAGE_CONTEXT_SERIALIZATION = (
    make_message_context(
        chat_id="test_chat",
        message_text="Hello",
        is_from_me=True,
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        sender_id="user_1",
        thread_id="thread_1",
        metadata={"key": "value"},
    ),
    {
        "chat_id": "test_chat",
        "message_text": "Hello",
        "is_from_me": True,
        "timestamp": datetime(2024, 1, 1, 12, 0, 0),
        "sender_id": "user_1",
        "thread_id": "thread_1",
        "metadata": {"key": "value"},
    },
)
"""(MessageContext, expected_dict) pair for serialization testing."""

CLASSIFICATION_RESULT_SERIALIZATION = (
    make_classification_result(
        intent=IntentType.QUESTION,
        category=CategoryType.FULL_RESPONSE,
        urgency=UrgencyLevel.HIGH,
        confidence=0.95,
        requires_knowledge=True,
        metadata={"source": "test"},
    ),
    {
        "intent": IntentType.QUESTION,
        "category": CategoryType.FULL_RESPONSE,
        "urgency": UrgencyLevel.HIGH,
        "confidence": 0.95,
        "requires_knowledge": True,
        "metadata": {"source": "test"},
    },
)
"""(ClassificationResult, expected_dict) pair for serialization testing."""

ENTITY_SERIALIZATION = (
    make_entity(
        name="Alice",
        label="PERSON",
        text="Alice",
        start_char=0,
        end_char=5,
    ),
    {
        "name": "Alice",
        "label": "PERSON",
        "text": "Alice",
        "start_char": 0,
        "end_char": 5,
    },
)
"""(Entity, expected_dict) pair for serialization testing."""

FACT_SERIALIZATION = (
    make_fact(
        subject="Alice",
        predicate="knows",
        object="Bob",
        confidence=0.88,
        source_text="Alice knows Bob well.",
    ),
    {
        "subject": "Alice",
        "predicate": "knows",
        "object": "Bob",
        "confidence": 0.88,
        "source_text": "Alice knows Bob well.",
    },
)
"""(Fact, expected_dict) pair for serialization testing."""

RELATIONSHIP_SERIALIZATION = (
    make_relationship(
        source_entity="Alice",
        target_entity="Bob",
        relation_type="friend_of",
        confidence=0.75,
    ),
    {
        "source_entity": "Alice",
        "target_entity": "Bob",
        "relation_type": "friend_of",
        "confidence": 0.75,
    },
)
"""(Relationship, expected_dict) pair for serialization testing."""

EXTRACTION_RESULT_SERIALIZATION = (
    make_extraction_result(
        entities=[],
        facts=[],
        relationships=[],
        topics=[],
    ),
    {
        "entities": [],
        "facts": [],
        "relationships": [],
        "topics": [],
    },
)
"""(ExtractionResult, expected_dict) pair for serialization testing."""

RAG_DOCUMENT_SERIALIZATION = (
    make_rag_document(
        content="Test content",
        source="test.md",
        score=0.85,
        metadata={"index": 1},
    ),
    {
        "content": "Test content",
        "source": "test.md",
        "score": 0.85,
        "metadata": {"index": 1},
    },
)
"""(RAGDocument, expected_dict) pair for serialization testing."""

GENERATION_RESPONSE_SERIALIZATION = (
    make_generation_response(
        response="Test response",
        confidence=0.9,
        used_kg_facts=["fact_1"],
        streaming=False,
        metadata={"model": "test"},
    ),
    {
        "response": "Test response",
        "confidence": 0.9,
        "used_kg_facts": ["fact_1"],
        "streaming": False,
        "metadata": {"model": "test"},
    },
)
"""(GenerationResponse, expected_dict) pair for serialization testing."""

# List of all serialization fixtures for batch testing
ALL_SERIALIZATION_PAIRS: list[tuple[Any, dict[str, Any]]] = [
    MESSAGE_CONTEXT_SERIALIZATION,
    CLASSIFICATION_RESULT_SERIALIZATION,
    ENTITY_SERIALIZATION,
    FACT_SERIALIZATION,
    RELATIONSHIP_SERIALIZATION,
    EXTRACTION_RESULT_SERIALIZATION,
    RAG_DOCUMENT_SERIALIZATION,
    GENERATION_RESPONSE_SERIALIZATION,
]
"""List of all (instance, expected_dict) pairs for batch serialization testing."""


# =============================================================================
# COLLECTIONS FOR BATCH TESTING
# =============================================================================

ALL_EMPTY_FIXTURES: list[Any] = [
    MESSAGE_CONTEXT_EMPTY,
    CLASSIFICATION_RESULT_EMPTY,
    ENTITY_EMPTY,
    FACT_EMPTY,
    RELATIONSHIP_EMPTY,
    EXTRACTION_RESULT_EMPTY,
    RAG_DOCUMENT_EMPTY,
    GENERATION_REQUEST_EMPTY,
    GENERATION_RESPONSE_EMPTY,
]
"""All empty/minimal fixtures for batch testing."""

ALL_MAXIMAL_FIXTURES: list[Any] = [
    MESSAGE_CONTEXT_MAXIMAL,
    CLASSIFICATION_RESULT_MAXIMAL,
    ENTITY_MAXIMAL,
    FACT_MAXIMAL,
    RELATIONSHIP_MAXIMAL,
    EXTRACTION_RESULT_MAXIMAL,
    RAG_DOCUMENT_MAXIMAL,
    GENERATION_REQUEST_MAXIMAL,
    GENERATION_RESPONSE_MAXIMAL,
]
"""All maximal fixtures for batch testing."""

ALL_UNICODE_FIXTURES: list[Any] = [
    MESSAGE_CONTEXT_UNICODE,
    CLASSIFICATION_RESULT_UNICODE,
    ENTITY_UNICODE,
    FACT_UNICODE,
    RELATIONSHIP_UNICODE,
    RAG_DOCUMENT_UNICODE,
    GENERATION_RESPONSE_UNICODE,
]
"""All unicode fixtures for batch testing."""

ALL_LONG_FIXTURES: list[Any] = [
    MESSAGE_CONTEXT_LONG,
    RAG_DOCUMENT_LONG,
    GENERATION_RESPONSE_LONG,
]
"""All long string fixtures for batch testing."""

ALL_BOUNDARY_FIXTURES: list[Any] = [
    CLASSIFICATION_RESULT_CONFIDENCE_0,
    CLASSIFICATION_RESULT_CONFIDENCE_1,
    FACT_CONFIDENCE_0,
    FACT_CONFIDENCE_1,
    RELATIONSHIP_CONFIDENCE_0,
    RELATIONSHIP_CONFIDENCE_1,
    RAG_DOCUMENT_SCORE_0,
    RAG_DOCUMENT_SCORE_1,
    GENERATION_RESPONSE_CONFIDENCE_0,
    GENERATION_RESPONSE_CONFIDENCE_1,
    ENTITY_ZERO_POSITIONS,
    ENTITY_LARGE_POSITIONS,
]
"""All boundary value fixtures for batch testing."""
