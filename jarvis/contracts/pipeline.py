from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

# --- Enums ---


class IntentType(StrEnum):
    QUESTION = "question"
    STATEMENT = "statement"
    REQUEST = "request"
    CLARIFICATION = "clarification"
    GREETING = "greeting"
    UNKNOWN = "unknown"


class CategoryType(StrEnum):
    ACKNOWLEDGE = "acknowledge"  # "Got it"
    CLOSING = "closing"  # "See you later"
    DEFER = "defer"  # "I'll look into that"
    FULL_RESPONSE = "full_response"  # Standard RAG/LLM reply
    OFF_TOPIC = "off_topic"  # Ignore/Low priority


class UrgencyLevel(StrEnum):
    LOW = "low"  # Can wait (e.g. general info)
    MEDIUM = "medium"  # Normal conversation
    HIGH = "high"  # Immediate attention needed


# --- Data Structures ---


@dataclass(slots=True)
class MessageContext:
    """Raw input context for the pipeline."""

    chat_id: str
    message_text: str
    is_from_me: bool
    timestamp: datetime
    sender_id: str | None = None
    thread_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ClassificationResult:
    """Result of the intent/category classification step."""

    intent: IntentType
    category: CategoryType
    urgency: UrgencyLevel
    confidence: float
    requires_knowledge: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Entity:
    """A named entity extracted from text."""

    name: str
    label: str  # PERSON, ORG, DATE, LOCATION, etc.
    text: str
    start_char: int
    end_char: int


@dataclass(slots=True)
class Fact:
    """An atomic fact extracted from text."""

    subject: str
    predicate: str
    object: str
    confidence: float
    source_text: str


@dataclass(slots=True)
class Relationship:
    """A relationship between two entities."""

    source_entity: str
    target_entity: str
    relation_type: str
    confidence: float


@dataclass(slots=True)
class ExtractionResult:
    """Consolidated knowledge extraction result."""

    entities: list[Entity] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RAGDocument:
    """A retrieved document/chunk for context."""

    content: str
    source: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationRequest:
    """The full packet sent to the generator service."""

    context: MessageContext
    classification: ClassificationResult
    extraction: ExtractionResult | None
    retrieved_docs: list[RAGDocument] = field(default_factory=list)
    few_shot_examples: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class GenerationResponse:
    """The final output from the pipeline."""

    response: str
    confidence: float
    used_kg_facts: list[str] = field(default_factory=list)
    streaming: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
