from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Any, Dict
from enum import Enum

# --- Enums ---

class IntentType(str, Enum):
    QUESTION = "question"
    STATEMENT = "statement"
    REQUEST = "request"
    CLARIFICATION = "clarification"
    GREETING = "greeting"
    UNKNOWN = "unknown"

class CategoryType(str, Enum):
    ACKNOWLEDGE = "acknowledge"      # "Got it"
    CLOSING = "closing"              # "See you later"
    DEFER = "defer"                  # "I'll look into that"
    FULL_RESPONSE = "full_response"  # Standard RAG/LLM reply
    OFF_TOPIC = "off_topic"          # Ignore/Low priority

class UrgencyLevel(str, Enum):
    LOW = "low"       # Can wait (e.g. general info)
    MEDIUM = "medium" # Normal conversation
    HIGH = "high"     # Immediate attention needed

# --- Data Structures ---

@dataclass
class MessageContext:
    """Raw input context for the pipeline."""
    chat_id: str
    message_text: str
    is_from_me: bool
    timestamp: datetime
    sender_id: Optional[str] = None
    thread_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ClassificationResult:
    """Result of the intent/category classification step."""
    intent: IntentType
    category: CategoryType
    urgency: UrgencyLevel
    confidence: float
    requires_knowledge: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Entity:
    """A named entity extracted from text."""
    name: str
    label: str  # PERSON, ORG, DATE, LOCATION, etc.
    text: str
    start_char: int
    end_char: int

@dataclass
class Fact:
    """An atomic fact extracted from text."""
    subject: str
    predicate: str
    object: str
    confidence: float
    source_text: str

@dataclass
class Relationship:
    """A relationship between two entities."""
    source_entity: str
    target_entity: str
    relation_type: str
    confidence: float

@dataclass
class ExtractionResult:
    """Consolidated knowledge extraction result."""
    entities: List[Entity] = field(default_factory=list)
    facts: List[Fact] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)

@dataclass
class RAGDocument:
    """A retrieved document/chunk for context."""
    content: str
    source: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GenerationRequest:
    """The full packet sent to the generator service."""
    context: MessageContext
    classification: ClassificationResult
    extraction: Optional[ExtractionResult]
    retrieved_docs: List[RAGDocument] = field(default_factory=list)
    few_shot_examples: List[Dict[str, str]] = field(default_factory=list)

@dataclass
class GenerationResponse:
    """The final output from the pipeline."""
    response: str
    confidence: float
    used_kg_facts: List[str] = field(default_factory=list)
    streaming: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
