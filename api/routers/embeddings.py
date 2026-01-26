"""Embeddings API endpoints.

Provides endpoints for semantic search and relationship profiling
using message embeddings.

Endpoints:
    POST /embeddings/index - Index messages from a conversation
    GET /embeddings/search - Semantic search across messages
    GET /embeddings/relationship/{contact_id} - Relationship profile
    GET /embeddings/stats - Index statistics
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from api.dependencies import get_imessage_reader
from integrations.imessage import ChatDBReader
from jarvis.embeddings import (
    EmbeddingError,
    EmbeddingStore,
    RelationshipProfile,
    SimilarMessage,
    get_embedding_store,
)

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


# =============================================================================
# Request/Response Models
# =============================================================================


class IndexConversationRequest(BaseModel):
    """Request to index messages from a conversation."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chat_id": "chat123456789",
                "limit": 500,
            }
        }
    )

    chat_id: str = Field(
        ...,
        description="Conversation ID to index messages from",
        examples=["chat123456789"],
    )
    limit: int = Field(
        default=500,
        ge=10,
        le=5000,
        description="Maximum number of messages to index",
    )


class IndexResponse(BaseModel):
    """Response after indexing messages."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "indexed": 450,
                "skipped": 30,
                "duplicates": 20,
                "chat_id": "chat123456789",
            }
        }
    )

    success: bool = Field(..., description="Whether indexing succeeded")
    indexed: int = Field(..., description="Number of messages indexed", ge=0)
    skipped: int = Field(..., description="Messages skipped (too short)", ge=0)
    duplicates: int = Field(..., description="Messages already indexed", ge=0)
    chat_id: str = Field(..., description="Conversation ID that was indexed")


class SimilarMessageResponse(BaseModel):
    """A message with its similarity score."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message_id": 12345,
                "chat_id": "chat123456789",
                "text": "Hey, want to grab dinner tonight?",
                "sender": "+15551234567",
                "sender_name": "John",
                "timestamp": "2024-01-15T18:30:00Z",
                "is_from_me": False,
                "similarity": 0.85,
            }
        }
    )

    message_id: int = Field(..., description="Message ID")
    chat_id: str = Field(..., description="Conversation ID")
    text: str = Field(..., description="Message text")
    sender: str | None = Field(None, description="Sender phone/email")
    sender_name: str | None = Field(None, description="Sender display name")
    timestamp: datetime = Field(..., description="Message timestamp")
    is_from_me: bool = Field(..., description="Whether sent by user")
    similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Semantic similarity score (0-1)",
    )


class SearchResponse(BaseModel):
    """Response for semantic search."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "dinner plans",
                "results": [
                    {
                        "message_id": 12345,
                        "chat_id": "chat123456789",
                        "text": "Hey, want to grab dinner tonight?",
                        "similarity": 0.85,
                    }
                ],
                "total_results": 1,
            }
        }
    )

    query: str = Field(..., description="Original search query")
    results: list[SimilarMessageResponse] = Field(
        ..., description="Matching messages sorted by similarity"
    )
    total_results: int = Field(..., description="Total number of results", ge=0)


class ResponsePatterns(BaseModel):
    """Response pattern statistics."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "avg_response_time_seconds": 180.5,
                "quick_responses": 45,
                "slow_responses": 10,
            }
        }
    )

    avg_response_time_seconds: float | None = Field(
        None, description="Average response time in seconds"
    )
    quick_responses: int | None = Field(None, description="Count of quick responses (<5 min)")
    slow_responses: int | None = Field(None, description="Count of slow responses (>1 hour)")


class RelationshipProfileResponse(BaseModel):
    """Relationship profile for a contact."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "contact_id": "chat123456789",
                "display_name": "John Doe",
                "total_messages": 500,
                "sent_count": 250,
                "received_count": 250,
                "common_topics": ["dinner", "work", "weekend"],
                "typical_tone": "casual",
                "avg_message_length": 45.5,
                "response_patterns": {
                    "avg_response_time_seconds": 180.5,
                    "quick_responses": 45,
                },
                "last_interaction": "2024-01-15T18:30:00Z",
            }
        }
    )

    contact_id: str = Field(..., description="Chat ID")
    display_name: str | None = Field(None, description="Contact display name")
    total_messages: int = Field(..., description="Total indexed messages", ge=0)
    sent_count: int = Field(..., description="Messages sent", ge=0)
    received_count: int = Field(..., description="Messages received", ge=0)
    common_topics: list[str] = Field(default_factory=list, description="Common conversation topics")
    typical_tone: str = Field(
        ...,
        description="Typical communication tone (casual/professional/mixed)",
    )
    avg_message_length: float = Field(..., description="Average message length in characters")
    response_patterns: ResponsePatterns | None = Field(None, description="Response time patterns")
    last_interaction: datetime | None = Field(None, description="Most recent interaction")


class IndexStatsResponse(BaseModel):
    """Statistics about the embedding index."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_embeddings": 5000,
                "unique_chats": 25,
                "oldest_message": "2023-01-01T00:00:00",
                "newest_message": "2024-01-15T18:30:00",
                "db_path": "/Users/user/.jarvis/embeddings.db",
                "db_size_bytes": 10485760,
            }
        }
    )

    total_embeddings: int = Field(..., description="Total number of indexed messages", ge=0)
    unique_chats: int = Field(..., description="Number of unique conversations indexed", ge=0)
    oldest_message: str | None = Field(None, description="Timestamp of oldest indexed message")
    newest_message: str | None = Field(None, description="Timestamp of newest indexed message")
    db_path: str = Field(..., description="Path to the embedding database")
    db_size_bytes: int = Field(..., description="Database file size in bytes", ge=0)


# =============================================================================
# Helper Functions
# =============================================================================


def _get_store() -> EmbeddingStore:
    """Get the embedding store singleton."""
    return get_embedding_store()


def _similar_to_response(msg: SimilarMessage) -> SimilarMessageResponse:
    """Convert SimilarMessage to response model."""
    return SimilarMessageResponse(
        message_id=msg.message_id,
        chat_id=msg.chat_id,
        text=msg.text,
        sender=msg.sender,
        sender_name=msg.sender_name,
        timestamp=msg.timestamp,
        is_from_me=msg.is_from_me,
        similarity=msg.similarity,
    )


def _profile_to_response(profile: RelationshipProfile) -> RelationshipProfileResponse:
    """Convert RelationshipProfile to response model."""
    patterns = None
    if profile.response_patterns:
        patterns = ResponsePatterns(
            avg_response_time_seconds=profile.response_patterns.get("avg_response_time_seconds"),
            quick_responses=profile.response_patterns.get("quick_responses"),
            slow_responses=profile.response_patterns.get("slow_responses"),
        )

    return RelationshipProfileResponse(
        contact_id=profile.contact_id,
        display_name=profile.display_name,
        total_messages=profile.total_messages,
        sent_count=profile.sent_count,
        received_count=profile.received_count,
        common_topics=profile.common_topics,
        typical_tone=profile.typical_tone,
        avg_message_length=profile.avg_message_length,
        response_patterns=patterns,
        last_interaction=profile.last_interaction,
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/index", response_model=IndexResponse)
def index_conversation(
    request: IndexConversationRequest,
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> IndexResponse:
    """Index messages from a conversation for semantic search.

    This endpoint retrieves messages from the specified conversation
    and creates embeddings for semantic search. Messages that are too
    short (<3 characters) or already indexed are skipped.

    Indexing is idempotent - calling multiple times with the same
    conversation will not create duplicate embeddings.

    Args:
        request: Index request with chat_id and optional limit.

    Returns:
        IndexResponse with counts of indexed, skipped, and duplicate messages.

    Raises:
        HTTPException 403: If Full Disk Access is not granted.
        HTTPException 404: If the conversation is not found.
        HTTPException 500: If indexing fails.

    Example:
        ```
        POST /embeddings/index
        {
            "chat_id": "chat123456789",
            "limit": 500
        }
        ```
    """
    try:
        # Get messages from iMessage
        messages = reader.get_messages(
            chat_id=request.chat_id,
            limit=request.limit,
        )

        if not messages:
            raise HTTPException(
                status_code=404,
                detail=f"No messages found for conversation: {request.chat_id}",
            )

        # Index messages
        store = _get_store()
        stats = store.index_messages(messages)

        return IndexResponse(
            success=True,
            indexed=stats["indexed"],
            skipped=stats["skipped"],
            duplicates=stats["duplicates"],
            chat_id=request.chat_id,
        )

    except HTTPException:
        raise
    except EmbeddingError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding error: {e.message}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to index conversation: {e}",
        ) from e


@router.get("/search", response_model=SearchResponse)
def semantic_search(
    query: str = Query(
        ...,
        min_length=2,
        description="Search query text",
        examples=["dinner plans tonight"],
    ),
    contact_id: str | None = Query(
        None,
        description="Optional chat ID to filter results",
        examples=["chat123456789"],
    ),
    limit: int = Query(
        10,
        ge=1,
        le=100,
        description="Maximum number of results",
    ),
    min_similarity: float = Query(
        0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold (0-1)",
    ),
) -> SearchResponse:
    """Search messages by semantic similarity.

    Finds messages that are semantically similar to the query text,
    regardless of exact keyword matches. Uses sentence embeddings
    to compute similarity.

    Results are sorted by similarity score in descending order.

    Args:
        query: Text to search for similar messages.
        contact_id: Optional chat ID to filter results to one conversation.
        limit: Maximum number of results to return.
        min_similarity: Minimum similarity threshold (0-1).

    Returns:
        SearchResponse with matching messages and similarity scores.

    Raises:
        HTTPException 500: If search fails.

    Example:
        ```
        GET /embeddings/search?query=dinner+plans&limit=5
        ```
    """
    try:
        store = _get_store()
        results = store.find_similar(
            query=query,
            chat_id=contact_id,
            limit=limit,
            min_similarity=min_similarity,
        )

        return SearchResponse(
            query=query,
            results=[_similar_to_response(r) for r in results],
            total_results=len(results),
        )

    except EmbeddingError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search error: {e.message}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {e}",
        ) from e


@router.get("/relationship/{contact_id}", response_model=RelationshipProfileResponse)
def get_relationship_profile(contact_id: str) -> RelationshipProfileResponse:
    """Get the relationship profile for a contact.

    Analyzes indexed messages to build a profile of communication
    patterns with a specific contact, including:
    - Message counts (sent/received)
    - Common conversation topics
    - Typical communication tone
    - Average message length
    - Response time patterns

    The profile is computed from indexed messages only. Call the
    /embeddings/index endpoint first to ensure messages are indexed.

    Args:
        contact_id: Chat ID to get the profile for.

    Returns:
        RelationshipProfileResponse with aggregated statistics.

    Raises:
        HTTPException 404: If no messages are indexed for this contact.
        HTTPException 500: If profile computation fails.

    Example:
        ```
        GET /embeddings/relationship/chat123456789
        ```
    """
    try:
        store = _get_store()
        profile = store.get_relationship_profile(contact_id)

        if profile.total_messages == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No indexed messages found for contact: {contact_id}. "
                "Use POST /embeddings/index to index messages first.",
            )

        return _profile_to_response(profile)

    except HTTPException:
        raise
    except EmbeddingError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Profile error: {e.message}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get relationship profile: {e}",
        ) from e


@router.get("/stats", response_model=IndexStatsResponse)
def get_index_stats() -> IndexStatsResponse:
    """Get statistics about the embedding index.

    Returns information about the embedding database including
    total indexed messages, unique conversations, date range,
    and database file size.

    Returns:
        IndexStatsResponse with index statistics.

    Raises:
        HTTPException 500: If stats retrieval fails.

    Example:
        ```
        GET /embeddings/stats
        ```
    """
    try:
        store = _get_store()
        stats = store.get_stats()

        return IndexStatsResponse(
            total_embeddings=stats["total_embeddings"],
            unique_chats=stats["unique_chats"],
            oldest_message=stats["oldest_message"],
            newest_message=stats["newest_message"],
            db_path=stats["db_path"],
            db_size_bytes=stats["db_size_bytes"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get index stats: {e}",
        ) from e
