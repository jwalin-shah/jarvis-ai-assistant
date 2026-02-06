"""Embeddings API endpoints.

Provides endpoints for semantic search and relationship profiling
using message embeddings (via sqlite-vec).

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
from jarvis.contacts.contact_profile import ContactProfile, get_contact_profile
from jarvis.search.vec_search import VecSearcher, get_vec_searcher

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
                "chat_id": "chat123456789",
            }
        }
    )

    success: bool = Field(..., description="Whether indexing succeeded")
    indexed: int = Field(..., description="Number of messages indexed", ge=0)
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

    message_id: int = Field(..., description="Message ID (rowid)")
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
                "updated_at": "2024-01-15T18:30:00Z",
            }
        }
    )

    contact_id: str = Field(..., description="Chat ID")
    display_name: str | None = Field(None, description="Contact display name")
    total_messages: int = Field(..., description="Total analyzed messages", ge=0)
    sent_count: int = Field(..., description="Messages sent (approx)", ge=0)
    received_count: int = Field(..., description="Messages received (approx)", ge=0)
    common_topics: list[str] = Field(default_factory=list, description="Common conversation topics")
    typical_tone: str = Field(
        ...,
        description="Typical communication tone (casual/professional/mixed)",
    )
    avg_message_length: float = Field(..., description="Average message length in characters")
    updated_at: str | None = Field(None, description="Last update timestamp")


class IndexStatsResponse(BaseModel):
    """Statistics about the embedding index."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_embeddings": 5000,
                "unique_chats": 25,
                "db_path": "/Users/user/.jarvis/jarvis.db",
            }
        }
    )

    total_embeddings: int = Field(..., description="Total number of indexed messages", ge=0)
    unique_chats: int = Field(..., description="Number of unique conversations indexed", ge=0)
    db_path: str = Field(..., description="Path to the database")


# =============================================================================
# Helper Functions
# =============================================================================


def _get_searcher() -> VecSearcher:
    """Get the vector searcher singleton."""
    return get_vec_searcher()


def _profile_to_response(profile: ContactProfile) -> RelationshipProfileResponse:
    """Convert ContactProfile to response model."""
    # Infer sent/received from my_message_count
    sent = profile.my_message_count
    received = profile.message_count - sent

    return RelationshipProfileResponse(
        contact_id=profile.contact_id,
        display_name=profile.contact_name,
        total_messages=profile.message_count,
        sent_count=sent,
        received_count=received,
        common_topics=profile.top_topics,
        typical_tone=profile.formality,  # Maps roughly (casual/formal)
        avg_message_length=profile.avg_message_length,
        updated_at=profile.updated_at,
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

    Args:
        request: Index request with chat_id and optional limit.

    Returns:
        IndexResponse with counts of indexed messages.
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
        searcher = _get_searcher()
        count = searcher.index_messages(messages)

        return IndexResponse(
            success=True,
            indexed=count,
            chat_id=request.chat_id,
        )

    except HTTPException:
        raise
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
) -> SearchResponse:
    """Search messages by semantic similarity.

    Args:
        query: Text to search for similar messages.
        contact_id: Optional chat ID to filter results.
        limit: Maximum number of results.

    Returns:
        SearchResponse with matching messages.
    """
    try:
        searcher = _get_searcher()
        results = searcher.search(
            query=query,
            chat_id=contact_id,
            limit=limit,
        )

        return SearchResponse(
            query=query,
            results=[
                SimilarMessageResponse(
                    message_id=r.rowid,
                    chat_id=r.chat_id or "",
                    text=r.text or "",
                    sender=r.sender,
                    sender_name=r.sender,  # We don't have separate sender_name in vec yet
                    timestamp=datetime.fromtimestamp(r.timestamp)
                    if r.timestamp
                    else datetime.now(),
                    is_from_me=bool(r.is_from_me),
                    similarity=r.score,
                )
                for r in results
            ],
            total_results=len(results),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {e}",
        ) from e


@router.get("/relationship/{contact_id}", response_model=RelationshipProfileResponse)
def get_profile(contact_id: str) -> RelationshipProfileResponse:
    """Get the relationship profile for a contact.

    Args:
        contact_id: Chat ID to get the profile for.

    Returns:
        RelationshipProfileResponse with aggregated statistics.
    """
    try:
        profile = get_contact_profile(contact_id)

        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"No profile found for contact: {contact_id}. "
                "Profile analysis usually happens during ingestion.",
            )

        return _profile_to_response(profile)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get relationship profile: {e}",
        ) from e


@router.get("/stats", response_model=IndexStatsResponse)
def get_index_stats() -> IndexStatsResponse:
    """Get statistics about the embedding index.

    Returns:
        IndexStatsResponse with index statistics.
    """
    try:
        searcher = _get_searcher()
        stats = searcher.get_stats()

        return IndexStatsResponse(
            total_embeddings=stats.get("total_embeddings", 0),
            unique_chats=stats.get("unique_chats", 0),
            db_path=stats.get("db_path", ""),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get index stats: {e}",
        ) from e
