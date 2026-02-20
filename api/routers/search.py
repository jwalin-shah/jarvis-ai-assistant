"""Semantic search API endpoints.

Provides endpoints for AI-powered semantic search across iMessage conversations
using sentence embeddings.

Uses the all-MiniLM-L6-v2 model for encoding messages and queries,
then finds matches by cosine similarity rather than keyword matching.
"""

import sqlite3
import threading
from datetime import datetime
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from api.schemas import ErrorResponse, MessageResponse
from jarvis.search.vec_search import VecSearcher, get_vec_searcher

router = APIRouter(prefix="/search", tags=["search"])

_searcher_lock = threading.Lock()
_vec_searcher: VecSearcher | None = None


def _get_vec_searcher() -> VecSearcher:
    """Get or create singleton VecSearcher instance."""
    global _vec_searcher

    if _vec_searcher is not None:
        return _vec_searcher

    with _searcher_lock:
        if _vec_searcher is None:
            _vec_searcher = get_vec_searcher()
    return _vec_searcher


class SemanticSearchRequest(BaseModel):
    """Request body for semantic search.

    Example:
        ```json
        {
            "query": "dinner plans this weekend",
            "limit": 20,
            "threshold": 0.3,
            "filters": {
                "sender": "+15551234567",
                "after": "2024-01-01T00:00:00Z"
            }
        }
        ```
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural language search query",
        examples=["dinner plans", "meeting tomorrow", "project deadline"],
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of results to return",
    )
    threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0.0-1.0). Higher = more relevant results only.",
    )
    index_limit: int = Field(
        default=1000,
        ge=100,
        le=5000,
        description="Maximum messages to search through (higher = slower but more thorough)",
    )


class SemanticSearchFilters(BaseModel):
    """Optional filters for semantic search.

    Example:
        ```json
        {
            "sender": "+15551234567",
            "chat_id": "chat123",
            "after": "2024-01-01T00:00:00Z",
            "before": "2024-12-31T23:59:59Z",
            "has_attachments": true
        }
        ```
    """

    sender: str | None = Field(
        default=None,
        description="Filter by sender phone/email. Use 'me' for your own messages.",
        examples=["+15551234567", "me"],
    )
    chat_id: str | None = Field(
        default=None,
        description="Filter to a specific conversation",
        examples=["chat123456789"],
    )
    after: datetime | None = Field(
        default=None,
        description="Only messages after this date (ISO 8601)",
        examples=["2024-01-01T00:00:00Z"],
    )
    before: datetime | None = Field(
        default=None,
        description="Only messages before this date (ISO 8601)",
        examples=["2024-12-31T23:59:59Z"],
    )
    has_attachments: bool | None = Field(
        default=None,
        description="Filter by presence of attachments",
    )


class SemanticSearchRequestWithFilters(SemanticSearchRequest):
    """Request body for semantic search with filters.

    Example:
        ```json
        {
            "query": "dinner plans this weekend",
            "limit": 20,
            "threshold": 0.3,
            "filters": {
                "sender": "+15551234567",
                "after": "2024-01-01T00:00:00Z"
            }
        }
        ```
    """

    filters: SemanticSearchFilters | None = Field(
        default=None,
        description="Optional filters to narrow search scope",
    )


class SemanticSearchResultItem(BaseModel):
    """A single semantic search result with similarity score.

    Example:
        ```json
        {
            "message": {
                "id": 12345,
                "chat_id": "chat123",
                "sender": "+15551234567",
                "text": "Let's get dinner this Saturday!",
                "date": "2024-01-15T18:30:00Z",
                "is_from_me": false
            },
            "similarity": 0.87
        }
        ```
    """

    message: MessageResponse = Field(
        ...,
        description="The matching message",
    )
    similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score (0.0-1.0). Higher = more relevant.",
        examples=[0.87, 0.72, 0.55],
    )


class SemanticSearchResponse(BaseModel):
    """Response containing semantic search results.

    Example:
        ```json
        {
            "query": "dinner plans",
            "results": [...],
            "total_results": 15,
            "threshold_used": 0.3,
            "messages_searched": 1000
        }
        ```
    """

    query: str = Field(
        ...,
        description="The original search query",
    )
    results: list[SemanticSearchResultItem] = Field(
        ...,
        description="List of matching messages with similarity scores",
    )
    total_results: int = Field(
        ...,
        description="Number of results returned",
        ge=0,
    )
    threshold_used: float = Field(
        ...,
        description="The similarity threshold that was applied",
    )
    messages_searched: int = Field(
        ...,
        description="Number of messages that were searched",
        ge=0,
    )


class CacheStatsResponse(BaseModel):
    """Embedding cache statistics.

    Example:
        ```json
        {
            "embedding_count": 5000,
            "size_bytes": 7680000,
            "size_mb": 7.32
        }
        ```
    """

    embedding_count: int = Field(
        ...,
        description="Number of cached embeddings",
        ge=0,
    )
    size_bytes: int = Field(
        ...,
        description="Total size of cached embeddings in bytes",
        ge=0,
    )
    size_mb: float = Field(
        ...,
        description="Total size of cached embeddings in megabytes",
        ge=0,
    )


@router.post(
    "/semantic",
    response_model=SemanticSearchResponse,
    response_model_exclude_unset=True,
    summary="Semantic search across messages",
    response_description="Search results ranked by semantic similarity",
    responses={
        200: {
            "description": "Search completed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "query": "dinner plans",
                        "results": [
                            {
                                "message": {
                                    "id": 12345,
                                    "chat_id": "chat123",
                                    "sender": "+15551234567",
                                    "sender_name": "John",
                                    "text": "Let's get dinner this Saturday!",
                                    "date": "2024-01-15T18:30:00Z",
                                    "is_from_me": False,
                                    "attachments": [],
                                    "reactions": [],
                                },
                                "similarity": 0.87,
                            }
                        ],
                        "total_results": 1,
                        "threshold_used": 0.3,
                        "messages_searched": 1000,
                    }
                }
            },
        },
        403: {
            "description": "Full Disk Access not granted",
            "model": ErrorResponse,
        },
    },
)
async def semantic_search(
    request: SemanticSearchRequestWithFilters,
) -> SemanticSearchResponse:
    """Search messages by semantic similarity.

    Uses AI-powered semantic search to find messages by meaning rather than
    exact keyword matching. This allows finding messages that are conceptually
    similar even if they don't contain the exact search terms.

    **How it works:**
    1. Your query is converted to a semantic embedding using all-MiniLM-L6-v2
    2. Messages are compared using cosine similarity
    3. Results are ranked by relevance score

    **Example Queries:**
    - "dinner plans" - finds messages about eating out, restaurants, food plans
    - "meeting tomorrow" - finds scheduling discussions, appointment confirmations
    - "running late" - finds messages about delays, being behind schedule

    **Performance Notes:**
    - First search may be slower as messages are indexed
    - Subsequent searches are faster due to embedding caching
    - Use `index_limit` to control search thoroughness vs speed

    **Filtering:**
    Use the `filters` object to narrow results by:
    - `sender`: Phone number, email, or "me" for your own messages
    - `chat_id`: Specific conversation only
    - `after`/`before`: Date range
    - `has_attachments`: Messages with/without attachments

    Args:
        request: Search query and options

    Returns:
        SemanticSearchResponse with ranked results and metadata

    Raises:
        HTTPException 403: Full Disk Access permission not granted
    """
    chat_id_filter = request.filters.chat_id if request.filters else None

    def _do_search() -> list[Any]:
        """Perform blocking search operation in thread pool."""
        searcher = _get_vec_searcher()
        results = searcher.search(
            query=request.query,
            chat_id=chat_id_filter,
            limit=request.limit,
        )

        # Post-filter by threshold
        results = [r for r in results if r.score >= request.threshold]

        # Post-filter by sender if requested
        if request.filters and request.filters.sender:
            sender = request.filters.sender
            results = [r for r in results if r.sender == sender]

        return results

    # Run blocking search in thread pool
    results = await run_in_threadpool(_do_search)

    # Convert to response format
    result_items = [
        SemanticSearchResultItem(
            message=MessageResponse(
                id=r.rowid,
                chat_id=r.chat_id or "",
                sender=r.sender or "",
                text=r.text or "",
                date=datetime.fromtimestamp(r.timestamp) if r.timestamp else datetime.min,
                is_from_me=r.is_from_me or False,
            ),
            similarity=round(r.score, 4),
        )
        for r in results
    ]

    return SemanticSearchResponse(
        query=request.query,
        results=result_items,
        total_results=len(result_items),
        threshold_used=request.threshold,
        messages_searched=request.limit,
    )


@router.get(
    "/semantic/cache/stats",
    response_model=CacheStatsResponse,
    summary="Get embedding cache statistics",
    response_description="Cache statistics including size and count",
)
async def get_cache_stats() -> CacheStatsResponse:
    """Get statistics about the vector search index.

    Returns information about indexed message embeddings including
    the count and estimated storage size.

    Returns:
        CacheStatsResponse with index statistics
    """
    from jarvis.db import get_db

    def _get_stats() -> dict[str, Any]:
        """Get vec table stats in thread pool."""
        db = get_db()
        with db.connection() as conn:
            try:
                row = conn.execute("SELECT COUNT(*) as cnt FROM vec_messages").fetchone()
                count = row["cnt"] if row else 0
            except (OSError, sqlite3.Error):
                count = 0
            # Estimate: each int8 embedding is 384 bytes + metadata
            size_bytes = count * 400
            return {
                "embedding_count": count,
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 2),
            }

    stats = await run_in_threadpool(_get_stats)
    return CacheStatsResponse(
        embedding_count=int(stats["embedding_count"]),
        size_bytes=int(stats["size_bytes"]),
        size_mb=stats["size_mb"],
    )


@router.delete(
    "/semantic/cache",
    summary="Clear embedding cache",
    response_description="Confirmation of cache clear",
)
async def clear_cache() -> dict[str, str]:
    """Clear the vector search index.

    This removes all indexed message embeddings. The next indexing
    operation will recompute them.

    Use this if:
    - Embeddings seem stale or incorrect
    - You want to free up disk space
    - You've updated the embedding model

    Returns:
        Confirmation message
    """
    from jarvis.db import get_db

    def _clear_cache() -> dict[str, Any]:
        """Clear vec table in thread pool."""
        db = get_db()
        with db.connection() as conn:
            try:
                row = conn.execute("SELECT COUNT(*) as cnt FROM vec_messages").fetchone()
                count = row["cnt"] if row else 0
                size_bytes = count * 400
                conn.execute("DELETE FROM vec_messages")
                conn.commit()
                return {
                    "embedding_count": count,
                    "size_mb": round(size_bytes / (1024 * 1024), 2),
                }
            except (OSError, sqlite3.Error):
                return {"embedding_count": 0, "size_mb": 0.0}

    stats_before = await run_in_threadpool(_clear_cache)
    return {
        "status": "success",
        "message": f"Cleared {stats_before['embedding_count']} indexed embeddings "
        f"({stats_before['size_mb']} MB)",
    }
