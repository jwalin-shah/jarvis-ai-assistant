"""Conversation Threading API endpoints.

Provides endpoints for getting threaded views of conversations,
where messages are grouped into logical threads based on time gaps,
topic shifts, and explicit reply references.

All endpoints require Full Disk Access permission to read the iMessage database.
"""

from datetime import datetime
from typing import cast

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_imessage_reader
from api.schemas import (
    ErrorResponse,
    MessageResponse,
    ThreadedMessageResponse,
    ThreadedViewResponse,
    ThreadingConfigRequest,
    ThreadResponse,
)
from integrations.imessage import ChatDBReader
from jarvis.cache import TTLCache
from jarvis.threading import (
    ThreadAnalyzer,
    ThreadingConfig,
    get_thread_analyzer,
)

router = APIRouter(prefix="/conversations", tags=["threads"])

# Cache for threaded views (5 minute TTL)
_thread_cache: TTLCache | None = None


def get_thread_cache() -> TTLCache:
    """Get the singleton thread cache."""
    global _thread_cache
    if _thread_cache is None:
        _thread_cache = TTLCache(ttl_seconds=300, maxsize=100)
    return _thread_cache


@router.get(
    "/{chat_id}/threads",
    response_model=ThreadedViewResponse,
    response_model_exclude_unset=True,
    response_description="Threaded view of conversation",
    summary="Get threaded view of a conversation",
    responses={
        200: {
            "description": "Threaded view retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "chat_id": "chat123456789",
                        "threads": [
                            {
                                "thread_id": "a1b2c3d4e5f67890",
                                "messages": [12345, 12346, 12347],
                                "topic_label": "Dinner Plans",
                                "message_count": 3,
                            }
                        ],
                        "messages": [
                            {
                                "id": 12345,
                                "text": "Are you free for dinner?",
                                "thread_id": "a1b2c3d4e5f67890",
                                "thread_position": 0,
                                "is_thread_start": True,
                            }
                        ],
                        "total_threads": 1,
                        "total_messages": 3,
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
def get_threaded_view(
    chat_id: str,
    limit: int = Query(
        default=200,
        ge=1,
        le=1000,
        description="Maximum number of messages to analyze",
        examples=[100, 200, 500],
    ),
    before: datetime | None = Query(
        default=None,
        description="Only analyze messages before this date (for pagination)",
        examples=["2024-01-15T10:30:00Z"],
    ),
    time_gap_minutes: int = Query(
        default=30,
        ge=5,
        le=1440,
        description="Minutes of silence to start new thread",
        examples=[30, 60],
    ),
    use_semantic: bool = Query(
        default=True,
        description="Use ML-based topic detection for thread boundaries",
    ),
    refresh: bool = Query(
        default=False,
        description="Force refresh, bypassing cache",
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> ThreadedViewResponse:
    """Get a threaded view of a conversation.

    Analyzes messages and groups them into logical threads based on:
    - **Time gaps**: Long pauses between messages start new threads
    - **Topic shifts**: Semantic analysis detects topic changes
    - **Reply references**: Explicit reply_to_id links keep messages together

    **Use Cases:**
    - Displaying conversations organized by topic
    - Finding related messages in a thread
    - Understanding conversation flow

    **Caching:**
    Results are cached for 5 minutes. Use `refresh=true` to bypass cache.

    **Pagination:**
    Use the `before` parameter with the `date` of the oldest message to
    fetch older threaded views.

    **Example Response:**
    ```json
    {
        "chat_id": "chat123456789",
        "threads": [
            {
                "thread_id": "a1b2c3d4e5f67890",
                "messages": [12345, 12346, 12347],
                "topic_label": "Dinner Plans",
                "start_time": "2024-01-15T18:00:00Z",
                "end_time": "2024-01-15T18:30:00Z",
                "participant_count": 2,
                "message_count": 3
            }
        ],
        "messages": [
            {
                "id": 12345,
                "chat_id": "chat123456789",
                "sender": "+15551234567",
                "text": "Are you free for dinner?",
                "date": "2024-01-15T18:00:00Z",
                "is_from_me": false,
                "thread_id": "a1b2c3d4e5f67890",
                "thread_position": 0,
                "is_thread_start": true
            }
        ],
        "total_threads": 1,
        "total_messages": 3
    }
    ```

    Args:
        chat_id: The unique conversation identifier
        limit: Maximum messages to analyze (1-1000, default 200)
        before: Only messages before this date (for pagination)
        time_gap_minutes: Minutes of silence to start new thread (5-1440, default 30)
        use_semantic: Whether to use ML topic detection (default true)
        refresh: Force cache refresh (default false)

    Returns:
        ThreadedViewResponse with threads and threaded messages

    Raises:
        HTTPException 403: Full Disk Access permission not granted
    """
    # Build cache key
    before_str = before.isoformat() if before else "none"
    cache_key = f"threads:{chat_id}:{limit}:{before_str}:{time_gap_minutes}:{use_semantic}"

    # Check cache unless refresh requested
    if not refresh:
        cache = get_thread_cache()
        found, cached = cache.get(cache_key)
        if found and cached is not None:
            return cast(ThreadedViewResponse, cached)

    # Fetch messages
    messages = reader.get_messages(chat_id=chat_id, limit=limit, before=before)

    if not messages:
        return ThreadedViewResponse(
            chat_id=chat_id,
            threads=[],
            messages=[],
            total_threads=0,
            total_messages=0,
        )

    # Configure and run thread analysis
    config = ThreadingConfig(
        time_gap_threshold_minutes=time_gap_minutes,
        use_semantic_analysis=use_semantic,
    )
    analyzer = ThreadAnalyzer(config)

    # Get threads
    threads = analyzer.analyze_threads(messages, chat_id)

    # Get threaded messages
    threaded_messages = analyzer.get_threaded_messages(messages, chat_id)

    # Convert to response models
    thread_responses = [
        ThreadResponse(
            thread_id=t.thread_id,
            messages=t.messages,
            topic_label=t.topic_label,
            start_time=t.start_time,
            end_time=t.end_time,
            participant_count=t.participant_count,
            message_count=t.message_count,
        )
        for t in threads
    ]

    message_responses = [
        ThreadedMessageResponse(
            id=tm.message.id,
            chat_id=tm.message.chat_id,
            sender=tm.message.sender,
            sender_name=tm.message.sender_name,
            text=tm.message.text,
            date=tm.message.date,
            is_from_me=tm.message.is_from_me,
            attachments=[],  # Simplified for threading view
            reply_to_id=tm.message.reply_to_id,
            reactions=[],  # Simplified for threading view
            is_system_message=tm.message.is_system_message,
            thread_id=tm.thread_id,
            thread_position=tm.thread_position,
            is_thread_start=tm.is_thread_start,
        )
        for tm in threaded_messages
    ]

    result = ThreadedViewResponse(
        chat_id=chat_id,
        threads=thread_responses,
        messages=message_responses,
        total_threads=len(threads),
        total_messages=len(messages),
    )

    # Cache result
    cache = get_thread_cache()
    cache.set(cache_key, result)

    return result


@router.get(
    "/{chat_id}/threads/{thread_id}",
    response_model=list[MessageResponse],
    response_model_exclude_unset=True,
    response_description="Messages in the specified thread",
    summary="Get messages for a specific thread",
    responses={
        200: {
            "description": "Thread messages retrieved successfully",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": 12345,
                            "chat_id": "chat123456789",
                            "sender": "+15551234567",
                            "text": "Are you free for dinner?",
                            "date": "2024-01-15T18:00:00Z",
                            "is_from_me": False,
                        }
                    ]
                }
            },
        },
        403: {
            "description": "Full Disk Access not granted",
            "model": ErrorResponse,
        },
        404: {
            "description": "Thread not found",
            "model": ErrorResponse,
        },
    },
)
def get_thread_messages(
    chat_id: str,
    thread_id: str,
    limit: int = Query(
        default=200,
        ge=1,
        le=1000,
        description="Maximum messages to fetch for thread detection",
        examples=[100, 200],
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> list[MessageResponse]:
    """Get all messages belonging to a specific thread.

    First performs thread analysis on the conversation, then returns
    only the messages belonging to the specified thread.

    **Use Cases:**
    - Viewing a single thread in isolation
    - Getting context for a specific topic

    **Example Response:**
    ```json
    [
        {
            "id": 12345,
            "chat_id": "chat123456789",
            "sender": "+15551234567",
            "text": "Are you free for dinner?",
            "date": "2024-01-15T18:00:00Z",
            "is_from_me": false,
            "attachments": [],
            "reactions": []
        },
        {
            "id": 12346,
            "chat_id": "chat123456789",
            "sender": "me",
            "text": "Sure! What time?",
            "date": "2024-01-15T18:01:00Z",
            "is_from_me": true,
            "attachments": [],
            "reactions": []
        }
    ]
    ```

    Args:
        chat_id: The conversation identifier
        thread_id: The thread identifier to fetch
        limit: Max messages to analyze for finding thread (1-1000)

    Returns:
        List of MessageResponse objects in the thread

    Raises:
        HTTPException 403: Full Disk Access permission not granted
        HTTPException 404: Thread not found
    """
    # Fetch messages
    messages = reader.get_messages(chat_id=chat_id, limit=limit)

    if not messages:
        return []

    # Run thread analysis
    analyzer = get_thread_analyzer()
    threads = analyzer.analyze_threads(messages, chat_id)

    # Find the requested thread
    target_thread = None
    for thread in threads:
        if thread.thread_id == thread_id:
            target_thread = thread
            break

    if target_thread is None:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")

    # Get message IDs in thread
    thread_message_ids = set(target_thread.messages)

    # Filter and return messages in thread order
    thread_messages = [
        MessageResponse.model_validate(m) for m in messages if m.id in thread_message_ids
    ]

    return thread_messages


@router.post(
    "/{chat_id}/threads/analyze",
    response_model=list[ThreadResponse],
    response_model_exclude_unset=True,
    response_description="List of detected threads",
    summary="Analyze conversation for threads",
    responses={
        200: {
            "description": "Thread analysis completed successfully",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "thread_id": "a1b2c3d4e5f67890",
                            "messages": [12345, 12346, 12347],
                            "topic_label": "Dinner Plans",
                            "message_count": 3,
                        }
                    ]
                }
            },
        },
        403: {
            "description": "Full Disk Access not granted",
            "model": ErrorResponse,
        },
    },
)
def analyze_threads(
    chat_id: str,
    config: ThreadingConfigRequest | None = None,
    limit: int = Query(
        default=200,
        ge=1,
        le=1000,
        description="Maximum messages to analyze",
        examples=[100, 200, 500],
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> list[ThreadResponse]:
    """Analyze a conversation and return detected threads.

    Performs thread analysis with custom configuration options.
    Returns only the thread metadata without full messages.

    **Configuration Options:**
    - `time_gap_threshold_minutes`: Minutes of silence to start new thread
    - `semantic_similarity_threshold`: Min similarity to group messages
    - `use_semantic_analysis`: Enable/disable ML topic detection

    **Use Cases:**
    - Getting an overview of conversation topics
    - Identifying thread boundaries for UI

    **Example Request:**
    ```json
    {
        "time_gap_threshold_minutes": 60,
        "semantic_similarity_threshold": 0.5,
        "use_semantic_analysis": true
    }
    ```

    **Example Response:**
    ```json
    [
        {
            "thread_id": "a1b2c3d4e5f67890",
            "messages": [12345, 12346, 12347],
            "topic_label": "Dinner Plans",
            "start_time": "2024-01-15T18:00:00Z",
            "end_time": "2024-01-15T18:30:00Z",
            "participant_count": 2,
            "message_count": 3
        },
        {
            "thread_id": "b2c3d4e5f6789012",
            "messages": [12348, 12349],
            "topic_label": "Work Discussion",
            "start_time": "2024-01-15T20:00:00Z",
            "end_time": "2024-01-15T20:15:00Z",
            "participant_count": 2,
            "message_count": 2
        }
    ]
    ```

    Args:
        chat_id: The conversation identifier
        config: Optional threading configuration
        limit: Max messages to analyze (1-1000)

    Returns:
        List of ThreadResponse objects

    Raises:
        HTTPException 403: Full Disk Access permission not granted
    """
    # Fetch messages
    messages = reader.get_messages(chat_id=chat_id, limit=limit)

    if not messages:
        return []

    # Configure analyzer
    threading_config = ThreadingConfig()
    if config:
        threading_config.time_gap_threshold_minutes = config.time_gap_threshold_minutes
        threading_config.semantic_similarity_threshold = config.semantic_similarity_threshold
        threading_config.use_semantic_analysis = config.use_semantic_analysis

    analyzer = ThreadAnalyzer(threading_config)
    threads = analyzer.analyze_threads(messages, chat_id)

    # Convert to response models
    return [
        ThreadResponse(
            thread_id=t.thread_id,
            messages=t.messages,
            topic_label=t.topic_label,
            start_time=t.start_time,
            end_time=t.end_time,
            participant_count=t.participant_count,
            message_count=t.message_count,
        )
        for t in threads
    ]
