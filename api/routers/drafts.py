"""AI-powered draft reply generation API endpoints.

Provides endpoints for generating draft replies and conversation summaries
using the MLX language model with conversation context via RAG (Retrieval
Augmented Generation).

These endpoints use the local MLX model to generate contextually appropriate
responses without sending any data to external services.

NOTE: All prompts and examples are imported from jarvis.prompts (jarvis/prompts/),
which is the single source of truth for all prompts in the JARVIS system.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from api.dependencies import get_imessage_reader
from api.ratelimit import (
    RATE_LIMIT_GENERATION,
    limiter,
)
from api.schemas import (
    ContextInfo,
    DateRange,
    DraftReplyRequest,
    DraftReplyResponse,
    DraftSummaryRequest,
    DraftSummaryResponse,
    ErrorResponse,
    RoutedReplyRequest,
    RoutedReplyResponse,
)
from api.services.drafts_helpers import sanitize_instruction
from api.services.drafts_service import (
    build_draft_suggestions,
    build_reply_context,
    build_smart_reply_input,
    ensure_messages_exist,
    fetch_messages,
    generate_summary_payload,
    route_smart_reply,
)
from integrations.imessage import ChatDBReader
from jarvis.prompts import API_REPLY_EXAMPLES, API_SUMMARY_EXAMPLES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/drafts", tags=["drafts"])

# Use centralized prompts from jarvis.prompts
REPLY_EXAMPLES = API_REPLY_EXAMPLES
SUMMARY_EXAMPLES = API_SUMMARY_EXAMPLES


def _sanitize_instruction(instruction: str | None) -> str | None:
    """Backward-compatible shim for legacy imports."""
    return sanitize_instruction(instruction)


@router.post(
    "/reply",
    response_model=DraftReplyResponse,
    response_model_exclude_unset=True,
    response_description="AI-generated reply suggestions with context metadata",
    summary="Generate draft reply suggestions",
    responses={
        200: {
            "description": "Reply suggestions generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "suggestions": [
                            {"text": "Yes, I'd love to! What time?", "confidence": 0.9},
                            {"text": "Sure! Let me know the details.", "confidence": 0.8},
                            {"text": "Sounds good to me!", "confidence": 0.7},
                        ],
                        "context_used": {
                            "num_messages": 20,
                            "participants": ["John Doe"],
                            "last_message": "Are you free for dinner tonight?",
                        },
                    }
                }
            },
        },
        403: {
            "description": "Full Disk Access not granted",
            "model": ErrorResponse,
        },
        404: {
            "description": "Conversation not found or no messages",
            "model": ErrorResponse,
        },
        408: {
            "description": "Request timed out",
            "model": ErrorResponse,
        },
        429: {
            "description": "Rate limit exceeded",
            "model": ErrorResponse,
        },
        500: {
            "description": "Failed to generate suggestions",
            "model": ErrorResponse,
        },
        503: {
            "description": "Model service unavailable",
            "model": ErrorResponse,
        },
    },
)
@limiter.limit(RATE_LIMIT_GENERATION)
async def generate_draft_reply(
    draft_request: DraftReplyRequest,
    request: Request,
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> DraftReplyResponse:
    """Generate AI-powered reply suggestions for a conversation.

    Uses the local MLX language model with conversation context to generate
    contextually appropriate reply suggestions. All processing is done locally
    without sending data to external services.

    **How It Works:**
    1. Retrieves recent messages from the conversation for context
    2. Formats the conversation as input for the language model
    3. Generates multiple reply suggestions with varying tones
    4. Returns suggestions ranked by confidence

    **Rate Limiting:**
    This endpoint is rate limited to 10 requests per minute to prevent
    resource exhaustion from CPU-intensive model generation.

    **Customizing Replies:**
    Use the `instruction` parameter to guide the tone or content:
    - "accept enthusiastically" - positive, excited response
    - "politely decline" - courteous refusal
    - "ask for more details" - clarifying questions
    - "be brief" - short, concise reply
    - "be formal" - professional tone

    **Example Request:**
    ```json
    {
        "chat_id": "chat123456789",
        "instruction": "accept enthusiastically",
        "num_suggestions": 3,
        "context_messages": 20
    }
    ```

    **Example Response:**
    ```json
    {
        "suggestions": [
            {"text": "Yes, I'd love to! What time works for you?", "confidence": 0.9},
            {"text": "Absolutely! Count me in!", "confidence": 0.8},
            {"text": "Sure thing! Looking forward to it!", "confidence": 0.7}
        ],
        "context_used": {
            "num_messages": 20,
            "participants": ["John Doe"],
            "last_message": "Are you free for dinner tonight?"
        }
    }
    ```

    Args:
        draft_request: DraftReplyRequest with chat_id, optional instruction,
                 num_suggestions (1-5), and context_messages (5-50)
        request: FastAPI request object (for rate limiting)

    Returns:
        DraftReplyResponse with list of suggestions and context metadata

    Raises:
        HTTPException 403: Full Disk Access not granted
        HTTPException 404: No messages found for the conversation
        HTTPException 408: Request timed out
        HTTPException 429: Rate limit exceeded
        HTTPException 500: Failed to generate suggestions
        HTTPException 503: Model service unavailable
    """
    messages = await fetch_messages(reader, draft_request.chat_id, draft_request.context_messages)
    ensure_messages_exist(draft_request.chat_id, messages)
    last_message, participants, thread = build_reply_context(
        messages,
        draft_request.context_messages,
    )

    suggestions = await build_draft_suggestions(
        chat_id=draft_request.chat_id,
        last_message=last_message,
        thread=thread,
        instruction=draft_request.instruction,
        num_suggestions=draft_request.num_suggestions,
    )

    if not suggestions:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate any reply suggestions",
        )

    return DraftReplyResponse(
        suggestions=suggestions,
        context_used=ContextInfo(
            num_messages=len(messages),
            participants=participants,
            last_message=last_message,
        ),
    )


@router.post(
    "/summarize",
    response_model=DraftSummaryResponse,
    response_model_exclude_unset=True,
    response_description="AI-generated conversation summary with key points",
    summary="Summarize a conversation",
    responses={
        200: {
            "description": "Summary generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "summary": "Discussion about planning a weekend trip to the beach.",
                        "key_points": [
                            "Decided on Saturday departure",
                            "Meeting at John's place at 9am",
                            "Everyone bringing snacks",
                        ],
                        "date_range": {"start": "2024-01-10", "end": "2024-01-15"},
                    }
                }
            },
        },
        403: {
            "description": "Full Disk Access not granted",
            "model": ErrorResponse,
        },
        404: {
            "description": "Conversation not found or no messages",
            "model": ErrorResponse,
        },
        408: {
            "description": "Request timed out",
            "model": ErrorResponse,
        },
        429: {
            "description": "Rate limit exceeded",
            "model": ErrorResponse,
        },
        500: {
            "description": "Failed to generate summary",
            "model": ErrorResponse,
        },
        503: {
            "description": "Model service unavailable",
            "model": ErrorResponse,
        },
    },
)
@limiter.limit(RATE_LIMIT_GENERATION)
async def summarize_conversation(
    summary_request: DraftSummaryRequest,
    request: Request,
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> DraftSummaryResponse:
    """Summarize a conversation using AI.

    Analyzes the specified number of messages from a conversation and generates
    a concise summary with key points. Uses the local MLX model for processing.

    **Rate Limiting:**
    This endpoint is rate limited to 10 requests per minute to prevent
    resource exhaustion from CPU-intensive model generation.

    **What You Get:**
    - A 1-2 sentence summary of the conversation
    - 2-4 key points extracted from the discussion
    - The date range of messages included in the summary

    **Use Cases:**
    - Catch up on a conversation you've missed
    - Get a quick overview of a long group chat
    - Find important decisions or action items

    **Example Request:**
    ```json
    {
        "chat_id": "chat123456789",
        "num_messages": 50
    }
    ```

    **Example Response:**
    ```json
    {
        "summary": "Discussion about planning a weekend trip to the beach.",
        "key_points": [
            "Decided on Saturday departure at 9am",
            "Meeting at John's place",
            "Everyone bringing snacks and sunscreen",
            "Return planned for Sunday evening"
        ],
        "date_range": {
            "start": "2024-01-10",
            "end": "2024-01-15"
        }
    }
    ```

    Args:
        summary_request: DraftSummaryRequest with chat_id and num_messages (10-200)
        request: FastAPI request object (for rate limiting)

    Returns:
        DraftSummaryResponse with summary, key_points, and date_range

    Raises:
        HTTPException 403: Full Disk Access not granted
        HTTPException 404: No messages found for the conversation
        HTTPException 408: Request timed out
        HTTPException 429: Rate limit exceeded
        HTTPException 500: Failed to generate summary
        HTTPException 503: Model service unavailable
    """
    messages = await fetch_messages(reader, summary_request.chat_id, summary_request.num_messages)
    ensure_messages_exist(summary_request.chat_id, messages)

    # Determine date range (messages are newest-first)
    newest_date = messages[0].date
    oldest_date = messages[-1].date

    summary, key_points = await generate_summary_payload(
        summary_request.chat_id,
        messages,
        SUMMARY_EXAMPLES,
    )

    return DraftSummaryResponse(
        summary=summary,
        key_points=key_points,
        date_range=DateRange(
            start=oldest_date.strftime("%Y-%m-%d"),
            end=newest_date.strftime("%Y-%m-%d"),
        ),
    )


@router.post(
    "/smart-reply",
    response_model=RoutedReplyResponse,
    response_model_exclude_unset=True,
    response_description="Smart routed reply with confidence and source metadata",
    summary="Generate smart routed reply",
    responses={
        200: {
            "description": "Reply generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "response": "Sure, sounds great!",
                        "response_type": "generated",
                        "confidence": "high",
                        "similarity_score": 0.92,
                    }
                }
            },
        },
        403: {
            "description": "Full Disk Access not granted",
            "model": ErrorResponse,
        },
        404: {
            "description": "Conversation not found or no messages",
            "model": ErrorResponse,
        },
        408: {
            "description": "Request timed out",
            "model": ErrorResponse,
        },
        429: {
            "description": "Rate limit exceeded",
            "model": ErrorResponse,
        },
        500: {
            "description": "Failed to generate reply",
            "model": ErrorResponse,
        },
    },
)
@limiter.limit(RATE_LIMIT_GENERATION)
async def generate_smart_reply(
    routed_request: RoutedReplyRequest,
    request: Request,
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> RoutedReplyResponse:
    """Generate a smart routed reply using the simplified router.

    Current behavior:
    - Non-empty messages: generated reply path.
    - Empty/invalid messages: clarify response.
    - Similarity and mobilization data influence prompt quality and confidence.

    Args:
        routed_request: RoutedReplyRequest with chat_id and optional last_message
        request: FastAPI request object (for rate limiting)

    Returns:
        RoutedReplyResponse with response, type, confidence, and metadata
    """
    messages = await fetch_messages(reader, routed_request.chat_id, routed_request.context_messages)
    ensure_messages_exist(routed_request.chat_id, messages)
    last_message, thread_context, context_info = build_smart_reply_input(
        messages=messages,
        requested_last_message=routed_request.last_message,
    )
    result = await route_smart_reply(routed_request.chat_id, last_message, thread_context)

    return RoutedReplyResponse(
        response=result["response"],
        response_type=result["type"],
        confidence=result["confidence"],
        similarity_score=result.get("similarity_score", 0.0),
        cluster_name=result.get("cluster_name"),
        similar_triggers=result.get("similar_triggers"),
        context_used=context_info,
    )
