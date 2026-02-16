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

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from api.dependencies import get_imessage_reader
from api.ratelimit import (
    RATE_LIMIT_GENERATION,
    get_timeout_generation,
    limiter,
)
from api.schemas import (
    ContextInfo,
    DateRange,
    DraftReplyRequest,
    DraftReplyResponse,
    DraftSuggestion,
    DraftSummaryRequest,
    DraftSummaryResponse,
    ErrorResponse,
    RoutedReplyRequest,
    RoutedReplyResponse,
)
from integrations.imessage import ChatDBReader
from jarvis.contracts.pipeline import MessageContext
from jarvis.errors import ModelError, iMessageQueryError
from jarvis.model_warmer import get_warm_generator
from jarvis.prompts import API_REPLY_EXAMPLES, API_SUMMARY_EXAMPLES
from api.services.drafts_helpers import (
    build_summary_prompt,
    format_messages_for_context,
    parse_summary_response,
    sanitize_instruction,
)
from api.services.drafts_pipeline import (
    generate_summary,
    route_reply_sync,
    run_classification_and_search,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/drafts", tags=["drafts"])

# Use centralized prompts from jarvis.prompts
REPLY_EXAMPLES = API_REPLY_EXAMPLES
SUMMARY_EXAMPLES = API_SUMMARY_EXAMPLES



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
    # Fetch conversation messages for context (I/O bound - run in threadpool)
    try:
        messages = await run_in_threadpool(
            reader.get_messages,
            chat_id=draft_request.chat_id,
            limit=draft_request.context_messages,
        )
    except Exception as e:
        logger.error("Failed to fetch messages for chat %s: %s", draft_request.chat_id, e)
        raise iMessageQueryError(
            f"Failed to fetch conversation context for chat: {draft_request.chat_id}",
            cause=e,
        )

    if not messages:
        from jarvis.errors import ConversationNotFoundError
        raise ConversationNotFoundError(f"No messages found for chat_id: {draft_request.chat_id}")

    # Last incoming message and thread (chronological, for RAG)
    last_message = None
    for msg in messages:
        if not msg.is_from_me and msg.text:
            last_message = msg.text
            break
    if not last_message:
        last_message = messages[0].text if messages else ""
    participants = list({m.sender_name or m.sender for m in messages if not m.is_from_me})
    thread = [m.text for m in reversed(messages) if m.text][: draft_request.context_messages]

    from jarvis.reply_service import get_reply_service

    reply_service = get_reply_service()

    try:
        classification, search_results = await run_in_threadpool(
            run_classification_and_search,
            last_message or "",
            thread,
        )
    except Exception as e:
        logger.error("Classification/search failed: %s", e)
        raise ModelError("Model service unavailable", cause=e) from e

    context = MessageContext(
        chat_id=draft_request.chat_id,
        message_text=last_message or "",
        is_from_me=False,
        timestamp=datetime.now(UTC),
        metadata={"thread": thread},
    )

    # Variety instructions for multiple suggestions
    base_instruction = sanitize_instruction(draft_request.instruction)

    variant_instructions: list[str | None] = [base_instruction]
    if draft_request.num_suggestions > 1:
        variant_instructions.append(
            (base_instruction + " (slightly more casual)")
            if base_instruction
            else "be slightly more casual"
        )
    if draft_request.num_suggestions > 2:
        variant_instructions.append(
            (base_instruction + " (concise)") if base_instruction else "be concise"
        )

    suggestions: list[DraftSuggestion] = []
    try:
        async with asyncio.timeout(get_timeout_generation()):
            for i in range(draft_request.num_suggestions):
                instruction = (
                    variant_instructions[i]
                    if i < len(variant_instructions)
                    else variant_instructions[0]
                )
                try:
                    gen_response = await run_in_threadpool(
                        reply_service.generate_reply,
                        context,
                        classification,
                        search_results,
                        thread,
                        None,
                        None,
                        instruction,
                    )
                except Exception as e:
                    logger.warning("Generation %d failed: %s", i, e)
                    continue
                if gen_response.response:
                    confidence = max(0.5, gen_response.confidence)
                    suggestions.append(
                        DraftSuggestion(
                            text=gen_response.response,
                            confidence=confidence,
                        )
                    )
                    await run_in_threadpool(
                        reply_service.log_custom_generation,
                        chat_id=draft_request.chat_id,
                        incoming_text=last_message or "",
                        final_prompt=gen_response.metadata.get("final_prompt", ""),
                        response_text=gen_response.response,
                        confidence=confidence,
                        category="draft_reply",
                        metadata={"suggestion_index": i},
                    )
    except TimeoutError:
        logger.warning("Generation timed out after %s seconds", get_timeout_generation())
        if not suggestions:
            raise HTTPException(
                status_code=408,
                detail=f"Request timed out after {get_timeout_generation()} seconds",
            ) from None

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
    # Fetch conversation messages (I/O bound - run in threadpool)
    try:
        messages = await run_in_threadpool(
            reader.get_messages,
            chat_id=summary_request.chat_id,
            limit=summary_request.num_messages,
        )
    except Exception as e:
        logger.error("Failed to fetch messages for chat %s: %s", summary_request.chat_id, e)
        raise iMessageQueryError(
            f"Failed to fetch conversation for chat: {summary_request.chat_id}",
            cause=e,
        )

    if not messages:
        from jarvis.errors import ConversationNotFoundError
        raise ConversationNotFoundError(f"No messages found for chat_id: {summary_request.chat_id}")

    # Build context
    context_text = format_messages_for_context(messages)

    # Determine date range (messages are newest-first)
    newest_date = messages[0].date
    oldest_date = messages[-1].date

    # Get the generator
    try:
        generator = await run_in_threadpool(get_warm_generator)
    except Exception as e:
        logger.error("Failed to get generator: %s", e)
        raise ModelError(
            "Model service unavailable",
            cause=e,
        )

    # Generate summary with timeout
    prompt = build_summary_prompt(context_text, len(messages))

    try:
        async with asyncio.timeout(get_timeout_generation()):
            response_text = await run_in_threadpool(
                generate_summary,
                generator,
                prompt,
                SUMMARY_EXAMPLES,
            )
            summary, key_points = parse_summary_response(response_text)

            # Log for traceability
            from jarvis.reply_service import get_reply_service

            reply_service = get_reply_service()
            await run_in_threadpool(
                reply_service.log_custom_generation,
                chat_id=summary_request.chat_id,
                incoming_text=f"Summarize {len(messages)} messages",
                final_prompt=prompt,
                response_text=response_text,
                category="summary",
                metadata={"num_messages": len(messages)},
            )
    except TimeoutError:
        logger.warning("Summary generation timed out after %s seconds", get_timeout_generation())
        raise HTTPException(
            status_code=408,
            detail=f"Request timed out after {get_timeout_generation()} seconds",
        ) from None
    except Exception as e:
        logger.error("Failed to generate summary: %s", e)
        raise ModelError(
            "Failed to generate conversation summary",
            cause=e,
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
    # Fetch conversation messages for context
    try:
        messages = await run_in_threadpool(
            reader.get_messages,
            chat_id=routed_request.chat_id,
            limit=routed_request.context_messages,
        )
    except Exception as e:
        logger.error("Failed to fetch messages for chat %s: %s", routed_request.chat_id, e)
        raise iMessageQueryError(
            f"Failed to fetch conversation context for chat: {routed_request.chat_id}",
            cause=e,
        )

    if not messages:
        from jarvis.errors import ConversationNotFoundError
        raise ConversationNotFoundError(f"No messages found for chat_id: {routed_request.chat_id}")

    # Determine the message to respond to
    last_message = routed_request.last_message
    if not last_message and messages:
        # Use the most recent message that's not from us
        for msg in messages:
            if not msg.is_from_me and msg.text:
                last_message = msg.text
                break

    if not last_message:
        # Fall back to the most recent message
        last_message = messages[0].text if messages[0].text else ""

    # Build thread context (reverse chronological -> chronological)
    thread_context = [msg.text for msg in reversed(messages) if msg.text and len(msg.text) > 0][
        -10:
    ]  # Last 10 messages

    # Build context info for response
    participants = list({m.sender_name or m.sender for m in messages if not m.is_from_me})
    context_info = ContextInfo(
        num_messages=len(messages),
        participants=participants,
        last_message=last_message,
    )

    # Route the reply with timeout
    try:
        async with asyncio.timeout(get_timeout_generation()):
            result = await run_in_threadpool(
                route_reply_sync,
                routed_request.chat_id,
                last_message,
                thread_context,
            )
    except TimeoutError:
        logger.warning("Routed reply timed out after %s seconds", get_timeout_generation())
        raise HTTPException(
            status_code=408,
            detail=f"Request timed out after {get_timeout_generation()} seconds",
        ) from None
    except Exception as e:
        logger.error("Failed to route reply: %s", e)
        raise ModelError(
            "Failed to generate smart reply",
            cause=e,
        )

    return RoutedReplyResponse(
        response=result["response"],
        response_type=result["type"],
        confidence=result["confidence"],
        similarity_score=result.get("similarity_score", 0.0),
        cluster_name=result.get("cluster_name"),
        similar_triggers=result.get("similar_triggers"),
        context_used=context_info,
    )
