"""AI-powered draft reply generation API endpoints.

Provides endpoints for generating draft replies and conversation summaries
using the MLX language model with conversation context via RAG (Retrieval
Augmented Generation).

These endpoints use the local MLX model to generate contextually appropriate
responses without sending any data to external services.

NOTE: All prompts and examples are imported from jarvis/prompts.py,
which is the single source of truth for all prompts in the JARVIS system.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
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
)
from contracts.imessage import Message
from contracts.models import GenerationRequest
from integrations.imessage import ChatDBReader
from jarvis.model_warmer import get_warm_generator
from jarvis.prompts import API_REPLY_EXAMPLES, API_SUMMARY_EXAMPLES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/drafts", tags=["drafts"])

# Use centralized prompts from jarvis/prompts.py
REPLY_EXAMPLES = API_REPLY_EXAMPLES
SUMMARY_EXAMPLES = API_SUMMARY_EXAMPLES


def _format_messages_for_context(messages: list[Message]) -> str:
    """Format messages as context string for RAG.

    Args:
        messages: List of messages (newest first from reader)

    Returns:
        Formatted context string with messages in chronological order
    """
    # Reverse to chronological order (oldest first)
    chronological = list(reversed(messages))

    lines = []
    for msg in chronological:
        sender = "You" if msg.is_from_me else (msg.sender_name or msg.sender)
        lines.append(f"[{sender}]: {msg.text}")

    return "\n".join(lines)


def _build_reply_prompt(
    last_message: str,
    instruction: str | None,
    suggestion_num: int,
) -> str:
    """Build a prompt for generating a reply suggestion.

    Args:
        last_message: The last message in the conversation
        instruction: Optional user instruction for reply tone/content
        suggestion_num: Which suggestion number (1, 2, 3...) for variety

    Returns:
        Formatted prompt string
    """
    base_prompt = f"Last message: '{last_message}'\nInstruction: {instruction or 'None'}"

    # Add variety hints for different suggestions
    variety_hints = [
        "\nGenerate a natural, conversational reply.",
        "\nGenerate a slightly more casual reply variant.",
        "\nGenerate a concise reply variant.",
    ]

    hint_idx = (suggestion_num - 1) % len(variety_hints)
    return base_prompt + variety_hints[hint_idx]


def _build_summary_prompt(context: str, num_messages: int) -> str:
    """Build a prompt for conversation summarization.

    Args:
        context: Formatted conversation context
        num_messages: Number of messages being summarized

    Returns:
        Formatted prompt string
    """
    return (
        f"Summarize this conversation of {num_messages} messages. "
        "Provide a brief summary and extract 2-4 key points.\n\n"
        f"Conversation:\n{context}\n\n"
        "Provide your response in this format:\n"
        "Summary: [1-2 sentence summary]\n"
        "Key points:\n- [point 1]\n- [point 2]"
    )


def _parse_summary_response(response_text: str) -> tuple[str, list[str]]:
    """Parse the LLM summary response into summary and key points.

    Handles variations in LLM output format including:
    - "Summary: ...", "Here is the summary: ...", "**Summary:** ..."
    - "Key points:", "Key Points:", "Here are the key points:"
    - Bullet points with -, *, or numbered lists (1., 2.)

    Args:
        response_text: Raw LLM response

    Returns:
        Tuple of (summary, key_points). Falls back to raw text if parsing fails.
    """
    import re as _re

    lines = response_text.strip().split("\n")
    summary = ""
    key_points: list[str] = []

    in_key_points = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Strip markdown bold markers for matching
        clean = stripped.replace("**", "")
        lower = clean.lower()

        # Match summary line: "Summary: ...", "Here is the summary: ...", etc.
        if not summary and not in_key_points:
            summary_match = _re.match(
                r"^(?:here\s+is\s+(?:the\s+)?)?summary\s*:\s*(.+)",
                lower,
            )
            if summary_match:
                # Extract from original line (preserving case) after the colon
                colon_idx = clean.find(":")
                if colon_idx >= 0:
                    summary = clean[colon_idx + 1 :].strip()
                continue

        # Match key points header (case-insensitive, with optional preamble)
        if _re.match(r"^(?:here\s+are\s+(?:the\s+)?)?key\s+points\s*:?\s*$", lower):
            in_key_points = True
            continue

        # Extract bullet points (-, *, bullet char, or numbered)
        if in_key_points:
            bullet_match = _re.match(r"^[-*\u2022]\s*(.+)", stripped)
            if bullet_match:
                point = bullet_match.group(1).strip()
                if point:
                    key_points.append(point)
                continue
            numbered_match = _re.match(r"^\d+[.)\-]\s*(.+)", stripped)
            if numbered_match:
                point = numbered_match.group(1).strip()
                if point:
                    key_points.append(point)
                continue

    # Fallback: if structured parsing found nothing, use raw text
    if not summary:
        raw = response_text.strip()
        if len(raw) > 200:
            summary = raw[:200].rsplit(" ", 1)[0] + "..."
        else:
            summary = raw
    if not key_points:
        key_points = ["See summary for details"]

    return summary, key_points


def _generate_single_suggestion(
    generator: object,
    prompt: str,
    context_text: str,
    temperature: float,
) -> str | None:
    """Generate a single reply suggestion synchronously.

    This runs in a thread pool to avoid blocking the event loop.

    Args:
        generator: The MLX generator instance.
        prompt: The prompt to generate from.
        context_text: Conversation context.
        temperature: Sampling temperature.

    Returns:
        Generated text or None if generation failed.
    """
    gen_request = GenerationRequest(
        prompt=prompt,
        context_documents=[context_text],
        few_shot_examples=REPLY_EXAMPLES,
        max_tokens=200,
        temperature=temperature,
    )
    try:
        response = generator.generate(gen_request)  # type: ignore[attr-defined]
        text: str = str(response.text) if response.text else ""
        return text.strip()
    except Exception as e:
        logger.warning("Generation failed: %s", e)
        return None


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
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch conversation context: {e}",
        ) from e

    if not messages:
        raise HTTPException(
            status_code=404,
            detail=f"No messages found for chat_id: {draft_request.chat_id}",
        )

    # Build context info
    last_message = messages[0].text if messages else None
    participants = list({m.sender_name or m.sender for m in messages if not m.is_from_me})
    context_text = _format_messages_for_context(messages)

    # Get the generator
    try:
        generator = await run_in_threadpool(get_warm_generator)
    except Exception as e:
        logger.error("Failed to get generator: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Model service unavailable",
        ) from e

    # Generate multiple suggestions with timeout
    suggestions: list[DraftSuggestion] = []

    try:
        async with asyncio.timeout(get_timeout_generation()):
            for i in range(draft_request.num_suggestions):
                prompt = _build_reply_prompt(
                    last_message=last_message or "",
                    instruction=draft_request.instruction,
                    suggestion_num=i + 1,
                )

                # Run generation in threadpool (CPU-bound)
                text = await run_in_threadpool(
                    _generate_single_suggestion,
                    generator,
                    prompt,
                    context_text,
                    0.7 + (i * 0.1),  # Vary temperature for diversity
                )

                if text:
                    confidence = max(0.5, 0.9 - (i * 0.1))
                    suggestions.append(
                        DraftSuggestion(
                            text=text,
                            confidence=confidence,
                        )
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


def _generate_summary(generator: object, prompt: str) -> str:
    """Generate a summary synchronously.

    This runs in a thread pool to avoid blocking the event loop.

    Args:
        generator: The MLX generator instance.
        prompt: The prompt to generate from.

    Returns:
        Generated summary text.
    """
    gen_request = GenerationRequest(
        prompt=prompt,
        context_documents=[],  # Context already in prompt
        few_shot_examples=SUMMARY_EXAMPLES,
        max_tokens=500,
        temperature=0.5,  # Lower temperature for more focused summary
    )
    response = generator.generate(gen_request)  # type: ignore[attr-defined]
    return str(response.text) if response.text else ""


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
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch conversation: {e}",
        ) from e

    if not messages:
        raise HTTPException(
            status_code=404,
            detail=f"No messages found for chat_id: {summary_request.chat_id}",
        )

    # Build context
    context_text = _format_messages_for_context(messages)

    # Determine date range (messages are newest-first)
    newest_date = messages[0].date
    oldest_date = messages[-1].date

    # Get the generator
    try:
        generator = await run_in_threadpool(get_warm_generator)
    except Exception as e:
        logger.error("Failed to get generator: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Model service unavailable",
        ) from e

    # Generate summary with timeout
    prompt = _build_summary_prompt(context_text, len(messages))

    try:
        async with asyncio.timeout(get_timeout_generation()):
            response_text = await run_in_threadpool(
                _generate_summary,
                generator,
                prompt,
            )
            summary, key_points = _parse_summary_response(response_text)
    except TimeoutError:
        logger.warning("Summary generation timed out after %s seconds", get_timeout_generation())
        raise HTTPException(
            status_code=408,
            detail=f"Request timed out after {get_timeout_generation()} seconds",
        ) from None
    except Exception as e:
        logger.error("Failed to generate summary: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate conversation summary",
        ) from e

    return DraftSummaryResponse(
        summary=summary,
        key_points=key_points,
        date_range=DateRange(
            start=oldest_date.strftime("%Y-%m-%d"),
            end=newest_date.strftime("%Y-%m-%d"),
        ),
    )


# =============================================================================
# Smart Reply (Routed) Endpoint
# =============================================================================


class RoutedReplyRequest(BaseModel):
    """Request for smart routed reply generation.

    Uses the ReplyRouter's simplified flow:
    all non-empty inputs route to generation, with mobilization and
    retrieval used as prompt context.
    """

    chat_id: str = Field(
        ...,
        description="Conversation ID to generate reply for",
    )
    last_message: str | None = Field(
        default=None,
        description="Override last message (uses latest from chat if not provided)",
    )
    instruction: str | None = Field(
        default=None,
        description="Reserved for future use. Currently ignored by /drafts/smart-reply.",
    )
    context_messages: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Number of previous messages for context (1-30)",
    )


class RoutedReplyResponse(BaseModel):
    """Response from smart routed reply generation.

    Includes the response type ('generated' or 'clarify'),
    confidence level, and routing metadata.
    """

    response: str = Field(
        ...,
        description="The generated response text",
    )
    response_type: Literal["generated", "clarify"] = Field(
        ...,
        description="How the response was produced: 'generated' or 'clarify'",
    )
    confidence: str = Field(
        ...,
        description="Confidence level: 'high', 'medium', or 'low'",
    )
    similarity_score: float = Field(
        default=0.0,
        description="Top similarity score from retrieved historical exchanges (0-1)",
    )
    cluster_name: str | None = Field(
        default=None,
        description="Legacy field retained for backward compatibility",
    )
    similar_triggers: list[str] | None = Field(
        default=None,
        description="Similar past triggers found during routing",
    )
    context_used: ContextInfo | None = Field(
        default=None,
        description="Information about the context used",
    )


def _route_reply_sync(
    chat_id: str,
    last_message: str,
    thread_context: list[str],
) -> dict[str, Any]:
    """Run the router synchronously (for thread pool execution).

    Args:
        chat_id: Chat ID for contact lookup.
        last_message: The message to respond to.
        thread_context: Previous messages for context.

    Returns:
        Routing result dict.
    """
    from jarvis.router import get_reply_router

    router = get_reply_router()
    return router.route(
        incoming=last_message,
        chat_id=chat_id,
        thread=thread_context,
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
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch conversation context: {e}",
        ) from e

    if not messages:
        raise HTTPException(
            status_code=404,
            detail=f"No messages found for chat_id: {routed_request.chat_id}",
        )

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
                _route_reply_sync,
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
        raise HTTPException(
            status_code=500,
            detail="Failed to generate smart reply",
        ) from e

    return RoutedReplyResponse(
        response=result["response"],
        response_type=result["type"],
        confidence=result["confidence"],
        similarity_score=result.get("similarity_score", 0.0),
        cluster_name=result.get("cluster_name"),
        similar_triggers=result.get("similar_triggers"),
        context_used=context_info,
    )


# =============================================================================
# Multi-Option Reply Endpoint
# =============================================================================


class ResponseOptionSchema(BaseModel):
    """A single response option with type and confidence."""

    type: str = Field(..., description="Response type: AGREE, DECLINE, DEFER, etc.")
    response: str = Field(..., description="The response text")
    confidence: float = Field(..., description="Confidence score 0-1")
    source: str = Field(default="template", description="Source: template, generated, fallback")


class MultiOptionRequest(BaseModel):
    """Request shape retained for backward compatibility.

    Current simplified routing does not generate true multi-option outputs,
    but legacy clients may still call this endpoint.
    """

    chat_id: str = Field(
        ...,
        description="Conversation ID to generate reply for",
    )
    last_message: str | None = Field(
        default=None,
        description="Override last message (uses latest from chat if not provided)",
    )
    force_multi: bool = Field(
        default=False,
        description="Compatibility flag. Accepted but currently has no effect.",
    )


class MultiOptionResponse(BaseModel):
    """Backward-compatible response for legacy multi-option clients.

    In simplified mode this endpoint returns a single generated response
    and keeps multi-option fields populated with empty/default values.
    """

    is_commitment: bool = Field(
        ...,
        description="Whether the incoming message is a commitment question",
    )
    trigger_da: str | None = Field(
        default=None,
        description="Classified trigger dialogue act type (INVITATION, REQUEST, etc.)",
    )
    options: list[ResponseOptionSchema] = Field(
        default_factory=list,
        description="Response options with type and confidence",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Just the response texts (backward compatible)",
    )
    response: str | None = Field(
        default=None,
        description="Single response (when not a commitment question)",
    )
    response_type: str | None = Field(
        default=None,
        description="Response type (when not a commitment question)",
    )
    confidence: str = Field(
        default="medium",
        description="Overall confidence level",
    )
    context_used: ContextInfo | None = Field(
        default=None,
        description="Information about the context used",
    )


def _route_multi_option_sync(
    chat_id: str,
    last_message: str,
    force_multi: bool = False,
) -> dict[str, Any]:
    """Run multi-option routing synchronously.

    Args:
        chat_id: Chat ID for contact lookup.
        last_message: The message to respond to.
        force_multi: Force multi-option even for non-commitment.

    Returns:
        Multi-option routing result dict.
    """
    from jarvis.router import get_reply_router

    router = get_reply_router()
    return router.route_multi_option(
        incoming=last_message,
        chat_id=chat_id,
        force_multi=force_multi,
    )


@router.post(
    "/multi-option",
    response_model=MultiOptionResponse,
    response_model_exclude_unset=True,
    response_description="Multiple reply options for commitment questions",
    summary="Generate multiple reply options",
    responses={
        200: {
            "description": "Multi-option reply generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "is_commitment": False,
                        "trigger_da": None,
                        "options": [],
                        "suggestions": ["Sure, sounds good."],
                        "response": "Sure, sounds good.",
                        "response_type": "generated",
                        "confidence": "medium",
                    }
                }
            },
        },
        404: {
            "description": "Conversation not found",
            "model": ErrorResponse,
        },
    },
)
@limiter.limit(RATE_LIMIT_GENERATION)
async def generate_multi_option_reply(
    multi_request: MultiOptionRequest,
    request: Request,
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> MultiOptionResponse:
    """Return compatibility multi-option payload using simplified routing.

    Current behavior:
    - Delegates to ReplyRouter.route_multi_option().
    - Returns `is_commitment=false`, empty `options`, and one suggestion.
    - `force_multi` is accepted for compatibility but does not alter behavior.

    Args:
        multi_request: MultiOptionRequest with chat_id and optional last_message

    Returns:
        MultiOptionResponse with options for commitment, or single response otherwise
    """
    # Fetch messages to get the last message if not provided
    try:
        messages = await run_in_threadpool(
            reader.get_messages,
            chat_id=multi_request.chat_id,
            limit=5,
        )
    except Exception as e:
        logger.error("Failed to fetch messages for chat %s: %s", multi_request.chat_id, e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch conversation context: {e}",
        ) from e

    if not messages:
        raise HTTPException(
            status_code=404,
            detail=f"No messages found for chat_id: {multi_request.chat_id}",
        )

    # Determine the message to respond to
    last_message = multi_request.last_message
    if not last_message:
        for msg in messages:
            if not msg.is_from_me and msg.text:
                last_message = msg.text
                break

    if not last_message:
        last_message = messages[0].text if messages[0].text else ""

    # Build context info
    participants = list({m.sender_name or m.sender for m in messages if not m.is_from_me})
    context_info = ContextInfo(
        num_messages=len(messages),
        participants=participants,
        last_message=last_message,
    )

    # Route with multi-option
    try:
        async with asyncio.timeout(get_timeout_generation()):
            result = await run_in_threadpool(
                _route_multi_option_sync,
                multi_request.chat_id,
                last_message,
                multi_request.force_multi,
            )
    except TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=f"Request timed out after {get_timeout_generation()} seconds",
        ) from None
    except Exception as e:
        logger.error("Failed to generate multi-option reply: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate multi-option reply",
        ) from e

    # Build response
    options = [
        ResponseOptionSchema(
            type=opt.get("type", "STATEMENT"),
            response=opt.get("response", ""),
            confidence=opt.get("confidence", 0.5),
            source=opt.get("source", "fallback"),
        )
        for opt in result.get("options", [])
    ]

    return MultiOptionResponse(
        is_commitment=result.get("is_commitment", False),
        trigger_da=result.get("trigger_da"),
        options=options,
        suggestions=result.get("suggestions", []),
        response=result.get("response"),
        response_type=result.get("type"),
        confidence=result.get("confidence", "medium"),
        context_used=context_info,
    )
