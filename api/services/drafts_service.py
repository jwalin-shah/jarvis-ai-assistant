"""Service-layer orchestration for draft endpoints."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

from api.ratelimit import get_timeout_generation
from api.schemas import ContextInfo, DraftSuggestion
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
from contracts.imessage import Message
from jarvis.contracts.pipeline import MessageContext
from jarvis.core.exceptions import (
    ConversationNotFoundError,
    ModelError,
    iMessageQueryError,
)
from jarvis.model_warmer import get_warm_generator
from jarvis.reply_service import get_reply_service

logger = logging.getLogger(__name__)


async def fetch_messages(reader: object, chat_id: str, limit: int) -> list[Message]:
    """Fetch chat messages with standardized error handling."""
    try:
        return await run_in_threadpool(reader.get_messages, chat_id=chat_id, limit=limit)  # type: ignore[attr-defined]
    except Exception as e:
        logger.error("Failed to fetch messages for chat %s: %s", chat_id, e)
        raise iMessageQueryError(
            f"Failed to fetch conversation context for chat: {chat_id}",
            cause=e,
        )


def ensure_messages_exist(chat_id: str, messages: list[Message]) -> None:
    if not messages:
        raise ConversationNotFoundError(f"No messages found for chat_id: {chat_id}")


def build_reply_context(
    messages: list[Message],
    context_messages: int,
) -> tuple[str, list[str], list[str]]:
    """Build last message, participants, and thread context for reply generation."""
    last_message = None
    for msg in messages:
        if not msg.is_from_me and msg.text:
            last_message = msg.text
            break
    if not last_message:
        last_message = messages[0].text if messages else ""

    participants = list({m.sender_name or m.sender for m in messages if not m.is_from_me})
    thread = [m.text for m in reversed(messages) if m.text][:context_messages]
    return last_message or "", participants, thread


async def build_draft_suggestions(
    *,
    chat_id: str,
    last_message: str,
    thread: list[str],
    instruction: str | None,
    num_suggestions: int,
) -> list[DraftSuggestion]:
    """Generate draft suggestions with retrieval and model generation."""
    reply_service = get_reply_service()
    try:
        classification, search_results = await run_in_threadpool(
            run_classification_and_search,
            last_message,
            thread,
        )
    except Exception as e:
        logger.error("Classification/search failed: %s", e)
        raise ModelError("Model service unavailable", cause=e) from e

    context = MessageContext(
        chat_id=chat_id,
        message_text=last_message,
        is_from_me=False,
        timestamp=datetime.now(UTC),
        metadata={"thread": thread},
    )

    base_instruction = sanitize_instruction(instruction)
    variant_instructions: list[str | None] = [base_instruction]
    if num_suggestions > 1:
        variant_instructions.append(
            (base_instruction + " (slightly more casual)")
            if base_instruction
            else "be slightly more casual"
        )
    if num_suggestions > 2:
        variant_instructions.append(
            (base_instruction + " (concise)") if base_instruction else "be concise"
        )

    suggestions: list[DraftSuggestion] = []
    try:
        async with asyncio.timeout(get_timeout_generation()):
            for i in range(num_suggestions):
                variant_instruction = (
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
                        variant_instruction,
                    )
                except Exception as e:
                    logger.warning("Generation %d failed: %s", i, e)
                    continue

                if gen_response.response:
                    confidence = max(0.5, gen_response.confidence)
                    suggestions.append(
                        DraftSuggestion(text=gen_response.response, confidence=confidence)
                    )
                    await run_in_threadpool(
                        reply_service.log_custom_generation,
                        chat_id=chat_id,
                        incoming_text=last_message,
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

    return suggestions


async def generate_summary_payload(
    chat_id: str,
    messages: list[Message],
    summary_examples: list[tuple[str, str]],
) -> tuple[str, list[str]]:
    """Generate summary + key points for conversation messages."""
    context_text = format_messages_for_context(messages)
    prompt = build_summary_prompt(context_text, len(messages))

    try:
        generator = await run_in_threadpool(get_warm_generator)
    except Exception as e:
        logger.error("Failed to get generator: %s", e)
        raise ModelError("Model service unavailable", cause=e)

    try:
        async with asyncio.timeout(get_timeout_generation()):
            response_text = await run_in_threadpool(
                generate_summary,
                generator,
                prompt,
                summary_examples,
            )
            summary, key_points = parse_summary_response(response_text)

            reply_service = get_reply_service()
            await run_in_threadpool(
                reply_service.log_custom_generation,
                chat_id=chat_id,
                incoming_text=f"Summarize {len(messages)} messages",
                final_prompt=prompt,
                response_text=response_text,
                category="summary",
                metadata={"num_messages": len(messages)},
            )
            return summary, key_points
    except TimeoutError:
        logger.warning("Summary generation timed out after %s seconds", get_timeout_generation())
        raise HTTPException(
            status_code=408,
            detail=f"Request timed out after {get_timeout_generation()} seconds",
        ) from None
    except Exception as e:
        logger.error("Failed to generate summary: %s", e)
        raise ModelError("Failed to generate conversation summary", cause=e)


def build_smart_reply_input(
    *,
    messages: list[Message],
    requested_last_message: str | None,
) -> tuple[str, list[str], ContextInfo]:
    """Build input payload for smart reply endpoint."""
    last_message = requested_last_message
    if not last_message and messages:
        for msg in messages:
            if not msg.is_from_me and msg.text:
                last_message = msg.text
                break
    if not last_message:
        last_message = messages[0].text if messages and messages[0].text else ""

    thread_context = [msg.text for msg in reversed(messages) if msg.text and len(msg.text) > 0][
        -10:
    ]
    participants = list({m.sender_name or m.sender for m in messages if not m.is_from_me})
    context_info = ContextInfo(
        num_messages=len(messages),
        participants=participants,
        last_message=last_message,
    )
    return last_message, thread_context, context_info


async def route_smart_reply(
    chat_id: str,
    last_message: str,
    thread_context: list[str],
) -> dict[str, Any]:
    """Generate smart reply with timeout handling."""
    try:
        async with asyncio.timeout(get_timeout_generation()):
            return await run_in_threadpool(route_reply_sync, chat_id, last_message, thread_context)
    except TimeoutError:
        logger.warning("Routed reply timed out after %s seconds", get_timeout_generation())
        raise HTTPException(
            status_code=408,
            detail=f"Request timed out after {get_timeout_generation()} seconds",
        ) from None
    except Exception as e:
        logger.error("Failed to route reply: %s", e)
        raise ModelError("Failed to generate smart reply", cause=e)
