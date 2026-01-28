"""Reply generation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import (
    GeneratedReplyResponse,
    GenerateRepliesRequest,
    GenerateRepliesResponse,
    GenerationDebugInfo,
    PastReplyResponse,
)

router = APIRouter()

# User's name for context
USER_NAME = "Jwalin"


@router.post("/replies", response_model=GenerateRepliesResponse)
async def generate_replies(request: GenerateRepliesRequest) -> GenerateRepliesResponse:
    """Generate reply suggestions for a conversation."""
    from core.generation import ReplyGenerator
    from core.imessage import MessageReader
    from core.models import get_model_loader

    # Get conversation messages
    reader = MessageReader()
    if not reader.check_access():
        raise HTTPException(
            status_code=503,
            detail="Cannot access iMessage database",
        )

    try:
        messages = reader.get_messages(chat_id=request.chat_id, limit=30)

        if not messages:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Convert to dict format for generator
        messages_dict = [
            {
                "text": m.text,
                "sender": m.sender,
                "sender_name": m.sender_name,  # Contact name (not phone number)
                "is_from_me": m.is_from_me,
                "timestamp": m.timestamp,
            }
            for m in reversed(messages)  # Oldest first for context
        ]

        # Generate replies
        loader = get_model_loader()
        generator = ReplyGenerator(loader)

        result = generator.generate_replies(
            messages=messages_dict,
            chat_id=request.chat_id,
            num_replies=1,  # Generate 1 reply for speed (was: request.num_replies)
            user_name=USER_NAME,
        )

        # Build debug info
        debug_info = GenerationDebugInfo(
            style_instructions=result.style_instructions,
            intent_detected=result.context.intent.value,
            past_replies_found=[
                PastReplyResponse(
                    their_message=their_msg,
                    your_reply=your_reply,
                    similarity=sim,
                )
                for their_msg, your_reply, sim in result.past_replies
            ],
            full_prompt=result.prompt_used,  # Our template prompt
            formatted_prompt=result.formatted_prompt,  # Actual ChatML sent to model
        )

        return GenerateRepliesResponse(
            replies=[
                GeneratedReplyResponse(
                    text=r.text,
                    reply_type=r.reply_type,
                    confidence=r.confidence,
                )
                for r in result.replies
            ],
            chat_id=request.chat_id,
            model_used=result.model_used,
            generation_time_ms=result.generation_time_ms,
            context_summary=result.context.summary,
            debug=debug_info,
        )

    finally:
        reader.close()
