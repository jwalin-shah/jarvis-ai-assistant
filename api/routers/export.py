"""Export API endpoints.

Provides endpoints for exporting conversations, search results, and backups.
"""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from api.dependencies import get_imessage_reader
from api.ratelimit import RATE_LIMIT_READ, get_timeout_read, limiter
from api.schemas import (
    ExportBackupRequest,
    ExportConversationRequest,
    ExportFormatEnum,
    ExportResponse,
    ExportSearchRequest,
)
from contracts.imessage import Conversation, Message
from integrations.imessage import ChatDBReader
from jarvis.errors import ExportError
from jarvis.export import (
    ExportFormat,
    export_backup,
    export_messages,
    export_search_results,
    get_export_filename,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/export", tags=["export"])


def _to_export_format(format_enum: ExportFormatEnum) -> ExportFormat:
    """Convert API format enum to export module format enum."""
    return ExportFormat(format_enum.value)


@router.post("/conversation/{chat_id}", response_model=ExportResponse)
@limiter.limit(RATE_LIMIT_READ)
async def export_conversation(
    chat_id: str,
    export_request: ExportConversationRequest,
    request: Request,
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> ExportResponse:
    """Export a single conversation.

    Exports all messages from a conversation in the specified format.
    Supports JSON (full data), CSV (flattened), and TXT (human-readable).

    **Rate Limiting:**
    This endpoint is rate limited to 60 requests per minute.

    Args:
        chat_id: The conversation ID to export.
        request: Export options including format and date range.
        http_request: FastAPI request object (for rate limiting)

    Returns:
        ExportResponse with exported data.

    Raises:
        HTTPException 408: Request timed out
        HTTPException 429: Rate limit exceeded
    """
    try:
        async with asyncio.timeout(get_timeout_read()):
            # Get conversation metadata by direct lookup
            conversation = await run_in_threadpool(reader.get_conversation, chat_id)

            if conversation is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Conversation not found: {chat_id}",
                )

            # Get messages with date filtering pushed to DB
            before = export_request.date_range.end if export_request.date_range else None
            after = export_request.date_range.start if export_request.date_range else None
            messages = await run_in_threadpool(
                reader.get_messages,
                chat_id=chat_id,
                limit=export_request.limit,
                before=before,
                after=after,
            )

            if not messages:
                raise HTTPException(
                    status_code=404,
                    detail="No messages found in the specified range",
                )

            # Export in requested format
            export_format = _to_export_format(export_request.format)
            exported_data = export_messages(
                messages=messages,
                format=export_format,
                conversation=conversation,
                include_attachments=export_request.include_attachments,
            )

            filename = get_export_filename(
                format=export_format,
                prefix="conversation",
                chat_id=chat_id,
            )

            return ExportResponse(
                success=True,
                format=export_request.format.value,
                filename=filename,
                data=exported_data,
                message_count=len(messages),
                export_type="conversation",
            )

    except TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=f"Request timed out after {get_timeout_read()} seconds",
        ) from None
    except HTTPException:
        raise
    except Exception as e:
        raise ExportError(
            "Failed to export conversation",
            export_format=export_request.format.value,
            cause=e,
        )


@router.post("/search", response_model=ExportResponse)
@limiter.limit(RATE_LIMIT_READ)
async def export_search(
    search_request: ExportSearchRequest,
    request: Request,
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> ExportResponse:
    """Export search results.

    Searches messages and exports the results in the specified format.

    **Rate Limiting:**
    This endpoint is rate limited to 60 requests per minute.

    Args:
        request: Search query and export options.
        request: FastAPI request object (for rate limiting)

    Returns:
        ExportResponse with exported search results.

    Raises:
        HTTPException 408: Request timed out
        HTTPException 429: Rate limit exceeded
    """
    try:
        async with asyncio.timeout(get_timeout_read()):
            # Perform search
            after = search_request.date_range.start if search_request.date_range else None
            before = search_request.date_range.end if search_request.date_range else None

            messages = await run_in_threadpool(
                reader.search,
                query=search_request.query,
                limit=search_request.limit,
                sender=search_request.sender,
                after=after,
                before=before,
            )

            if not messages:
                raise HTTPException(
                    status_code=404,
                    detail=f"No messages found matching query: {search_request.query}",
                )

            # Export in requested format
            export_format = _to_export_format(search_request.format)
            exported_data = export_search_results(
                messages=messages,
                query=search_request.query,
                format=export_format,
            )

            filename = get_export_filename(
                format=export_format,
                prefix="search_results",
            )

            return ExportResponse(
                success=True,
                format=search_request.format.value,
                filename=filename,
                data=exported_data,
                message_count=len(messages),
                export_type="search",
            )

    except TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=f"Request timed out after {get_timeout_read()} seconds",
        ) from None
    except HTTPException:
        raise
    except Exception as e:
        raise ExportError(
            "Failed to export search results",
            export_format=search_request.format.value,
            cause=e,
        )


TIMEOUT_BACKUP = 60.0  # Backups can take longer


@router.post("/backup", response_model=ExportResponse)
@limiter.limit(RATE_LIMIT_READ)
async def export_full_backup(
    backup_request: ExportBackupRequest,
    request: Request,
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> ExportResponse:
    """Create a full backup of accessible conversations.

    Exports multiple conversations with their messages in JSON format.
    Only JSON format is supported for full backups.

    **Rate Limiting:**
    This endpoint is rate limited to 60 requests per minute.

    Args:
        backup_request: Backup options including limits.
        request: FastAPI request object (for rate limiting)

    Returns:
        ExportResponse with backup data.

    Raises:
        HTTPException 408: Request timed out
        HTTPException 429: Rate limit exceeded
    """
    try:
        async with asyncio.timeout(TIMEOUT_BACKUP):
            # Fetch enough conversations to satisfy offset + limit
            fetch_limit = backup_request.offset + backup_request.conversation_limit
            conversations = await run_in_threadpool(
                reader.get_conversations, limit=fetch_limit
            )

            # Apply offset pagination
            conversations = conversations[backup_request.offset:]

            if not conversations:
                raise HTTPException(
                    status_code=404,
                    detail="No conversations found to backup",
                )

            # Collect messages for each conversation with per-conversation timeout
            conversation_data: list[tuple[Conversation, list[Message]]] = []
            total_messages = 0

            for conv in conversations:
                before = backup_request.date_range.end if backup_request.date_range else None
                after = backup_request.date_range.start if backup_request.date_range else None
                try:
                    async with asyncio.timeout(10.0):
                        messages = await run_in_threadpool(
                            reader.get_messages,
                            chat_id=conv.chat_id,
                            limit=backup_request.messages_per_conversation,
                            before=before,
                            after=after,
                        )
                except TimeoutError:
                    logger.warning(
                        "Backup: skipping conversation %s (message fetch timed out)",
                        conv.chat_id,
                    )
                    continue

                if messages:
                    conversation_data.append((conv, messages))
                    total_messages += len(messages)

            if not conversation_data:
                raise HTTPException(
                    status_code=404,
                    detail="No messages found in the specified range",
                )

            # Export backup (JSON only)
            exported_data = export_backup(conversation_data)

            filename = get_export_filename(
                format=ExportFormat.JSON,
                prefix="backup",
            )

            return ExportResponse(
                success=True,
                format="json",
                filename=filename,
                data=exported_data,
                message_count=total_messages,
                export_type="backup",
            )

    except TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=f"Request timed out after {TIMEOUT_BACKUP} seconds",
        ) from None
    except HTTPException:
        raise
    except Exception as e:
        raise ExportError(
            "Failed to create backup",
            cause=e,
        ) from e
