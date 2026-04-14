"""Export API endpoints.

Provides endpoints for exporting conversations, search results, backups, and PDFs.
"""

import asyncio
import base64
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Path, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
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
from jarvis.contracts.imessage import Conversation, Message
from integrations.imessage import ChatDBReader
from jarvis.core.exceptions import ExportError
from jarvis.export import (
    ExportFormat,
    export_backup,
    export_messages,
    export_search_results,
    get_export_filename,
)
from jarvis.pdf_generator import PDFExportOptions, generate_pdf

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/export", tags=["export"])


# =============================================================================
# PDF Export Models
# =============================================================================


class PDFExportDateRange(BaseModel):
    """Date range filter for PDF exports."""

    start: datetime | None = Field(default=None, description="Start date (inclusive)")
    end: datetime | None = Field(default=None, description="End date (inclusive)")


class PDFExportRequest(BaseModel):
    """Request to export a conversation to PDF."""

    include_attachments: bool = Field(
        default=True,
        description="Include attachment thumbnails in PDF",
    )
    include_reactions: bool = Field(
        default=True,
        description="Include message reactions in PDF",
    )
    date_range: PDFExportDateRange | None = Field(
        default=None,
        description="Optional date range filter",
    )
    limit: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum messages to export",
    )


class PDFExportResponse(BaseModel):
    """Response containing PDF export data."""

    success: bool = Field(..., description="Whether the export succeeded")
    filename: str = Field(..., description="Suggested filename for the PDF")
    data: str = Field(..., description="Base64-encoded PDF data")
    message_count: int = Field(..., description="Number of messages in the PDF")
    size_bytes: int = Field(..., description="Size of the PDF in bytes")


def _to_export_format(format_enum: ExportFormatEnum) -> ExportFormat:
    """Convert API format enum to export module format enum."""
    return ExportFormat(format_enum.value)


@router.post("/conversation/{chat_id}", response_model=ExportResponse)
@limiter.limit(RATE_LIMIT_READ)
async def export_conversation(
    request: Request,
    export_request: ExportConversationRequest,
    chat_id: str = Path(
        ...,
        min_length=1,
        max_length=255,
        description="The unique conversation identifier",
        examples=["chat123456789"],
    ),
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
            conversations = await run_in_threadpool(reader.get_conversations, limit=fetch_limit)

            # Apply offset pagination
            conversations = conversations[backup_request.offset :]

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


# =============================================================================
# PDF Export Endpoints
# =============================================================================


@router.post("/pdf/{chat_id}", response_model=PDFExportResponse)
async def export_conversation_pdf(
    pdf_request: PDFExportRequest,
    chat_id: str = Path(
        ...,
        min_length=1,
        max_length=255,
        description="The unique conversation identifier",
        examples=["chat123456789"],
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> PDFExportResponse:
    """Export a conversation to PDF format.

    Generates a beautifully formatted PDF document with:
    - Header with conversation name, participants, and date range
    - Messages with styled bubbles (grayscale for printing)
    - Inline image attachment thumbnails
    - Message reactions
    - Page numbers in footer

    Args:
        chat_id: The conversation ID to export.
        request: Export options including attachments, reactions, and date range.

    Returns:
        PDFExportResponse with base64-encoded PDF data.

    Raises:
        HTTPException: If conversation not found or export fails.
    """
    try:
        # Get conversation metadata via direct lookup (avoids N+1)
        conversation = await run_in_threadpool(reader.get_conversation, chat_id)

        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation not found: {chat_id}",
            )

        before = pdf_request.date_range.end if pdf_request.date_range else None
        after = pdf_request.date_range.start if pdf_request.date_range else None
        messages = await run_in_threadpool(
            reader.get_messages,
            chat_id=chat_id,
            limit=pdf_request.limit,
            before=before,
            after=after,
        )

        if not messages:
            raise HTTPException(
                status_code=404,
                detail="No messages found in the specified range",
            )

        options = PDFExportOptions(
            include_attachments=pdf_request.include_attachments,
            include_reactions=pdf_request.include_reactions,
            start_date=pdf_request.date_range.start if pdf_request.date_range else None,
            end_date=pdf_request.date_range.end if pdf_request.date_range else None,
        )

        # Generate PDF
        pdf_bytes = await run_in_threadpool(generate_pdf, messages, conversation, options)
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_chat_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in chat_id)
        safe_chat_id = safe_chat_id[:30]
        filename = f"conversation_{safe_chat_id}_{timestamp}.pdf"

        return PDFExportResponse(
            success=True,
            filename=filename,
            data=pdf_base64,
            message_count=len(messages),
            size_bytes=len(pdf_bytes),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate PDF: {e}",
        ) from e


@router.post("/pdf/{chat_id}/download")
async def download_conversation_pdf(
    pdf_request: PDFExportRequest,
    chat_id: str = Path(
        ...,
        min_length=1,
        max_length=255,
        description="The unique conversation identifier",
        examples=["chat123456789"],
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> Response:
    """Download a conversation as a PDF file.

    Returns the PDF file directly for download instead of base64-encoded data.

    Args:
        chat_id: The conversation ID to export.
        pdf_request: Export options.

    Returns:
        PDF file response for direct download.
    """
    try:
        # Get conversation metadata via direct lookup (avoids N+1)
        conversation = await run_in_threadpool(reader.get_conversation, chat_id)

        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation not found: {chat_id}",
            )

        # Get messages with optional date filtering
        before = pdf_request.date_range.end if pdf_request.date_range else None
        after = pdf_request.date_range.start if pdf_request.date_range else None
        messages = await run_in_threadpool(
            reader.get_messages,
            chat_id=chat_id,
            limit=pdf_request.limit,
            before=before,
            after=after,
        )

        if not messages:
            raise HTTPException(
                status_code=404,
                detail="No messages found in the specified range",
            )

        # Create export options
        options = PDFExportOptions(
            include_attachments=pdf_request.include_attachments,
            include_reactions=pdf_request.include_reactions,
            start_date=pdf_request.date_range.start if pdf_request.date_range else None,
            end_date=pdf_request.date_range.end if pdf_request.date_range else None,
        )

        # Generate PDF
        pdf_bytes = await run_in_threadpool(generate_pdf, messages, conversation, options)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_chat_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in chat_id)
        safe_chat_id = safe_chat_id[:30]
        filename = f"conversation_{safe_chat_id}_{timestamp}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate PDF: {e}",
        ) from e
