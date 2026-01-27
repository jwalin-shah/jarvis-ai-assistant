"""MCP Tool handlers for JARVIS functionality.

Implements the actual logic for each MCP tool, connecting to JARVIS
services for iMessage access, AI generation, and contact lookup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution."""

    success: bool
    data: dict[str, Any] | list[Any] | None = None
    error: str | None = None


def _parse_datetime(date_str: str | None) -> datetime | None:
    """Parse an ISO 8601 datetime string.

    Args:
        date_str: ISO 8601 formatted date string or None.

    Returns:
        datetime object with UTC timezone or None.
    """
    if not date_str:
        return None

    try:
        # Try parsing with timezone
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except ValueError:
        # Try simpler formats
        for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S"]:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.replace(tzinfo=UTC)
            except ValueError:
                continue
        logger.warning("Could not parse datetime: %s", date_str)
        return None


def _check_imessage_access() -> bool:
    """Check if iMessage database is accessible.

    Returns:
        True if accessible, False otherwise.
    """
    try:
        from integrations.imessage import ChatDBReader

        with ChatDBReader() as reader:
            return reader.check_access()
    except Exception as e:
        logger.warning("iMessage access check failed: %s", e)
        return False


def _find_chat_id_by_name(name: str) -> str | None:
    """Find a chat_id by person name.

    Args:
        name: Person name to search for.

    Returns:
        chat_id if found, None otherwise.
    """
    from integrations.imessage import ChatDBReader
    from jarvis.context import ContextFetcher

    with ChatDBReader() as reader:
        fetcher = ContextFetcher(reader)
        return fetcher.find_conversation_by_name(name)


def handle_search_messages(params: dict[str, Any]) -> ToolResult:
    """Handle search_messages tool call.

    Args:
        params: Tool parameters including query and filters.

    Returns:
        ToolResult with matching messages or error.
    """
    if not _check_imessage_access():
        return ToolResult(
            success=False,
            error="Cannot access iMessage database. Grant Full Disk Access permission.",
        )

    query = params.get("query", "")
    if not query:
        return ToolResult(success=False, error="Query parameter is required")

    limit = min(params.get("limit", 20), 100)
    sender = params.get("sender")
    start_date = _parse_datetime(params.get("start_date"))
    end_date = _parse_datetime(params.get("end_date"))
    has_attachments = params.get("has_attachments")
    chat_id = params.get("chat_id")

    try:
        from integrations.imessage import ChatDBReader

        with ChatDBReader() as reader:
            messages = reader.search(
                query=query,
                limit=limit,
                sender=sender,
                after=start_date,
                before=end_date,
                chat_id=chat_id,
                has_attachments=has_attachments,
            )

            results = []
            for msg in messages:
                results.append(
                    {
                        "id": msg.id,
                        "chat_id": msg.chat_id,
                        "sender": msg.sender,
                        "sender_name": msg.sender_name,
                        "text": msg.text,
                        "date": msg.date.isoformat() if msg.date else None,
                        "is_from_me": msg.is_from_me,
                        "has_attachments": len(msg.attachments) > 0,
                    }
                )

            return ToolResult(
                success=True,
                data={
                    "messages": results,
                    "count": len(results),
                    "query": query,
                },
            )
    except Exception as e:
        logger.exception("Error searching messages")
        return ToolResult(success=False, error=str(e))


def handle_get_summary(params: dict[str, Any]) -> ToolResult:
    """Handle get_summary tool call.

    Args:
        params: Tool parameters including person_name or chat_id.

    Returns:
        ToolResult with conversation summary or error.
    """
    if not _check_imessage_access():
        return ToolResult(
            success=False,
            error="Cannot access iMessage database. Grant Full Disk Access permission.",
        )

    person_name = params.get("person_name")
    chat_id = params.get("chat_id")
    num_messages = min(params.get("num_messages", 50), 200)

    if not person_name and not chat_id:
        return ToolResult(
            success=False,
            error="Either person_name or chat_id is required",
        )

    try:
        # Find chat_id if person_name provided
        if person_name and not chat_id:
            chat_id = _find_chat_id_by_name(person_name)
            if not chat_id:
                return ToolResult(
                    success=False,
                    error=f"Could not find conversation with '{person_name}'",
                )

        # Verify chat_id was successfully resolved
        if chat_id is None:
            return ToolResult(
                success=False,
                error="Failed to resolve chat_id",
            )

        from contracts.models import GenerationRequest
        from integrations.imessage import ChatDBReader
        from jarvis.context import ContextFetcher
        from jarvis.prompts import SUMMARY_EXAMPLES, build_summary_prompt
        from models import get_generator

        with ChatDBReader() as reader:
            fetcher = ContextFetcher(reader)
            context = fetcher.get_summary_context(chat_id, num_messages=num_messages)

            if len(context.messages) < 3:
                return ToolResult(
                    success=False,
                    error="Not enough messages to summarize",
                )

            # Generate summary
            formatted_prompt = build_summary_prompt(context=context.formatted_context)

            generator = get_generator()
            request = GenerationRequest(
                prompt=formatted_prompt,
                context_documents=[context.formatted_context],
                few_shot_examples=SUMMARY_EXAMPLES,
                max_tokens=500,
                temperature=0.5,
            )
            response = generator.generate(request)

            # Parse summary
            summary_text = response.text.strip()
            key_points = []

            lines = summary_text.split("\n")
            summary = ""
            in_key_points = False

            for line in lines:
                line = line.strip()
                if line.lower().startswith("summary:"):
                    summary = line[8:].strip()
                elif "key points" in line.lower():
                    in_key_points = True
                elif in_key_points and (line.startswith("-") or line.startswith("•")):
                    key_points.append(line[1:].strip())

            if not summary:
                summary = summary_text[:200] if len(summary_text) > 200 else summary_text
            if not key_points:
                key_points = ["See summary for details"]

            return ToolResult(
                success=True,
                data={
                    "summary": summary,
                    "key_points": key_points,
                    "message_count": len(context.messages),
                    "date_range": {
                        "start": context.date_range[0].isoformat(),
                        "end": context.date_range[1].isoformat(),
                    },
                    "participants": context.participant_names,
                    "chat_id": chat_id,
                },
            )
    except Exception as e:
        logger.exception("Error generating summary")
        return ToolResult(success=False, error=str(e))


def handle_generate_reply(params: dict[str, Any]) -> ToolResult:
    """Handle generate_reply tool call.

    Args:
        params: Tool parameters including person_name or chat_id.

    Returns:
        ToolResult with reply suggestions or error.
    """
    if not _check_imessage_access():
        return ToolResult(
            success=False,
            error="Cannot access iMessage database. Grant Full Disk Access permission.",
        )

    person_name = params.get("person_name")
    chat_id = params.get("chat_id")
    instruction = params.get("instruction")
    num_suggestions = min(params.get("num_suggestions", 3), 5)
    context_messages = min(params.get("context_messages", 20), 50)

    if not person_name and not chat_id:
        return ToolResult(
            success=False,
            error="Either person_name or chat_id is required",
        )

    try:
        # Find chat_id if person_name provided
        if person_name and not chat_id:
            chat_id = _find_chat_id_by_name(person_name)
            if not chat_id:
                return ToolResult(
                    success=False,
                    error=f"Could not find conversation with '{person_name}'",
                )

        # Verify chat_id was successfully resolved
        if chat_id is None:
            return ToolResult(
                success=False,
                error="Failed to resolve chat_id",
            )

        from contracts.models import GenerationRequest
        from integrations.imessage import ChatDBReader
        from jarvis.context import ContextFetcher
        from jarvis.prompts import REPLY_EXAMPLES, build_reply_prompt
        from models import get_generator

        with ChatDBReader() as reader:
            fetcher = ContextFetcher(reader)
            context = fetcher.get_reply_context(chat_id, num_messages=context_messages)

            if not context.last_received_message:
                return ToolResult(
                    success=False,
                    error="No recent messages to reply to",
                )

            last_msg = context.last_received_message
            suggestions = []

            generator = get_generator()

            for i in range(num_suggestions):
                formatted_prompt = build_reply_prompt(
                    context=context.formatted_context,
                    last_message=last_msg.text,
                    instruction=instruction or "Generate a natural, friendly reply",
                )

                request = GenerationRequest(
                    prompt=formatted_prompt,
                    context_documents=[context.formatted_context],
                    few_shot_examples=REPLY_EXAMPLES,
                    max_tokens=150,
                    temperature=0.7 + (i * 0.1),
                )

                try:
                    response = generator.generate(request)
                    suggestions.append(
                        {
                            "text": response.text.strip(),
                            "confidence": round(max(0.5, 0.9 - (i * 0.1)), 2),
                        }
                    )
                except Exception as e:
                    logger.warning("Failed to generate suggestion %d: %s", i + 1, e)
                    continue

            if not suggestions:
                return ToolResult(
                    success=False,
                    error="Failed to generate any reply suggestions",
                )

            return ToolResult(
                success=True,
                data={
                    "suggestions": suggestions,
                    "context": {
                        "last_message": last_msg.text,
                        "sender": last_msg.sender_name or last_msg.sender,
                        "participants": context.participant_names,
                        "message_count": len(context.messages),
                    },
                    "chat_id": chat_id,
                },
            )
    except Exception as e:
        logger.exception("Error generating reply")
        return ToolResult(success=False, error=str(e))


def handle_get_contact_info(params: dict[str, Any]) -> ToolResult:
    """Handle get_contact_info tool call.

    Args:
        params: Tool parameters including identifier.

    Returns:
        ToolResult with contact information or error.
    """
    identifier = params.get("identifier", "")
    if not identifier:
        return ToolResult(success=False, error="Identifier parameter is required")

    try:
        from integrations.imessage import ContactAvatarData, get_contact_avatar
        from integrations.imessage.parser import normalize_phone_number

        # Normalize the identifier
        is_email = "@" in identifier
        if is_email:
            normalized = identifier.lower().strip()
        else:
            normalized = normalize_phone_number(identifier)
            if normalized is None:
                return ToolResult(
                    success=False,
                    error="Invalid phone number format",
                )

        # Get contact info
        avatar_data: ContactAvatarData | None = None
        try:
            avatar_data = get_contact_avatar(normalized)
        except Exception as e:
            logger.warning("Error fetching contact info: %s", e)

        display_name = None
        has_avatar = False
        initials = "?"

        if avatar_data:
            has_avatar = avatar_data.image_data is not None
            initials = avatar_data.initials
            if avatar_data.display_name:
                display_name = avatar_data.display_name
            elif avatar_data.first_name or avatar_data.last_name:
                parts = []
                if avatar_data.first_name:
                    parts.append(avatar_data.first_name)
                if avatar_data.last_name:
                    parts.append(avatar_data.last_name)
                display_name = " ".join(parts)
        else:
            # Generate initials from identifier
            if identifier.startswith("+") or identifier.replace("-", "").isdigit():
                digits = "".join(c for c in identifier if c.isdigit())
                initials = digits[-2:] if len(digits) >= 2 else digits[-1:]
            elif "@" in identifier:
                local_part = identifier.split("@")[0]
                initials = local_part[0].upper() if local_part else "?"

        return ToolResult(
            success=True,
            data={
                "identifier": normalized,
                "display_name": display_name,
                "has_avatar": has_avatar,
                "initials": initials,
            },
        )
    except Exception as e:
        logger.exception("Error getting contact info")
        return ToolResult(success=False, error=str(e))


def handle_list_conversations(params: dict[str, Any]) -> ToolResult:
    """Handle list_conversations tool call.

    Args:
        params: Tool parameters including limit and since.

    Returns:
        ToolResult with conversation list or error.
    """
    if not _check_imessage_access():
        return ToolResult(
            success=False,
            error="Cannot access iMessage database. Grant Full Disk Access permission.",
        )

    limit = min(params.get("limit", 20), 100)
    since = _parse_datetime(params.get("since"))

    try:
        from integrations.imessage import ChatDBReader

        with ChatDBReader() as reader:
            conversations = reader.get_conversations(limit=limit, since=since)

            results = []
            for conv in conversations:
                results.append(
                    {
                        "chat_id": conv.chat_id,
                        "display_name": conv.display_name,
                        "participants": conv.participants,
                        "last_message_date": (
                            conv.last_message_date.isoformat() if conv.last_message_date else None
                        ),
                        "message_count": conv.message_count,
                        "is_group": conv.is_group,
                        "last_message_text": conv.last_message_text,
                    }
                )

            return ToolResult(
                success=True,
                data={
                    "conversations": results,
                    "count": len(results),
                },
            )
    except Exception as e:
        logger.exception("Error listing conversations")
        return ToolResult(success=False, error=str(e))


def handle_get_conversation_messages(params: dict[str, Any]) -> ToolResult:
    """Handle get_conversation_messages tool call.

    Args:
        params: Tool parameters including person_name or chat_id.

    Returns:
        ToolResult with messages or error.
    """
    if not _check_imessage_access():
        return ToolResult(
            success=False,
            error="Cannot access iMessage database. Grant Full Disk Access permission.",
        )

    person_name = params.get("person_name")
    chat_id = params.get("chat_id")
    limit = min(params.get("limit", 20), 100)

    if not person_name and not chat_id:
        return ToolResult(
            success=False,
            error="Either person_name or chat_id is required",
        )

    try:
        # Find chat_id if person_name provided
        if person_name and not chat_id:
            chat_id = _find_chat_id_by_name(person_name)
            if not chat_id:
                return ToolResult(
                    success=False,
                    error=f"Could not find conversation with '{person_name}'",
                )

        # Verify chat_id was successfully resolved
        if chat_id is None:
            return ToolResult(
                success=False,
                error="Failed to resolve chat_id",
            )

        from integrations.imessage import ChatDBReader

        with ChatDBReader() as reader:
            messages = reader.get_messages(chat_id=chat_id, limit=limit)

            results = []
            for msg in messages:
                results.append(
                    {
                        "id": msg.id,
                        "sender": msg.sender,
                        "sender_name": msg.sender_name,
                        "text": msg.text,
                        "date": msg.date.isoformat() if msg.date else None,
                        "is_from_me": msg.is_from_me,
                        "has_attachments": len(msg.attachments) > 0,
                        "reply_to_id": msg.reply_to_id,
                    }
                )

            return ToolResult(
                success=True,
                data={
                    "messages": results,
                    "count": len(results),
                    "chat_id": chat_id,
                },
            )
    except Exception as e:
        logger.exception("Error getting conversation messages")
        return ToolResult(success=False, error=str(e))


# Tool handler registry
TOOL_HANDLERS = {
    "search_messages": handle_search_messages,
    "get_summary": handle_get_summary,
    "generate_reply": handle_generate_reply,
    "get_contact_info": handle_get_contact_info,
    "list_conversations": handle_list_conversations,
    "get_conversation_messages": handle_get_conversation_messages,
}


def execute_tool(name: str, params: dict[str, Any]) -> ToolResult:
    """Execute a tool by name with the given parameters.

    Args:
        name: The tool name to execute.
        params: Parameters to pass to the tool.

    Returns:
        ToolResult with the execution result.
    """
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return ToolResult(success=False, error=f"Unknown tool: {name}")

    return handler(params)
