"""Service-layer logic for priority inbox routes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from jarvis.core.exceptions import iMessageQueryError
from jarvis.priority import PriorityLevel, get_priority_scorer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PriorityInboxPayload:
    """Computed payload for the priority inbox response."""

    messages: list[dict[str, Any]]
    total_count: int
    unhandled_count: int
    needs_response_count: int
    critical_count: int
    high_count: int


def _parse_min_level(min_level: str | None) -> PriorityLevel | None:
    if not min_level:
        return None
    try:
        return PriorityLevel(min_level.lower())
    except ValueError:
        logger.warning("Invalid priority level filter: %s", min_level)
        return None


def _meets_min_level(level: PriorityLevel, min_level: PriorityLevel | None) -> bool:
    if not min_level:
        return True
    level_order = {
        PriorityLevel.CRITICAL: 4,
        PriorityLevel.HIGH: 3,
        PriorityLevel.MEDIUM: 2,
        PriorityLevel.LOW: 1,
    }
    return level_order[level] >= level_order[min_level]


def build_priority_inbox_payload(
    *,
    reader: Any,
    limit: int,
    include_handled: bool,
    min_level: str | None,
) -> PriorityInboxPayload:
    """Build and score priority inbox messages with aggregate counts."""
    scorer = get_priority_scorer()

    try:
        conversations = reader.get_conversations(limit=15)
    except Exception as e:
        logger.exception("Failed to get conversations")
        raise iMessageQueryError(
            "Failed to read conversations for priority inbox",
            cause=e,
        )

    conv_cache: dict[str, str] = {
        conv.chat_id: conv.display_name or ", ".join(conv.participants[:3])
        for conv in conversations
    }

    all_messages = []
    chat_ids = [conv.chat_id for conv in conversations[:15]]
    limit_per_chat = max(1, limit // 5)

    try:
        batch_result = reader.get_messages_batch(chat_ids, limit_per_chat=limit_per_chat)
        for msgs in batch_result.values():
            all_messages.extend(msgs)
    except Exception:
        for chat_id in chat_ids:
            try:
                all_messages.extend(reader.get_messages(chat_id, limit=limit_per_chat))
            except Exception:  # nosec B112
                continue

    priority_scores = scorer.score_messages(all_messages)
    minimum_level = _parse_min_level(min_level)

    filtered_scores = [
        score
        for score in priority_scores
        if (include_handled or not score.handled) and _meets_min_level(score.level, minimum_level)
    ]

    message_lookup = {(m.chat_id, m.id): m for m in all_messages}

    response_messages: list[dict[str, Any]] = []
    critical_count = 0
    high_count = 0
    needs_response_count = 0
    unhandled_count = 0

    for score in filtered_scores:
        if score.level == PriorityLevel.CRITICAL:
            critical_count += 1
        elif score.level == PriorityLevel.HIGH:
            high_count += 1

        if score.needs_response:
            needs_response_count += 1
        if not score.handled:
            unhandled_count += 1

        if len(response_messages) >= limit:
            continue

        message = message_lookup.get((score.chat_id, score.message_id))
        if not message:
            continue

        response_messages.append(
            {
                "message_id": score.message_id,
                "chat_id": score.chat_id,
                "sender": message.sender,
                "sender_name": message.sender_name,
                "text": message.text,
                "date": message.date.isoformat(),
                "priority_score": score.score,
                "priority_level": score.level.value,
                "reasons": [r.value for r in score.reasons],
                "needs_response": score.needs_response,
                "handled": score.handled,
                "conversation_name": conv_cache.get(score.chat_id),
            }
        )

    return PriorityInboxPayload(
        messages=response_messages,
        total_count=len(priority_scores),
        unhandled_count=unhandled_count,
        needs_response_count=needs_response_count,
        critical_count=critical_count,
        high_count=high_count,
    )


def mark_handled_state(*, chat_id: str, message_id: int, handled: bool) -> None:
    """Mark or unmark a message as handled."""
    scorer = get_priority_scorer()
    if handled:
        scorer.mark_handled(chat_id, message_id)
    else:
        scorer.unmark_handled(chat_id, message_id)


def mark_contact_importance(*, identifier: str, important: bool) -> None:
    """Mark or unmark a contact as important."""
    scorer = get_priority_scorer()
    scorer.mark_contact_important(identifier, important)


def get_priority_stats_payload() -> dict[str, int]:
    """Return aggregate scorer stats for API output."""
    scorer = get_priority_scorer()
    return {
        "handled_count": scorer.get_handled_count(),
        "important_contacts_count": len(scorer._important_contacts),
    }


def clear_handled_items() -> None:
    """Clear all handled markers in scorer state."""
    get_priority_scorer().clear_handled()
