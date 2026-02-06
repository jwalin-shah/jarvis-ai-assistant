"""Exchange-Based Pair Extraction - Extract (trigger, response) pairs from iMessage.

Uses validity gates (Gate A/B/C) to ensure high-quality training pairs.
Groups consecutive messages from the same person into "spans" (turns).

Example:
    Them: "hey"
    Them: "want to grab lunch?"
    You: "sounds good!"

    Becomes: trigger="hey\nwant to grab lunch?"
             response="sounds good!"

Usage:
    jarvis db extract                  # Extract from all conversations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from contracts.imessage import Message

if TYPE_CHECKING:
    from jarvis.exchange import CandidateExchange

logger = logging.getLogger(__name__)


@dataclass
class ExchangeBuilderConfig:
    """Configuration for exchange-based extraction.

    Attributes:
        time_gap_boundary_minutes: Time gap that marks a new conversation thread.
        trigger_max_messages: Max messages to include in trigger span.
        trigger_max_duration_minutes: Max duration for trigger span.
        response_max_messages: Max messages to include in response span.
        response_max_duration_minutes: Max duration for response span.
        context_window_size: Number of previous messages for context.
        max_response_delay_hours: Hard cutoff for response time.
        min_trigger_length: Min chars for trigger.
        min_response_length: Min chars for response.
        max_trigger_length: Max chars for trigger.
        max_response_length: Max chars for response.
    """

    time_gap_boundary_minutes: float = 30.0
    trigger_max_messages: int = 5
    trigger_max_duration_minutes: float = 3.0
    response_max_messages: int = 5
    response_max_duration_minutes: float = 3.0
    context_window_size: int = 20
    max_response_delay_hours: float = 24.0
    min_trigger_length: int = 2
    min_response_length: int = 15
    max_trigger_length: int = 500
    max_response_length: int = 400


@dataclass
class ExchangeExtractionStats:
    """Statistics from exchange-based extraction."""

    total_messages_scanned: int = 0
    exchanges_built: int = 0
    gate_a_passed: int = 0
    gate_a_rejected: int = 0
    gate_b_accepted: int = 0
    gate_b_borderline: int = 0
    gate_b_rejected: int = 0
    gate_c_accepted: int = 0
    gate_c_rejected: int = 0
    gate_c_uncertain: int = 0
    final_valid: int = 0
    final_invalid: int = 0
    final_uncertain: int = 0
    gate_a_rejection_reasons: dict[str, int] = field(default_factory=dict)


class ExchangeBuilder:
    """Build candidate exchanges with proper boundaries."""

    def __init__(self, config: ExchangeBuilderConfig | None = None) -> None:
        """Initialize exchange builder."""
        self.config = config or ExchangeBuilderConfig()

    def build_candidates(
        self,
        messages: list[Message],
        chat_id: str,
        contact_id: int | None = None,
    ) -> list[CandidateExchange]:
        """Build candidate exchanges from messages."""
        from jarvis.exchange import CandidateExchange, ContextMessage
        from jarvis.text_normalizer import get_attachment_token, is_reaction, normalize_for_task

        if not messages:
            return []

        sorted_messages = sorted(messages, key=lambda m: m.date)
        context_msgs: list[ContextMessage] = []
        for msg in sorted_messages:
            if msg.is_system_message:
                continue

            flags: set[str] = set()
            raw_text = msg.text or ""

            if is_reaction(raw_text):
                flags.add("reaction")
                normalized = ""
            else:
                normalized = normalize_for_task(raw_text, "extraction")

            if not normalized and msg.attachments:
                att_type = msg.attachments[0].mime_type if msg.attachments else None
                normalized = get_attachment_token(att_type)
                flags.add("attachment")

            from jarvis.text_normalizer import is_emoji_only

            if normalized and is_emoji_only(normalized):
                flags.add("emoji_only")

            context_msgs.append(
                ContextMessage(
                    speaker="me" if msg.is_from_me else "them",
                    timestamp=msg.date,
                    text=normalized,
                    flags=flags,
                    raw_text=raw_text if raw_text != normalized else None,
                )
            )

        candidates: list[CandidateExchange] = []
        i = 0
        n = len(context_msgs)

        while i < n:
            msg = context_msgs[i]
            if msg.speaker == "me" or "reaction" in msg.flags:
                i += 1
                continue

            trigger_span: list[ContextMessage] = [msg]
            trigger_msg_ids: list[int] = [sorted_messages[i].id]
            trigger_start = msg.timestamp
            j = i + 1

            while j < n and len(trigger_span) < self.config.trigger_max_messages:
                next_msg = context_msgs[j]
                if next_msg.speaker != "them":
                    break
                if "reaction" in next_msg.flags:
                    j += 1
                    continue
                duration_mins = (next_msg.timestamp - trigger_start).total_seconds() / 60
                if duration_mins > self.config.trigger_max_duration_minutes:
                    break
                trigger_span.append(next_msg)
                trigger_msg_ids.append(sorted_messages[j].id)
                j += 1

            if j >= n or context_msgs[j].speaker != "me":
                i = j if j < n else i + 1
                continue

            time_gap = context_msgs[j].timestamp - trigger_span[-1].timestamp
            gap_hours = time_gap.total_seconds() / 3600
            if gap_hours > self.config.max_response_delay_hours:
                i = j
                continue

            gap_mins = time_gap.total_seconds() / 60
            if gap_mins > self.config.time_gap_boundary_minutes:
                i = j
                continue

            response_span: list[ContextMessage] = [context_msgs[j]]
            response_msg_ids: list[int] = [sorted_messages[j].id]
            response_start = context_msgs[j].timestamp
            k = j + 1

            while k < n and len(response_span) < self.config.response_max_messages:
                next_msg = context_msgs[k]
                if next_msg.speaker != "me":
                    break
                if "reaction" in next_msg.flags:
                    k += 1
                    continue
                duration_mins = (next_msg.timestamp - response_start).total_seconds() / 60
                if duration_mins > self.config.response_max_duration_minutes:
                    break
                response_span.append(next_msg)
                response_msg_ids.append(sorted_messages[k].id)
                k += 1

            context_start = max(0, i - self.config.context_window_size)
            context_window = context_msgs[context_start:i]

            candidates.append(
                CandidateExchange(
                    trigger_span=trigger_span,
                    response_span=response_span,
                    context_window=context_window,
                    chat_id=chat_id,
                    contact_id=contact_id,
                    trigger_msg_ids=trigger_msg_ids,
                    response_msg_ids=response_msg_ids,
                )
            )
            i = k

        return candidates


def extract_all_pairs(
    chat_db_reader: Any,
    jarvis_db: Any,
    config: ExchangeBuilderConfig | None = None,
    embedder: Any | None = None,
    nli_model: Any | None = None,
    progress_callback: Any | None = None,
    skip_nli: bool = False,
) -> dict[str, Any]:
    """Extract high-quality pairs using exchange-based pipeline."""
    import json as json_module

    from jarvis.nlp.validity_gate import GateConfig, ValidityGate

    builder = ExchangeBuilder(config)
    gate = ValidityGate(
        embedder=embedder,
        nli_model=None if skip_nli else nli_model,
        config=GateConfig(),
    )

    aggregate_stats = {
        "conversations_processed": 0,
        "total_messages_scanned": 0,
        "exchanges_built": 0,
        "pairs_added": 0,
        "pairs_skipped_duplicate": 0,
        "gate_a_rejected": 0,
        "gate_b_rejected": 0,
        "gate_c_rejected": 0,
        "final_valid": 0,
        "final_invalid": 0,
        "final_uncertain": 0,
        "gate_a_reasons": {},
        "errors": [],
    }

    conversations = chat_db_reader.get_conversations(limit=1000)
    total = len(conversations)

    for idx, conv in enumerate(conversations):
        if progress_callback:
            progress_callback(idx, total, conv.chat_id)

        try:
            contact = jarvis_db.get_contact_by_chat_id(conv.chat_id)
            contact_id = contact.id if contact else None
            is_group = getattr(conv, "is_group", False)

            messages = chat_db_reader.get_messages(conv.chat_id, limit=10000)
            aggregate_stats["total_messages_scanned"] += len(messages)

            exchanges = builder.build_candidates(messages, conv.chat_id, contact_id)
            aggregate_stats["exchanges_built"] += len(exchanges)

            for exchange in exchanges:
                result = gate.validate(exchange)

                if not result.gate_a_passed:
                    aggregate_stats["gate_a_rejected"] += 1
                    reason = result.gate_a_reason or "unknown"
                    aggregate_stats["gate_a_reasons"][reason] = (
                        aggregate_stats["gate_a_reasons"].get(reason, 0) + 1
                    )

                if result.gate_b_band == "reject":
                    aggregate_stats["gate_b_rejected"] += 1
                if result.gate_c_verdict == "reject":
                    aggregate_stats["gate_c_rejected"] += 1

                if result.final_status == "valid":
                    aggregate_stats["final_valid"] += 1
                elif result.final_status == "invalid":
                    aggregate_stats["final_invalid"] += 1
                else:
                    aggregate_stats["final_uncertain"] += 1

                pair = jarvis_db.add_validated_pair(
                    trigger_text=exchange.trigger_text,
                    response_text=exchange.response_text,
                    trigger_timestamp=exchange.trigger_start_time,
                    response_timestamp=exchange.response_start_time,
                    chat_id=exchange.chat_id,
                    contact_id=exchange.contact_id,
                    trigger_msg_id=exchange.primary_trigger_msg_id,
                    response_msg_id=exchange.primary_response_msg_id,
                    trigger_msg_ids=exchange.trigger_msg_ids,
                    response_msg_ids=exchange.response_msg_ids,
                    is_group=is_group,
                    gate_a_passed=result.gate_a_passed,
                    gate_b_score=result.gate_b_score,
                    gate_c_verdict=result.gate_c_verdict,
                    validity_status=result.final_status,
                    context_json=json_module.dumps(exchange.context_to_json()),
                    gate_a_reason=result.gate_a_reason,
                    gate_c_scores_json=json_module.dumps(result.gate_c_scores)
                    if result.gate_c_scores
                    else None,
                )

                if pair:
                    aggregate_stats["pairs_added"] += 1
                else:
                    aggregate_stats["pairs_skipped_duplicate"] += 1

            aggregate_stats["conversations_processed"] += 1

        except Exception as e:
            logger.warning("Error extracting from %s: %s", conv.chat_id, e)
            aggregate_stats["errors"].append({"chat_id": conv.chat_id, "error": str(e)})

    return aggregate_stats
