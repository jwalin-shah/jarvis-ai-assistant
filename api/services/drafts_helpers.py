"""Helper utilities for draft generation endpoints."""

from __future__ import annotations

import re

from jarvis.contracts.imessage import Message

_ALLOWED_INSTRUCTION_PATTERN = re.compile(r"^[a-zA-Z0-9 ,.'!?;:\-/()]+$")


def format_messages_for_context(messages: list[Message]) -> str:
    """Format messages as context string (chronological order), combining consecutive turns."""
    if not messages:
        return ""

    chronological = list(reversed(messages))
    turns: list[str] = []

    current_sender: str | None = None
    current_text_parts: list[str] = []

    for msg in chronological:
        sender = "You" if msg.is_from_me else (msg.sender_name or msg.sender or "Contact")
        text = (msg.text or "").strip()
        if not text:
            continue

        if sender == current_sender:
            current_text_parts.append(text)
        else:
            if current_sender is not None:
                turns.append(f"{current_sender}: {' '.join(current_text_parts)}")
            current_sender = sender
            current_text_parts = [text]

    if current_sender is not None:
        turns.append(f"{current_sender}: {' '.join(current_text_parts)}")

    return "\n".join(turns)


def sanitize_instruction(instruction: str | None) -> str | None:
    """Sanitize user instruction to prevent prompt injection."""
    if not instruction:
        return None

    instruction = instruction[:200].strip()
    if not instruction:
        return None
    if not _ALLOWED_INSTRUCTION_PATTERN.match(instruction):
        return None
    return instruction


def build_summary_prompt(context: str, num_messages: int) -> str:
    return (
        f"Summarize this conversation of {num_messages} messages. "
        "Provide a brief summary and extract 2-4 key points.\n\n"
        f"Conversation:\n{context}\n\n"
        "Provide your response in this format:\n"
        "Summary: [1-2 sentence summary]\n"
        "Key points:\n- [point 1]\n- [point 2]"
    )


def parse_summary_response(response_text: str) -> tuple[str, list[str]]:
    """Parse LLM summary response into (summary, key_points)."""
    import re as _re

    lines = response_text.strip().split("\n")
    summary = ""
    key_points: list[str] = []

    in_key_points = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        clean = stripped.replace("**", "")
        lower = clean.lower()

        if not summary and not in_key_points:
            summary_match = _re.match(
                r"^(?:here\s+is\s+(?:the\s+)?)?summary\s*:\s*(.+)",
                lower,
            )
            if summary_match:
                colon_idx = clean.find(":")
                if colon_idx >= 0:
                    summary = clean[colon_idx + 1 :].strip()
                continue

        if _re.match(r"^(?:here\s+are\s+(?:the\s+)?)?key\s+points\s*:?\s*$", lower):
            in_key_points = True
            continue

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

    if not summary:
        raw = response_text.strip()
        if len(raw) > 200:
            summary = raw[:200].rsplit(" ", 1)[0] + "..."
        else:
            summary = raw

    if not key_points:
        key_points = ["See summary for details"]

    return summary, key_points
