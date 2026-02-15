#!/usr/bin/env python3
"""A/B compare pass-1 extraction modes on the same chat slices.

Mode A (legacy):
- Free-text [Segment N] lines
- max_tokens=800

Mode B (strict):
- JSONL schema with evidence quote
- max_tokens=240
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, ".")

from integrations.imessage import ChatDBReader
from jarvis.contacts.instruction_extractor import (
    NEGATIVE_CONSTRAINTS,
    _PASS1_META_LINE_RE,
    _PASS1_PLACEHOLDER_RE,
    get_instruction_extractor,
)
from jarvis.contacts.junk_filters import is_junk_message
from jarvis.text_normalizer import normalize_text
from jarvis.topics.topic_segmenter import TopicSegment

LEGACY_SYSTEM_PROMPT = """You extract durable personal facts from chat turns.

Task:
- Return ONLY stable personal claims that are useful for a long-lived profile.

Allowed claim types:
- identity/relationship (family, partner, close friend)
- work/school/role/company
- home/current location (not temporary travel)
- durable preferences (likes/dislikes/habits)
- stable schedule preference (e.g., "usually free after 6")

Do NOT extract:
- one-off logistics (meetups, ETAs, "on my way", "can we call", orders, deliveries)
- sports/news chatter, jokes, reactions, tapbacks
- meta speech ("X said", "X asked", "X mentioned")
- facts about group chats/platforms/companies unless clearly about a person
- speculative/uncertain claims

Output format:
- One line per claim with this exact structure:
  [Segment N] | <Person Name> | <durable_fact_sentence>
- Max 3 claims per segment.
- If a segment has no durable claims, output nothing for that segment.
- If no claims at all, output exactly: NONE

Hard rules:
- Only explicit claims from text (no inference).
- Use 3rd-person wording.
- Subject must be a person, never a group/chat.
- Do not use placeholders, brackets, or variables (no "[City]", "[Job Title]", "<unknown>").
- No headings, markdown, commentary, or extra labels."""

LEGACY_INFER_RE = re.compile(
    r"\b(likely|suggest(?:s|ing)?|imply(?:ing)?|possibly|maybe|appears to|history of)\b",
    re.IGNORECASE,
)


@dataclass
class EvalStats:
    chats: int = 0
    segments: int = 0
    claims: int = 0
    suspicious: int = 0
    inferred: int = 0
    prefilter_messages_skipped: int = 0


def _is_suspicious_claim(text: str) -> bool:
    t = " ".join(text.lower().split())
    if not t:
        return True
    low_info = {"me", "you", "that", "this", "it", "them", "someone", "something", "none"}
    if t in low_info:
        return True
    if any(k in t for k in ("group", "chat", "added", "removed", "left the group")):
        return True
    if t.startswith(("liked ", "loved ", "laughed at ", "emphasized ", "questioned ")):
        return True
    return False


def _build_segments(reader: ChatDBReader, limit: int, window: int) -> tuple[list[tuple[str, str, list[TopicSegment]]], str]:
    convos = reader.get_conversations(limit=max(50, min(1000, limit * 10)))
    user_name = reader.get_user_name()
    active = [
        c
        for c in convos
        if c.message_count >= 5 and ("iMessage" in c.chat_id or "RCS" in c.chat_id or "SMS" in c.chat_id)
    ][:limit]

    out: list[tuple[str, str, list[TopicSegment]]] = []
    for conv in active:
        chat_id = conv.chat_id
        contact_name = conv.display_name or "Contact"
        messages = reader.get_messages(chat_id, limit=window)
        messages.reverse()
        if not messages:
            continue
        segments: list[TopicSegment] = []
        for j in range(0, len(messages), 20):
            window_msgs = messages[j : j + 25]
            if not window_msgs:
                break
            segments.append(
                TopicSegment(
                    chat_id=chat_id,
                    contact_id=chat_id,
                    messages=window_msgs,
                    start_time=window_msgs[0].date,
                    end_time=window_msgs[-1].date,
                    message_count=len(window_msgs),
                    segment_id=f"ab_{chat_id}_{j}",
                    text="\n".join((m.text or "") for m in window_msgs),
                )
            )
        if segments:
            out.append((chat_id, contact_name, segments))
    return out, user_name


def _format_batch_text(segments: list[TopicSegment], contact_id: str, contact_name: str, user_name: str) -> tuple[str, int]:
    segment_texts = []
    skipped = 0
    for i, segment in enumerate(segments):
        messages = getattr(segment, "messages", [])
        prompt_lines = []
        if messages:
            current_label = None
            current_block: list[str] = []
            for m in messages:
                label = user_name if m.is_from_me else (getattr(m, "sender_name", None) or contact_name)
                raw_msg = " ".join((m.text or "").splitlines()).strip()
                if not raw_msg:
                    continue
                clean_msg = normalize_text(
                    raw_msg,
                    filter_garbage=True,
                    filter_attributed_artifacts=True,
                    strip_signatures=True,
                )
                if not clean_msg:
                    skipped += 1
                    continue
                if is_junk_message(clean_msg, contact_id):
                    skipped += 1
                    continue
                if label == current_label:
                    current_block.append(clean_msg)
                else:
                    if current_block and current_label:
                        prompt_lines.append(f"{current_label}: {' '.join(current_block)}")
                    current_label = label
                    current_block = [clean_msg]
            if current_block and current_label:
                prompt_lines.append(f"{current_label}: {' '.join(current_block)}")
        segment_texts.append(f"[Segment {i}]\n" + "\n".join(prompt_lines))
    return "\n\n".join(segment_texts), skipped


def _legacy_extract(extractor, segments: list[TopicSegment], contact_id: str, contact_name: str, user_name: str) -> tuple[list[list[str]], int]:
    claims_by_segment: list[list[str]] = [[] for _ in segments]
    batch_text, skipped = _format_batch_text(segments, contact_id, contact_name, user_name)
    p1_user = f"Conversation:\n{batch_text}\n\nFactual Claims (prefix with [Segment N]):\n- "
    messages_p1 = [
        {"role": "system", "content": LEGACY_SYSTEM_PROMPT},
        {"role": "user", "content": p1_user},
    ]
    formatted = extractor._loader._tokenizer.apply_chat_template(messages_p1, tokenize=False, add_generation_prompt=True)
    if not formatted.endswith("- "):
        formatted += "- "

    res = extractor._loader.generate_sync(
        prompt=formatted,
        max_tokens=800,
        temperature=0.0,
        stop_sequences=["<|im_end|>", "###"],
        pre_formatted=True,
        negative_constraints=NEGATIVE_CONSTRAINTS,
    )
    raw_facts = "- " + res.text.strip()
    saw_segment_tags = False
    for raw_line in raw_facts.split("\n"):
        clean = raw_line.strip()
        if not clean:
            continue
        seg_match = re.search(r"\[Segment\s*(\d+)\]", clean, re.IGNORECASE)
        if not seg_match:
            continue
        saw_segment_tags = True
        seg_idx = int(seg_match.group(1))
        if seg_idx < 0 or seg_idx >= len(segments):
            continue
        claim = re.sub(r"^[\s\-\*\d\.]+\s*", "", clean).strip()
        claim = re.sub(r"\[Segment\s*\d+\]\s*", "", claim, flags=re.IGNORECASE).strip()
        claim = re.sub(r"^[\)\]\:\-]+\s*", "", claim).strip()
        if claim and not _PASS1_META_LINE_RE.search(claim) and not _PASS1_PLACEHOLDER_RE.search(claim):
            claims_by_segment[seg_idx].append(claim)

    if not saw_segment_tags and claims_by_segment:
        fallback_claims = []
        for raw_line in raw_facts.split("\n"):
            claim = re.sub(r"^[\s\-\*\d\.]+\s*", "", raw_line).strip()
            if not claim:
                continue
            if claim.lower() in {"none", "personal factual claims:"}:
                continue
            claim = re.sub(r"^[\)\]\:\-]+\s*", "", claim).strip()
            if _PASS1_META_LINE_RE.search(claim) or _PASS1_PLACEHOLDER_RE.search(claim):
                continue
            fallback_claims.append(claim)
        claims_by_segment[0].extend(fallback_claims)

    return claims_by_segment, skipped


def _accumulate(stats: EvalStats, claims_by_segment: list[list[str]], skipped: int) -> None:
    stats.prefilter_messages_skipped += skipped
    for claims in claims_by_segment:
        for claim in claims:
            stats.claims += 1
            if _is_suspicious_claim(claim):
                stats.suspicious += 1
            if LEGACY_INFER_RE.search(claim):
                stats.inferred += 1


def _run_compare(limit: int, window: int, tier: str) -> None:
    extractor = get_instruction_extractor(tier=tier)
    if not extractor.is_loaded():
        extractor.load()

    strict = EvalStats()
    legacy = EvalStats()

    with ChatDBReader() as reader:
        datasets, user_name = _build_segments(reader, limit, window)
        for chat_id, contact_name, segments in datasets:
            strict.chats += 1
            legacy.chats += 1
            strict.segments += len(segments)
            legacy.segments += len(segments)

            strict_claims = extractor.extract_pass1_claims_from_batch(
                segments, contact_id=chat_id, contact_name=contact_name, user_name=user_name
            )
            strict_skipped = extractor.get_last_batch_stats().get("prefilter_messages_skipped", 0)
            _accumulate(strict, strict_claims, strict_skipped)

            legacy_claims, legacy_skipped = _legacy_extract(
                extractor, segments, contact_id=chat_id, contact_name=contact_name, user_name=user_name
            )
            _accumulate(legacy, legacy_claims, legacy_skipped)

    def pct(a: int, b: int) -> float:
        return (100.0 * a / b) if b else 0.0

    print("\n" + "=" * 88)
    print(f"PASS-1 A/B COMPARISON  (tier={tier}, chats={limit}, window={window})")
    print("=" * 88)
    print(f"{'Metric':28} {'Legacy(800,text)':>22} {'Strict(240,json)':>22}")
    print("-" * 88)
    print(f"{'chats':28} {legacy.chats:22d} {strict.chats:22d}")
    print(f"{'segments':28} {legacy.segments:22d} {strict.segments:22d}")
    print(f"{'claims':28} {legacy.claims:22d} {strict.claims:22d}")
    print(
        f"{'claims/segment':28} "
        f"{(legacy.claims / legacy.segments if legacy.segments else 0.0):22.3f} "
        f"{(strict.claims / strict.segments if strict.segments else 0.0):22.3f}"
    )
    print(
        f"{'suspicious ratio':28} "
        f"{pct(legacy.suspicious, legacy.claims):21.2f}% "
        f"{pct(strict.suspicious, strict.claims):21.2f}%"
    )
    print(
        f"{'inference-word ratio':28} "
        f"{pct(legacy.inferred, legacy.claims):21.2f}% "
        f"{pct(strict.inferred, strict.claims):21.2f}%"
    )
    print(f"{'prefilter msgs skipped':28} {legacy.prefilter_messages_skipped:22d} {strict.prefilter_messages_skipped:22d}")


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B compare pass-1 extraction modes")
    parser.add_argument("--limit", type=int, default=20, help="Number of chats")
    parser.add_argument("--window", type=int, default=100, help="Messages per chat")
    parser.add_argument("--tier", default="0.7b", help="Instruction model tier")
    args = parser.parse_args()
    _run_compare(limit=args.limit, window=args.window, tier=args.tier)


if __name__ == "__main__":
    main()
