#!/usr/bin/env python3
"""Offline chat analytics/backfill prepass with batched message loading.

Purpose:
- Pull many chat messages efficiently in chunked batches.
- Keep messages in memory only per chunk (bounded RAM).
- Score/filter windows before sending anything to LLM.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

sys.path.insert(0, ".")

from integrations.imessage import ChatDBReader
from jarvis.contacts.instruction_extractor import get_instruction_extractor
from jarvis.contacts.junk_filters import is_junk_message
from jarvis.text_normalizer import normalize_text
from jarvis.topics.topic_segmenter import TopicSegment


@dataclass
class WindowStat:
    total_messages: int = 0
    nonempty_messages: int = 0
    nonjunk_messages: int = 0


def _build_segment(chat_id: str, messages: list[Any], start_idx: int, size: int) -> TopicSegment | None:
    window_msgs = messages[start_idx : start_idx + size]
    if not window_msgs:
        return None
    return TopicSegment(
        chat_id=chat_id,
        contact_id=chat_id,
        messages=window_msgs,
        start_time=window_msgs[0].date,
        end_time=window_msgs[-1].date,
        message_count=len(window_msgs),
        segment_id=str(uuid.uuid4()),
        text="\n".join((m.text or "") for m in window_msgs),
    )


def _score_window(messages: list[Any], chat_id: str) -> WindowStat:
    stat = WindowStat(total_messages=len(messages))
    for m in messages:
        raw = " ".join((m.text or "").splitlines()).strip()
        if not raw:
            continue
        stat.nonempty_messages += 1
        clean = normalize_text(
            raw,
            filter_garbage=True,
            filter_attributed_artifacts=True,
            strip_signatures=True,
        )
        if not clean:
            continue
        if is_junk_message(clean, chat_id):
            continue
        stat.nonjunk_messages += 1
    return stat


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline chat analytics prepass")
    parser.add_argument("--limit", type=int, default=50, help="Number of chats")
    parser.add_argument("--window", type=int, default=0, help="Messages per chat (0=all)")
    parser.add_argument("--chat-chunk-size", type=int, default=50, help="Chats per batch fetch")
    parser.add_argument("--segment-window-size", type=int, default=25, help="Window size")
    parser.add_argument("--segment-window-stride", type=int, default=25, help="Window stride")
    parser.add_argument(
        "--min-signal-messages",
        type=int,
        default=4,
        help="Minimum non-junk messages required for a window to be LLM-eligible",
    )
    parser.add_argument(
        "--extract-pass1",
        action="store_true",
        help="Also run pass-1 extractor only on eligible windows",
    )
    parser.add_argument(
        "--max-windows-per-chat",
        type=int,
        default=4,
        help="Cap eligible windows sent to pass-1 per chat",
    )
    parser.add_argument("--tier", default="0.7b", help="Instruction model tier for pass-1")
    args = parser.parse_args()

    t0 = time.time()
    totals = {
        "chats": 0,
        "messages": 0,
        "windows_total": 0,
        "windows_eligible": 0,
        "pass1_claims": 0,
    }

    extractor = get_instruction_extractor(tier=args.tier) if args.extract_pass1 else None

    with ChatDBReader() as reader:
        convo_fetch_limit = max(50, min(1000, args.limit * 10))
        convos = reader.get_conversations(limit=convo_fetch_limit)
        user_name = reader.get_user_name()
        active = [
            c
            for c in convos
            if c.message_count >= 5 and ("iMessage" in c.chat_id or "RCS" in c.chat_id or "SMS" in c.chat_id)
        ][: args.limit]
        totals["chats"] = len(active)

        for i in range(0, len(active), max(1, args.chat_chunk_size)):
            batch = active[i : i + args.chat_chunk_size]
            chat_ids = [c.chat_id for c in batch]
            msgs_by_chat = reader.get_messages_batch(
                chat_ids,
                limit_per_chat=None if args.window <= 0 else args.window,
            )

            for conv in batch:
                chat_id = conv.chat_id
                contact_name = conv.display_name or "Contact"
                messages = msgs_by_chat.get(chat_id, [])
                if not messages:
                    continue
                # Convert from newest-first to chronological.
                messages = sorted(messages, key=lambda m: m.date or datetime.min)
                totals["messages"] += len(messages)

                eligible_segments: list[TopicSegment] = []
                for j in range(0, len(messages), max(1, args.segment_window_stride)):
                    seg = _build_segment(chat_id, messages, j, max(1, args.segment_window_size))
                    if not seg:
                        continue
                    totals["windows_total"] += 1
                    stat = _score_window(seg.messages, chat_id)
                    if stat.nonjunk_messages >= args.min_signal_messages:
                        totals["windows_eligible"] += 1
                        if len(eligible_segments) < args.max_windows_per_chat:
                            eligible_segments.append(seg)

                if args.extract_pass1 and eligible_segments:
                    claims = extractor.extract_pass1_claims_from_batch(
                        eligible_segments,
                        contact_id=chat_id,
                        contact_name=contact_name,
                        user_name=user_name,
                    )
                    claim_count = sum(len(x) for x in claims)
                    totals["pass1_claims"] += claim_count

    elapsed = time.time() - t0
    print("\n=== OFFLINE ANALYTICS SUMMARY ===")
    print(f"chats={totals['chats']} messages={totals['messages']}")
    print(f"windows_total={totals['windows_total']} windows_eligible={totals['windows_eligible']}")
    if args.extract_pass1:
        print(f"pass1_claims={totals['pass1_claims']}")
    print(f"elapsed_s={elapsed:.2f}")


if __name__ == "__main__":
    main()
