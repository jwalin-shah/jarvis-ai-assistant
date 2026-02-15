#!/usr/bin/env python3
"""Evaluate pass-1 (natural-claim) extraction quality on a small chat slice."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

sys.path.insert(0, ".")

from integrations.imessage import ChatDBReader
from jarvis.contacts.instruction_extractor import get_instruction_extractor
from jarvis.topics.topic_segmenter import TopicSegment


@dataclass
class EvalStats:
    chats: int = 0
    segments: int = 0
    claims: int = 0
    suspicious: int = 0
    prefilter_messages_skipped: int = 0


def _is_suspicious_claim(text: str) -> bool:
    t = " ".join(text.lower().split())
    if not t:
        return True
    low_info = {
        "me",
        "you",
        "that",
        "this",
        "it",
        "them",
        "someone",
        "something",
        "none",
    }
    if t in low_info:
        return True
    if any(k in t for k in ("group", "chat", "added", "removed", "left the group")):
        return True
    if t.startswith(("liked ", "loved ", "laughed at ", "emphasized ", "questioned ")):
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Pass-1 claims evaluator")
    parser.add_argument("--limit", type=int, default=5, help="Number of chats")
    parser.add_argument("--window", type=int, default=100, help="Messages per chat")
    parser.add_argument("--tier", default="0.7b", help="Instruction model tier")
    parser.add_argument(
        "--convo-fetch-limit",
        type=int,
        default=0,
        help="Raw get_conversations limit override (0=auto from --limit)",
    )
    args = parser.parse_args()

    stats = EvalStats()
    extractor = get_instruction_extractor(tier=args.tier)

    with ChatDBReader() as reader:
        convo_fetch_limit = (
            args.convo_fetch_limit
            if args.convo_fetch_limit > 0
            else max(50, min(1000, args.limit * 10))
        )
        convos = reader.get_conversations(limit=convo_fetch_limit)
        user_name = reader.get_user_name()

        active = [
            c
            for c in convos
            if c.message_count >= 5 and ("iMessage" in c.chat_id or "RCS" in c.chat_id or "SMS" in c.chat_id)
        ][: args.limit]

        for idx, conv in enumerate(active, 1):
            chat_id = conv.chat_id
            contact_name = conv.display_name or "Contact"
            messages = reader.get_messages(chat_id, limit=args.window)
            messages.reverse()
            if not messages:
                continue

            # same windowing strategy as backfill_complete
            segments = []
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
                        segment_id=f"eval_{chat_id}_{j}",
                        text="\n".join((m.text or "") for m in window_msgs),
                    )
                )

            if not segments:
                continue

            stats.chats += 1
            stats.segments += len(segments)

            claims_by_segment = extractor.extract_pass1_claims_from_batch(
                segments,
                contact_id=chat_id,
                contact_name=contact_name,
                user_name=user_name,
            )
            batch_stats = extractor.get_last_batch_stats()
            stats.prefilter_messages_skipped += batch_stats.get("prefilter_messages_skipped", 0)

            print(f"\n[{idx}/{len(active)}] {chat_id} ({contact_name})")
            for sidx, claims in enumerate(claims_by_segment):
                if not claims:
                    continue
                print(f"  Segment {sidx}: {len(claims)} claims")
                for claim in claims[:5]:
                    flag = " [SUS]" if _is_suspicious_claim(claim) else ""
                    print(f"    - {claim}{flag}")
                    stats.claims += 1
                    if flag:
                        stats.suspicious += 1

    print("\n" + "=" * 72)
    print("PASS-1 SUMMARY")
    print("=" * 72)
    print(f"chats={stats.chats} segments={stats.segments}")
    print(f"claims_shown={stats.claims} suspicious_shown={stats.suspicious}")
    ratio = (stats.suspicious / stats.claims) if stats.claims else 0.0
    print(f"suspicious_ratio={ratio:.3f}")
    print(f"prefilter_messages_skipped={stats.prefilter_messages_skipped}")


if __name__ == "__main__":
    main()
