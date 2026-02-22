#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def is_reaction(text: str) -> bool:
    lowered = text.lower().strip()
    patterns = ["liked", "loved", "emphasized", "reacted", "tapback", "👍", "❤️"]
    return any(p in lowered for p in patterns) and len(lowered.split()) <= 6


def build_turns_streaming(path: Path) -> dict[str, list[dict]]:
    """Build per-contact turns in one pass over already time-ordered message JSONL."""
    by_contact_turns: dict[str, list[dict]] = defaultdict(list)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            msg = json.loads(line)
            contact = msg.get("contact")
            if not contact:
                continue

            text = normalize_text(msg.get("text", ""))
            if not text:
                continue

            is_from_me = bool(msg.get("is_from_me"))
            ts = msg.get("timestamp")
            if not ts:
                continue
            dt = datetime.fromisoformat(ts)

            turns = by_contact_turns[contact]
            if turns and turns[-1]["is_from_me"] == is_from_me:
                turns[-1]["texts"].append(text)
                turns[-1]["end_dt"] = dt
                turns[-1]["end_timestamp"] = ts
            else:
                turns.append(
                    {
                        "is_from_me": is_from_me,
                        "texts": [text],
                        "start_dt": dt,
                        "end_dt": dt,
                        "start_timestamp": ts,
                        "end_timestamp": ts,
                    }
                )

    return by_contact_turns


def parse_pairs_from_turns(
    by_contact_turns: dict[str, list[dict]],
    max_reply_window: timedelta,
) -> list[dict]:
    pairs: list[dict] = []

    for contact, turns in by_contact_turns.items():
        for i, turn in enumerate(turns):
            turn_text = "\n".join(turn["texts"])
            if turn["is_from_me"]:
                continue
            if is_reaction(turn_text):
                continue
            if i + 1 >= len(turns):
                continue

            candidate = turns[i + 1]
            candidate_text = "\n".join(candidate["texts"])
            if not candidate["is_from_me"]:
                continue
            if is_reaction(candidate_text):
                continue

            delta = candidate["start_dt"] - turn["end_dt"]
            if delta > max_reply_window:
                continue

            context_slice = turns[max(0, i - 5) : i]
            context = ["\n".join(c["texts"]) for c in context_slice]

            pairs.append(
                {
                    "contact": contact,
                    "context": context,
                    "their_message": turn_text,
                    "my_reply": candidate_text,
                    "timestamp": candidate["start_timestamp"],
                    "response_time_seconds": int(delta.total_seconds()),
                    "their_turn_messages": len(turn["texts"]),
                    "my_turn_messages": len(candidate["texts"]),
                }
            )

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse exported messages into turn-level reply pairs")
    parser.add_argument("--input", default="data/raw/messages.jsonl")
    parser.add_argument("--output", default="data/processed/conversation_pairs.jsonl")
    parser.add_argument("--window-hours", type=int, default=24)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[02] Loading {in_path}")
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}. Run scripts/01_export_imessage.py first.")

    by_contact_turns = build_turns_streaming(in_path)
    pairs = parse_pairs_from_turns(by_contact_turns, max_reply_window=timedelta(hours=args.window_hours))

    with out_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"[02] Contacts processed: {len(by_contact_turns)}")
    print(f"[02] Parsed {len(pairs)} turn-level conversation pairs -> {out_path}")


if __name__ == "__main__":
    main()
