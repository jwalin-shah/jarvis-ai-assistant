#!/usr/bin/env python3
"""Evaluate normalization impact on labeled data slices.

Reports how many samples are dropped/changed and basic length stats.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from jarvis.text_normalizer import extract_text_features, normalize_for_task

CHAT_DB_PATH = Path.home() / "Library/Messages/chat.db"


def _load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
            if limit and len(data) >= limit:
                break
    return data


def _summarize(label: str, rows: list[dict], task: str, text_key: str = "text") -> None:
    total = len(rows)
    if total == 0:
        print(f"{label}: no rows")
        return

    dropped = 0
    changed = 0
    short = 0
    lengths_raw = []
    lengths_norm = []
    tokens = Counter()

    for r in rows:
        text = (r.get(text_key) or "").strip()
        if not text:
            dropped += 1
            continue

        normalized = normalize_for_task(text, task)
        if not normalized:
            dropped += 1
            continue

        if normalized != text:
            changed += 1

        lengths_raw.append(len(text))
        lengths_norm.append(len(normalized))
        feats = extract_text_features(normalized)
        if feats.is_short:
            short += 1

        for token in ("<EMAIL>", "<PHONE>", "<PERSON>", "<CITY>", "<TEAM>", "<CODE>"):
            if token in normalized:
                tokens[token] += 1

        if "<URL:" in normalized or "<URL>" in normalized:
            tokens["<URL>"] += 1

        if "<EMOJI_" in normalized:
            tokens["<EMOJI>"] += 1

    avg_raw = sum(lengths_raw) / len(lengths_raw) if lengths_raw else 0
    avg_norm = sum(lengths_norm) / len(lengths_norm) if lengths_norm else 0

    print(f"\n{label} (task={task})")
    print(f"  total:   {total}")
    print(f"  dropped: {dropped} ({dropped / total:.1%})")
    print(f"  changed: {changed} ({changed / total:.1%})")
    print(f"  short:   {short} ({short / total:.1%})")
    print(f"  avg_len_raw:  {avg_raw:.1f}")
    print(f"  avg_len_norm: {avg_norm:.1f}")
    if tokens:
        print("  token_hits:")
        for k, v in tokens.most_common():
            print(f"    {k:<10} {v}")


def _load_from_chat_db(limit: int | None = None, message_type: str = "incoming") -> list[dict]:
    if not CHAT_DB_PATH.exists():
        raise FileNotFoundError(f"iMessage database not found: {CHAT_DB_PATH}")

    conn = sqlite3.connect(f"file:{CHAT_DB_PATH}?mode=ro", uri=True)
    cursor = conn.cursor()

    query = """
        SELECT
            message.text,
            message.attributedBody,
            message.is_from_me
        FROM message
        WHERE (message.text IS NOT NULL AND message.text != '')
           OR message.attributedBody IS NOT NULL
        ORDER BY message.date DESC
    """

    if limit:
        query += f" LIMIT {limit * 3}"

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    messages: list[dict] = []
    for text, attributed_body, is_from_me in rows:
        if message_type == "incoming" and is_from_me:
            continue
        if message_type == "outgoing" and not is_from_me:
            continue

        if not text and attributed_body:
            try:
                from integrations.imessage.parser import parse_attributed_body

                text = parse_attributed_body(attributed_body)
            except Exception:
                text = None

        if not text:
            continue

        messages.append({"text": text})
        if limit and len(messages) >= limit:
            break

    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate normalization impact on labeled data")
    parser.add_argument("--limit", type=int, default=500, help="Max samples per dataset")
    parser.add_argument(
        "--task",
        choices=["classification", "topic_modeling", "extraction"],
        default="classification",
    )
    parser.add_argument("--trigger", type=Path, default=Path("data/trigger_labeling.jsonl"))
    parser.add_argument("--response", type=Path, default=Path("data/response_labeling.jsonl"))
    parser.add_argument("--input", type=Path, default=None, help="JSONL input path (optional)")
    parser.add_argument("--text-field", type=str, default="text", help="Text field name in JSONL")
    parser.add_argument("--from-chatdb", action="store_true", help="Sample directly from chat.db")
    parser.add_argument("--type", choices=["incoming", "outgoing", "both"], default="incoming")
    args = parser.parse_args()

    if args.from_chatdb:
        if args.type in ("incoming", "both"):
            rows = _load_from_chat_db(args.limit, message_type="incoming")
            _summarize("ChatDB incoming", rows, args.task, text_key="text")
        if args.type in ("outgoing", "both"):
            rows = _load_from_chat_db(args.limit, message_type="outgoing")
            _summarize("ChatDB outgoing", rows, args.task, text_key="text")
        return

    if args.input:
        if args.input.exists():
            rows = _load_jsonl(args.input, args.limit)
            _summarize(f"Input {args.input.name}", rows, args.task, text_key=args.text_field)
        else:
            print(f"Input file not found: {args.input}")
        return

    if args.trigger.exists():
        rows = _load_jsonl(args.trigger, args.limit)
        _summarize("Trigger labels", rows, args.task)
    else:
        print(f"Trigger file not found: {args.trigger}")

    if args.response.exists():
        rows = _load_jsonl(args.response, args.limit)
        _summarize("Response labels", rows, args.task)
    else:
        print(f"Response file not found: {args.response}")


if __name__ == "__main__":
    main()
