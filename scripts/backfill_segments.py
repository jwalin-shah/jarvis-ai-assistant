#!/usr/bin/env python3
"""Backfill conversation_segments from iMessage chat.db.

Migrates the JARVIS DB schema (creates conversation_segments + segment_messages
tables if needed), then iterates over all active chats and runs the topic
segmenter + persist pipeline.

Usage:
    uv run python scripts/backfill_segments.py
    uv run python scripts/backfill_segments.py --limit 10   # only first 10 chats
    uv run python scripts/backfill_segments.py --window 200  # messages per chat
"""
import argparse
import sys
import time

sys.path.insert(0, ".")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill conversation segments")
    parser.add_argument("--limit", type=int, default=0, help="Max chats to process (0=all)")
    parser.add_argument("--window", type=int, default=200, help="Messages per chat to segment")
    args = parser.parse_args()

    from integrations.imessage import ChatDBReader
    from jarvis.db import get_db
    from jarvis.nlp.ner_client import is_service_running
    from jarvis.topics.segment_pipeline import process_segments
    from jarvis.topics.segment_storage import delete_segments_for_chat
    from jarvis.topics.topic_segmenter import segment_conversation

    # Step 0: Ensure NER service is running (auto-starts if venv exists)
    print("Checking NER service...", flush=True)
    if is_service_running():
        print("  NER service is running (entities will be used for topic labels)", flush=True)
    else:
        print("  NER service not available (topic labels will use word frequency fallback)", flush=True)

    # Step 1: Migrate schema
    print("Migrating DB schema...", flush=True)
    db = get_db()
    updated = db.init_schema()
    print(f"  Schema {'updated' if updated else 'already current'}", flush=True)

    # Step 2: Get all active chat_ids
    print("Fetching active conversations...", flush=True)
    with ChatDBReader() as reader:
        convos = reader.get_conversations(limit=500)
    chat_ids = [c.chat_id for c in convos]
    print(f"  Found {len(chat_ids)} active chats", flush=True)

    if args.limit > 0:
        chat_ids = chat_ids[: args.limit]
        print(f"  Limited to {len(chat_ids)} chats", flush=True)

    # Step 3: Process each chat
    total_segments = 0
    total_persisted = 0
    errors = 0
    t0 = time.time()

    for i, chat_id in enumerate(chat_ids):
        print(f"[{i + 1}/{len(chat_ids)}] {chat_id[:40]}...", end=" ", flush=True)
        try:
            with ChatDBReader() as reader:
                messages = reader.get_messages(chat_id, limit=args.window)
                messages.reverse()  # chronological order

            if not messages:
                print("no messages, skipping", flush=True)
                continue

            segments = segment_conversation(messages, contact_id=chat_id)
            if not segments:
                print("no segments", flush=True)
                continue

            # Clear old segments for this chat
            with db.connection() as conn:
                delete_segments_for_chat(conn, chat_id)

            # Persist + index (no fact extraction)
            stats = process_segments(
                segments, chat_id, contact_id=chat_id, extract_facts=False
            )

            total_segments += len(segments)
            total_persisted += stats["persisted"]
            print(
                f"{len(segments)} segments, {stats['persisted']} persisted, "
                f"{stats['indexed']} indexed",
                flush=True,
            )
        except Exception as e:
            errors += 1
            print(f"ERROR: {e}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s", flush=True)
    print(f"  Chats processed: {len(chat_ids)}", flush=True)
    print(f"  Total segments: {total_segments}", flush=True)
    print(f"  Total persisted: {total_persisted}", flush=True)
    print(f"  Errors: {errors}", flush=True)


if __name__ == "__main__":
    main()
