#!/usr/bin/env python3
"""Complete backfill: segments + embeddings + facts.

This script runs the full V4 pipeline on historical messages:
1. Topic segmentation (creates conversation_segments)
2. Embedding indexing (creates vec_chunks for semantic search)
3. Fact extraction (creates contact_facts)
4. Contact creation (ensures contacts exist in jarvis.db)

Usage:
    uv run python scripts/backfill_complete.py
    uv run python scripts/backfill_complete.py --limit 50    # only top 50 chats
    uv run python scripts/backfill_complete.py --window 100  # messages per chat
"""
import argparse
import os
import sys
import time
from typing import Any

sys.path.insert(0, ".")


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete backfill (segments + facts)")
    parser.add_argument("--limit", type=int, default=0, help="Max chats to process (0=all)")
    parser.add_argument("--window", type=int, default=0, help="Messages per chat (0=all)")
    parser.add_argument("--force", action="store_true", help="Force re-processing of already completed chats")
    parser.add_argument("--skip-segments", action="store_true", help="Skip segment creation")
    parser.add_argument("--skip-facts", action="store_true", help="Skip fact extraction")
    parser.add_argument(
        "--segment-window-size",
        type=int,
        default=max(1, int(os.getenv("BACKFILL_SEGMENT_WINDOW_SIZE", "25"))),
        help="Messages per extraction segment window (default: 25)",
    )
    parser.add_argument(
        "--segment-window-stride",
        type=int,
        default=max(1, int(os.getenv("BACKFILL_SEGMENT_WINDOW_STRIDE", "25"))),
        help="Stride for sliding windows (default: 25, no overlap)",
    )
    parser.add_argument(
        "--extract-slots",
        type=int,
        default=max(1, int(os.getenv("BACKFILL_EXTRACT_SLOTS", "1"))),
        help="Concurrent extraction slots across workers (default: 1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, int(os.getenv("BACKFILL_MAX_WORKERS", "2"))),
        help="Parallel chat workers for backfill (default: BACKFILL_MAX_WORKERS or 2)",
    )
    args = parser.parse_args()

    from integrations.imessage import ChatDBReader
    from jarvis.db import get_db

    db = get_db()
    db.init_schema()
    print("✓ Database schema initialized")

    # Step 0: Sync contacts from AddressBook
    print("\n[Step 0] Syncing contacts from AddressBook...")
    with ChatDBReader() as reader:
        # Avoid scanning the full chat list when a small --limit is requested.
        convo_fetch_limit = 1000 if args.limit <= 0 else max(50, min(1000, args.limit * 10))
        convos = reader.get_conversations(limit=convo_fetch_limit)
        user_name = reader.get_user_name()
    
    # Filter for real chats
    active_chats = [
        c for c in convos 
        if c.message_count >= 5 and ("iMessage" in c.chat_id or "RCS" in c.chat_id or "SMS" in c.chat_id)
    ]
    
    contacts_synced = 0
    for conv in active_chats:
        chat_id = conv.chat_id
        display_name = conv.display_name or "Contact"
        phone_or_email = chat_id.split(';')[-1] if ';' in chat_id else chat_id
        
        try:
            db.add_contact(
                display_name=display_name,
                chat_id=chat_id,
                phone_or_email=phone_or_email
            )
            contacts_synced += 1
        except Exception as e:
            print(f"  Warning: Could not sync contact {chat_id}: {e}")
    
    print(f"✓ Synced {contacts_synced} contacts from {len(active_chats)} active chats (user: {user_name})")
    
    if args.limit > 0:
        active_chats = active_chats[:args.limit]
        print(f"Limited to {len(active_chats)} chats for processing")

        import threading
        from concurrent.futures import ThreadPoolExecutor

        

        total_stats = {

            "chats": 0,

            "segments": 0,

            "indexed": 0,

            "facts": 0,

            "errors": 0

        }

        

        t0 = time.time()
        extraction_semaphore = threading.Semaphore(max(1, args.extract_slots))

    

        def process_chat(chat_data: tuple[int, Any]) -> dict[str, int] | str | None:
            i, conv = chat_data
            chat_id = conv.chat_id

            print(f"\n[{i+1}/{len(active_chats)}] Processing {chat_id[:50]}...")

            

            try:

                with ChatDBReader() as reader:

                    limit = args.window if args.window > 0 else 1000000 # Practical uncap

                    messages = reader.get_messages(chat_id, limit=limit)

                    messages.reverse()  # chronological

                    

                if not messages:

                    return None

                

                # Step 1: Create/update contact

                contact_name = conv.display_name or "Contact"

                db.add_contact(

                    display_name=contact_name,

                    chat_id=chat_id,

                    phone_or_email=chat_id.split(';')[-1] if ';' in chat_id else chat_id

                )

    

                # Step 2: Unified Pipeline (Segment + Embed + Extract Facts)

                if not args.skip_segments or not args.skip_facts:

                    from jarvis.topics.segment_pipeline import process_segments

                    from jarvis.topics.segment_storage import delete_segments_for_chat

                    from jarvis.topics.topic_segmenter import TopicSegment

                    import uuid

    

                    # Check if already done (unless --force)

                    already_done = False

                    if not args.force:

                        with db.connection() as conn:

                            row = conn.execute(

                                "SELECT last_extracted_rowid FROM contacts WHERE chat_id = ?",

                                (chat_id,)

                            ).fetchone()

                            if row and row[0]:

                                already_done = True

                    

                    if already_done:

                        print(f"  ⊘ {contact_name}: Already processed, skipping")

                        return {"segments": 0, "indexed": 0, "facts": 0}

                    

                    # 1. Create Sliding Window Segments

                    window_segments = []

                    messages = sorted(messages, key=lambda m: m.date)

                    for j in range(0, len(messages), args.segment_window_stride):

                        window_msgs = messages[j : j + args.segment_window_size]

                        if not window_msgs:

                            break

                        

                        seg = TopicSegment(

                            chat_id=chat_id,

                            contact_id=chat_id,

                            messages=window_msgs,

                            start_time=window_msgs[0].date,

                            end_time=window_msgs[-1].date,

                            message_count=len(window_msgs),

                            segment_id=str(uuid.uuid4()),

                            text="\n".join([m.text or "" for m in window_msgs])

                        )

                        window_segments.append(seg)

    

                    if window_segments:

                        # Clear existing segments/facts for this chat if re-running

                        with db.connection() as conn:

                            delete_segments_for_chat(conn, chat_id)

                            if args.force:

                                conn.execute("DELETE FROM contact_facts WHERE contact_id = ?", (chat_id,))

    

                        # 2. Run through Unified Pipeline

                        with extraction_semaphore:
                            stats = process_segments(
                                window_segments,
                                chat_id,
                                contact_id=chat_id,
                                extract_facts=not args.skip_facts,
                            )

                        

                        print(f"  ✓ {contact_name}: {len(messages)} messages -> {stats['persisted']} segments, {stats['facts_extracted']} facts")

                        

                        # Track progress

                        last_msg_id = max(m.id for m in messages if m.id)

                        with db.connection() as conn:

                            conn.execute(

                                """UPDATE contacts 

                                   SET last_extracted_rowid = ?,

                                       last_extracted_at = CURRENT_TIMESTAMP

                                   WHERE chat_id = ?""",

                                (last_msg_id, chat_id)

                            )

                        return {"segments": stats['persisted'], "indexed": stats['indexed'], "facts": stats['facts_extracted']}

                

                return {"segments": 0, "indexed": 0, "facts": 0}

    

            except Exception as e:

                print(f"  ✗ {chat_id[:20]} ERROR: {e}")

                return "error"

    

        # Run chats in parallel (2-4 at a time balances CPU vs RAM)

        max_workers = max(1, min(args.workers, 4))
        print(f"Using {max_workers} worker(s) for chat processing")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            results = list(executor.map(process_chat, enumerate(active_chats)))

    

        for res in results:
            if res == "error":
                total_stats["errors"] += 1
            elif isinstance(res, dict):
                total_stats["chats"] += 1
                total_stats["segments"] += res.get("segments", 0)
                total_stats["indexed"] += res.get("indexed", 0)
                total_stats["facts"] += res.get("facts", 0)

    elapsed = time.time() - t0
    
    print(f"\n{'='*50}")
    print(f"Complete backfill finished in {elapsed:.1f}s")
    print(f"  Chats processed: {total_stats['chats']}")
    print(f"  Segments created: {total_stats['segments']}")
    print(f"  Chunks embedded: {total_stats['indexed']}")
    print(f"  Facts extracted: {total_stats['facts']}")
    print(f"  Errors: {total_stats['errors']}")
    print(f"{'='*50}")
    
    print("\nNext steps:")
    print("  - Start watcher: uv run python -m jarvis.watcher")
    print("  - Start API: make api-dev")
    print("  - Semantic search is now available for historical messages")


if __name__ == "__main__":
    main()
