#!/usr/bin/env python3
"""Complete backfill: semantic segments + embeddings + facts.

This script runs the full V4 pipeline on historical messages:
1. Semantic topic segmentation (creates conversation_segments)
2. Embedding indexing (creates vec_chunks for semantic search)
3. Fact extraction (creates contact_facts)
4. Contact creation (ensures contacts exist in jarvis.db)

Usage:
    uv run python scripts/backfill_complete.py
    uv run python scripts/backfill_complete.py --limit 50    # only top 50 chats
    uv run python scripts/backfill_complete.py --window 100  # messages per chat

    # Resume partial contacts (window-level progress tracking)
    uv run python scripts/backfill_complete.py --resume-partial
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
    parser.add_argument("--offset", type=int, default=0, help="Skip first N chats (for resuming)")
    parser.add_argument("--window", type=int, default=0, help="Messages per chat (0=all)")
    parser.add_argument(
        "--resume-partial",
        action="store_true",
        help="Resume contacts with partial window progress (skips fully completed)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-processing of already completed chats"
    )
    parser.add_argument("--skip-segments", action="store_true", help="Skip segment creation")
    parser.add_argument("--skip-facts", action="store_true", help="Skip fact extraction")
    parser.add_argument(
        "--segment-window-size",
        type=int,
        default=max(1, int(os.getenv("BACKFILL_SEGMENT_WINDOW_SIZE", "25"))),
        help="Messages per extraction window (default: 25)",
    )
    parser.add_argument(
        "--segment-overlap",
        type=int,
        default=max(0, int(os.getenv("BACKFILL_SEGMENT_OVERLAP", "5"))),
        help=(
            "Message overlap between consecutive windows "
            "(default: 5; with size=25 this implies stride=20)"
        ),
    )
    parser.add_argument(
        "--segment-window-stride",
        type=int,
        default=int(os.getenv("BACKFILL_SEGMENT_WINDOW_STRIDE", "0")),
        help=(
            "Optional explicit stride. If <=0, stride is derived as "
            "(segment_window_size - segment_overlap)."
        ),
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

    # Derive effective stride from overlap unless explicitly overridden.
    if args.segment_overlap >= args.segment_window_size:
        args.segment_overlap = max(0, args.segment_window_size - 1)
    if args.segment_window_stride <= 0:
        args.segment_window_stride = max(1, args.segment_window_size - args.segment_overlap)

    from integrations.imessage import ChatDBReader
    from jarvis.db import get_db

    db = get_db()
    db.init_schema()
    print("✓ Database schema initialized")

    # Step 0: Sync contacts from AddressBook
    print("\n[Step 0] Syncing contacts from AddressBook...")
    with ChatDBReader() as reader:
        # Always fetch full active-chat scope for contact sync so small --limit
        # values do not silently reduce the synced contact set.
        convos = reader.get_conversations(limit=1000)
        user_name = reader.get_user_name()

    # Filter for real chats
    active_chats = [
        c
        for c in convos
        if c.message_count >= 5
        and ("iMessage" in c.chat_id or "RCS" in c.chat_id or "SMS" in c.chat_id)
    ]

    def _resolve_display_name(chat_id: str, candidate: str | None) -> str:
        """Prefer explicit candidate; otherwise preserve existing non-generic name."""
        name = (candidate or "").strip()
        if name:
            return name
        existing = db.get_contact_by_chat_id(chat_id)
        if existing and existing.display_name and existing.display_name.lower() != "contact":
            return existing.display_name
        return "Contact"

    contacts_synced = 0
    for conv in active_chats:
        chat_id = conv.chat_id
        display_name = _resolve_display_name(chat_id, conv.display_name)
        phone_or_email = chat_id.split(";")[-1] if ";" in chat_id else chat_id

        try:
            db.add_contact(
                display_name=display_name, chat_id=chat_id, phone_or_email=phone_or_email
            )
            contacts_synced += 1
        except Exception as e:
            print(f"  Warning: Could not sync contact {chat_id}: {e}")

    print(
        f"✓ Synced {contacts_synced} contacts from {len(active_chats)} "
        f"active chats (user: {user_name})"
    )

    if args.offset > 0:
        skipped = min(args.offset, len(active_chats))
        active_chats = active_chats[args.offset :]
        print(
            f"Skipped first {skipped} chats (offset={args.offset}), {len(active_chats)} remaining"
        )

    if args.limit > 0:
        active_chats = active_chats[: args.limit]
        print(f"Limited to {len(active_chats)} chats for processing")

    # Handle --resume-partial: prioritize contacts with partial window progress
    if args.resume_partial:
        from jarvis.db.window_progress import get_contacts_with_partial_progress

        with db.connection() as conn:
            partial_contacts = get_contacts_with_partial_progress(conn, min_windows=1)

        if partial_contacts:
            # Get chat_ids of partial contacts
            partial_chat_ids = {c[0] for c in partial_contacts}

            # Reorder: partial contacts first, then remaining unprocessed
            partial_chats = [c for c in active_chats if c.chat_id in partial_chat_ids]
            other_chats = [c for c in active_chats if c.chat_id not in partial_chat_ids]

            # Filter out completed contacts (have last_extracted_rowid)
            with db.connection() as conn:
                cursor = conn.execute(
                    "SELECT chat_id FROM contacts WHERE last_extracted_rowid IS NOT NULL"
                )
                completed_chat_ids = {row[0] for row in cursor.fetchall()}

            other_chats = [c for c in other_chats if c.chat_id not in completed_chat_ids]
            active_chats = partial_chats + other_chats

            print(
                f"Resume partial: {len(partial_chats)} partial + {len(other_chats)} "
                f"new = {len(active_chats)} to process"
            )
            for pc, pw, pf in partial_contacts[:5]:
                print(f"  - {pc[:50]}: {pw} windows, {pf} facts")
            if len(partial_contacts) > 5:
                print(f"  ... and {len(partial_contacts) - 5} more")

    import threading
    from concurrent.futures import ThreadPoolExecutor

    total_stats = {"chats": 0, "segments": 0, "indexed": 0, "facts": 0, "errors": 0}

    t0 = time.time()
    extraction_semaphore = threading.Semaphore(max(1, args.extract_slots))

    def process_chat(chat_data: tuple[int, Any]) -> dict[str, int] | str | None:
        i, conv = chat_data
        chat_id = conv.chat_id
        contact_name = _resolve_display_name(chat_id, conv.display_name)
        print(f"\n[{i + 1}/{len(active_chats)}] Processing {contact_name} ({chat_id[:50]})...")

        try:
            with ChatDBReader() as reader:
                limit = args.window if args.window > 0 else 1000000  # Practical uncap

                messages = reader.get_messages(chat_id, limit=limit)

                messages.reverse()  # chronological

            if not messages:
                return None

            # Step 1: Create/update contact

            db.add_contact(
                display_name=contact_name,
                chat_id=chat_id,
                phone_or_email=chat_id.split(";")[-1] if ";" in chat_id else chat_id,
            )

            # Step 2: Unified Pipeline (Segment + Embed + Extract Facts)

            if not args.skip_segments or not args.skip_facts:
                from jarvis.contacts.fact_storage import log_pass1_claims, save_facts
                from jarvis.contacts.instruction_extractor import get_instruction_extractor
                from jarvis.topics.segment_pipeline import process_segments
                from jarvis.topics.segment_storage import delete_segments_for_chat
                from jarvis.topics.topic_segmenter import segment_conversation

                # Check if already done (unless --force)

                already_done = False

                if not args.force:
                    with db.connection() as conn:
                        row = conn.execute(
                            "SELECT last_extracted_rowid FROM contacts WHERE chat_id = ?",
                            (chat_id,),
                        ).fetchone()

                        if row and row[0]:
                            already_done = True

                if already_done:
                    print(f"  ⊘ {contact_name}: Already processed, skipping")

                    return {"segments": 0, "indexed": 0, "facts": 0}

                # Ensure messages are sorted chronologically
                messages = sorted(messages, key=lambda m: m.date)

                stats = {"persisted": 0, "indexed": 0}
                msg_to_seg: dict[int, int] = {}

                # 1. Build semantic topic segments for persistence/indexing.
                if not args.skip_segments:
                    segments_to_process = segment_conversation(messages, contact_id=chat_id)

                    if segments_to_process:
                        # Clear existing segments for this chat if re-running
                        with db.connection() as conn:
                            delete_segments_for_chat(conn, chat_id)

                        # 2. Persist/index semantic segments (no fact extraction here).
                        with extraction_semaphore:
                            stats = process_segments(
                                segments_to_process,
                                chat_id,
                                contact_id=chat_id,
                                extract_facts=False,
                            )

                        # Map semantic segment UUIDs to DB row IDs for fact traceability.
                        segment_db_ids: list[int] = []
                        with db.connection() as conn:
                            seg_rows = conn.execute(
                                """
                                SELECT id, segment_id
                                FROM conversation_segments
                                WHERE chat_id = ?
                                """,
                                (chat_id,),
                            ).fetchall()
                        db_id_by_uuid = {row["segment_id"]: row["id"] for row in seg_rows}
                        for seg in segments_to_process:
                            seg_db_id = db_id_by_uuid.get(getattr(seg, "segment_id", None))
                            if seg_db_id is not None:
                                segment_db_ids.append(seg_db_id)

                        # Build source_message_id -> segment_db_id lookup.
                        for seg, seg_db_id in zip(segments_to_process, segment_db_ids):
                            for msg in getattr(seg, "messages", []):
                                msg_id = getattr(msg, "id", None)
                                if msg_id is not None:
                                    msg_to_seg[msg_id] = seg_db_id

                # 3. Extract facts from overlapping fixed windows.
                facts_extracted = 0
                if not args.skip_facts:
                    # Clear existing facts for this chat if re-running
                    if args.force:
                        with db.connection() as conn:
                            conn.execute(
                                "DELETE FROM contact_facts WHERE contact_id = ?", (chat_id,)
                            )

                    from dataclasses import dataclass

                    @dataclass
                    class ExtractionWindow:
                        messages: list[Any]
                        text: str

                    extraction_windows: list[ExtractionWindow] = []
                    for j in range(0, len(messages), args.segment_window_stride):
                        window_msgs = messages[j : j + args.segment_window_size]
                        if len(window_msgs) < 5:
                            break
                        window_text = "\n".join([m.text or "" for m in window_msgs])
                        extraction_windows.append(
                            ExtractionWindow(messages=window_msgs, text=window_text)
                        )

                    if extraction_windows:
                        from jarvis.db.window_progress import (
                            get_completed_windows,
                            init_window_progress_table,
                            record_window_progress,
                        )

                        # Initialize window progress tracking
                        with db.connection() as conn:
                            init_window_progress_table(conn)
                            completed_windows = get_completed_windows(conn, chat_id)

                        extractor = get_instruction_extractor(
                            tier=os.getenv("FACT_EXTRACT_TIER", "0.7b")
                        )
                        total_windows = len(extraction_windows)
                        windows_to_skip = len(completed_windows)

                        # BATCH: Collect all facts first, save once at the end
                        # RAM usage: ~1KB per fact, ~5 facts/window = ~5KB/window
                        # For 127 windows = ~635KB - totally negligible
                        all_facts: list[Any] = []
                        total_raw_claims = 0
                        windows_actually_processed = 0

                        for w_idx, extraction_window in enumerate(extraction_windows, start=1):
                            # Skip already-completed windows (resume capability)
                            if w_idx in completed_windows:
                                continue

                            print(
                                f"    - Extracting facts windows {w_idx}/{total_windows}...",
                                flush=True,
                            )
                            window_results = extractor.extract_facts_from_batch(
                                [extraction_window],
                                contact_id=chat_id,
                                contact_name=contact_name,
                                user_name=user_name,
                            )
                            windows_actually_processed += 1

                            claims_by_segment = extractor.get_last_batch_pass1_claims()
                            facts_in_window = 0
                            if claims_by_segment:
                                total_raw_claims += sum(len(c) for c in claims_by_segment)
                                log_pass1_claims(
                                    contact_id=chat_id,
                                    chat_id=chat_id,
                                    segment_db_ids=[None],
                                    claims_by_segment=claims_by_segment,
                                    stage=f"backfill_window_{w_idx}",
                                )
                            if window_results and window_results[0]:
                                window_facts = window_results[0]
                                facts_in_window = len(window_facts)
                                for fact in window_facts:
                                    msg_id = getattr(fact, "source_message_id", None)
                                    if msg_id is not None and msg_id in msg_to_seg:
                                        setattr(fact, "_segment_db_id", msg_to_seg[msg_id])
                                all_facts.extend(window_facts)

                            # Record progress after each window (for resume capability)
                            with db.connection() as conn:
                                record_window_progress(
                                    conn,
                                    contact_id=chat_id,
                                    window_number=w_idx,
                                    facts_found=facts_in_window,
                                    messages_in_window=len(extraction_window.messages),
                                )

                        if windows_to_skip > 0:
                            print(f"    - Skipped {windows_to_skip} already-processed windows")

                        # Single batch save at the end - one transaction, no lock contention
                        if all_facts:
                            print(
                                f"    - Saving {len(all_facts)} facts in single batch...",
                                flush=True,
                            )
                            facts_extracted = save_facts(
                                all_facts,
                                chat_id,
                                log_raw_facts=True,
                                log_chat_id=chat_id,
                                log_stage="backfill_batched",
                                raw_count=total_raw_claims,
                            )

                print(
                    f"  ✓ {contact_name}: {len(messages)} messages -> "
                    f"{stats['persisted']} segments, {facts_extracted} facts"
                )

                # Track progress

                last_msg_id = max(m.id for m in messages if m.id)

                with db.connection() as conn:
                    conn.execute(
                        """UPDATE contacts

                           SET last_extracted_rowid = ?,

                               last_extracted_at = CURRENT_TIMESTAMP

                           WHERE chat_id = ?""",
                        (last_msg_id, chat_id),
                    )

                return {
                    "segments": stats["persisted"],
                    "indexed": stats["indexed"],
                    "facts": facts_extracted,
                }

            return {"segments": 0, "indexed": 0, "facts": 0}

        except Exception as e:
            print(f"  ✗ {chat_id[:20]} ERROR: {e}")

            return "error"

    # Run chats in parallel (2-4 at a time balances CPU vs RAM)

    max_workers = max(1, min(args.workers, 4))
    print(f"Using {max_workers} worker(s) for chat processing")
    print("Segmentation mode: semantic topic boundaries")
    print(
        f"Fact extraction windows: size={args.segment_window_size}, "
        f"overlap={args.segment_overlap}, stride={args.segment_window_stride}"
    )

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

    print(f"\n{'=' * 50}")
    print(f"Complete backfill finished in {elapsed:.1f}s")
    print(f"  Chats processed: {total_stats['chats']}")
    print(f"  Segments created: {total_stats['segments']}")
    print(f"  Chunks embedded: {total_stats['indexed']}")
    print(f"  Facts extracted: {total_stats['facts']}")
    print(f"  Errors: {total_stats['errors']}")
    print(f"{'=' * 50}")

    print("\nNext steps:")
    print("  - Start watcher: uv run python -m jarvis.watcher")
    print("  - Start API: make api-dev")
    print("  - Semantic search is now available for historical messages")


if __name__ == "__main__":
    main()
