#!/usr/bin/env python3
"""Segmentation-only backfill using CPU embedder.

This runs in parallel to the main backfill (which uses GPU for LLM).
Uses CPU embedder so there's no GPU contention.

Usage:
    uv run python scripts/segment_only_backfill.py --limit 100

    # Run in parallel with fact backfill:
    # Terminal 1: uv run python scripts/backfill_complete.py --skip-segments
    # Terminal 2: uv run python scripts/segment_only_backfill.py
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")


def main():
    parser = argparse.ArgumentParser(description="Segmentation-only backfill (CPU)")
    parser.add_argument("--limit", type=int, default=0, help="Max chats to process")
    parser.add_argument("--force", action="store_true", help="Reprocess already done chats")
    parser.add_argument("--use-cpu", action="store_true", default=True, help="Use CPU embedder")
    parser.add_argument(
        "--drift-threshold", type=float, default=0.55, help="Segmentation threshold"
    )
    args = parser.parse_args()

    # Check CPU embedder availability
    if args.use_cpu:
        from jarvis.embedding import is_cpu_embedder_available

        if not is_cpu_embedder_available():
            print("✗ CPU embedder not available (onnxruntime not installed)")
            print("  Install: uv pip install onnxruntime")
            print("  Or export model: optimum-cli export onnx ...")
            return 1

        print("✓ CPU embedder available")
        print("  This will run in parallel to GPU fact extraction!")
    else:
        print("⚠ Using MLX embedder (will contend with GPU LLM)")

    from integrations.imessage import ChatDBReader
    from jarvis.db import get_db
    from jarvis.topics.segment_pipeline import process_segments
    from jarvis.topics.topic_segmenter import segment_conversation

    db = get_db()
    db.init_schema()

    # Get chats
    with ChatDBReader() as reader:
        convos = reader.get_conversations(limit=1000)

    active_chats = [
        c
        for c in convos
        if c.message_count >= 5
        and ("iMessage" in c.chat_id or "RCS" in c.chat_id or "SMS" in c.chat_id)
    ]

    if args.limit:
        active_chats = active_chats[: args.limit]

    print(f"\nProcessing {len(active_chats)} chats for segmentation...")
    print(f"Drift threshold: {args.drift_threshold}")

    # Import embedder
    if args.use_cpu:
        from jarvis.embedding import get_parallel_embedder

        embedder = get_parallel_embedder()
        print(f"Using CPU embedder: {type(embedder).__name__}")
    else:
        from jarvis.embedding_adapter import get_embedder

        embedder = get_embedder()
        print(f"Using MLX embedder: {type(embedder).__name__}")

    processed = 0
    skipped = 0
    errors = 0

    for i, conv in enumerate(active_chats, 1):
        chat_id = conv.chat_id
        print(f"\n[{i}/{len(active_chats)}] {chat_id[:40]}...")

        try:
            # Check if already done
            if not args.force:
                with db.connection() as conn:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM conversation_segments WHERE chat_id = ?",
                        (chat_id,),
                    ).fetchone()
                    if row and row[0] > 0:
                        print(f"  ⊘ Already segmented ({row[0]} segments), skipping")
                        skipped += 1
                        continue

            # Load messages
            with ChatDBReader() as reader:
                messages = reader.get_messages(chat_id, limit=10000)
                messages.reverse()  # Chronological

            if len(messages) < 5:
                print(f"  ⊘ Too few messages ({len(messages)}), skipping")
                skipped += 1
                continue

            # Segment
            print(f"  Segmenting {len(messages)} messages...")
            segments = segment_conversation(
                messages,
                contact_id=chat_id,
                drift_threshold=args.drift_threshold,
            )

            if not segments:
                print("  ⚠ No segments created")
                skipped += 1
                continue

            # Process (persist + index) - skip fact extraction
            print(f"  Persisting {len(segments)} segments...")
            stats = process_segments(
                segments,
                chat_id=chat_id,
                contact_id=chat_id,
                extract_facts=False,  # Don't extract facts here
            )

            persisted = stats.get("persisted", 0)
            indexed = stats.get("indexed", 0)

            print(f"  ✓ Segments: {persisted}, Indexed: {indexed}")
            processed += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            errors += 1
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Segmentation Backfill Complete")
    print("=" * 60)
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Errors:    {errors}")

    # Cleanup
    if args.use_cpu:
        from jarvis.embedding import CPUEmbedder

        CPUEmbedder.get_instance().unload()

    return 0


if __name__ == "__main__":
    sys.exit(main())
