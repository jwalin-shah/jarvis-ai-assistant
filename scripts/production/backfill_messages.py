#!/usr/bin/env python3
"""Backfill script to index historical messages for semantic search.

This script iterates through all active chats and indexes all
messages into the 'vec_messages' table for semantic search.
"""

import argparse
import sys
import time

sys.path.insert(0, ".")


def main():
    parser = argparse.ArgumentParser(description="Backfill message search index")
    parser.add_argument("--chats", type=int, default=0, help="Number of chats to process (0=all)")
    parser.add_argument("--messages", type=int, default=0, help="Messages per chat (0=all)")
    args = parser.parse_args()

    import numpy as np

    from integrations.imessage import ChatDBReader
    from jarvis.db import get_db
    from jarvis.search.vec_search import get_vec_searcher

    db = get_db()
    db.init_schema()
    searcher = get_vec_searcher()

    chat_limit = args.chats if args.chats > 0 else 1000
    msg_limit = args.messages if args.messages > 0 else 1000000

    print(f"Starting GPU-accelerated message backfill (Top {chat_limit} chats)...")
    print("Using MLX backend with float16 precision and length-sorted batching.")

    with ChatDBReader() as reader:
        conversations = reader.get_conversations(limit=chat_limit)

        total_indexed = 0
        t0 = time.time()

        for i, conv in enumerate(conversations):
            name = conv.display_name or conv.chat_id
            print(f"[{i + 1}/{len(conversations)}] Indexing: {name}")

            # Get messages for this chat
            messages = reader.get_messages(chat_id=conv.chat_id, limit=msg_limit)
            if not messages:
                continue

            # Filter valid messages here to avoid overhead
            valid_messages = [m for m in messages if m.text and len(m.text.strip()) >= 3]
            if not valid_messages:
                continue

            # Index messages - VecSearcher will use the MLX GPU backend
            count = searcher.index_messages(valid_messages, dtype=np.float16)
            total_indexed += count

        elapsed = time.time() - t0
        print(f"\n{'=' * 50}")
        print(f"Backfill complete in {elapsed:.1f}s")
        print(f"Total messages indexed: {total_indexed}")
        print(f"Speed: {total_indexed / elapsed:.1f} msgs/sec")
        print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
