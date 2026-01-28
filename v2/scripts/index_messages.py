#!/usr/bin/env python3
"""Index all iMessage history for style learning.

This is a one-time setup that indexes your message history so JARVIS can:
1. Learn your texting style (lowercase, emojis, vocabulary)
2. Find your past replies to similar messages
3. Generate responses that match how YOU actually text

Run with: python -m v2.scripts.index_messages

Time estimates:
- 10,000 messages: ~1-2 minutes
- 50,000 messages: ~5-10 minutes
- 100,000 messages: ~15-20 minutes
"""

from __future__ import annotations

import sys


def main():
    print()
    print("=" * 60)
    print("JARVIS v2 - Message Indexer")
    print("=" * 60)
    print()
    print("This will index your iMessage history to learn your style.")
    print("Your data stays local - nothing is sent to the cloud.")
    print()

    # Check for --yes flag to skip confirmation
    if "--yes" not in sys.argv and "-y" not in sys.argv:
        response = input("Continue? [y/N] ").strip().lower()
        if response != "y":
            print("Cancelled.")
            return

    print()

    from v2.core.embeddings.indexer import run_indexing

    stats = run_indexing(verbose=True)

    if stats.messages_indexed > 0:
        print("Your message history is now indexed!")
        print("JARVIS will use your past replies to match your texting style.")
    else:
        print("No new messages to index.")


if __name__ == "__main__":
    main()
