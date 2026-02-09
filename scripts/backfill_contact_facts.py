"""Backfill contact_facts table from historical iMessage conversations.

Extracts facts (relationships, locations, work, preferences) from existing
messages and populates the knowledge graph database.

Usage:
    uv run python scripts/backfill_contact_facts.py [--max-contacts 50] [--messages-per-contact 500] [--no-nli]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("backfill_facts.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def backfill(
    max_contacts: int = 50,
    messages_per_contact: int = 500,
    use_nli: bool = True,
) -> None:
    """Extract facts from historical messages for top contacts."""
    from integrations.imessage import ChatDBReader
    from jarvis.contacts.fact_extractor import FactExtractor
    from jarvis.contacts.fact_storage import get_fact_count, save_facts
    from jarvis.db import get_db

    # Ensure schema is up to date
    db = get_db()
    db.init_schema()

    print(f"Starting backfill: max_contacts={max_contacts}, "
          f"messages_per_contact={messages_per_contact}, nli={use_nli}", flush=True)

    # Get conversations sorted by message count
    reader = ChatDBReader()
    conversations = reader.get_conversations(limit=max_contacts * 2)
    print(f"Found {len(conversations)} conversations", flush=True)

    extractor = FactExtractor(use_nli=use_nli, entailment_threshold=0.6)

    total_facts = 0
    total_inserted = 0
    start_time = time.time()

    # Sort by participant count (prefer 1:1 chats)
    conversations.sort(key=lambda c: len(c.participants))

    processed = 0
    for conv in conversations:
        if processed >= max_contacts:
            break

        chat_id = conv.chat_id
        if not chat_id:
            continue

        try:
            messages = reader.get_messages(chat_id, limit=messages_per_contact)
            # Only process incoming messages
            incoming = [m for m in messages if not m.is_from_me and m.text]

            if len(incoming) < 3:
                continue

            processed += 1
            elapsed = time.time() - start_time
            eta = (elapsed / processed * (min(max_contacts, len(conversations)) - processed)
                   if processed > 0 else 0)

            print(
                f"[{processed}/{max_contacts}] {chat_id[:30]:30s} "
                f"({len(incoming)} msgs) "
                f"ETA: {eta:.0f}s",
                flush=True,
            )

            facts = extractor.extract_facts(incoming, chat_id)
            total_facts += len(facts)

            if facts:
                inserted = save_facts(facts, chat_id)
                total_inserted += inserted
                if inserted:
                    print(f"  -> {len(facts)} facts extracted, {inserted} new", flush=True)

        except Exception as e:
            logger.warning("Error processing %s: %s", chat_id[:20], e)
            continue

    elapsed = time.time() - start_time
    total_in_db = get_fact_count()
    print(f"\nBackfill complete in {elapsed:.1f}s:", flush=True)
    print(f"  Contacts processed: {processed}", flush=True)
    print(f"  Facts extracted:    {total_facts}", flush=True)
    print(f"  New facts saved:    {total_inserted}", flush=True)
    print(f"  Total facts in DB:  {total_in_db}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill contact facts from iMessage history")
    parser.add_argument("--max-contacts", type=int, default=50, help="Max contacts to process")
    parser.add_argument(
        "--messages-per-contact", type=int, default=500, help="Messages per contact"
    )
    parser.add_argument("--no-nli", action="store_true", help="Skip NLI verification")
    args = parser.parse_args()

    backfill(
        max_contacts=args.max_contacts,
        messages_per_contact=args.messages_per_contact,
        use_nli=not args.no_nli,
    )


if __name__ == "__main__":
    main()
