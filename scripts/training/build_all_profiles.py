#!/usr/bin/env python3
"""Build contact profiles for all contacts with extracted facts.

This script:
1. Loads all contacts from the database
2. Builds enriched profiles using ContactProfileBuilder
3. Saves profiles to ~/.jarvis/profiles/
4. Reports statistics on profile completeness

Usage:
    uv run python scripts/build_all_profiles.py
    uv run python scripts/build_all_profiles.py --min-messages 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from collections import Counter

from integrations.imessage import ChatDBReader
from jarvis.contacts.contact_profile import ContactProfileBuilder, save_profile
from jarvis.contacts.fact_storage import count_facts_for_contact
from jarvis.db import get_db


def main():
    parser = argparse.ArgumentParser(description="Build profiles for all contacts")
    parser.add_argument("--min-messages", type=int, default=5, help="Minimum messages for profile")
    parser.add_argument("--limit", type=int, default=0, help="Max profiles to build (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Don't save, just report")
    args = parser.parse_args()

    db = get_db()
    db.init_schema()
    reader = ChatDBReader()
    builder = ContactProfileBuilder(min_messages=args.min_messages)

    # Get all contacts with chat_ids
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT chat_id, display_name, relationship FROM contacts WHERE chat_id IS NOT NULL"
        ).fetchall()

    print(f"Database: {db.db_path}")
    print(f"Found {len(rows)} contacts in database")
    print(f"Building profiles with min_messages={args.min_messages}")
    print("=" * 60)

    if not rows:
        print("No contacts found. Run `uv run python scripts/sync_contacts.py` first.")

    stats = {
        "processed": 0,
        "saved": 0,
        "skipped": 0,
        "errors": 0,
        "facts_total": 0,
        "relationships": Counter(),
    }

    t0 = time.time()

    targets = rows[: args.limit] if args.limit > 0 else rows

    for i, row in enumerate(targets):
        chat_id = row["chat_id"]
        display_name = row["display_name"] or "Contact"

        try:
            # Check if has facts
            fact_count = count_facts_for_contact(chat_id)
            stats["facts_total"] += fact_count

            # Get messages
            messages = reader.get_messages(chat_id, limit=300)
            if len(messages) < args.min_messages:
                stats["skipped"] += 1
                continue

            # Build profile
            profile = builder.build_profile(chat_id, messages, contact_name=display_name)

            # Track relationship
            stats["relationships"][profile.relationship] += 1

            # Save if not dry run
            if not args.dry_run:
                save_profile(profile)

                # Persist relationship back to SQLite database for visibility in UI/Bakeoff
                with db.connection() as conn:
                    conn.execute(
                        "UPDATE contacts SET relationship = ?, relationship_reasoning = ? WHERE chat_id = ?",
                        (profile.relationship, profile.relationship_reasoning, chat_id),
                    )
                stats["saved"] += 1
            else:
                stats["saved"] += 1  # Count as saved for dry-run reporting

            stats["processed"] += 1

            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"  ... processed {i + 1}/{len(targets)}")

        except Exception as e:
            print(f"  ✗ Error processing {chat_id[:30]}: {e}")
            stats["errors"] += 1

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print(f"Profile building complete in {elapsed:.1f}s")
    print(f"  Processed: {stats['processed']}")
    print(f"  Saved:     {stats['saved']}")
    print(f"  Skipped:   {stats['skipped']} (too few messages)")
    print(f"  Errors:    {stats['errors']}")
    print(f"  Total facts linked: {stats['facts_total']}")
    print("\nRelationship distribution:")
    for rel, count in stats["relationships"].most_common():
        print(f"  • {rel}: {count}")

    print("\nNext steps:")
    print("  - Start API: make api-dev")
    print("  - View profiles: GET /graph/contact/{contact_id}")
    print("  - View network:  GET /graph/network")


if __name__ == "__main__":
    main()
