#!/usr/bin/env python3
"""Sync contacts from macOS AddressBook to jarvis.db.

This ensures contacts exist in jarvis.db with proper display names from AddressBook,
so fact extraction can properly attribute facts and resolve contact names.

Usage:
    uv run python scripts/sync_contacts.py
    uv run python scripts/sync_contacts.py --dry-run  # Preview changes
"""

import argparse
import sys

sys.path.insert(0, ".")


def main():
    parser = argparse.ArgumentParser(description="Sync AddressBook contacts to jarvis.db")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    args = parser.parse_args()

    from integrations.imessage import ChatDBReader
    from jarvis.db import get_db

    db = get_db()
    db.init_schema()

    print("Loading contacts from AddressBook...")

    # Get all conversations from iMessage
    with ChatDBReader() as reader:
        convos = reader.get_conversations(limit=1000)
        user_name = reader.get_user_name()

    print(f"Found {len(convos)} conversations")
    print(f"User name resolved as: {user_name}")

    synced = 0
    skipped = 0
    errors = 0

    for conv in convos:
        chat_id = conv.chat_id
        display_name = conv.display_name or "Contact"

        # Extract phone/email from chat_id
        phone_or_email = chat_id.split(";")[-1] if ";" in chat_id else chat_id

        try:
            # Check if contact already exists
            with db.connection() as conn:
                row = conn.execute(
                    "SELECT id, display_name FROM contacts WHERE chat_id = ?", (chat_id,)
                ).fetchone()

                if row:
                    # Contact exists - check if name needs update
                    existing_name = row["display_name"]
                    if existing_name != display_name and not args.dry_run:
                        conn.execute(
                            """UPDATE contacts 
                               SET display_name = ?, phone_or_email = ?, updated_at = CURRENT_TIMESTAMP
                               WHERE chat_id = ?""",
                            (display_name, phone_or_email, chat_id),
                        )
                        conn.commit()
                        print(f"  Updated: {existing_name} -> {display_name}")
                        synced += 1
                    else:
                        skipped += 1
                else:
                    # New contact
                    if not args.dry_run:
                        conn.execute(
                            """INSERT INTO contacts 
                               (chat_id, display_name, phone_or_email, created_at, updated_at)
                               VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
                            (chat_id, display_name, phone_or_email),
                        )
                        conn.commit()
                        print(f"  Added: {display_name}")
                        synced += 1
                    else:
                        print(f"  Would add: {display_name}")

        except Exception as e:
            print(f"  Error with {chat_id}: {e}")
            errors += 1

    print(f"\n{'=' * 50}")
    print("Contact sync complete:")
    print(f"  Synced/Updated: {synced}")
    print(f"  Skipped (no change): {skipped}")
    print(f"  Errors: {errors}")
    print(f"{'=' * 50}")

    if args.dry_run:
        print("\nThis was a dry run. Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
