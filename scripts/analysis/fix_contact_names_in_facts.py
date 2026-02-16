#!/usr/bin/env python3
"""Fix 'Contact' fallback names in existing facts.

This script:
1. Finds all facts with subject='Contact'
2. Resolves the proper display name from the contacts table
3. Updates the facts with proper names

Usage:
    uv run python scripts/fix_contact_names_in_facts.py
    uv run python scripts/fix_contact_names_in_facts.py --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis.contacts.fact_storage import get_all_facts
from jarvis.db import get_db


def resolve_contact_name(contact_id: str) -> str | None:
    """Get display name for a contact from the database."""
    from jarvis.db import get_db

    db = get_db()

    with db.connection() as conn:
        # Try exact match
        row = conn.execute(
            "SELECT display_name FROM contacts WHERE chat_id = ?", (contact_id,)
        ).fetchone()
        if row and row[0] and row[0] not in ["Contact", "None", "Unknown", None]:
            return row[0]

        # Try partial match on phone/email
        clean_id = contact_id.split(";")[-1] if ";" in contact_id else contact_id
        row = conn.execute(
            "SELECT display_name FROM contacts WHERE phone_or_email LIKE ? OR chat_id LIKE ?",
            (f"%{clean_id}%", f"%{clean_id}%"),
        ).fetchone()
        if row and row[0] and row[0] not in ["Contact", "None", "Unknown", None]:
            return row[0]

    return None


def main():
    parser = argparse.ArgumentParser(description="Fix 'Contact' names in facts")
    parser.add_argument("--dry-run", action="store_true", help="Don't modify DB")
    args = parser.parse_args()

    print("=" * 60)
    print("FIXING CONTACT NAMES IN FACTS")
    print("=" * 60)

    # Load all facts
    print("\n1. Loading facts...")
    facts = get_all_facts()
    print(f"   Total facts: {len(facts)}")

    # Find facts with 'Contact' subject
    contact_facts = [f for f in facts if f.subject == "Contact" or f.subject in ["them", "they"]]
    print(f"   Facts with 'Contact' subject: {len(contact_facts)}")

    if not contact_facts:
        print("   ✓ No facts to fix!")
        return

    # Group by contact_id
    print("\n2. Resolving proper names...")
    by_contact = {}
    for f in contact_facts:
        by_contact.setdefault(f.contact_id, []).append(f)

    fixes = []
    for contact_id, facts_list in by_contact.items():
        proper_name = resolve_contact_name(contact_id)
        if proper_name and proper_name != "Contact":
            fixes.append((contact_id, proper_name, len(facts_list)))
            # Update facts
            for f in facts_list:
                f.subject = proper_name

    print(f"   Can fix {len(fixes)} contacts:")
    for contact_id, name, count in fixes[:10]:
        print(f"     • {name} ({contact_id[:40]}...): {count} facts")
    if len(fixes) > 10:
        print(f"     ... and {len(fixes) - 10} more")

    if args.dry_run:
        print("\n⚠️  DRY RUN - No changes made")
        return

    # Apply fixes
    print("\n3. Applying fixes...")

    # We need to delete and re-insert because subject is part of unique constraint
    fixed_count = 0
    db = get_db()

    with db.connection() as conn:
        for contact_id, proper_name, count in fixes:
            # Get all facts for this contact
            contact_all_facts = [f for f in facts if f.contact_id == contact_id]

            # Delete old facts
            conn.execute("DELETE FROM contact_facts WHERE contact_id = ?", (contact_id,))

            # Re-insert all with updated names
            for f in contact_all_facts:
                # Update subject if it was 'Contact'
                if f.subject in ["Contact", "them", "they"]:
                    f.subject = proper_name

            # Batch insert
            fact_data = [
                (
                    f.contact_id,
                    f.category,
                    f.subject,
                    f.predicate,
                    f.value,
                    f.confidence,
                    f.source_message_id,
                    f.source_text[:500] if f.source_text else "",
                    f.extracted_at or "",
                    None,  # linked_contact_id
                    None,  # valid_from
                    None,  # valid_until
                    f.attribution,
                    None,  # segment_id
                )
                for f in contact_all_facts
            ]

            conn.executemany(
                """
                INSERT OR IGNORE INTO contact_facts
                (contact_id, category, subject, predicate, value, confidence,
                 source_message_id, source_text, extracted_at, linked_contact_id,
                 valid_from, valid_until, attribution, segment_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                fact_data,
            )
            fixed_count += count

        conn.commit()

    print(f"   ✅ Fixed {fixed_count} facts across {len(fixes)} contacts")

    # Verify
    print("\n4. Verification:")
    final_facts = get_all_facts()
    remaining_contact = sum(1 for f in final_facts if f.subject == "Contact")
    print(f"   Facts remaining with 'Contact' subject: {remaining_contact}")


if __name__ == "__main__":
    main()
