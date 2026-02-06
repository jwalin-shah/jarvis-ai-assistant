"""Contact Ingestion - Import contacts from macOS Address Book.

Reads from the local AddressBook SQLite database and populates the
JarvisDB contacts table. This enables:
1. Resolving names for phone numbers/emails
2. Linking multiple handles to the same person
3. Pre-populating contacts for profile building

Usage:
    from jarvis.search.ingest import ingest_contacts
    from jarvis.db import get_db
    ingest_contacts(get_db())
"""

import logging
import sqlite3
from pathlib import Path

from integrations.imessage.parser import normalize_phone_number
from jarvis.db import JarvisDB

logger = logging.getLogger(__name__)

# Address Book path
ADDRESS_BOOK_PATH = Path.home() / "Library/Application Support/AddressBook/AddressBook-v22.abcddb"

# Query to get all contacts (same as in desktop/src/lib/db/queries.ts)
ALL_CONTACTS_QUERY = """
  SELECT
    p.ZFULLNUMBER as identifier,
    c.ZFIRSTNAME as first_name,
    c.ZLASTNAME as last_name,
    c.ZORGANIZATION as org_name
  FROM ZABCDPHONENUMBER p
  JOIN ZABCDRECORD c ON p.ZOWNER = c.Z_PK
  UNION ALL
  SELECT
    e.ZADDRESS as identifier,
    c.ZFIRSTNAME as first_name,
    c.ZLASTNAME as last_name,
    c.ZORGANIZATION as org_name
  FROM ZABCDEMAILADDRESS e
  JOIN ZABCDRECORD c ON e.ZOWNER = c.Z_PK
"""

# Re-export for backward compatibility
__all__ = ["normalize_phone_number", "ingest_contacts"]


def ingest_contacts(db: JarvisDB) -> dict[str, int]:
    """Ingest contacts from Address Book into JarvisDB.

    Args:
        db: JarvisDB instance.

    Returns:
        Stats dictionary.
    """
    if not ADDRESS_BOOK_PATH.exists():
        logger.warning(f"Address Book database not found at {ADDRESS_BOOK_PATH}")
        return {"error": "Address Book not found"}

    stats = {
        "processed": 0,
        "updated": 0,
        "created": 0,
        "skipped": 0,
    }

    try:
        # Open Address Book DB (read-only)
        conn = sqlite3.connect(f"file:{ADDRESS_BOOK_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(ALL_CONTACTS_QUERY)
        rows = cursor.fetchall()
        conn.close()

        # Group by name to collect all handles for a person
        # Key: (first_name, last_name, org_name) -> set(handles)
        people: dict[tuple[str | None, str | None, str | None], set[str]] = {}

        for row in rows:
            identifier = row["identifier"]
            normalized = normalize_phone_number(identifier)
            if not normalized:
                continue

            key = (row["first_name"], row["last_name"], row["org_name"])
            if key not in people:
                people[key] = set()
            people[key].add(normalized)

        logger.info(f"Found {len(people)} unique contacts in Address Book")

        # Process each person
        for (first, last, org), handles in people.items():
            stats["processed"] += 1

            # Construct display name
            display_name = ""
            if first and last:
                display_name = f"{first} {last}"
            elif first:
                display_name = first
            elif last:
                display_name = last
            elif org:
                display_name = org
            else:
                continue  # Skip empty names

            handle_list = list(handles)
            primary_handle = handle_list[0] if handle_list else None

            # Check if this contact already exists in JarvisDB
            # We check by any of their handles
            existing_contact = None
            for handle in handle_list:
                existing = db.get_contact_by_handle(handle)
                if existing:
                    existing_contact = existing
                    break

            if existing_contact:
                # Update existing contact
                # Merge handles
                current_handles = set(existing_contact.handles)
                new_handles = current_handles.union(handles)

                # Only update if changed
                if existing_contact.display_name != display_name or len(new_handles) > len(
                    current_handles
                ):
                    db.add_contact(
                        display_name=display_name,
                        chat_id=existing_contact.chat_id,
                        phone_or_email=existing_contact.phone_or_email,  # Keep existing primary
                        relationship=existing_contact.relationship,
                        style_notes=existing_contact.style_notes,
                        handles=list(new_handles),
                    )
                    stats["updated"] += 1
                else:
                    stats["skipped"] += 1
            else:
                # Create new contact
                # We don't have a chat_id (GUID) yet, so we use the primary handle
                # This ensures extract.py can find it if the chat_id matches the handle
                # (which is true for SMS and many 1:1 chats)
                db.add_contact(
                    display_name=display_name,
                    chat_id=None,  # Let DB gen ID, but extract.py needs to find it.
                    # Actually, extract.py uses get_contact_by_chat_id.
                    # If we set chat_id=primary_handle, it might match some chats.
                    # But chat.db GUIDs often have prefixes.
                    # Ideally we leave chat_id null and let a linker script fix it later.
                    # For now, we populate phone_or_email and handles.
                    phone_or_email=primary_handle,
                    handles=handle_list,
                )
                stats["created"] += 1

    except Exception as e:
        logger.error(f"Error ingesting contacts: {e}")
        return {"error": str(e)}

    return stats
