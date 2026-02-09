"""Contact Ingestion - Import contacts from macOS Address Book.

Reads from the per-source AddressBook SQLite databases under
~/Library/Application Support/AddressBook/Sources/, which contain
the actual iCloud-synced contact data (unlike the root-level DB
which is often empty on modern macOS).

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

# Root sources directory containing per-account AddressBook DBs
SOURCES_DIR = Path.home() / "Library/Application Support/AddressBook/Sources"

# Query to get all contacts with phone numbers and emails
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


def _read_all_source_dbs() -> list[dict]:
    """Read contacts from all AddressBook source databases.

    Returns:
        List of row dicts with keys: identifier, first_name, last_name, org_name.
    """
    if not SOURCES_DIR.exists():
        logger.warning("AddressBook Sources directory not found at %s", SOURCES_DIR)
        return []

    all_rows: list[dict] = []
    for source_dir in SOURCES_DIR.iterdir():
        db_path = source_dir / "AddressBook-v22.abcddb"
        if not db_path.exists():
            continue

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(ALL_CONTACTS_QUERY).fetchall()
            finally:
                conn.close()

            for row in rows:
                all_rows.append(
                    {
                        "identifier": row["identifier"],
                        "first_name": row["first_name"],
                        "last_name": row["last_name"],
                        "org_name": row["org_name"],
                    }
                )
        except Exception as e:
            logger.warning("Skipping source %s: %s", source_dir.name, e)

    return all_rows


def ingest_contacts(db: JarvisDB) -> dict[str, int]:
    """Ingest contacts from macOS Address Book source databases into JarvisDB.

    Args:
        db: JarvisDB instance.

    Returns:
        Stats dictionary.
    """
    stats = {
        "processed": 0,
        "updated": 0,
        "created": 0,
        "skipped": 0,
    }

    rows = _read_all_source_dbs()
    if not rows:
        return {"error": "No contacts found in Address Book sources"}

    # Group by name to collect all handles for a person
    people: dict[tuple[str | None, str | None, str | None], set[str]] = {}
    for row in rows:
        normalized = normalize_phone_number(row["identifier"])
        if not normalized:
            continue

        key = (row["first_name"], row["last_name"], row["org_name"])
        if key not in people:
            people[key] = set()
        people[key].add(normalized)

    logger.info("Found %d unique contacts from Address Book sources", len(people))

    for (first, last, org), handles in people.items():
        stats["processed"] += 1

        # Build display name
        if first and last:
            display_name = f"{first} {last}"
        elif first:
            display_name = first
        elif last:
            display_name = last
        elif org:
            display_name = org
        else:
            continue

        handle_list = list(handles)

        # Check if this contact already exists by any handle
        existing_contact = None
        for handle in handle_list:
            existing = db.get_contact_by_handle(handle)
            if existing:
                existing_contact = existing
                break

        if existing_contact:
            # Update if name is just a phone number (not a real name yet)
            is_phone_name = existing_contact.display_name.startswith("+")
            name_changed = is_phone_name or existing_contact.display_name != display_name

            if name_changed:
                db.add_contact(
                    display_name=display_name,
                    chat_id=existing_contact.chat_id,
                    phone_or_email=existing_contact.phone_or_email,
                    relationship=existing_contact.relationship,
                    style_notes=existing_contact.style_notes,
                )
                stats["updated"] += 1
            else:
                stats["skipped"] += 1
        else:
            db.add_contact(
                display_name=display_name,
                phone_or_email=handle_list[0],
            )
            stats["created"] += 1

    return stats
