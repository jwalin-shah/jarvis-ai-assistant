"""Migrate contact profiles from JSON files to SQLite.

Reads existing JSON profiles from ~/.jarvis/profiles/ and inserts
them into the contact_profiles table in jarvis.db.

Usage:
    uv run python scripts/migrate_profiles.py [--dry-run]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def migrate_profiles(dry_run: bool = False) -> None:
    profiles_dir = Path.home() / ".jarvis" / "profiles"

    if not profiles_dir.exists():
        print("No profiles directory found, nothing to migrate.", flush=True)
        return

    json_files = list(profiles_dir.glob("*.json"))
    if not json_files:
        print("No JSON profile files found.", flush=True)
        return

    print(f"Found {len(json_files)} profile files to migrate.", flush=True)

    if dry_run:
        for f in json_files:
            data = json.loads(f.read_text())
            cid = data.get("contact_id", "unknown")
            name = data.get("contact_name", "?")
            msgs = data.get("message_count", 0)
            print(f"  [DRY RUN] Would migrate: {cid[:16]}... ({name}, {msgs} msgs)", flush=True)
        return

    from jarvis.contacts.contact_profile import ContactProfile
    from jarvis.db import get_db

    db = get_db()

    migrated = 0
    errors = 0
    for f in json_files:
        try:
            data = json.loads(f.read_text())
            profile = ContactProfile.from_dict(data)
            with db.connection() as conn:
                if profile.save_to_db(conn):
                    migrated += 1
                    print(
                        f"  Migrated: {profile.contact_id[:16]}... "
                        f"({profile.contact_name or '?'}, {profile.message_count} msgs)",
                        flush=True,
                    )
                else:
                    errors += 1
                    print(f"  Failed to save: {f.name}", flush=True)
        except Exception as e:
            errors += 1
            print(f"  Error migrating {f.name}: {e}", flush=True)

    print(f"\nMigration complete: {migrated} migrated, {errors} errors.", flush=True)


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    migrate_profiles(dry_run=dry)
