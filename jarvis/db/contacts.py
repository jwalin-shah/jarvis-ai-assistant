"""Contact CRUD operations mixin."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING, Any

from jarvis.db.models import Contact

if TYPE_CHECKING:
    from jarvis.db.core import JarvisDBBase


class ContactMixin:
    """Mixin providing contact CRUD operations."""

    # Type hints for attributes provided by JarvisDBBase
    _contact_cache: Any
    _stats_cache: Any
    _trigger_pattern_cache: Any

    def add_contact(
        self: JarvisDBBase,
        display_name: str,
        chat_id: str | None = None,
        phone_or_email: str | None = None,
        relationship: str | None = None,
        style_notes: str | None = None,
        handles: list[str] | None = None,
    ) -> Contact:
        """Add or update a contact.

        If a contact with the same chat_id exists, it will be updated.

        Args:
            display_name: Contact's display name.
            chat_id: Optional chat ID from iMessage.
            phone_or_email: Phone number or email.
            relationship: Relationship type (e.g., 'sister', 'boss').
            style_notes: Communication style notes.
            handles: List of all handles (phones/emails) for this person.

        Returns:
            The created or updated Contact.
        """
        with self.connection() as conn:
            now = datetime.now()
            handles_json = json.dumps(handles) if handles else None

            if chat_id:
                # Check for existing contact
                cursor = conn.execute("SELECT id FROM contacts WHERE chat_id = ?", (chat_id,))
                existing = cursor.fetchone()

                if existing:
                    # Update existing
                    conn.execute(
                        """
                        UPDATE contacts
                        SET display_name = ?, phone_or_email = ?, relationship = ?,
                            style_notes = ?, handles_json = ?, updated_at = ?
                        WHERE chat_id = ?
                        """,
                        (
                            display_name,
                            phone_or_email,
                            relationship,
                            style_notes,
                            handles_json,
                            now,
                            chat_id,
                        ),
                    )
                    # Invalidate cache entries for this contact
                    self._contact_cache.invalidate(("contact_id", existing["id"]))
                    self._contact_cache.invalidate(("chat_id", chat_id))
                    return Contact(
                        id=existing["id"],
                        chat_id=chat_id,
                        display_name=display_name,
                        phone_or_email=phone_or_email,
                        relationship=relationship,
                        style_notes=style_notes,
                        handles_json=handles_json,
                        updated_at=now,
                    )

            # Insert new contact
            cursor = conn.execute(
                """
                INSERT INTO contacts
                (chat_id, display_name, phone_or_email, relationship, style_notes, handles_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (chat_id, display_name, phone_or_email, relationship, style_notes, handles_json),
            )

            new_contact = Contact(
                id=cursor.lastrowid,
                chat_id=chat_id,
                display_name=display_name,
                phone_or_email=phone_or_email,
                relationship=relationship,
                style_notes=style_notes,
                handles_json=handles_json,
                created_at=now,
                updated_at=now,
            )
            # Invalidate stats cache since we added a new contact
            self._stats_cache.invalidate("db_stats")
            return new_contact

    def get_contact(self: JarvisDBBase, contact_id: int) -> Contact | None:
        """Get a contact by ID.

        Results are cached with 30-second TTL for performance.
        """
        cache_key = ("contact_id", contact_id)
        hit, cached = self._contact_cache.get(cache_key)
        if hit:
            return cached

        with self.connection() as conn:
            cursor = conn.execute(
                """SELECT id, chat_id, display_name, phone_or_email, relationship,
                          style_notes, handles_json, created_at, updated_at
                   FROM contacts WHERE id = ?""",
                (contact_id,),
            )
            row = cursor.fetchone()
            result = self._row_to_contact(row) if row else None
            self._contact_cache.set(cache_key, result)
            return result

    def get_contact_by_chat_id(self: JarvisDBBase, chat_id: str) -> Contact | None:
        """Get a contact by their chat ID.

        Handles iMessage-format chat_ids (e.g., "iMessage;-;+15551234567")
        by extracting the identifier and matching against stored chat_ids.

        Results are cached with 30-second TTL for performance.
        """
        cache_key = ("chat_id", chat_id)
        hit, cached = self._contact_cache.get(cache_key)
        if hit:
            return cached

        with self.connection() as conn:
            # Try exact match first
            cursor = conn.execute(
                """SELECT id, chat_id, display_name, phone_or_email, relationship,
                          style_notes, handles_json, created_at, updated_at
                   FROM contacts WHERE chat_id = ?""",
                (chat_id,),
            )
            row = cursor.fetchone()

            # If no match and chat_id has iMessage/SMS prefix, extract the identifier
            if row is None and ";" in chat_id:
                identifier = chat_id.rsplit(";", 1)[-1]
                if identifier:
                    cursor = conn.execute(
                        """SELECT id, chat_id, display_name, phone_or_email, relationship,
                                  style_notes, handles_json, created_at, updated_at
                           FROM contacts WHERE chat_id = ?""",
                        (identifier,),
                    )
                    row = cursor.fetchone()

            result = self._row_to_contact(row) if row else None
            self._contact_cache.set(cache_key, result)
            return result

    def get_contact_by_handle(self: JarvisDBBase, handle: str) -> Contact | None:
        """Get a contact by phone number or email handle.

        Checks chat_id and phone_or_email fields.
        """
        with self.connection() as conn:
            cursor = conn.execute(
                """SELECT id, chat_id, display_name, phone_or_email, relationship,
                          style_notes, handles_json, created_at, updated_at
                   FROM contacts WHERE chat_id = ? OR phone_or_email = ?""",
                (handle, handle),
            )
            row = cursor.fetchone()
            return self._row_to_contact(row) if row else None

    def get_contact_by_name(self: JarvisDBBase, name: str) -> Contact | None:
        """Get a contact by display name (case-insensitive partial match)."""
        with self.connection() as conn:
            # Try exact match first
            cursor = conn.execute(
                """SELECT id, chat_id, display_name, phone_or_email, relationship,
                          style_notes, handles_json, created_at, updated_at
                   FROM contacts WHERE LOWER(display_name) = LOWER(?)""",
                (name,),
            )
            row = cursor.fetchone()

            if not row:
                # Try partial match
                cursor = conn.execute(
                    """SELECT id, chat_id, display_name, phone_or_email, relationship,
                              style_notes, handles_json, created_at, updated_at
                       FROM contacts WHERE LOWER(display_name) LIKE LOWER(?)""",
                    (f"%{name}%",),
                )
                row = cursor.fetchone()

            if row:
                return self._row_to_contact(row)
            return None

    def list_contacts(self: JarvisDBBase, limit: int = 100) -> list[Contact]:
        """List all contacts."""
        with self.connection() as conn:
            cursor = conn.execute(
                """SELECT id, chat_id, display_name, phone_or_email, relationship,
                          style_notes, handles_json, created_at, updated_at
                   FROM contacts ORDER BY display_name LIMIT ?""",
                (limit,),
            )
            return [self._row_to_contact(row) for row in cursor]

    def delete_contact(self: JarvisDBBase, contact_id: int) -> bool:
        """Delete a contact and their associated pairs."""
        with self.connection() as conn:
            # Get chat_id before deletion for cache invalidation
            cursor = conn.execute("SELECT chat_id FROM contacts WHERE id = ?", (contact_id,))
            row = cursor.fetchone()
            chat_id = row["chat_id"] if row else None

            # Delete embeddings for this contact's pairs
            conn.execute(
                """DELETE FROM pair_embeddings WHERE pair_id IN
                   (SELECT id FROM pairs WHERE contact_id = ?)""",
                (contact_id,),
            )
            # Delete pairs
            conn.execute("DELETE FROM pairs WHERE contact_id = ?", (contact_id,))
            cursor = conn.execute("DELETE FROM contacts WHERE id = ?", (contact_id,))
            deleted = cursor.rowcount > 0

            if deleted:
                # Invalidate caches
                self._contact_cache.invalidate(("contact_id", contact_id))
                if chat_id:
                    self._contact_cache.invalidate(("chat_id", chat_id))
                self._stats_cache.invalidate("db_stats")
                # Clear trigger pattern cache for this contact
                self._trigger_pattern_cache.clear()  # Simpler than tracking individual keys

            return deleted

    @staticmethod
    def _row_to_contact(row: sqlite3.Row) -> Contact:
        """Convert a database row to a Contact object."""
        return Contact(
            id=row["id"],
            chat_id=row["chat_id"],
            display_name=row["display_name"],
            phone_or_email=row["phone_or_email"],
            relationship=row["relationship"],
            style_notes=row["style_notes"],
            handles_json=row["handles_json"] if "handles_json" in row.keys() else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
