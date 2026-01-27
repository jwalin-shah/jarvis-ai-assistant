"""Contact avatar retrieval from macOS AddressBook database.

Provides functionality to fetch contact photos from the Contacts/AddressBook
database, with support for both phone numbers and email addresses.
"""

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .parser import normalize_phone_number

logger = logging.getLogger(__name__)

# Path to macOS AddressBook database
ADDRESSBOOK_DB_PATH = Path.home() / "Library" / "Application Support" / "AddressBook" / "Sources"

# Database connection timeout
DB_TIMEOUT_SECONDS = 5.0


@dataclass
class ContactAvatarData:
    """Contact avatar data with associated name info."""

    image_data: bytes | None
    first_name: str | None
    last_name: str | None
    display_name: str | None

    @property
    def initials(self) -> str:
        """Generate initials from the contact name.

        Returns:
            1-2 character string of initials, or "?" if no name available
        """
        if self.first_name and self.last_name:
            return f"{self.first_name[0]}{self.last_name[0]}".upper()
        elif self.first_name:
            return self.first_name[0].upper()
        elif self.last_name:
            return self.last_name[0].upper()
        elif self.display_name:
            parts = self.display_name.split()
            if len(parts) >= 2:
                return f"{parts[0][0]}{parts[-1][0]}".upper()
            elif parts:
                return parts[0][0].upper()
        return "?"


def get_contact_avatar(identifier: str) -> ContactAvatarData | None:
    """Get contact avatar data for a phone number or email.

    Queries the macOS AddressBook database for the contact's thumbnail
    image and name information.

    Args:
        identifier: Phone number (e.g., "+15551234567") or email address

    Returns:
        ContactAvatarData if contact found, None otherwise
    """
    if not identifier:
        return None

    # Try to find AddressBook database
    if not ADDRESSBOOK_DB_PATH.exists():
        logger.debug("AddressBook path not found")
        return None

    try:
        # Find the first AddressBook source database
        for source_dir in ADDRESSBOOK_DB_PATH.iterdir():
            ab_db = source_dir / "AddressBook-v22.abcddb"
            if ab_db.exists():
                result = _query_contact_avatar(ab_db, identifier)
                if result:
                    return result
    except PermissionError:
        logger.debug("Permission denied accessing AddressBook database")
    except Exception as e:
        logger.debug(f"Error accessing AddressBook: {e}")

    return None


def _query_contact_avatar(db_path: Path, identifier: str) -> ContactAvatarData | None:
    """Query a specific AddressBook database for contact avatar.

    Args:
        db_path: Path to the AddressBook database
        identifier: Phone number or email address

    Returns:
        ContactAvatarData if found, None otherwise
    """
    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=DB_TIMEOUT_SECONDS)
        try:
            conn.row_factory = sqlite3.Row

            # Check if this looks like a phone number or email
            is_email = "@" in identifier
            normalized = identifier.lower() if is_email else normalize_phone_number(identifier)

            if normalized is None:
                return None

            if is_email:
                return _query_by_email(conn, normalized)
            else:
                return _query_by_phone(conn, normalized)

        finally:
            conn.close()

    except (sqlite3.Error, OSError) as e:
        logger.debug(f"Error querying contact avatar: {e}")
        return None


def _query_by_phone(conn: sqlite3.Connection, phone: str) -> ContactAvatarData | None:
    """Query contact by phone number.

    Args:
        conn: Database connection
        phone: Normalized phone number

    Returns:
        ContactAvatarData if found, None otherwise
    """
    cursor = conn.cursor()

    try:
        # Query all phone contacts and match normalized phone numbers
        cursor.execute(
            """
            SELECT
                ZABCDRECORD.ZTHUMBNAILIMAGEDATA as image_data,
                ZABCDRECORD.ZFIRSTNAME as first_name,
                ZABCDRECORD.ZLASTNAME as last_name,
                ZABCDRECORD.ZDISPLAYNAME as display_name,
                ZABCDPHONENUMBER.ZFULLNUMBER as phone_number
            FROM ZABCDPHONENUMBER
            JOIN ZABCDRECORD ON ZABCDPHONENUMBER.ZOWNER = ZABCDRECORD.Z_PK
            WHERE ZABCDPHONENUMBER.ZFULLNUMBER IS NOT NULL
            """,
        )

        for row in cursor.fetchall():
            row_phone = normalize_phone_number(row["phone_number"])
            if row_phone == phone:
                return ContactAvatarData(
                    image_data=row["image_data"],
                    first_name=row["first_name"],
                    last_name=row["last_name"],
                    display_name=row["display_name"],
                )

        return None

    except sqlite3.OperationalError as e:
        logger.debug(f"Phone query error: {e}")
        return None
    finally:
        cursor.close()


def _query_by_email(conn: sqlite3.Connection, email: str) -> ContactAvatarData | None:
    """Query contact by email address.

    Args:
        conn: Database connection
        email: Lowercase email address

    Returns:
        ContactAvatarData if found, None otherwise
    """
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT
                ZABCDRECORD.ZTHUMBNAILIMAGEDATA as image_data,
                ZABCDRECORD.ZFIRSTNAME as first_name,
                ZABCDRECORD.ZLASTNAME as last_name,
                ZABCDRECORD.ZDISPLAYNAME as display_name
            FROM ZABCDEMAILADDRESS
            JOIN ZABCDRECORD ON ZABCDEMAILADDRESS.ZOWNER = ZABCDRECORD.Z_PK
            WHERE LOWER(ZABCDEMAILADDRESS.ZADDRESS) = ?
            """,
            (email,),
        )

        row = cursor.fetchone()
        if row:
            return ContactAvatarData(
                image_data=row["image_data"],
                first_name=row["first_name"],
                last_name=row["last_name"],
                display_name=row["display_name"],
            )

        return None

    except sqlite3.OperationalError as e:
        logger.debug(f"Email query error: {e}")
        return None
    finally:
        cursor.close()
