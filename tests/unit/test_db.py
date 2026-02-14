"""Unit tests for active JARVIS Database Management (JarvisDB).

Covers database initialization, schema versioning, and contact CRUD operations.
Legacy pair/cluster operations are tested in the archived test suite.
"""

from datetime import datetime
from pathlib import Path

import pytest

from jarvis.db import (
    CURRENT_SCHEMA_VERSION,
    Contact,
    JarvisDB,
    get_db,
    reset_db,
)


class TestDatabaseInitialization:
    """Tests for database initialization and schema creation."""

    def test_init_creates_directory(self, tmp_path: Path) -> None:
        """Test that JarvisDB creates parent directory if it doesn't exist."""
        db_path = tmp_path / "nested" / "dir" / "jarvis.db"
        db = JarvisDB(db_path)

        assert db.db_path == db_path
        assert db_path.parent.exists()

    def test_exists_returns_false_when_no_file(self, tmp_path: Path) -> None:
        """Test exists() returns False when database file doesn't exist."""
        db_path = tmp_path / "nonexistent.db"
        db = JarvisDB(db_path)

        assert db.exists() is False

    def test_exists_returns_true_after_init_schema(self, tmp_path: Path) -> None:
        """Test exists() returns True after schema initialization."""
        db_path = tmp_path / "jarvis.db"
        db = JarvisDB(db_path)
        db.init_schema()

        assert db.exists() is True

    def test_init_schema_creates_tables(self, tmp_path: Path) -> None:
        """Test that init_schema creates all required tables."""
        db_path = tmp_path / "jarvis.db"
        db = JarvisDB(db_path)
        db.init_schema()

        with db.connection() as conn:
            # Check all expected tables exist
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = {row["name"] for row in cursor}

            expected_tables = {
                "schema_version",
                "contacts",
                "contact_style_targets",
                "conversation_segments",
                "segment_messages",
                "index_versions",
            }
            assert expected_tables.issubset(tables)

    def test_init_schema_sets_version(self, tmp_path: Path) -> None:
        """Test that init_schema sets the correct schema version."""
        db_path = tmp_path / "jarvis.db"
        db = JarvisDB(db_path)
        db.init_schema()

        with db.connection() as conn:
            cursor = conn.execute("SELECT version FROM schema_version")
            row = cursor.fetchone()
            assert row["version"] == CURRENT_SCHEMA_VERSION


class TestContactOperations:
    """Tests for contact CRUD operations."""

    @pytest.fixture
    def db(self, tmp_path: Path) -> JarvisDB:
        """Provide a fresh, initialized database."""
        db_path = tmp_path / "jarvis.db"
        db = JarvisDB(db_path)
        db.init_schema()
        return db

    def test_add_and_get_contact(self, db: JarvisDB) -> None:
        """Test adding and retrieving a contact."""
        contact = db.add_contact(
            display_name="John Doe",
            chat_id="chat123",
            relationship="friend",
            style_notes="uses emojis",
        )

        assert contact is not None
        assert contact.id is not None
        assert contact.display_name == "John Doe"
        assert contact.chat_id == "chat123"
        assert contact.relationship == "friend"
        assert contact.style_notes == "uses emojis"

        # Retrieve by contact_id
        retrieved = db.get_contact(contact.id)
        assert retrieved is not None
        assert retrieved.display_name == "John Doe"

        # Retrieve by chat_id
        by_chat = db.get_contact_by_chat_id("chat123")
        assert by_chat is not None
        assert by_chat.id == contact.id

    def test_list_contacts(self, db: JarvisDB) -> None:
        """Test listing all contacts."""
        db.add_contact(display_name="Alice", chat_id="chat1")
        db.add_contact(display_name="Bob", chat_id="chat2")

        contacts = db.list_contacts()
        assert len(contacts) == 2
        names = {c.display_name for c in contacts}
        assert names == {"Alice", "Bob"}
