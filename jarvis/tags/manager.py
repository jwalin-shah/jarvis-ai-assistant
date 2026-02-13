"""Tag manager for CRUD operations on tags, smart folders, and tag rules.

Provides a complete interface for managing the tagging system including:
- Tag CRUD with hierarchical support
- Conversation tagging with bulk operations
- Smart folder management
- Auto-tagging rule management
- Tag suggestions based on user behavior

Usage:
    from jarvis.tags.manager import TagManager

    manager = TagManager()
    manager.init_schema()

    tag = manager.create_tag("Work", color="#0066cc")
    manager.add_tag_to_conversation("chat123", tag.id)
    tags = manager.get_tags_for_conversation("chat123")
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jarvis.cache import TTLCache
from jarvis.tags.models import (
    ConversationTag,
    SmartFolder,
    SmartFolderRules,
    Tag,
    TagColor,
    TagIcon,
    TagRule,
)

logger = logging.getLogger(__name__)

# Default database path (same directory as main jarvis.db)
TAGS_DB_PATH = Path.home() / ".jarvis" / "jarvis.db"


def _validate_placeholders(placeholders: str) -> None:
    """Validate SQL placeholder string contains only safe characters.

    SECURITY: Ensures placeholder strings like "?,?,?" don't contain SQL injection.
    Raises ValueError if placeholders contain anything other than '?' and ','.

    Args:
        placeholders: The placeholder string to validate (e.g., "?,?,?")

    Raises:
        ValueError: If placeholders contain invalid characters
    """
    if not placeholders:
        return
    allowed_chars = set("?,")
    if not set(placeholders).issubset(allowed_chars):
        raise ValueError(f"Invalid characters in SQL placeholders: {placeholders}")


# Schema SQL for tags system
TAGS_SCHEMA_SQL = """
-- Tags table with hierarchical support
CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    color TEXT DEFAULT '#3b82f6',
    icon TEXT DEFAULT 'tag',
    parent_id INTEGER REFERENCES tags(id) ON DELETE SET NULL,
    description TEXT,
    aliases_json TEXT,              -- JSON array of alias strings
    sort_order INTEGER DEFAULT 0,
    is_system BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, parent_id)
);

-- Conversation-tag junction table
CREATE TABLE IF NOT EXISTS conversation_tags (
    chat_id TEXT NOT NULL,
    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    added_by TEXT DEFAULT 'user',   -- 'user', 'auto', or rule name
    confidence REAL DEFAULT 1.0,    -- confidence for auto-assigned tags
    PRIMARY KEY (chat_id, tag_id)
);

-- Smart folders with rule-based queries
CREATE TABLE IF NOT EXISTS smart_folders (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    icon TEXT DEFAULT 'folder',
    color TEXT DEFAULT '#64748b',
    rules_json TEXT,                -- JSON SmartFolderRules
    sort_order INTEGER DEFAULT 0,
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Auto-tagging rules
CREATE TABLE IF NOT EXISTS tag_rules (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    trigger TEXT DEFAULT 'on_new_message',
    conditions_json TEXT,           -- JSON array of conditions
    tag_ids_json TEXT,              -- JSON array of tag IDs to apply
    priority INTEGER DEFAULT 0,
    is_enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_triggered_at TIMESTAMP,
    trigger_count INTEGER DEFAULT 0
);

-- Tag usage history for learning suggestions
CREATE TABLE IF NOT EXISTS tag_usage_history (
    id INTEGER PRIMARY KEY,
    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    chat_id TEXT NOT NULL,
    action TEXT NOT NULL,           -- 'add', 'remove', 'suggest_accepted', 'suggest_rejected'
    context_json TEXT,              -- Context at time of action (keywords, sentiment, etc)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fast lookup
CREATE INDEX IF NOT EXISTS idx_tags_parent ON tags(parent_id);
CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);
CREATE INDEX IF NOT EXISTS idx_conv_tags_chat ON conversation_tags(chat_id);
CREATE INDEX IF NOT EXISTS idx_conv_tags_tag ON conversation_tags(tag_id);
CREATE INDEX IF NOT EXISTS idx_tag_rules_trigger ON tag_rules(trigger, is_enabled);
CREATE INDEX IF NOT EXISTS idx_tag_usage_tag ON tag_usage_history(tag_id);
CREATE INDEX IF NOT EXISTS idx_tag_usage_chat ON tag_usage_history(chat_id);
"""

# Default smart folders to create
DEFAULT_SMART_FOLDERS = [
    {
        "name": "All Messages",
        "icon": "inbox",
        "color": TagColor.BLUE.value,
        "rules": {"match": "all", "conditions": [], "sort_by": "last_message_date"},
        "is_default": True,
    },
    {
        "name": "Unread",
        "icon": "mail",
        "color": TagColor.RED.value,
        "rules": {
            "match": "all",
            "conditions": [{"field": "unread_count", "operator": "greater_than", "value": 0}],
        },
        "is_default": True,
    },
    {
        "name": "Flagged",
        "icon": "flag",
        "color": TagColor.AMBER.value,
        "rules": {
            "match": "all",
            "conditions": [{"field": "is_flagged", "operator": "equals", "value": True}],
        },
        "is_default": True,
    },
    {
        "name": "Recent",
        "icon": "clock",
        "color": TagColor.GREEN.value,
        "rules": {
            "match": "all",
            "conditions": [{"field": "last_message_date", "operator": "in_last_days", "value": 7}],
        },
        "is_default": True,
    },
]

# Default system tags
DEFAULT_SYSTEM_TAGS = [
    {"name": "Important", "color": TagColor.RED.value, "icon": TagIcon.STAR.value},
    {"name": "Work", "color": TagColor.BLUE.value, "icon": TagIcon.BRIEFCASE.value},
    {"name": "Personal", "color": TagColor.GREEN.value, "icon": TagIcon.HOME.value},
    {"name": "Family", "color": TagColor.PURPLE.value, "icon": TagIcon.USERS.value},
    {"name": "Follow Up", "color": TagColor.AMBER.value, "icon": TagIcon.CLOCK.value},
]


class TagManager:
    """Manager for tags, smart folders, and auto-tagging rules.

    Thread-safe connection management with caching for frequently accessed data.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize tag manager.

        Args:
            db_path: Path to database file. Uses default if None.
        """
        self.db_path = db_path or TAGS_DB_PATH
        self._local = threading.local()
        self._ensure_directory()

        # Query result caches
        self._tag_cache = TTLCache(maxsize=256, ttl_seconds=60.0)
        self._folder_cache = TTLCache(maxsize=64, ttl_seconds=60.0)

    def _ensure_directory(self) -> None:
        """Ensure the database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a thread-local connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                timeout=30.0,
                check_same_thread=False,
            )
            self._local.connection.row_factory = sqlite3.Row
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            self._local.connection.execute("PRAGMA journal_mode = WAL")
        return self._local.connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection with automatic commit/rollback."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            logger.debug("Transaction rolled back", exc_info=True)
            conn.rollback()
            raise

    def close(self) -> None:
        """Close the thread-local connection."""
        if hasattr(self._local, "connection") and self._local.connection is not None:
            self._local.connection.close()
            self._local.connection = None
        self.clear_caches()

    def clear_caches(self) -> None:
        """Clear all caches."""
        self._tag_cache.clear()
        self._folder_cache.clear()

    def init_schema(self, create_defaults: bool = True) -> bool:
        """Initialize the tags schema.

        Args:
            create_defaults: If True, create default tags and smart folders.

        Returns:
            True if schema was created, False if already exists.
        """
        with self.connection() as conn:
            # Check if tags table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='tags'"
            )
            exists = cursor.fetchone() is not None

            # Create schema
            conn.executescript(TAGS_SCHEMA_SQL)

            if not exists and create_defaults:
                # Create default smart folders
                for folder_data in DEFAULT_SMART_FOLDERS:
                    rules = SmartFolderRules.from_dict(folder_data.get("rules", {}))
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO smart_folders
                        (name, icon, color, rules_json, is_default, sort_order)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            folder_data["name"],
                            folder_data.get("icon", "folder"),
                            folder_data.get("color", TagColor.SLATE.value),
                            rules.to_json(),
                            folder_data.get("is_default", False),
                            DEFAULT_SMART_FOLDERS.index(folder_data),
                        ),
                    )

                # Create default system tags
                for i, tag_data in enumerate(DEFAULT_SYSTEM_TAGS):
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO tags
                        (name, color, icon, is_system, sort_order)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            tag_data["name"],
                            tag_data["color"],
                            tag_data["icon"],
                            True,
                            i,
                        ),
                    )

                logger.info("Created tags schema with default tags and folders")
                return True

            return not exists

    # -------------------------------------------------------------------------
    # Tag CRUD Operations
    # -------------------------------------------------------------------------

    def create_tag(
        self,
        name: str,
        color: str = TagColor.BLUE.value,
        icon: str = TagIcon.TAG.value,
        parent_id: int | None = None,
        description: str | None = None,
        aliases: list[str] | None = None,
    ) -> Tag:
        """Create a new tag.

        Args:
            name: Tag display name.
            color: Hex color code.
            icon: Icon name.
            parent_id: Parent tag ID for hierarchy.
            description: Optional description.
            aliases: Alternative names for the tag.

        Returns:
            The created Tag.

        Raises:
            ValueError: If a tag with the same name and parent exists.
        """
        with self.connection() as conn:
            now = datetime.now(UTC)
            aliases_json = json.dumps(aliases) if aliases else None

            try:
                cursor = conn.execute(
                    """
                    INSERT INTO tags (name, color, icon, parent_id, description, aliases_json,
                                      created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (name, color, icon, parent_id, description, aliases_json, now, now),
                )
                tag_id = cursor.lastrowid
            except sqlite3.IntegrityError as e:
                raise ValueError(f"Tag '{name}' already exists") from e

            self._tag_cache.clear()

            return Tag(
                id=tag_id,
                name=name,
                color=color,
                icon=icon,
                parent_id=parent_id,
                description=description,
                aliases_json=aliases_json,
                created_at=now,
                updated_at=now,
            )

    def get_tag(self, tag_id: int) -> Tag | None:
        """Get a tag by ID."""
        hit, cached = self._tag_cache.get(f"tag:{tag_id}")
        if hit:
            return cached

        with self.connection() as conn:
            cursor = conn.execute("SELECT * FROM tags WHERE id = ?", (tag_id,))
            row = cursor.fetchone()
            if not row:
                return None

            tag = self._row_to_tag(row)
            self._tag_cache.set(f"tag:{tag_id}", tag)
            return tag

    def get_tag_by_name(self, name: str, parent_id: int | None = None) -> Tag | None:
        """Get a tag by name and optional parent."""
        with self.connection() as conn:
            if parent_id is None:
                cursor = conn.execute(
                    "SELECT * FROM tags WHERE name = ? AND parent_id IS NULL", (name,)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM tags WHERE name = ? AND parent_id = ?", (name, parent_id)
                )
            row = cursor.fetchone()
            return self._row_to_tag(row) if row else None

    def list_tags(
        self,
        parent_id: int | None = None,
        include_children: bool = False,
        include_system: bool = True,
    ) -> list[Tag]:
        """List tags with optional filtering.

        Args:
            parent_id: Filter by parent (None for root tags only, -1 for all).
            include_children: If True and parent_id is set, include all descendants.
            include_system: If True, include system tags.

        Returns:
            List of matching tags.
        """
        with self.connection() as conn:
            conditions = []
            params: list[Any] = []

            if parent_id is not None and parent_id != -1:
                conditions.append("parent_id = ?")
                params.append(parent_id)
            elif parent_id is None:
                conditions.append("parent_id IS NULL")

            if not include_system:
                conditions.append("is_system = FALSE")

            # SECURITY: where_clause is built from hardcoded, safe SQL fragments only.
            # All conditions use parameterized queries ("parent_id = ?", "is_system = FALSE").
            # No user input is interpolated into the SQL string.
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            cursor = conn.execute(
                f"SELECT * FROM tags WHERE {where_clause} ORDER BY sort_order, name",
                params,
            )

            tags = [self._row_to_tag(row) for row in cursor]

            if include_children and parent_id is not None and parent_id != -1:
                # Recursively get children
                for tag in list(tags):
                    children = self.list_tags(
                        parent_id=tag.id, include_children=True, include_system=include_system
                    )
                    tags.extend(children)

            return tags

    def get_all_tags(self) -> list[Tag]:
        """Get all tags as a flat list."""
        return self.list_tags(parent_id=-1)

    def update_tag(
        self,
        tag_id: int,
        name: str | None = None,
        color: str | None = None,
        icon: str | None = None,
        parent_id: int | None = None,
        description: str | None = None,
        aliases: list[str] | None = None,
        sort_order: int | None = None,
    ) -> Tag | None:
        """Update a tag's properties.

        Only provided fields are updated. Pass None to leave unchanged.

        Returns:
            The updated Tag, or None if not found.
        """
        tag = self.get_tag(tag_id)
        if not tag:
            return None

        with self.connection() as conn:
            updates = []
            params: list[Any] = []

            if name is not None:
                updates.append("name = ?")
                params.append(name)
            if color is not None:
                updates.append("color = ?")
                params.append(color)
            if icon is not None:
                updates.append("icon = ?")
                params.append(icon)
            if parent_id is not None:
                updates.append("parent_id = ?")
                params.append(parent_id if parent_id != -1 else None)
            if description is not None:
                updates.append("description = ?")
                params.append(description)
            if aliases is not None:
                updates.append("aliases_json = ?")
                params.append(json.dumps(aliases) if aliases else None)
            if sort_order is not None:
                updates.append("sort_order = ?")
                params.append(sort_order)

            if updates:
                updates.append("updated_at = ?")
                params.append(datetime.now(UTC))
                params.append(tag_id)

                # SECURITY: updates list contains only hardcoded SQL fragments ("name = ?", etc).
                # All values are passed through parameterized queries. No user input is
                # interpolated into column names.
                conn.execute(
                    f"UPDATE tags SET {', '.join(updates)} WHERE id = ?",
                    params,
                )
                self._tag_cache.invalidate(f"tag:{tag_id}")

        return self.get_tag(tag_id)

    def delete_tag(self, tag_id: int) -> bool:
        """Delete a tag and remove it from all conversations.

        Returns:
            True if deleted, False if not found.
        """
        with self.connection() as conn:
            cursor = conn.execute("DELETE FROM tags WHERE id = ?", (tag_id,))
            self._tag_cache.invalidate(f"tag:{tag_id}")
            return cursor.rowcount > 0

    def get_tag_hierarchy(self, tag_id: int) -> list[Tag]:
        """Get the full hierarchy path from root to this tag.

        Returns:
            List of tags from root to the specified tag.
        """
        path = []
        current_id = tag_id

        while current_id is not None:
            tag = self.get_tag(current_id)
            if tag is None:
                break
            path.insert(0, tag)
            current_id = tag.parent_id

        return path

    def get_tag_path(self, tag_id: int) -> str:
        """Get the hierarchical path as a string (e.g., 'Work/Projects/Alpha')."""
        hierarchy = self.get_tag_hierarchy(tag_id)
        return "/".join(tag.name for tag in hierarchy)

    def search_tags(self, query: str, limit: int = 10) -> list[Tag]:
        """Search tags by name or alias."""
        query_lower = query.lower()
        with self.connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM tags
                WHERE LOWER(name) LIKE ? OR LOWER(aliases_json) LIKE ?
                ORDER BY
                    CASE WHEN LOWER(name) = ? THEN 0
                         WHEN LOWER(name) LIKE ? THEN 1
                         ELSE 2
                    END,
                    name
                LIMIT ?
                """,
                (f"%{query_lower}%", f"%{query_lower}%", query_lower, f"{query_lower}%", limit),
            )
            return [self._row_to_tag(row) for row in cursor]

    # -------------------------------------------------------------------------
    # Conversation Tagging
    # -------------------------------------------------------------------------

    def add_tag_to_conversation(
        self,
        chat_id: str,
        tag_id: int,
        added_by: str = "user",
        confidence: float = 1.0,
    ) -> bool:
        """Add a tag to a conversation.

        Args:
            chat_id: Conversation identifier.
            tag_id: Tag to add.
            added_by: Who/what added it ('user', 'auto', or rule name).
            confidence: Confidence score for auto-tags.

        Returns:
            True if added, False if already exists.
        """
        with self.connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO conversation_tags (chat_id, tag_id, added_by, confidence, added_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (chat_id, tag_id, added_by, confidence, datetime.now(UTC)),
                )

                # Record usage for learning
                self._record_tag_usage(conn, tag_id, chat_id, "add", None)
                return True
            except sqlite3.IntegrityError:
                return False

    def remove_tag_from_conversation(self, chat_id: str, tag_id: int) -> bool:
        """Remove a tag from a conversation.

        Returns:
            True if removed, False if not found.
        """
        with self.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM conversation_tags WHERE chat_id = ? AND tag_id = ?",
                (chat_id, tag_id),
            )

            if cursor.rowcount > 0:
                self._record_tag_usage(conn, tag_id, chat_id, "remove", None)
                return True
            return False

    def get_tags_for_conversation(self, chat_id: str) -> list[tuple[Tag, ConversationTag]]:
        """Get all tags for a conversation with their assignment details.

        Returns:
            List of (Tag, ConversationTag) tuples.
        """
        with self.connection() as conn:
            cursor = conn.execute(
                """
                SELECT t.*, ct.added_at, ct.added_by, ct.confidence
                FROM tags t
                JOIN conversation_tags ct ON t.id = ct.tag_id
                WHERE ct.chat_id = ?
                ORDER BY t.sort_order, t.name
                """,
                (chat_id,),
            )

            results = []
            for row in cursor:
                tag = self._row_to_tag(row)
                conv_tag = ConversationTag(
                    chat_id=chat_id,
                    tag_id=tag.id,
                    added_at=row["added_at"],
                    added_by=row["added_by"],
                    confidence=row["confidence"],
                )
                results.append((tag, conv_tag))

            return results

    def get_conversations_with_tag(self, tag_id: int) -> list[str]:
        """Get all conversation IDs that have a specific tag."""
        with self.connection() as conn:
            cursor = conn.execute(
                "SELECT chat_id FROM conversation_tags WHERE tag_id = ?",
                (tag_id,),
            )
            return [row["chat_id"] for row in cursor]

    def get_conversations_with_tags(
        self,
        tag_ids: list[int],
        match_all: bool = True,
    ) -> list[str]:
        """Get conversations that have specified tags.

        Args:
            tag_ids: List of tag IDs to match.
            match_all: If True, conversation must have all tags. If False, any tag.

        Returns:
            List of chat_ids.
        """
        if not tag_ids:
            return []

        with self.connection() as conn:
            placeholders = ",".join("?" * len(tag_ids))
            # SECURITY: Validate placeholders only contain "?" and "," before SQL interpolation
            _validate_placeholders(placeholders)

            if match_all:
                cursor = conn.execute(
                    f"""
                    SELECT chat_id FROM conversation_tags
                    WHERE tag_id IN ({placeholders})
                    GROUP BY chat_id
                    HAVING COUNT(DISTINCT tag_id) = ?
                    """,
                    [*tag_ids, len(tag_ids)],
                )
            else:
                cursor = conn.execute(
                    f"""
                    SELECT DISTINCT chat_id FROM conversation_tags
                    WHERE tag_id IN ({placeholders})
                    """,
                    tag_ids,
                )

            return [row["chat_id"] for row in cursor]

    def bulk_add_tags(
        self,
        chat_ids: list[str],
        tag_ids: list[int],
        added_by: str = "user",
    ) -> int:
        """Add multiple tags to multiple conversations.

        Returns:
            Number of tag assignments created.
        """
        now = datetime.now(UTC)

        with self.connection() as conn:
            # Build batch of all (chat_id, tag_id) pairs
            batch = [(chat_id, tag_id, added_by, now) for chat_id in chat_ids for tag_id in tag_ids]

            # Get count before batch insert
            count_before = conn.execute(
                "SELECT COUNT(*) FROM conversation_tags"
            ).fetchone()[0]

            # Batch insert with INSERT OR IGNORE
            conn.executemany(
                """
                INSERT OR IGNORE INTO conversation_tags (chat_id, tag_id, added_by, added_at)
                VALUES (?, ?, ?, ?)
                """,
                batch,
            )

            # Get count after to determine how many were actually inserted
            count_after = conn.execute(
                "SELECT COUNT(*) FROM conversation_tags"
            ).fetchone()[0]

        return count_after - count_before

    def bulk_remove_tags(
        self,
        chat_ids: list[str],
        tag_ids: list[int],
    ) -> int:
        """Remove multiple tags from multiple conversations.

        Returns:
            Number of tag assignments removed.
        """
        with self.connection() as conn:
            chat_placeholders = ",".join("?" * len(chat_ids))
            tag_placeholders = ",".join("?" * len(tag_ids))
            # SECURITY: Validate placeholders only contain "?" and "," before SQL interpolation
            _validate_placeholders(chat_placeholders)
            _validate_placeholders(tag_placeholders)

            cursor = conn.execute(
                f"""
                DELETE FROM conversation_tags
                WHERE chat_id IN ({chat_placeholders}) AND tag_id IN ({tag_placeholders})
                """,
                [*chat_ids, *tag_ids],
            )
            return cursor.rowcount

    def set_conversation_tags(
        self,
        chat_id: str,
        tag_ids: list[int],
        added_by: str = "user",
    ) -> None:
        """Set the exact tags for a conversation (replaces existing).

        Args:
            chat_id: Conversation identifier.
            tag_ids: Tags to set (replaces all existing).
            added_by: Who set the tags.
        """
        with self.connection() as conn:
            # Remove existing tags
            conn.execute("DELETE FROM conversation_tags WHERE chat_id = ?", (chat_id,))

            # Add new tags in batch
            if tag_ids:
                now = datetime.now(UTC)
                batch = [(chat_id, tag_id, added_by, now) for tag_id in tag_ids]
                conn.executemany(
                    """
                    INSERT INTO conversation_tags (chat_id, tag_id, added_by, added_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    batch,
                )

    # -------------------------------------------------------------------------
    # Smart Folder Operations
    # -------------------------------------------------------------------------

    def create_smart_folder(
        self,
        name: str,
        rules: SmartFolderRules,
        icon: str = TagIcon.FOLDER.value,
        color: str = TagColor.SLATE.value,
    ) -> SmartFolder:
        """Create a new smart folder."""
        with self.connection() as conn:
            now = datetime.now(UTC)

            cursor = conn.execute(
                """
                INSERT INTO smart_folders (name, icon, color, rules_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (name, icon, color, rules.to_json(), now, now),
            )

            self._folder_cache.clear()

            return SmartFolder(
                id=cursor.lastrowid,
                name=name,
                icon=icon,
                color=color,
                rules_json=rules.to_json(),
                created_at=now,
                updated_at=now,
            )

    def get_smart_folder(self, folder_id: int) -> SmartFolder | None:
        """Get a smart folder by ID."""
        hit, cached = self._folder_cache.get(f"folder:{folder_id}")
        if hit:
            return cached

        with self.connection() as conn:
            cursor = conn.execute("SELECT * FROM smart_folders WHERE id = ?", (folder_id,))
            row = cursor.fetchone()
            if not row:
                return None

            folder = self._row_to_smart_folder(row)
            self._folder_cache.set(f"folder:{folder_id}", folder)
            return folder

    def list_smart_folders(self, include_defaults: bool = True) -> list[SmartFolder]:
        """List all smart folders."""
        with self.connection() as conn:
            if include_defaults:
                cursor = conn.execute(
                    "SELECT * FROM smart_folders ORDER BY is_default DESC, sort_order, name"
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM smart_folders
                    WHERE is_default = FALSE
                    ORDER BY sort_order, name
                    """
                )
            return [self._row_to_smart_folder(row) for row in cursor]

    def update_smart_folder(
        self,
        folder_id: int,
        name: str | None = None,
        rules: SmartFolderRules | None = None,
        icon: str | None = None,
        color: str | None = None,
        sort_order: int | None = None,
    ) -> SmartFolder | None:
        """Update a smart folder."""
        folder = self.get_smart_folder(folder_id)
        if not folder:
            return None

        with self.connection() as conn:
            updates = []
            params: list[Any] = []

            if name is not None:
                updates.append("name = ?")
                params.append(name)
            if rules is not None:
                updates.append("rules_json = ?")
                params.append(rules.to_json())
            if icon is not None:
                updates.append("icon = ?")
                params.append(icon)
            if color is not None:
                updates.append("color = ?")
                params.append(color)
            if sort_order is not None:
                updates.append("sort_order = ?")
                params.append(sort_order)

            if updates:
                updates.append("updated_at = ?")
                params.append(datetime.now(UTC))
                params.append(folder_id)

                conn.execute(
                    f"UPDATE smart_folders SET {', '.join(updates)} WHERE id = ?",
                    params,
                )
                self._folder_cache.invalidate(f"folder:{folder_id}")

        return self.get_smart_folder(folder_id)

    def delete_smart_folder(self, folder_id: int) -> bool:
        """Delete a smart folder.

        Returns:
            True if deleted, False if not found or is a default folder.
        """
        with self.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM smart_folders WHERE id = ? AND is_default = FALSE",
                (folder_id,),
            )
            self._folder_cache.invalidate(f"folder:{folder_id}")
            return cursor.rowcount > 0

    # -------------------------------------------------------------------------
    # Tag Rule Operations
    # -------------------------------------------------------------------------

    def create_tag_rule(self, rule: TagRule) -> TagRule:
        """Create a new auto-tagging rule."""
        with self.connection() as conn:
            now = datetime.now(UTC)

            cursor = conn.execute(
                """
                INSERT INTO tag_rules
                (name, trigger, conditions_json, tag_ids_json, priority, is_enabled, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rule.name,
                    rule.trigger,
                    rule.conditions_json,
                    rule.tag_ids_json,
                    rule.priority,
                    rule.is_enabled,
                    now,
                ),
            )

            rule.id = cursor.lastrowid
            rule.created_at = now
            return rule

    def get_tag_rule(self, rule_id: int) -> TagRule | None:
        """Get a tag rule by ID."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT * FROM tag_rules WHERE id = ?", (rule_id,))
            row = cursor.fetchone()
            return self._row_to_tag_rule(row) if row else None

    def list_tag_rules(
        self,
        trigger: str | None = None,
        enabled_only: bool = False,
    ) -> list[TagRule]:
        """List tag rules with optional filtering."""
        with self.connection() as conn:
            conditions = []
            params: list[Any] = []

            if trigger:
                conditions.append("trigger = ?")
                params.append(trigger)
            if enabled_only:
                conditions.append("is_enabled = TRUE")

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            cursor = conn.execute(
                f"SELECT * FROM tag_rules WHERE {where_clause} ORDER BY priority DESC, name",
                params,
            )
            return [self._row_to_tag_rule(row) for row in cursor]

    def update_tag_rule(self, rule: TagRule) -> TagRule | None:
        """Update an existing tag rule."""
        if rule.id is None:
            return None

        with self.connection() as conn:
            conn.execute(
                """
                UPDATE tag_rules SET
                    name = ?, trigger = ?, conditions_json = ?, tag_ids_json = ?,
                    priority = ?, is_enabled = ?
                WHERE id = ?
                """,
                (
                    rule.name,
                    rule.trigger,
                    rule.conditions_json,
                    rule.tag_ids_json,
                    rule.priority,
                    rule.is_enabled,
                    rule.id,
                ),
            )

        return self.get_tag_rule(rule.id)

    def delete_tag_rule(self, rule_id: int) -> bool:
        """Delete a tag rule."""
        with self.connection() as conn:
            cursor = conn.execute("DELETE FROM tag_rules WHERE id = ?", (rule_id,))
            return cursor.rowcount > 0

    def record_rule_trigger(self, rule_id: int) -> None:
        """Record that a rule was triggered."""
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE tag_rules SET
                    last_triggered_at = ?,
                    trigger_count = trigger_count + 1
                WHERE id = ?
                """,
                (datetime.now(UTC), rule_id),
            )

    # -------------------------------------------------------------------------
    # Tag Statistics and Suggestions
    # -------------------------------------------------------------------------

    def get_tag_statistics(self) -> dict[str, Any]:
        """Get statistics about tag usage."""
        with self.connection() as conn:
            # Total tags
            total_tags = conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]

            # Tags per conversation distribution
            tag_counts = conn.execute(
                """
                SELECT chat_id, COUNT(*) as count
                FROM conversation_tags
                GROUP BY chat_id
                """
            ).fetchall()

            avg_tags = sum(r["count"] for r in tag_counts) / len(tag_counts) if tag_counts else 0

            # Most used tags
            most_used = conn.execute(
                """
                SELECT t.id, t.name, COUNT(ct.chat_id) as usage_count
                FROM tags t
                LEFT JOIN conversation_tags ct ON t.id = ct.tag_id
                GROUP BY t.id
                ORDER BY usage_count DESC
                LIMIT 10
                """
            ).fetchall()

            return {
                "total_tags": total_tags,
                "total_tagged_conversations": len(tag_counts),
                "average_tags_per_conversation": round(avg_tags, 2),
                "most_used_tags": [
                    {"id": r["id"], "name": r["name"], "count": r["usage_count"]} for r in most_used
                ],
            }

    def get_frequently_used_with(self, tag_id: int, limit: int = 5) -> list[tuple[Tag, int]]:
        """Get tags frequently used together with the specified tag.

        Returns:
            List of (Tag, co-occurrence count) tuples.
        """
        with self.connection() as conn:
            cursor = conn.execute(
                """
                SELECT t.*, COUNT(*) as co_count
                FROM tags t
                JOIN conversation_tags ct ON t.id = ct.tag_id
                WHERE ct.chat_id IN (
                    SELECT chat_id FROM conversation_tags WHERE tag_id = ?
                )
                AND t.id != ?
                GROUP BY t.id
                ORDER BY co_count DESC
                LIMIT ?
                """,
                (tag_id, tag_id, limit),
            )

            return [(self._row_to_tag(row), row["co_count"]) for row in cursor]

    # -------------------------------------------------------------------------
    # Private Helpers
    # -------------------------------------------------------------------------

    def _row_to_tag(self, row: sqlite3.Row) -> Tag:
        """Convert a database row to a Tag object."""
        return Tag(
            id=row["id"],
            name=row["name"],
            color=row["color"],
            icon=row["icon"],
            parent_id=row["parent_id"],
            description=row["description"],
            aliases_json=row["aliases_json"],
            sort_order=row["sort_order"],
            is_system=bool(row["is_system"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_smart_folder(self, row: sqlite3.Row) -> SmartFolder:
        """Convert a database row to a SmartFolder object."""
        return SmartFolder(
            id=row["id"],
            name=row["name"],
            icon=row["icon"],
            color=row["color"],
            rules_json=row["rules_json"],
            sort_order=row["sort_order"],
            is_default=bool(row["is_default"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_tag_rule(self, row: sqlite3.Row) -> TagRule:
        """Convert a database row to a TagRule object."""
        return TagRule(
            id=row["id"],
            name=row["name"],
            trigger=row["trigger"],
            conditions_json=row["conditions_json"],
            tag_ids_json=row["tag_ids_json"],
            priority=row["priority"],
            is_enabled=bool(row["is_enabled"]),
            created_at=row["created_at"],
            last_triggered_at=row["last_triggered_at"],
            trigger_count=row["trigger_count"],
        )

    def _record_tag_usage(
        self,
        conn: sqlite3.Connection,
        tag_id: int,
        chat_id: str,
        action: str,
        context: dict | None,
    ) -> None:
        """Record tag usage for learning suggestions."""
        conn.execute(
            """
            INSERT INTO tag_usage_history (tag_id, chat_id, action, context_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                tag_id,
                chat_id,
                action,
                json.dumps(context) if context else None,
                datetime.now(UTC),
            ),
        )


# Export all public symbols
__all__ = ["TagManager"]
