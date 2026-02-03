"""Tag and SmartFolder data models for conversation organization.

Defines the core data structures for the tagging system including:
- Tag: Hierarchical tags with colors and icons
- TagAlias: Alternative names for tags
- ConversationTag: Links conversations to tags
- SmartFolder: Rule-based dynamic folders
- TagRule: Auto-tagging rules

Usage:
    from jarvis.tags.models import Tag, SmartFolder, TagRule

    tag = Tag(name="Work", color="#0066cc", icon="briefcase")
    folder = SmartFolder(name="Urgent", rules={"match": "all", "conditions": [...]})
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class TagColor(str, Enum):
    """Predefined tag colors."""

    RED = "#ef4444"
    ORANGE = "#f97316"
    AMBER = "#f59e0b"
    YELLOW = "#eab308"
    LIME = "#84cc16"
    GREEN = "#22c55e"
    EMERALD = "#10b981"
    TEAL = "#14b8a6"
    CYAN = "#06b6d4"
    SKY = "#0ea5e9"
    BLUE = "#3b82f6"
    INDIGO = "#6366f1"
    VIOLET = "#8b5cf6"
    PURPLE = "#a855f7"
    FUCHSIA = "#d946ef"
    PINK = "#ec4899"
    ROSE = "#f43f5e"
    SLATE = "#64748b"


class TagIcon(str, Enum):
    """Predefined tag icons (using common icon names)."""

    STAR = "star"
    HEART = "heart"
    FLAG = "flag"
    BOOKMARK = "bookmark"
    FOLDER = "folder"
    TAG = "tag"
    INBOX = "inbox"
    BRIEFCASE = "briefcase"
    HOME = "home"
    USERS = "users"
    USER = "user"
    CLOCK = "clock"
    CALENDAR = "calendar"
    BELL = "bell"
    ALERT = "alert"
    CHECK = "check"
    CIRCLE = "circle"
    SQUARE = "square"
    ARROW_UP = "arrow-up"
    ARROW_DOWN = "arrow-down"
    MESSAGE = "message"
    MAIL = "mail"
    PHONE = "phone"
    SPARKLES = "sparkles"
    FIRE = "fire"
    ZAP = "zap"


class RuleOperator(str, Enum):
    """Operators for smart folder rule conditions."""

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    IS_EMPTY = "is_empty"
    IS_NOT_EMPTY = "is_not_empty"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    IN_LAST_DAYS = "in_last_days"
    BEFORE = "before"
    AFTER = "after"
    HAS_TAG = "has_tag"
    HAS_ANY_TAG = "has_any_tag"
    HAS_ALL_TAGS = "has_all_tags"
    HAS_NO_TAGS = "has_no_tags"


class RuleField(str, Enum):
    """Fields that can be used in smart folder rules."""

    # Conversation fields
    CHAT_ID = "chat_id"
    DISPLAY_NAME = "display_name"
    LAST_MESSAGE_DATE = "last_message_date"
    MESSAGE_COUNT = "message_count"
    IS_GROUP = "is_group"
    UNREAD_COUNT = "unread_count"
    IS_FLAGGED = "is_flagged"

    # Contact fields
    RELATIONSHIP = "relationship"
    CONTACT_NAME = "contact_name"

    # Message content
    LAST_MESSAGE_TEXT = "last_message_text"
    HAS_ATTACHMENTS = "has_attachments"

    # Tag fields
    TAGS = "tags"

    # Sentiment/priority (from ML)
    SENTIMENT = "sentiment"
    PRIORITY = "priority"
    NEEDS_RESPONSE = "needs_response"


class AutoTagTrigger(str, Enum):
    """Triggers for auto-tagging rules."""

    ON_NEW_MESSAGE = "on_new_message"
    ON_CONVERSATION_START = "on_conversation_start"
    ON_KEYWORD_MATCH = "on_keyword_match"
    ON_CONTACT_MATCH = "on_contact_match"
    ON_SENTIMENT_CHANGE = "on_sentiment_change"
    ON_SCHEDULE = "on_schedule"
    MANUAL = "manual"


@dataclass
class Tag:
    """A tag for organizing conversations.

    Supports hierarchical organization (e.g., Work/Projects/ProjectA)
    via the parent_id field.

    Attributes:
        id: Unique tag identifier.
        name: Display name for the tag.
        color: Hex color code for display.
        icon: Icon name for display.
        parent_id: ID of parent tag (for hierarchy).
        description: Optional description.
        aliases_json: JSON array of alias strings.
        sort_order: Order within siblings.
        is_system: True if this is a system-generated tag.
        created_at: When the tag was created.
        updated_at: When the tag was last modified.
    """

    name: str
    id: int | None = None
    color: str = TagColor.BLUE.value
    icon: str = TagIcon.TAG.value
    parent_id: int | None = None
    description: str | None = None
    aliases_json: str | None = None
    sort_order: int = 0
    is_system: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def aliases(self) -> list[str]:
        """Get list of aliases from JSON."""
        if self.aliases_json:
            try:
                return json.loads(self.aliases_json)
            except json.JSONDecodeError:
                return []
        return []

    @aliases.setter
    def aliases(self, value: list[str]) -> None:
        """Set aliases as JSON."""
        self.aliases_json = json.dumps(value) if value else None

    def matches_term(self, term: str) -> bool:
        """Check if tag name or aliases match a search term."""
        term_lower = term.lower()
        if term_lower in self.name.lower():
            return True
        return any(term_lower in alias.lower() for alias in self.aliases)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "color": self.color,
            "icon": self.icon,
            "parent_id": self.parent_id,
            "description": self.description,
            "aliases": self.aliases,
            "sort_order": self.sort_order,
            "is_system": self.is_system,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Tag:
        """Create a Tag from a dictionary."""
        return cls(
            id=data.get("id"),
            name=data["name"],
            color=data.get("color", TagColor.BLUE.value),
            icon=data.get("icon", TagIcon.TAG.value),
            parent_id=data.get("parent_id"),
            description=data.get("description"),
            aliases_json=json.dumps(data.get("aliases", [])) if data.get("aliases") else None,
            sort_order=data.get("sort_order", 0),
            is_system=data.get("is_system", False),
            created_at=(
                datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None
            ),
        )


@dataclass
class ConversationTag:
    """Links a conversation to a tag.

    Attributes:
        chat_id: The conversation identifier.
        tag_id: The tag identifier.
        added_at: When the tag was added.
        added_by: Who/what added the tag (user, auto-tagger, rule name).
        confidence: Confidence score for auto-assigned tags (0-1).
    """

    chat_id: str
    tag_id: int
    added_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    added_by: str = "user"
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chat_id": self.chat_id,
            "tag_id": self.tag_id,
            "added_at": self.added_at.isoformat() if self.added_at else None,
            "added_by": self.added_by,
            "confidence": self.confidence,
        }


@dataclass
class RuleCondition:
    """A single condition in a smart folder rule.

    Attributes:
        field: The field to evaluate.
        operator: The comparison operator.
        value: The value to compare against.
    """

    field: str
    operator: str
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field": self.field,
            "operator": self.operator,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuleCondition:
        """Create from dictionary."""
        return cls(
            field=data["field"],
            operator=data["operator"],
            value=data.get("value"),
        )


@dataclass
class SmartFolderRules:
    """Rules configuration for a smart folder.

    Attributes:
        match: "all" requires all conditions to match, "any" requires at least one.
        conditions: List of rule conditions.
        sort_by: Field to sort results by.
        sort_order: "asc" or "desc".
        limit: Maximum number of results (0 for unlimited).
    """

    match: str = "all"  # "all" or "any"
    conditions: list[RuleCondition] = field(default_factory=list)
    sort_by: str = "last_message_date"
    sort_order: str = "desc"
    limit: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "match": self.match,
            "conditions": [c.to_dict() for c in self.conditions],
            "sort_by": self.sort_by,
            "sort_order": self.sort_order,
            "limit": self.limit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SmartFolderRules:
        """Create from dictionary."""
        return cls(
            match=data.get("match", "all"),
            conditions=[RuleCondition.from_dict(c) for c in data.get("conditions", [])],
            sort_by=data.get("sort_by", "last_message_date"),
            sort_order=data.get("sort_order", "desc"),
            limit=data.get("limit", 0),
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> SmartFolderRules:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class SmartFolder:
    """A dynamic folder based on rules.

    Smart folders automatically include conversations that match
    their rule conditions.

    Attributes:
        id: Unique folder identifier.
        name: Display name for the folder.
        icon: Icon name for display.
        color: Hex color code.
        rules_json: JSON string of SmartFolderRules.
        sort_order: Order in folder list.
        is_default: True if this is a default/system folder.
        created_at: When the folder was created.
        updated_at: When the folder was last modified.
    """

    name: str
    id: int | None = None
    icon: str = TagIcon.FOLDER.value
    color: str = TagColor.SLATE.value
    rules_json: str | None = None
    sort_order: int = 0
    is_default: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def rules(self) -> SmartFolderRules:
        """Get rules from JSON."""
        if self.rules_json:
            try:
                return SmartFolderRules.from_json(self.rules_json)
            except (json.JSONDecodeError, KeyError):
                return SmartFolderRules()
        return SmartFolderRules()

    @rules.setter
    def rules(self, value: SmartFolderRules) -> None:
        """Set rules as JSON."""
        self.rules_json = value.to_json()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "icon": self.icon,
            "color": self.color,
            "rules": self.rules.to_dict(),
            "sort_order": self.sort_order,
            "is_default": self.is_default,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SmartFolder:
        """Create a SmartFolder from a dictionary."""
        rules = SmartFolderRules.from_dict(data.get("rules", {}))
        return cls(
            id=data.get("id"),
            name=data["name"],
            icon=data.get("icon", TagIcon.FOLDER.value),
            color=data.get("color", TagColor.SLATE.value),
            rules_json=rules.to_json(),
            sort_order=data.get("sort_order", 0),
            is_default=data.get("is_default", False),
            created_at=(
                datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None
            ),
        )


@dataclass
class TagRule:
    """An auto-tagging rule.

    Automatically assigns tags to conversations based on conditions.

    Attributes:
        id: Unique rule identifier.
        name: Display name for the rule.
        trigger: When the rule should be evaluated.
        conditions_json: JSON string of rule conditions.
        tag_ids_json: JSON array of tag IDs to assign.
        priority: Higher priority rules are evaluated first.
        is_enabled: Whether the rule is active.
        created_at: When the rule was created.
        last_triggered_at: When the rule last matched.
        trigger_count: Number of times the rule has triggered.
    """

    name: str
    trigger: str = AutoTagTrigger.ON_NEW_MESSAGE.value
    id: int | None = None
    conditions_json: str | None = None
    tag_ids_json: str | None = None
    priority: int = 0
    is_enabled: bool = True
    created_at: datetime | None = None
    last_triggered_at: datetime | None = None
    trigger_count: int = 0

    @property
    def conditions(self) -> list[RuleCondition]:
        """Get conditions from JSON."""
        if self.conditions_json:
            try:
                data = json.loads(self.conditions_json)
                return [RuleCondition.from_dict(c) for c in data]
            except (json.JSONDecodeError, KeyError):
                return []
        return []

    @conditions.setter
    def conditions(self, value: list[RuleCondition]) -> None:
        """Set conditions as JSON."""
        self.conditions_json = json.dumps([c.to_dict() for c in value]) if value else None

    @property
    def tag_ids(self) -> list[int]:
        """Get tag IDs from JSON."""
        if self.tag_ids_json:
            try:
                return json.loads(self.tag_ids_json)
            except json.JSONDecodeError:
                return []
        return []

    @tag_ids.setter
    def tag_ids(self, value: list[int]) -> None:
        """Set tag IDs as JSON."""
        self.tag_ids_json = json.dumps(value) if value else None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "trigger": self.trigger,
            "conditions": [c.to_dict() for c in self.conditions],
            "tag_ids": self.tag_ids,
            "priority": self.priority,
            "is_enabled": self.is_enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_triggered_at": (
                self.last_triggered_at.isoformat() if self.last_triggered_at else None
            ),
            "trigger_count": self.trigger_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TagRule:
        """Create a TagRule from a dictionary."""
        conditions = [RuleCondition.from_dict(c) for c in data.get("conditions", [])]
        return cls(
            id=data.get("id"),
            name=data["name"],
            trigger=data.get("trigger", AutoTagTrigger.ON_NEW_MESSAGE.value),
            conditions_json=json.dumps([c.to_dict() for c in conditions]) if conditions else None,
            tag_ids_json=json.dumps(data.get("tag_ids", [])) if data.get("tag_ids") else None,
            priority=data.get("priority", 0),
            is_enabled=data.get("is_enabled", True),
            created_at=(
                datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
            ),
            last_triggered_at=(
                datetime.fromisoformat(data["last_triggered_at"])
                if data.get("last_triggered_at")
                else None
            ),
            trigger_count=data.get("trigger_count", 0),
        )


@dataclass
class TagSuggestion:
    """A suggested tag for a conversation.

    Generated by the auto-tagger based on content analysis.

    Attributes:
        tag_id: The suggested tag ID (None for new tag suggestions).
        tag_name: Display name for the tag.
        confidence: Confidence score (0-1).
        reason: Why this tag is suggested.
        source: What suggested this tag (content, contact, sentiment, etc).
    """

    tag_name: str
    confidence: float
    tag_id: int | None = None
    reason: str | None = None
    source: str = "content"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tag_id": self.tag_id,
            "tag_name": self.tag_name,
            "confidence": self.confidence,
            "reason": self.reason,
            "source": self.source,
        }


# Export all public symbols
__all__ = [
    # Enums
    "AutoTagTrigger",
    "RuleField",
    "RuleOperator",
    "TagColor",
    "TagIcon",
    # Models
    "ConversationTag",
    "RuleCondition",
    "SmartFolder",
    "SmartFolderRules",
    "Tag",
    "TagRule",
    "TagSuggestion",
]
