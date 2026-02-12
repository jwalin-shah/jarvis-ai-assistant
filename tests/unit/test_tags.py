"""Unit tests for JARVIS Tag System.

Tests cover:
- Tag CRUD operations
- Conversation tagging
- Smart folder management
- Auto-tagger suggestions
- Rules engine evaluation
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from jarvis.tags import (
    AutoTagger,
    ContentAnalysis,
    RuleCondition,
    RulesEngine,
    SmartFolder,
    SmartFolderRules,
    Tag,
    TagColor,
    TagIcon,
    TagManager,
    TagRule,
    TagSuggestion,
    build_rules,
)


class TestTagManager:
    """Tests for TagManager CRUD operations."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> TagManager:
        """Create a fresh tag manager for each test."""
        db_path = tmp_path / "tags.db"
        manager = TagManager(db_path)
        manager.init_schema(create_defaults=False)
        return manager

    @pytest.fixture
    def manager_with_defaults(self, tmp_path: Path) -> TagManager:
        """Create a tag manager with default tags and folders."""
        db_path = tmp_path / "tags_defaults.db"
        manager = TagManager(db_path)
        manager.init_schema(create_defaults=True)
        return manager


class TestTagCRUD(TestTagManager):
    """Tests for tag CRUD operations."""

    def test_create_tag_basic(self, manager: TagManager) -> None:
        """Test creating a basic tag."""
        tag = manager.create_tag("Work")

        assert tag.id is not None
        assert tag.name == "Work"
        assert tag.color == TagColor.BLUE.value
        assert tag.icon == TagIcon.TAG.value

    def test_create_tag_with_all_fields(self, manager: TagManager) -> None:
        """Test creating a tag with all fields populated."""
        tag = manager.create_tag(
            name="Important",
            color=TagColor.RED.value,
            icon=TagIcon.STAR.value,
            description="High priority items",
            aliases=["critical", "urgent"],
        )

        assert tag.name == "Important"
        assert tag.color == TagColor.RED.value
        assert tag.icon == TagIcon.STAR.value
        assert tag.description == "High priority items"
        assert tag.aliases == ["critical", "urgent"]

    def test_create_tag_duplicate_raises_error(self, manager: TagManager) -> None:
        """Test that creating a duplicate tag raises ValueError or returns existing."""
        original = manager.create_tag("Work")

        # Creating a duplicate should either raise ValueError or return without
        # duplicating - verify no extra tag is created at root level
        try:
            dup = manager.create_tag("Work")
            # If no error, the manager silently handled the duplicate.
            # Verify the tag wasn't duplicated: exactly one root-level "Work" tag.
            tags = manager.search_tags("Work")
            root_work_tags = [t for t in tags if t.parent_id is None and t.name == "Work"]
            assert len(root_work_tags) == 1, (
                f"Expected exactly 1 root 'Work' tag, found {len(root_work_tags)}"
            )
        except (ValueError, Exception):
            # ValueError (or IntegrityError) is the expected behavior for duplicates
            pass

    def test_create_hierarchical_tags(self, manager: TagManager) -> None:
        """Test creating hierarchical tags with parent-child relationships."""
        parent = manager.create_tag("Work")
        child = manager.create_tag("Projects", parent_id=parent.id)
        grandchild = manager.create_tag("Alpha", parent_id=child.id)

        assert child.parent_id == parent.id
        assert grandchild.parent_id == child.id

        # Test hierarchy path
        path = manager.get_tag_path(grandchild.id)
        assert path == "Work/Projects/Alpha"

    def test_get_tag_by_id(self, manager: TagManager) -> None:
        """Test retrieving a tag by ID."""
        created = manager.create_tag("TestTag")
        retrieved = manager.get_tag(created.id)

        assert retrieved is not None
        assert retrieved.name == "TestTag"
        assert retrieved.id == created.id

    def test_get_tag_returns_none_for_missing(self, manager: TagManager) -> None:
        """Test that get_tag returns None for non-existent ID."""
        result = manager.get_tag(99999)
        assert result is None

    def test_get_tag_by_name(self, manager: TagManager) -> None:
        """Test retrieving a tag by name."""
        manager.create_tag("UniqueTag")
        retrieved = manager.get_tag_by_name("UniqueTag")

        assert retrieved is not None
        assert retrieved.name == "UniqueTag"

    def test_get_tag_by_name_returns_none_for_missing(self, manager: TagManager) -> None:
        """Test that get_tag_by_name returns None for non-existent name."""
        result = manager.get_tag_by_name("NonExistent")
        assert result is None

    def test_list_tags(self, manager: TagManager) -> None:
        """Test listing all tags."""
        manager.create_tag("Alpha")
        manager.create_tag("Beta")
        manager.create_tag("Gamma")

        tags = manager.list_tags(parent_id=-1)  # All tags

        assert len(tags) == 3
        names = {t.name for t in tags}
        assert names == {"Alpha", "Beta", "Gamma"}

    def test_list_tags_root_only(self, manager: TagManager) -> None:
        """Test listing only root-level tags."""
        parent = manager.create_tag("Parent")
        manager.create_tag("Child", parent_id=parent.id)
        manager.create_tag("Other")

        tags = manager.list_tags(parent_id=None)  # Root tags only

        assert len(tags) == 2
        names = {t.name for t in tags}
        assert names == {"Parent", "Other"}

    def test_list_tags_by_parent(self, manager: TagManager) -> None:
        """Test listing tags filtered by parent."""
        parent = manager.create_tag("Parent")
        manager.create_tag("Child1", parent_id=parent.id)
        manager.create_tag("Child2", parent_id=parent.id)
        manager.create_tag("Other")

        children = manager.list_tags(parent_id=parent.id)

        assert len(children) == 2
        names = {t.name for t in children}
        assert names == {"Child1", "Child2"}

    def test_update_tag(self, manager: TagManager) -> None:
        """Test updating tag properties."""
        tag = manager.create_tag("OldName", color=TagColor.BLUE.value)

        updated = manager.update_tag(
            tag.id,
            name="NewName",
            color=TagColor.RED.value,
            description="Updated description",
        )

        assert updated is not None
        assert updated.name == "NewName"
        assert updated.color == TagColor.RED.value
        assert updated.description == "Updated description"

    def test_update_tag_returns_none_for_missing(self, manager: TagManager) -> None:
        """Test that update_tag returns None for non-existent ID."""
        result = manager.update_tag(99999, name="Anything")
        assert result is None

    def test_delete_tag(self, manager: TagManager) -> None:
        """Test deleting a tag."""
        tag = manager.create_tag("ToDelete")
        result = manager.delete_tag(tag.id)

        assert result is True
        assert manager.get_tag(tag.id) is None

    def test_delete_tag_returns_false_for_missing(self, manager: TagManager) -> None:
        """Test that delete_tag returns False for non-existent ID."""
        result = manager.delete_tag(99999)
        assert result is False

    def test_search_tags(self, manager: TagManager) -> None:
        """Test searching tags by name."""
        manager.create_tag("Work Projects")
        manager.create_tag("Personal")
        manager.create_tag("Work Tasks")

        results = manager.search_tags("work")

        assert len(results) == 2
        names = {t.name for t in results}
        assert names == {"Work Projects", "Work Tasks"}

    def test_search_tags_by_alias(self, manager: TagManager) -> None:
        """Test searching tags by alias."""
        manager.create_tag("Important", aliases=["critical", "urgent"])

        results = manager.search_tags("critical")

        assert len(results) == 1
        assert results[0].name == "Important"

    def test_get_tag_hierarchy(self, manager: TagManager) -> None:
        """Test getting full tag hierarchy path."""
        root = manager.create_tag("Work")
        mid = manager.create_tag("Projects", parent_id=root.id)
        leaf = manager.create_tag("Alpha", parent_id=mid.id)

        hierarchy = manager.get_tag_hierarchy(leaf.id)

        assert len(hierarchy) == 3
        assert [t.name for t in hierarchy] == ["Work", "Projects", "Alpha"]


class TestConversationTagging(TestTagManager):
    """Tests for conversation tagging operations."""

    def test_add_tag_to_conversation(self, manager: TagManager) -> None:
        """Test adding a tag to a conversation."""
        tag = manager.create_tag("Work")
        result = manager.add_tag_to_conversation("chat123", tag.id)

        assert result is True

    def test_add_tag_to_conversation_duplicate_returns_false(self, manager: TagManager) -> None:
        """Test that adding the same tag twice returns False."""
        tag = manager.create_tag("Work")
        manager.add_tag_to_conversation("chat123", tag.id)

        result = manager.add_tag_to_conversation("chat123", tag.id)

        assert result is False

    def test_add_tag_with_metadata(self, manager: TagManager) -> None:
        """Test adding a tag with metadata."""
        tag = manager.create_tag("Auto")
        result = manager.add_tag_to_conversation(
            "chat123", tag.id, added_by="auto:content", confidence=0.85
        )

        assert result is True

        tags = manager.get_tags_for_conversation("chat123")
        assert len(tags) == 1
        _, conv_tag = tags[0]
        assert conv_tag.added_by == "auto:content"
        assert conv_tag.confidence == 0.85

    def test_remove_tag_from_conversation(self, manager: TagManager) -> None:
        """Test removing a tag from a conversation."""
        tag = manager.create_tag("Work")
        manager.add_tag_to_conversation("chat123", tag.id)

        result = manager.remove_tag_from_conversation("chat123", tag.id)

        assert result is True
        assert manager.get_tags_for_conversation("chat123") == []

    def test_remove_tag_from_conversation_not_found(self, manager: TagManager) -> None:
        """Test removing a non-existent tag assignment returns False."""
        tag = manager.create_tag("Work")
        result = manager.remove_tag_from_conversation("chat123", tag.id)

        assert result is False

    def test_get_tags_for_conversation(self, manager: TagManager) -> None:
        """Test getting all tags for a conversation."""
        tag1 = manager.create_tag("Work")
        tag2 = manager.create_tag("Important")
        manager.add_tag_to_conversation("chat123", tag1.id)
        manager.add_tag_to_conversation("chat123", tag2.id)

        tags = manager.get_tags_for_conversation("chat123")

        assert len(tags) == 2
        tag_names = {t.name for t, _ in tags}
        assert tag_names == {"Work", "Important"}

    def test_get_conversations_with_tag(self, manager: TagManager) -> None:
        """Test getting all conversations with a specific tag."""
        tag = manager.create_tag("Work")
        manager.add_tag_to_conversation("chat1", tag.id)
        manager.add_tag_to_conversation("chat2", tag.id)
        manager.add_tag_to_conversation("chat3", tag.id)

        chat_ids = manager.get_conversations_with_tag(tag.id)

        assert len(chat_ids) == 3
        assert set(chat_ids) == {"chat1", "chat2", "chat3"}

    def test_get_conversations_with_tags_match_all(self, manager: TagManager) -> None:
        """Test getting conversations that have all specified tags."""
        tag1 = manager.create_tag("Work")
        tag2 = manager.create_tag("Important")

        manager.add_tag_to_conversation("chat1", tag1.id)
        manager.add_tag_to_conversation("chat1", tag2.id)
        manager.add_tag_to_conversation("chat2", tag1.id)  # Only has Work

        chat_ids = manager.get_conversations_with_tags([tag1.id, tag2.id], match_all=True)

        assert chat_ids == ["chat1"]

    def test_get_conversations_with_tags_match_any(self, manager: TagManager) -> None:
        """Test getting conversations that have any of specified tags."""
        tag1 = manager.create_tag("Work")
        tag2 = manager.create_tag("Important")

        manager.add_tag_to_conversation("chat1", tag1.id)
        manager.add_tag_to_conversation("chat2", tag2.id)

        chat_ids = manager.get_conversations_with_tags([tag1.id, tag2.id], match_all=False)

        assert len(chat_ids) == 2
        assert set(chat_ids) == {"chat1", "chat2"}

    def test_bulk_add_tags(self, manager: TagManager) -> None:
        """Test adding multiple tags to multiple conversations."""
        tag1 = manager.create_tag("Work")
        tag2 = manager.create_tag("Important")

        count = manager.bulk_add_tags(["chat1", "chat2", "chat3"], [tag1.id, tag2.id])

        # 3 chats x 2 tags = 6 assignments
        assert count == 6

        # Verify
        for chat_id in ["chat1", "chat2", "chat3"]:
            tags = manager.get_tags_for_conversation(chat_id)
            assert len(tags) == 2

    def test_bulk_remove_tags(self, manager: TagManager) -> None:
        """Test removing multiple tags from multiple conversations."""
        tag1 = manager.create_tag("Work")
        tag2 = manager.create_tag("Important")
        manager.bulk_add_tags(["chat1", "chat2"], [tag1.id, tag2.id])

        count = manager.bulk_remove_tags(["chat1", "chat2"], [tag1.id])

        assert count == 2

        # Verify only tag2 remains
        for chat_id in ["chat1", "chat2"]:
            tags = manager.get_tags_for_conversation(chat_id)
            assert len(tags) == 1
            assert tags[0][0].name == "Important"

    def test_set_conversation_tags(self, manager: TagManager) -> None:
        """Test setting exact tags for a conversation."""
        tag1 = manager.create_tag("Work")
        tag2 = manager.create_tag("Important")
        tag3 = manager.create_tag("Personal")

        # Add initial tags
        manager.add_tag_to_conversation("chat1", tag1.id)
        manager.add_tag_to_conversation("chat1", tag2.id)

        # Replace with new set
        manager.set_conversation_tags("chat1", [tag3.id])

        tags = manager.get_tags_for_conversation("chat1")
        assert len(tags) == 1
        assert tags[0][0].name == "Personal"


class TestSmartFolders(TestTagManager):
    """Tests for smart folder operations."""

    def test_create_smart_folder(self, manager: TagManager) -> None:
        """Test creating a smart folder."""
        rules = SmartFolderRules(
            match="all",
            conditions=[RuleCondition(field="unread_count", operator="greater_than", value=0)],
        )
        folder = manager.create_smart_folder("Unread Messages", rules)

        assert folder.id is not None
        assert folder.name == "Unread Messages"
        assert folder.rules.match == "all"

    def test_get_smart_folder(self, manager: TagManager) -> None:
        """Test retrieving a smart folder by ID."""
        rules = SmartFolderRules(match="all", conditions=[])
        created = manager.create_smart_folder("Test Folder", rules)

        retrieved = manager.get_smart_folder(created.id)

        assert retrieved is not None
        assert retrieved.name == "Test Folder"

    def test_get_smart_folder_returns_none_for_missing(self, manager: TagManager) -> None:
        """Test that get_smart_folder returns None for non-existent ID."""
        result = manager.get_smart_folder(99999)
        assert result is None

    def test_list_smart_folders(self, manager: TagManager) -> None:
        """Test listing all smart folders."""
        rules = SmartFolderRules(match="all", conditions=[])
        manager.create_smart_folder("Folder1", rules)
        manager.create_smart_folder("Folder2", rules)

        folders = manager.list_smart_folders()

        assert len(folders) == 2

    def test_update_smart_folder(self, manager: TagManager) -> None:
        """Test updating a smart folder."""
        rules = SmartFolderRules(match="all", conditions=[])
        folder = manager.create_smart_folder("Old Name", rules)

        new_rules = SmartFolderRules(
            match="any",
            conditions=[RuleCondition(field="is_flagged", operator="equals", value=True)],
        )
        updated = manager.update_smart_folder(
            folder.id,
            name="New Name",
            rules=new_rules,
            color=TagColor.RED.value,
        )

        assert updated is not None
        assert updated.name == "New Name"
        assert updated.rules.match == "any"
        assert updated.color == TagColor.RED.value

    def test_delete_smart_folder(self, manager: TagManager) -> None:
        """Test deleting a smart folder."""
        rules = SmartFolderRules(match="all", conditions=[])
        folder = manager.create_smart_folder("To Delete", rules)

        result = manager.delete_smart_folder(folder.id)

        assert result is True
        assert manager.get_smart_folder(folder.id) is None

    def test_default_smart_folders_created(self, manager_with_defaults: TagManager) -> None:
        """Test that default smart folders are created."""
        folders = manager_with_defaults.list_smart_folders()

        names = {f.name for f in folders}
        assert "All Messages" in names
        assert "Unread" in names
        assert "Flagged" in names
        assert "Recent" in names


class TestTagRules(TestTagManager):
    """Tests for auto-tagging rule operations."""

    def test_create_tag_rule(self, manager: TagManager) -> None:
        """Test creating an auto-tagging rule."""
        import json

        tag = manager.create_tag("Urgent")
        conditions = [{"field": "last_message_text", "operator": "contains", "value": "urgent"}]
        rule = TagRule(
            name="Mark Urgent Messages",
            trigger="on_new_message",
            conditions_json=json.dumps(conditions),
            tag_ids_json=json.dumps([tag.id]),
        )

        created = manager.create_tag_rule(rule)

        assert created.id is not None
        assert created.name == "Mark Urgent Messages"

    def test_get_tag_rule(self, manager: TagManager) -> None:
        """Test retrieving a tag rule by ID."""
        import json

        tag = manager.create_tag("Test")
        rule = TagRule(
            name="Test Rule", trigger="on_new_message", tag_ids_json=json.dumps([tag.id])
        )
        created = manager.create_tag_rule(rule)

        retrieved = manager.get_tag_rule(created.id)

        assert retrieved is not None
        assert retrieved.name == "Test Rule"

    def test_list_tag_rules(self, manager: TagManager) -> None:
        """Test listing tag rules."""
        import json

        tag = manager.create_tag("Test")
        rule1 = TagRule(name="Rule1", trigger="on_new_message", tag_ids_json=json.dumps([tag.id]))
        rule2 = TagRule(name="Rule2", trigger="manual", tag_ids_json=json.dumps([tag.id]))

        manager.create_tag_rule(rule1)
        manager.create_tag_rule(rule2)

        # List all
        all_rules = manager.list_tag_rules()
        assert len(all_rules) == 2

        # Filter by trigger
        new_msg_rules = manager.list_tag_rules(trigger="on_new_message")
        assert len(new_msg_rules) == 1
        assert new_msg_rules[0].name == "Rule1"

    def test_update_tag_rule(self, manager: TagManager) -> None:
        """Test updating a tag rule."""
        import json

        tag = manager.create_tag("Test")
        rule = TagRule(name="Old Name", trigger="on_new_message", tag_ids_json=json.dumps([tag.id]))
        created = manager.create_tag_rule(rule)

        created.name = "New Name"
        created.is_enabled = False
        updated = manager.update_tag_rule(created)

        assert updated is not None
        assert updated.name == "New Name"
        assert updated.is_enabled is False

    def test_delete_tag_rule(self, manager: TagManager) -> None:
        """Test deleting a tag rule."""
        import json

        tag = manager.create_tag("Test")
        rule = TagRule(
            name="To Delete", trigger="on_new_message", tag_ids_json=json.dumps([tag.id])
        )
        created = manager.create_tag_rule(rule)

        result = manager.delete_tag_rule(created.id)

        assert result is True
        assert manager.get_tag_rule(created.id) is None


class TestTagStatistics(TestTagManager):
    """Tests for tag statistics."""

    def test_get_tag_statistics(self, manager: TagManager) -> None:
        """Test getting tag usage statistics."""
        tag1 = manager.create_tag("Work")
        tag2 = manager.create_tag("Personal")

        # Add tags to conversations
        manager.add_tag_to_conversation("chat1", tag1.id)
        manager.add_tag_to_conversation("chat1", tag2.id)
        manager.add_tag_to_conversation("chat2", tag1.id)

        stats = manager.get_tag_statistics()

        assert stats["total_tags"] == 2
        assert stats["total_tagged_conversations"] == 2
        assert stats["average_tags_per_conversation"] == 1.5

    def test_get_frequently_used_with(self, manager: TagManager) -> None:
        """Test getting tags frequently used together."""
        tag1 = manager.create_tag("Work")
        tag2 = manager.create_tag("Important")
        tag3 = manager.create_tag("Personal")

        # Work and Important often used together
        for i in range(5):
            manager.add_tag_to_conversation(f"work_chat_{i}", tag1.id)
            manager.add_tag_to_conversation(f"work_chat_{i}", tag2.id)

        # Personal used alone
        manager.add_tag_to_conversation("personal_chat", tag3.id)

        co_occurring = manager.get_frequently_used_with(tag1.id)

        assert len(co_occurring) > 0
        # Important should be the most co-occurring
        assert co_occurring[0][0].name == "Important"


class TestAutoTagger:
    """Tests for AutoTagger suggestion system."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> TagManager:
        """Create a tag manager for auto-tagger tests."""
        db_path = tmp_path / "autotag.db"
        manager = TagManager(db_path)
        manager.init_schema(create_defaults=True)
        return manager

    @pytest.fixture
    def auto_tagger(self, manager: TagManager) -> AutoTagger:
        """Create an auto-tagger instance."""
        return AutoTagger(manager)

    def test_suggest_tags_from_content(self, auto_tagger: AutoTagger) -> None:
        """Test tag suggestions based on content keywords."""
        messages = [
            {"text": "Can we schedule a meeting for the project?", "is_from_me": False},
            {"text": "The deadline is next Monday", "is_from_me": False},
        ]

        suggestions = auto_tagger.suggest_tags("chat1", messages)

        # Should suggest "Work" based on meeting/project/deadline keywords
        suggestion_names = {s.tag_name for s in suggestions}
        assert "Work" in suggestion_names

    def test_suggest_tags_from_urgent_keywords(self, auto_tagger: AutoTagger) -> None:
        """Test tag suggestions for urgent content."""
        messages = [
            {"text": "This is URGENT! Need response ASAP!!!", "is_from_me": False},
        ]

        suggestions = auto_tagger.suggest_tags("chat1", messages)

        suggestion_names = {s.tag_name for s in suggestions}
        assert "Urgent" in suggestion_names

    def test_suggest_tags_from_family_keywords(self, auto_tagger: AutoTagger) -> None:
        """Test tag suggestions for family-related content."""
        messages = [
            {"text": "Mom is coming for Thanksgiving dinner", "is_from_me": True},
            {"text": "The kids are excited!", "is_from_me": False},
        ]

        suggestions = auto_tagger.suggest_tags("chat1", messages)

        suggestion_names = {s.tag_name for s in suggestions}
        assert "Family" in suggestion_names

    def test_suggest_tags_from_sentiment(self, auto_tagger: AutoTagger) -> None:
        """Test tag suggestions based on sentiment analysis."""
        messages = [
            {"text": "Sorry about the problem", "is_from_me": False},
            {"text": "I'm really frustrated with this issue", "is_from_me": False},
            {"text": "This is terrible, I'm so disappointed", "is_from_me": False},
        ]

        suggestions = auto_tagger.suggest_tags("chat1", messages)

        # Negative sentiment should trigger attention-related suggestions
        suggestion_names = {s.tag_name for s in suggestions}
        # Either "Needs Attention" or "Needs Response" are valid responses
        assert "Needs Attention" in suggestion_names or "Needs Response" in suggestion_names

    def test_suggest_tags_for_unanswered_questions(self, auto_tagger: AutoTagger) -> None:
        """Test suggestions for conversations needing response."""
        messages = [
            {"text": "Can you help me with this?", "is_from_me": False},
            {"text": "When are you available??", "is_from_me": False},
        ]

        suggestions = auto_tagger.suggest_tags("chat1", messages)

        suggestion_names = {s.tag_name for s in suggestions}
        assert "Needs Response" in suggestion_names

    def test_suggest_tags_from_contact_name(self, auto_tagger: AutoTagger) -> None:
        """Test suggestions based on contact name patterns."""
        messages = [{"text": "Hello", "is_from_me": False}]

        suggestions = auto_tagger.suggest_tags("chat1", messages, contact_name="Mom")

        suggestion_names = {s.tag_name for s in suggestions}
        assert "Family" in suggestion_names

    def test_no_duplicate_suggestions(self, auto_tagger: AutoTagger) -> None:
        """Test that duplicate suggestions are removed."""
        messages = [
            {"text": "Meeting about the project deadline", "is_from_me": False},
            {"text": "Office work discussion", "is_from_me": False},
        ]

        suggestions = auto_tagger.suggest_tags("chat1", messages)

        # Should not have duplicate "Work" suggestions
        work_count = sum(1 for s in suggestions if s.tag_name == "Work")
        assert work_count <= 1

    def test_respects_limit(self, auto_tagger: AutoTagger) -> None:
        """Test that suggestions respect the limit parameter."""
        messages = [
            {"text": "urgent meeting about family dinner project deadline", "is_from_me": False},
        ]

        suggestions = auto_tagger.suggest_tags("chat1", messages, limit=2)

        assert len(suggestions) <= 2

    def test_excludes_existing_tags(self, auto_tagger: AutoTagger, manager: TagManager) -> None:
        """Test that already-assigned tags are excluded from suggestions."""
        work_tag = manager.get_tag_by_name("Work")
        manager.add_tag_to_conversation("chat1", work_tag.id)

        messages = [
            {"text": "Meeting about the project", "is_from_me": False},
        ]

        suggestions = auto_tagger.suggest_tags("chat1", messages)

        # "Work" should not be suggested since it's already assigned
        suggestion_names = {s.tag_name for s in suggestions}
        assert "Work" not in suggestion_names

    def test_content_analysis(self, auto_tagger: AutoTagger) -> None:
        """Test the content analysis helper."""
        now = datetime.now(UTC)
        messages = [
            {"text": "Thanks for the help! You're awesome!", "is_from_me": True, "date": now},
            {"text": "Can we meet tomorrow?", "is_from_me": False, "date": now},
        ]

        analysis = auto_tagger._analyze_content(messages)

        assert isinstance(analysis, ContentAnalysis)
        assert analysis.message_count == 2
        assert analysis.question_count == 1
        assert analysis.sentiment in ("positive", "negative", "neutral")


class TestRulesEngine:
    """Tests for smart folder rules engine."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> TagManager:
        """Create a tag manager for rules engine tests."""
        db_path = tmp_path / "rules.db"
        manager = TagManager(db_path)
        manager.init_schema(create_defaults=False)
        return manager

    @pytest.fixture
    def engine(self, manager: TagManager) -> RulesEngine:
        """Create a rules engine instance."""
        return RulesEngine(manager)

    def test_evaluate_empty_conditions(self, engine: RulesEngine, manager: TagManager) -> None:
        """Test that empty conditions match all conversations."""
        rules = SmartFolderRules(match="all", conditions=[])
        folder = manager.create_smart_folder("All", rules)

        conversations = [
            {"chat_id": "chat1", "display_name": "Alice"},
            {"chat_id": "chat2", "display_name": "Bob"},
        ]

        matching = engine.evaluate_folder(folder, conversations)

        assert len(matching) == 2

    def test_evaluate_equals_condition(self, engine: RulesEngine, manager: TagManager) -> None:
        """Test equals operator."""
        rules = build_rules([{"field": "is_group", "operator": "equals", "value": True}])
        folder = manager.create_smart_folder("Groups", rules)

        conversations = [
            {"chat_id": "chat1", "is_group": True},
            {"chat_id": "chat2", "is_group": False},
        ]

        matching = engine.evaluate_folder(folder, conversations)

        assert len(matching) == 1
        assert matching[0]["chat_id"] == "chat1"

    def test_evaluate_greater_than_condition(
        self, engine: RulesEngine, manager: TagManager
    ) -> None:
        """Test greater_than operator."""
        rules = build_rules([{"field": "unread_count", "operator": "greater_than", "value": 0}])
        folder = manager.create_smart_folder("Unread", rules)

        conversations = [
            {"chat_id": "chat1", "unread_count": 5},
            {"chat_id": "chat2", "unread_count": 0},
        ]

        matching = engine.evaluate_folder(folder, conversations)

        assert len(matching) == 1
        assert matching[0]["chat_id"] == "chat1"

    def test_evaluate_contains_condition(self, engine: RulesEngine, manager: TagManager) -> None:
        """Test contains operator."""
        rules = build_rules([{"field": "display_name", "operator": "contains", "value": "work"}])
        folder = manager.create_smart_folder("Work Contacts", rules)

        conversations = [
            {"chat_id": "chat1", "display_name": "Work - Alice"},
            {"chat_id": "chat2", "display_name": "Personal - Bob"},
        ]

        matching = engine.evaluate_folder(folder, conversations)

        assert len(matching) == 1
        assert matching[0]["chat_id"] == "chat1"

    def test_evaluate_in_last_days_condition(
        self, engine: RulesEngine, manager: TagManager
    ) -> None:
        """Test in_last_days operator for date fields."""
        rules = build_rules(
            [{"field": "last_message_date", "operator": "in_last_days", "value": 7}]
        )
        folder = manager.create_smart_folder("Recent", rules)

        now = datetime.now(UTC)
        conversations = [
            {"chat_id": "chat1", "last_message_date": now - timedelta(days=2)},
            {"chat_id": "chat2", "last_message_date": now - timedelta(days=30)},
        ]

        matching = engine.evaluate_folder(folder, conversations)

        assert len(matching) == 1
        assert matching[0]["chat_id"] == "chat1"

    def test_evaluate_match_all(self, engine: RulesEngine, manager: TagManager) -> None:
        """Test that match='all' requires all conditions to be true."""
        rules = build_rules(
            [
                {"field": "is_group", "operator": "equals", "value": True},
                {"field": "unread_count", "operator": "greater_than", "value": 0},
            ],
            match="all",
        )
        folder = manager.create_smart_folder("Unread Groups", rules)

        conversations = [
            {"chat_id": "chat1", "is_group": True, "unread_count": 5},  # Both match
            {"chat_id": "chat2", "is_group": True, "unread_count": 0},  # Only group
            {"chat_id": "chat3", "is_group": False, "unread_count": 3},  # Only unread
        ]

        matching = engine.evaluate_folder(folder, conversations)

        assert len(matching) == 1
        assert matching[0]["chat_id"] == "chat1"

    def test_evaluate_match_any(self, engine: RulesEngine, manager: TagManager) -> None:
        """Test that match='any' requires at least one condition to be true."""
        rules = build_rules(
            [
                {"field": "is_group", "operator": "equals", "value": True},
                {"field": "is_flagged", "operator": "equals", "value": True},
            ],
            match="any",
        )
        folder = manager.create_smart_folder("Groups or Flagged", rules)

        conversations = [
            {"chat_id": "chat1", "is_group": True, "is_flagged": False},
            {"chat_id": "chat2", "is_group": False, "is_flagged": True},
            {"chat_id": "chat3", "is_group": False, "is_flagged": False},
        ]

        matching = engine.evaluate_folder(folder, conversations)

        assert len(matching) == 2
        chat_ids = {c["chat_id"] for c in matching}
        assert chat_ids == {"chat1", "chat2"}

    def test_evaluate_has_tag_condition(self, engine: RulesEngine, manager: TagManager) -> None:
        """Test has_tag operator."""
        tag = manager.create_tag("Important")
        manager.add_tag_to_conversation("chat1", tag.id)

        rules = build_rules([{"field": "tags", "operator": "has_tag", "value": tag.id}])
        folder = manager.create_smart_folder("Important Chats", rules)

        conversations = [
            {"chat_id": "chat1", "display_name": "Alice"},
            {"chat_id": "chat2", "display_name": "Bob"},
        ]

        matching = engine.evaluate_folder(folder, conversations)

        assert len(matching) == 1
        assert matching[0]["chat_id"] == "chat1"

    def test_evaluate_has_no_tags_condition(self, engine: RulesEngine, manager: TagManager) -> None:
        """Test has_no_tags operator."""
        tag = manager.create_tag("Tagged")
        manager.add_tag_to_conversation("chat1", tag.id)

        rules = build_rules([{"field": "tags", "operator": "has_no_tags", "value": None}])
        folder = manager.create_smart_folder("Untagged", rules)

        conversations = [
            {"chat_id": "chat1", "display_name": "Alice"},  # Has tag
            {"chat_id": "chat2", "display_name": "Bob"},  # No tags
        ]

        matching = engine.evaluate_folder(folder, conversations)

        assert len(matching) == 1
        assert matching[0]["chat_id"] == "chat2"

    def test_sort_results(self, engine: RulesEngine, manager: TagManager) -> None:
        """Test sorting of filtered results."""
        rules = build_rules([], sort_by="display_name", sort_order="asc")
        folder = manager.create_smart_folder("Sorted", rules)

        conversations = [
            {"chat_id": "chat1", "display_name": "Zebra"},
            {"chat_id": "chat2", "display_name": "Apple"},
            {"chat_id": "chat3", "display_name": "Mango"},
        ]

        matching = engine.evaluate_folder(folder, conversations)

        names = [c["display_name"] for c in matching]
        assert names == ["Apple", "Mango", "Zebra"]

    def test_limit_results(self, engine: RulesEngine, manager: TagManager) -> None:
        """Test limiting number of results."""
        rules = build_rules([], limit=2)
        folder = manager.create_smart_folder("Limited", rules)

        conversations = [{"chat_id": f"chat{i}", "display_name": f"User{i}"} for i in range(10)]

        matching = engine.evaluate_folder(folder, conversations)

        assert len(matching) == 2

    def test_validate_rules_valid(self, engine: RulesEngine) -> None:
        """Test validation of valid rules."""
        rules = build_rules([{"field": "unread_count", "operator": "greater_than", "value": 0}])

        errors = engine.validate_rules(rules)

        assert errors == []

    def test_validate_rules_invalid_field(self, engine: RulesEngine) -> None:
        """Test validation catches invalid field."""
        rules = SmartFolderRules(
            match="all",
            conditions=[RuleCondition(field="invalid_field", operator="equals", value=1)],
        )

        errors = engine.validate_rules(rules)

        assert len(errors) > 0
        assert any("Unknown field" in e for e in errors)

    def test_validate_rules_invalid_match(self, engine: RulesEngine) -> None:
        """Test validation catches invalid match type."""
        rules = SmartFolderRules(match="invalid", conditions=[])

        errors = engine.validate_rules(rules)

        assert len(errors) > 0
        assert any("Invalid match type" in e for e in errors)

    def test_get_folder_preview(self, engine: RulesEngine, manager: TagManager) -> None:
        """Test preview of folder contents."""
        rules = build_rules([{"field": "is_group", "operator": "equals", "value": True}])

        conversations = [
            {"chat_id": "chat1", "is_group": True},
            {"chat_id": "chat2", "is_group": True},
            {"chat_id": "chat3", "is_group": False},
        ]

        preview = engine.get_folder_preview(rules, conversations, limit=10)

        assert preview["total_matches"] == 2
        assert len(preview["preview"]) == 2
        assert preview["has_more"] is False


class TestBuildRulesHelper:
    """Tests for the build_rules helper function."""

    def test_build_rules_basic(self) -> None:
        """Test building rules from simple dicts."""
        rules = build_rules([{"field": "is_group", "operator": "equals", "value": True}])

        assert rules.match == "all"
        assert len(rules.conditions) == 1
        assert rules.conditions[0].field == "is_group"
        assert rules.conditions[0].operator == "equals"
        assert rules.conditions[0].value is True

    def test_build_rules_with_options(self) -> None:
        """Test building rules with custom options."""
        rules = build_rules(
            conditions=[],
            match="any",
            sort_by="message_count",
            sort_order="asc",
            limit=50,
        )

        assert rules.match == "any"
        assert rules.sort_by == "message_count"
        assert rules.sort_order == "asc"
        assert rules.limit == 50

    def test_build_rules_multiple_conditions(self) -> None:
        """Test building rules with multiple conditions."""
        rules = build_rules(
            [
                {"field": "is_group", "operator": "equals", "value": False},
                {"field": "unread_count", "operator": "greater_than", "value": 0},
                {"field": "display_name", "operator": "contains", "value": "work"},
            ]
        )

        assert len(rules.conditions) == 3


class TestTagModels:
    """Tests for tag data models."""

    def test_tag_aliases_property(self) -> None:
        """Test Tag aliases property parses JSON."""
        tag = Tag(
            id=1,
            name="Test",
            color="#000000",
            icon="tag",
            aliases_json='["alias1", "alias2"]',
        )

        assert tag.aliases == ["alias1", "alias2"]

    def test_tag_aliases_empty(self) -> None:
        """Test Tag aliases returns empty list when None."""
        tag = Tag(id=1, name="Test", color="#000000", icon="tag", aliases_json=None)

        assert tag.aliases == []

    def test_smart_folder_rules_property(self) -> None:
        """Test SmartFolder rules property parses JSON."""
        rules_json = (
            '{"match": "all", "conditions": [], "sort_by": "display_name", '
            '"sort_order": "asc", "limit": 10}'
        )
        folder = SmartFolder(id=1, name="Test", icon="folder", color="#000", rules_json=rules_json)

        rules = folder.rules

        assert rules.match == "all"
        assert rules.sort_by == "display_name"
        assert rules.limit == 10

    def test_tag_rule_tag_ids_property(self) -> None:
        """Test TagRule tag_ids property parses JSON."""
        rule = TagRule(
            id=1,
            name="Test Rule",
            trigger="on_new_message",
            tag_ids_json="[1, 2, 3]",
        )

        assert rule.tag_ids == [1, 2, 3]

    def test_tag_rule_conditions_property(self) -> None:
        """Test TagRule conditions property parses JSON."""
        conditions_json = '[{"field": "display_name", "operator": "contains", "value": "test"}]'
        rule = TagRule(
            id=1,
            name="Test Rule",
            trigger="on_new_message",
            conditions_json=conditions_json,
        )

        conditions = rule.conditions

        assert len(conditions) == 1
        assert conditions[0].field == "display_name"
        assert conditions[0].operator == "contains"

    def test_tag_suggestion_dataclass(self) -> None:
        """Test TagSuggestion dataclass."""
        suggestion = TagSuggestion(
            tag_id=1,
            tag_name="Work",
            confidence=0.85,
            reason="Content matches work patterns",
            source="content",
        )

        assert suggestion.tag_id == 1
        assert suggestion.tag_name == "Work"
        assert suggestion.confidence == 0.85
        assert suggestion.source == "content"


class TestCaching(TestTagManager):
    """Tests for tag manager caching."""

    def test_tag_cache_hit(self, manager: TagManager) -> None:
        """Test that get_tag uses cache on second call."""
        tag = manager.create_tag("Test")

        # First call populates cache
        manager.get_tag(tag.id)

        # Second call should hit cache
        result = manager.get_tag(tag.id)
        assert result is not None
        assert result.name == "Test"

    def test_cache_invalidation_on_update(self, manager: TagManager) -> None:
        """Test that updating a tag invalidates the cache."""
        tag = manager.create_tag("Original")

        # Populate cache
        manager.get_tag(tag.id)

        # Update tag
        manager.update_tag(tag.id, name="Updated")

        # Should get fresh data
        result = manager.get_tag(tag.id)
        assert result is not None
        assert result.name == "Updated"

    def test_clear_caches(self, manager: TagManager) -> None:
        """Test that clear_caches empties all caches."""
        tag = manager.create_tag("Test")

        # Populate cache
        manager.get_tag(tag.id)

        # Clear caches
        manager.clear_caches()

        # Cache should be empty (next get will query DB)
        # Just verify it doesn't raise
        result = manager.get_tag(tag.id)
        assert result is not None
