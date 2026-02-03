"""Tag and SmartFolder system for conversation organization.

This module provides:
- Tag: Hierarchical tags with colors and icons
- SmartFolder: Rule-based dynamic folders
- TagManager: CRUD operations for tags
- AutoTagger: ML-based tag suggestions
- RulesEngine: Smart folder rule evaluation

Usage:
    from jarvis.tags import TagManager, Tag, SmartFolder

    manager = TagManager()
    tag = manager.create_tag("Work", color="#0066cc")
    manager.add_tag_to_conversation("chat123", tag.id)
"""

from jarvis.tags.auto_tagger import AutoTagger, ContentAnalysis
from jarvis.tags.manager import TagManager
from jarvis.tags.models import (
    AutoTagTrigger,
    ConversationTag,
    RuleCondition,
    RuleField,
    RuleOperator,
    SmartFolder,
    SmartFolderRules,
    Tag,
    TagColor,
    TagIcon,
    TagRule,
    TagSuggestion,
)
from jarvis.tags.rules import RulesEngine, build_rules

__all__ = [
    # Enums
    "AutoTagTrigger",
    "RuleField",
    "RuleOperator",
    "TagColor",
    "TagIcon",
    # Models
    "ConversationTag",
    "ContentAnalysis",
    "RuleCondition",
    "SmartFolder",
    "SmartFolderRules",
    "Tag",
    "TagRule",
    "TagSuggestion",
    # Managers
    "AutoTagger",
    "RulesEngine",
    "TagManager",
    # Helpers
    "build_rules",
]
