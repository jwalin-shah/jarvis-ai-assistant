"""Smart folder rules engine for dynamic conversation filtering.

Evaluates smart folder rules against conversations to determine
which conversations belong in each smart folder.

Usage:
    from jarvis.tags.rules import RulesEngine

    engine = RulesEngine(tag_manager)
    chat_ids = engine.evaluate_folder(smart_folder, conversations)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from jarvis.tags.models import (
    RuleCondition,
    RuleField,
    RuleOperator,
    SmartFolder,
    SmartFolderRules,
)

if TYPE_CHECKING:
    from jarvis.tags.manager import TagManager

logger = logging.getLogger(__name__)


class RulesEngine:
    """Engine for evaluating smart folder rules against conversations.

    Provides efficient filtering of conversations based on rule conditions
    with support for caching frequently accessed smart folders.
    """

    def __init__(self, tag_manager: TagManager) -> None:
        """Initialize rules engine.

        Args:
            tag_manager: TagManager instance for tag lookups.
        """
        self.tag_manager = tag_manager
        # Note: caching removed as it was never populated or used

    def evaluate_folder(
        self,
        folder: SmartFolder,
        conversations: list[dict[str, Any]],
        use_cache: bool = True,  # noqa: ARG002 - kept for API compat
    ) -> list[dict[str, Any]]:
        """Evaluate a smart folder's rules against conversations.

        Args:
            folder: The smart folder to evaluate.
            conversations: List of conversation dicts to filter.
            use_cache: Whether to use cached results.

        Returns:
            List of conversations that match the folder's rules.
        """
        if not conversations:
            return []

        rules = folder.rules

        # Filter conversations
        matching = []
        for conv in conversations:
            if self._matches_rules(conv, rules):
                matching.append(conv)

        # Sort results
        matching = self._sort_results(matching, rules)

        # Apply limit
        if rules.limit > 0:
            matching = matching[: rules.limit]

        return matching

    def evaluate_folder_chat_ids(
        self,
        folder: SmartFolder,
        conversations: list[dict[str, Any]],
    ) -> list[str]:
        """Evaluate a folder and return just the chat IDs.

        Args:
            folder: The smart folder to evaluate.
            conversations: List of conversation dicts to filter.

        Returns:
            List of chat_ids that match the folder's rules.
        """
        matching = self.evaluate_folder(folder, conversations, use_cache=False)
        return [c.get("chat_id", "") for c in matching if c.get("chat_id")]

    def _matches_rules(
        self,
        conversation: dict[str, Any],
        rules: SmartFolderRules,
    ) -> bool:
        """Check if a conversation matches the folder rules."""
        if not rules.conditions:
            # No conditions means all conversations match
            return True

        results = []
        for condition in rules.conditions:
            result = self._evaluate_condition(conversation, condition)
            results.append(result)

        if rules.match == "all":
            return all(results)
        else:  # "any"
            return any(results)

    def _evaluate_condition(
        self,
        conversation: dict[str, Any],
        condition: RuleCondition,
    ) -> bool:
        """Evaluate a single condition against a conversation."""
        field = condition.field
        operator = condition.operator
        value = condition.value

        # Get the field value from conversation
        field_value = self._get_field_value(conversation, field)

        # Handle tag-specific operators
        if operator in (
            RuleOperator.HAS_TAG.value,
            RuleOperator.HAS_ANY_TAG.value,
            RuleOperator.HAS_ALL_TAGS.value,
            RuleOperator.HAS_NO_TAGS.value,
        ):
            return self._evaluate_tag_condition(conversation, operator, value)

        # Evaluate based on operator
        return self._compare_values(field_value, operator, value)

    def _get_field_value(
        self,
        conversation: dict[str, Any],
        field: str,
    ) -> Any:
        """Get the value of a field from a conversation dict."""
        # Direct field access
        if field in conversation:
            return conversation[field]

        # Handle nested/computed fields
        field_mappings = {
            RuleField.CHAT_ID.value: "chat_id",
            RuleField.DISPLAY_NAME.value: "display_name",
            RuleField.LAST_MESSAGE_DATE.value: "last_message_date",
            RuleField.MESSAGE_COUNT.value: "message_count",
            RuleField.IS_GROUP.value: "is_group",
            RuleField.UNREAD_COUNT.value: "unread_count",
            RuleField.IS_FLAGGED.value: "is_flagged",
            RuleField.RELATIONSHIP.value: "relationship",
            RuleField.CONTACT_NAME.value: "contact_name",
            RuleField.LAST_MESSAGE_TEXT.value: "last_message_text",
            RuleField.HAS_ATTACHMENTS.value: "has_attachments",
            RuleField.SENTIMENT.value: "sentiment",
            RuleField.PRIORITY.value: "priority",
            RuleField.NEEDS_RESPONSE.value: "needs_response",
        }

        mapped_field = field_mappings.get(field, field)
        return conversation.get(mapped_field)

    def _compare_values(
        self,
        field_value: Any,
        operator: str,
        rule_value: Any,
    ) -> bool:
        """Compare a field value against a rule value using the operator."""
        # Handle None field values
        if field_value is None:
            if operator == RuleOperator.IS_EMPTY.value:
                return True
            if operator == RuleOperator.IS_NOT_EMPTY.value:
                return False
            return False

        # String comparisons (case-insensitive)
        if isinstance(field_value, str):
            field_lower = field_value.lower()
            rule_lower = str(rule_value).lower() if rule_value else ""

            if operator == RuleOperator.EQUALS.value:
                return field_lower == rule_lower
            elif operator == RuleOperator.NOT_EQUALS.value:
                return field_lower != rule_lower
            elif operator == RuleOperator.CONTAINS.value:
                return rule_lower in field_lower
            elif operator == RuleOperator.NOT_CONTAINS.value:
                return rule_lower not in field_lower
            elif operator == RuleOperator.STARTS_WITH.value:
                return field_lower.startswith(rule_lower)
            elif operator == RuleOperator.ENDS_WITH.value:
                return field_lower.endswith(rule_lower)
            elif operator == RuleOperator.IS_EMPTY.value:
                return len(field_value.strip()) == 0
            elif operator == RuleOperator.IS_NOT_EMPTY.value:
                return len(field_value.strip()) > 0

        # Boolean comparisons (must be before int/float since bool is subclass of int)
        if isinstance(field_value, bool):
            if operator == RuleOperator.EQUALS.value:
                return field_value == bool(rule_value)
            elif operator == RuleOperator.NOT_EQUALS.value:
                return field_value != bool(rule_value)
            return False

        # Numeric comparisons
        if isinstance(field_value, (int, float)):
            try:
                rule_num = float(rule_value)
                if operator == RuleOperator.EQUALS.value:
                    return field_value == rule_num
                elif operator == RuleOperator.NOT_EQUALS.value:
                    return field_value != rule_num
                elif operator == RuleOperator.GREATER_THAN.value:
                    return field_value > rule_num
                elif operator == RuleOperator.LESS_THAN.value:
                    return field_value < rule_num
            except (TypeError, ValueError):
                return False

        # Date/time comparisons
        if isinstance(field_value, datetime):
            return self._compare_dates(field_value, operator, rule_value)

        # Try parsing field_value as datetime string
        if isinstance(field_value, str):
            try:
                parsed_date = datetime.fromisoformat(field_value.replace("Z", "+00:00"))
                return self._compare_dates(parsed_date, operator, rule_value)
            except ValueError:
                pass

        return False

    def _compare_dates(
        self,
        field_date: datetime,
        operator: str,
        rule_value: Any,
    ) -> bool:
        """Compare date field against rule value."""
        # Ensure timezone awareness
        if field_date.tzinfo is None:
            field_date = field_date.replace(tzinfo=UTC)

        now = datetime.now(UTC)

        if operator == RuleOperator.IN_LAST_DAYS.value:
            try:
                days = int(rule_value)
                cutoff = now - timedelta(days=days)
                return field_date >= cutoff
            except (TypeError, ValueError):
                return False

        elif operator == RuleOperator.BEFORE.value:
            try:
                if isinstance(rule_value, str):
                    rule_date = datetime.fromisoformat(rule_value.replace("Z", "+00:00"))
                else:
                    rule_date = rule_value
                if rule_date.tzinfo is None:
                    rule_date = rule_date.replace(tzinfo=UTC)
                return field_date < rule_date
            except (TypeError, ValueError):
                return False

        elif operator == RuleOperator.AFTER.value:
            try:
                if isinstance(rule_value, str):
                    rule_date = datetime.fromisoformat(rule_value.replace("Z", "+00:00"))
                else:
                    rule_date = rule_value
                if rule_date.tzinfo is None:
                    rule_date = rule_date.replace(tzinfo=UTC)
                return field_date > rule_date
            except (TypeError, ValueError):
                return False

        return False

    def _evaluate_tag_condition(
        self,
        conversation: dict[str, Any],
        operator: str,
        value: Any,
    ) -> bool:
        """Evaluate tag-specific conditions."""
        chat_id = conversation.get("chat_id")
        if not chat_id:
            return False

        # Get conversation's tag IDs
        conv_tags = self.tag_manager.get_tags_for_conversation(chat_id)
        conv_tag_ids = {t.id for t, _ in conv_tags}

        if operator == RuleOperator.HAS_TAG.value:
            # value is a single tag ID
            return int(value) in conv_tag_ids

        elif operator == RuleOperator.HAS_ANY_TAG.value:
            # value is a list of tag IDs
            rule_tag_ids = set(int(v) for v in value) if isinstance(value, list) else {int(value)}
            return bool(conv_tag_ids & rule_tag_ids)

        elif operator == RuleOperator.HAS_ALL_TAGS.value:
            # value is a list of tag IDs
            rule_tag_ids = set(int(v) for v in value) if isinstance(value, list) else {int(value)}
            return rule_tag_ids <= conv_tag_ids

        elif operator == RuleOperator.HAS_NO_TAGS.value:
            return len(conv_tag_ids) == 0

        return False

    def _sort_results(
        self,
        conversations: list[dict[str, Any]],
        rules: SmartFolderRules,
    ) -> list[dict[str, Any]]:
        """Sort conversations according to folder rules."""
        sort_field = rules.sort_by
        reverse = rules.sort_order == "desc"

        def get_sort_key(conv: dict[str, Any]) -> Any:
            value = conv.get(sort_field)
            if value is None:
                # Put None values at the end
                return (1, "")
            if isinstance(value, str):
                # Try parsing as datetime
                try:
                    return (0, datetime.fromisoformat(value.replace("Z", "+00:00")))
                except ValueError:
                    return (0, value.lower())
            return (0, value)

        return sorted(conversations, key=get_sort_key, reverse=reverse)

    def get_folder_preview(
        self,
        rules: SmartFolderRules,
        conversations: list[dict[str, Any]],
        limit: int = 10,
    ) -> dict[str, Any]:
        """Get a preview of what a folder would contain with given rules.

        Useful for previewing rules before saving a smart folder.

        Args:
            rules: The rules to evaluate.
            conversations: All conversations to filter.
            limit: Maximum number of preview results.

        Returns:
            Dictionary with preview results and stats.
        """
        # Create temporary folder
        temp_folder = SmartFolder(name="_preview")
        temp_folder.rules = rules

        matching = self.evaluate_folder(temp_folder, conversations, use_cache=False)
        total = len(matching)

        return {
            "total_matches": total,
            "preview": matching[:limit],
            "has_more": total > limit,
        }

    def validate_rules(self, rules: SmartFolderRules) -> list[str]:
        """Validate smart folder rules and return any errors.

        Args:
            rules: The rules to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        # Validate match type
        if rules.match not in ("all", "any"):
            errors.append(f"Invalid match type: {rules.match}. Must be 'all' or 'any'.")

        # Validate each condition
        valid_fields = {f.value for f in RuleField}
        valid_operators = {o.value for o in RuleOperator}

        for i, condition in enumerate(rules.conditions):
            if condition.field not in valid_fields:
                errors.append(f"Condition {i + 1}: Unknown field '{condition.field}'")

            if condition.operator not in valid_operators:
                errors.append(f"Condition {i + 1}: Unknown operator '{condition.operator}'")

            # Validate operator/field combinations
            if condition.operator in (
                RuleOperator.HAS_TAG.value,
                RuleOperator.HAS_ANY_TAG.value,
                RuleOperator.HAS_ALL_TAGS.value,
                RuleOperator.HAS_NO_TAGS.value,
            ):
                if condition.field != RuleField.TAGS.value:
                    errors.append(
                        f"Condition {i + 1}: Tag operators can only be used with 'tags' field"
                    )

            # Validate value requirements
            if condition.operator not in (
                RuleOperator.IS_EMPTY.value,
                RuleOperator.IS_NOT_EMPTY.value,
                RuleOperator.HAS_NO_TAGS.value,
            ):
                if condition.value is None:
                    errors.append(f"Condition {i + 1}: Value is required for operator")

        # Validate sort field
        if rules.sort_by and rules.sort_by not in valid_fields:
            # Allow common computed fields too
            allowed_sort = valid_fields | {"last_message_date", "message_count", "display_name"}
            if rules.sort_by not in allowed_sort:
                errors.append(f"Invalid sort field: {rules.sort_by}")

        # Validate sort order
        if rules.sort_order not in ("asc", "desc"):
            errors.append(f"Invalid sort order: {rules.sort_order}. Must be 'asc' or 'desc'.")

        # Validate limit
        if rules.limit < 0:
            errors.append("Limit must be non-negative")

        return errors


def build_rules(
    conditions: list[dict[str, Any]],
    match: str = "all",
    sort_by: str = "last_message_date",
    sort_order: str = "desc",
    limit: int = 0,
) -> SmartFolderRules:
    """Helper to build SmartFolderRules from simple dictionaries.

    Args:
        conditions: List of condition dicts with 'field', 'operator', 'value'.
        match: "all" or "any".
        sort_by: Field to sort by.
        sort_order: "asc" or "desc".
        limit: Maximum results (0 for unlimited).

    Returns:
        SmartFolderRules instance.

    Example:
        rules = build_rules([
            {"field": "unread_count", "operator": "greater_than", "value": 0},
            {"field": "is_group", "operator": "equals", "value": False}
        ], match="all")
    """
    rule_conditions = [
        RuleCondition(
            field=c["field"],
            operator=c["operator"],
            value=c.get("value"),
        )
        for c in conditions
    ]

    return SmartFolderRules(
        match=match,
        conditions=rule_conditions,
        sort_by=sort_by,
        sort_order=sort_order,
        limit=limit,
    )


# Export all public symbols
__all__ = ["RulesEngine", "build_rules"]
