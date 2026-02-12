"""Tags and Smart Folders API endpoints.

Provides comprehensive tag management for conversation organization:
- Tag CRUD operations
- Conversation tagging (single and bulk)
- Smart folder management
- Auto-tagging rules
- Tag suggestions

All endpoints are under /tags prefix.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_imessage_reader
from api.schemas.tags import (
    BulkTagRequest,
    BulkTagResponse,
    ConversationTagRequest,
    ConversationTagResponse,
    ConversationTagsResponse,
    RuleConditionSchema,
    SmartFolderCreate,
    SmartFolderListResponse,
    SmartFolderPreviewRequest,
    SmartFolderPreviewResponse,
    SmartFolderResponse,
    SmartFolderRulesSchema,
    SmartFolderUpdate,
    SuggestionFeedbackRequest,
    TagCreate,
    TagListResponse,
    TagResponse,
    TagRuleCreate,
    TagRuleListResponse,
    TagRuleResponse,
    TagRuleUpdate,
    TagStatisticsResponse,
    TagSuggestionResponse,
    TagSuggestionsRequest,
    TagSuggestionsResponse,
    TagUpdate,
    TagWithPath,
)
from integrations.imessage import ChatDBReader
from jarvis.tags import (
    AutoTagger,
    RuleCondition,
    RulesEngine,
    SmartFolderRules,
    TagManager,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tags", tags=["tags"])

# =============================================================================
# Singleton TagManager
# =============================================================================

_tag_manager: TagManager | None = None
_tag_manager_lock = threading.Lock()


def get_tag_manager() -> TagManager:
    """Get or create the singleton TagManager.

    Returns:
        TagManager instance with initialized schema.
    """
    global _tag_manager

    if _tag_manager is None:
        with _tag_manager_lock:
            if _tag_manager is None:
                _tag_manager = TagManager()
                _tag_manager.init_schema()

    return _tag_manager


def get_auto_tagger() -> AutoTagger:
    """Get an AutoTagger instance."""
    return AutoTagger(get_tag_manager())


def get_rules_engine() -> RulesEngine:
    """Get a RulesEngine instance."""
    return RulesEngine(get_tag_manager())


# =============================================================================
# Tag Endpoints
# =============================================================================


@router.get(
    "",
    response_model=TagListResponse,
    summary="List tags",
    description="Get all tags, optionally filtered by parent.",
)
async def list_tags(
    parent_id: int | None = Query(
        default=None,
        description="Filter by parent tag ID. Use -1 to get all tags including nested.",
    ),
    include_system: bool = Query(
        default=True,
        description="Include system-generated tags",
    ),
) -> TagListResponse:
    """List all tags with optional filtering."""
    manager = get_tag_manager()

    # Run blocking DB queries in executor to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    if parent_id == -1:
        tags = await loop.run_in_executor(None, manager.get_all_tags)
    else:
        tags = await loop.run_in_executor(
            None,
            functools.partial(
                manager.list_tags, parent_id=parent_id, include_system=include_system
            ),
        )

    return TagListResponse(
        tags=[_tag_to_response(t) for t in tags],
        total=len(tags),
    )


@router.post(
    "",
    response_model=TagResponse,
    summary="Create tag",
    description="Create a new tag.",
)
def create_tag(request: TagCreate) -> TagResponse:
    """Create a new tag."""
    manager = get_tag_manager()

    try:
        tag = manager.create_tag(
            name=request.name,
            color=request.color,
            icon=request.icon,
            parent_id=request.parent_id,
            description=request.description,
            aliases=request.aliases,
        )
        return _tag_to_response(tag)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get(
    "/{tag_id}",
    response_model=TagWithPath,
    summary="Get tag",
    description="Get a tag by ID with its hierarchical path.",
)
def get_tag(tag_id: int) -> TagWithPath:
    """Get a tag by ID."""
    manager = get_tag_manager()
    tag = manager.get_tag(tag_id)

    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")

    path = manager.get_tag_path(tag_id)
    response = _tag_to_response(tag)

    return TagWithPath(
        **response.model_dump(),
        path=path,
    )


@router.patch(
    "/{tag_id}",
    response_model=TagResponse,
    summary="Update tag",
    description="Update a tag's properties.",
)
def update_tag(tag_id: int, request: TagUpdate) -> TagResponse:
    """Update a tag."""
    manager = get_tag_manager()

    # Convert -1 to None for parent_id (root level)
    parent_id = request.parent_id
    if parent_id == -1:
        parent_id = None

    tag = manager.update_tag(
        tag_id=tag_id,
        name=request.name,
        color=request.color,
        icon=request.icon,
        parent_id=parent_id,
        description=request.description,
        aliases=request.aliases,
        sort_order=request.sort_order,
    )

    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")

    return _tag_to_response(tag)


@router.delete(
    "/{tag_id}",
    summary="Delete tag",
    description="Delete a tag and remove it from all conversations.",
)
def delete_tag(tag_id: int) -> dict[str, str]:
    """Delete a tag."""
    manager = get_tag_manager()

    if not manager.delete_tag(tag_id):
        raise HTTPException(status_code=404, detail="Tag not found")

    return {"status": "deleted"}


@router.get(
    "/search",
    response_model=TagListResponse,
    summary="Search tags",
    description="Search tags by name or alias.",
)
def search_tags(
    q: str = Query(..., description="Search query", min_length=1),
    limit: int = Query(default=10, ge=1, le=50),
) -> TagListResponse:
    """Search tags by name or alias."""
    manager = get_tag_manager()
    tags = manager.search_tags(q, limit=limit)

    return TagListResponse(
        tags=[_tag_to_response(t) for t in tags],
        total=len(tags),
    )


# =============================================================================
# Conversation Tagging Endpoints
# =============================================================================


@router.get(
    "/conversations/{chat_id}",
    response_model=ConversationTagsResponse,
    summary="Get conversation tags",
    description="Get all tags assigned to a conversation.",
)
def get_conversation_tags(chat_id: str) -> ConversationTagsResponse:
    """Get all tags for a conversation."""
    manager = get_tag_manager()
    tag_assignments = manager.get_tags_for_conversation(chat_id)

    return ConversationTagsResponse(
        chat_id=chat_id,
        tags=[
            ConversationTagResponse(
                chat_id=chat_id,
                tag=_tag_to_response(tag),
                added_at=conv_tag.added_at,
                added_by=conv_tag.added_by,
                confidence=conv_tag.confidence,
            )
            for tag, conv_tag in tag_assignments
        ],
    )


@router.post(
    "/conversations/{chat_id}",
    response_model=ConversationTagResponse,
    summary="Add tag to conversation",
    description="Add a tag to a conversation.",
)
def add_tag_to_conversation(
    chat_id: str,
    request: ConversationTagRequest,
) -> ConversationTagResponse:
    """Add a tag to a conversation."""
    manager = get_tag_manager()

    tag = manager.get_tag(request.tag_id)
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")

    manager.add_tag_to_conversation(chat_id, request.tag_id)

    # Get the assignment details
    tag_assignments = manager.get_tags_for_conversation(chat_id)
    for t, conv_tag in tag_assignments:
        if t.id == request.tag_id:
            return ConversationTagResponse(
                chat_id=chat_id,
                tag=_tag_to_response(t),
                added_at=conv_tag.added_at,
                added_by=conv_tag.added_by,
                confidence=conv_tag.confidence,
            )

    raise HTTPException(status_code=500, detail="Failed to add tag")


@router.delete(
    "/conversations/{chat_id}/{tag_id}",
    summary="Remove tag from conversation",
    description="Remove a tag from a conversation.",
)
def remove_tag_from_conversation(chat_id: str, tag_id: int) -> dict[str, str]:
    """Remove a tag from a conversation."""
    manager = get_tag_manager()

    if not manager.remove_tag_from_conversation(chat_id, tag_id):
        raise HTTPException(status_code=404, detail="Tag not assigned to conversation")

    return {"status": "removed"}


@router.put(
    "/conversations/{chat_id}",
    response_model=ConversationTagsResponse,
    summary="Set conversation tags",
    description="Replace all tags on a conversation with the specified set.",
)
def set_conversation_tags(
    chat_id: str,
    tag_ids: list[int],
) -> ConversationTagsResponse:
    """Set the exact tags for a conversation (replaces existing)."""
    manager = get_tag_manager()

    # Validate all tag IDs exist
    for tag_id in tag_ids:
        if not manager.get_tag(tag_id):
            raise HTTPException(status_code=404, detail=f"Tag {tag_id} not found")

    manager.set_conversation_tags(chat_id, tag_ids)

    return get_conversation_tags(chat_id)


@router.post(
    "/bulk/add",
    response_model=BulkTagResponse,
    summary="Bulk add tags",
    description="Add tags to multiple conversations at once.",
)
def bulk_add_tags(request: BulkTagRequest) -> BulkTagResponse:
    """Add tags to multiple conversations."""
    manager = get_tag_manager()

    count = manager.bulk_add_tags(
        chat_ids=request.chat_ids,
        tag_ids=request.tag_ids,
    )

    return BulkTagResponse(
        affected_count=count,
        chat_ids=request.chat_ids,
    )


@router.post(
    "/bulk/remove",
    response_model=BulkTagResponse,
    summary="Bulk remove tags",
    description="Remove tags from multiple conversations at once.",
)
def bulk_remove_tags(request: BulkTagRequest) -> BulkTagResponse:
    """Remove tags from multiple conversations."""
    manager = get_tag_manager()

    count = manager.bulk_remove_tags(
        chat_ids=request.chat_ids,
        tag_ids=request.tag_ids,
    )

    return BulkTagResponse(
        affected_count=count,
        chat_ids=request.chat_ids,
    )


# =============================================================================
# Smart Folder Endpoints
# =============================================================================


@router.get(
    "/folders",
    response_model=SmartFolderListResponse,
    summary="List smart folders",
    description="Get all smart folders.",
)
def list_smart_folders(
    include_defaults: bool = Query(default=True, description="Include default system folders"),
) -> SmartFolderListResponse:
    """List all smart folders."""
    manager = get_tag_manager()
    folders = manager.list_smart_folders(include_defaults=include_defaults)

    return SmartFolderListResponse(
        folders=[_folder_to_response(f) for f in folders],
        total=len(folders),
    )


@router.post(
    "/folders",
    response_model=SmartFolderResponse,
    summary="Create smart folder",
    description="Create a new smart folder with rules.",
)
def create_smart_folder(request: SmartFolderCreate) -> SmartFolderResponse:
    """Create a new smart folder."""
    manager = get_tag_manager()

    # Validate rules
    rules = _schema_to_rules(request.rules)
    engine = get_rules_engine()
    errors = engine.validate_rules(rules)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    folder = manager.create_smart_folder(
        name=request.name,
        rules=rules,
        icon=request.icon,
        color=request.color,
    )

    return _folder_to_response(folder)


@router.get(
    "/folders/{folder_id}",
    response_model=SmartFolderResponse,
    summary="Get smart folder",
    description="Get a smart folder by ID.",
)
def get_smart_folder(folder_id: int) -> SmartFolderResponse:
    """Get a smart folder by ID."""
    manager = get_tag_manager()
    folder = manager.get_smart_folder(folder_id)

    if not folder:
        raise HTTPException(status_code=404, detail="Smart folder not found")

    return _folder_to_response(folder)


@router.patch(
    "/folders/{folder_id}",
    response_model=SmartFolderResponse,
    summary="Update smart folder",
    description="Update a smart folder's properties.",
)
def update_smart_folder(folder_id: int, request: SmartFolderUpdate) -> SmartFolderResponse:
    """Update a smart folder."""
    manager = get_tag_manager()

    rules = None
    if request.rules:
        rules = _schema_to_rules(request.rules)
        engine = get_rules_engine()
        errors = engine.validate_rules(rules)
        if errors:
            raise HTTPException(status_code=400, detail={"errors": errors})

    folder = manager.update_smart_folder(
        folder_id=folder_id,
        name=request.name,
        rules=rules,
        icon=request.icon,
        color=request.color,
        sort_order=request.sort_order,
    )

    if not folder:
        raise HTTPException(status_code=404, detail="Smart folder not found")

    return _folder_to_response(folder)


@router.delete(
    "/folders/{folder_id}",
    summary="Delete smart folder",
    description="Delete a smart folder. Default folders cannot be deleted.",
)
def delete_smart_folder(folder_id: int) -> dict[str, str]:
    """Delete a smart folder."""
    manager = get_tag_manager()

    if not manager.delete_smart_folder(folder_id):
        raise HTTPException(
            status_code=400,
            detail="Smart folder not found or is a default folder",
        )

    return {"status": "deleted"}


@router.get(
    "/folders/{folder_id}/conversations",
    summary="Get folder conversations",
    description="Get conversations that match a smart folder's rules.",
)
def get_folder_conversations(
    folder_id: int,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> dict[str, Any]:
    """Get conversations matching a smart folder."""
    manager = get_tag_manager()
    folder = manager.get_smart_folder(folder_id)

    if not folder:
        raise HTTPException(status_code=404, detail="Smart folder not found")

    # Get all conversations from iMessage
    conversations = reader.get_conversations(limit=1000)  # Get all for filtering

    # Convert to dicts for the rules engine
    conv_dicts = [_conversation_to_dict(c) for c in conversations]

    # Evaluate folder rules
    engine = get_rules_engine()
    matching = engine.evaluate_folder(folder, conv_dicts)

    # Apply pagination
    total = len(matching)
    paginated = matching[offset : offset + limit]

    return {
        "folder_id": folder_id,
        "conversations": paginated,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.post(
    "/folders/preview",
    response_model=SmartFolderPreviewResponse,
    summary="Preview smart folder rules",
    description="Preview what conversations would match given rules without saving.",
)
def preview_smart_folder_rules(
    request: SmartFolderPreviewRequest,
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> SmartFolderPreviewResponse:
    """Preview smart folder rules."""
    rules = _schema_to_rules(request.rules)

    # Validate rules
    engine = get_rules_engine()
    errors = engine.validate_rules(rules)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})
    conversations = reader.get_conversations(limit=500)
    conv_dicts = [_conversation_to_dict(c) for c in conversations]

    # Preview
    result = engine.get_folder_preview(rules, conv_dicts, limit=request.limit)

    return SmartFolderPreviewResponse(**result)


# =============================================================================
# Tag Rule Endpoints
# =============================================================================


@router.get(
    "/rules",
    response_model=TagRuleListResponse,
    summary="List tag rules",
    description="Get all auto-tagging rules.",
)
def list_tag_rules(
    trigger: str | None = Query(default=None, description="Filter by trigger type"),
    enabled_only: bool = Query(default=False, description="Only return enabled rules"),
) -> TagRuleListResponse:
    """List all tag rules."""
    manager = get_tag_manager()
    rules = manager.list_tag_rules(trigger=trigger, enabled_only=enabled_only)

    return TagRuleListResponse(
        rules=[_rule_to_response(r) for r in rules],
        total=len(rules),
    )


@router.post(
    "/rules",
    response_model=TagRuleResponse,
    summary="Create tag rule",
    description="Create a new auto-tagging rule.",
)
def create_tag_rule(request: TagRuleCreate) -> TagRuleResponse:
    """Create a new tag rule."""
    manager = get_tag_manager()

    from jarvis.tags import TagRule

    rule = TagRule(
        name=request.name,
        trigger=request.trigger,
        priority=request.priority,
        is_enabled=request.is_enabled,
    )
    rule.conditions = [
        RuleCondition(
            field=c.field,
            operator=c.operator,
            value=c.value,
        )
        for c in request.conditions
    ]
    rule.tag_ids = request.tag_ids

    created = manager.create_tag_rule(rule)
    return _rule_to_response(created)


@router.get(
    "/rules/{rule_id}",
    response_model=TagRuleResponse,
    summary="Get tag rule",
    description="Get a tag rule by ID.",
)
def get_tag_rule(rule_id: int) -> TagRuleResponse:
    """Get a tag rule by ID."""
    manager = get_tag_manager()
    rule = manager.get_tag_rule(rule_id)

    if not rule:
        raise HTTPException(status_code=404, detail="Tag rule not found")

    return _rule_to_response(rule)


@router.patch(
    "/rules/{rule_id}",
    response_model=TagRuleResponse,
    summary="Update tag rule",
    description="Update a tag rule's properties.",
)
def update_tag_rule(rule_id: int, request: TagRuleUpdate) -> TagRuleResponse:
    """Update a tag rule."""
    manager = get_tag_manager()
    rule = manager.get_tag_rule(rule_id)

    if not rule:
        raise HTTPException(status_code=404, detail="Tag rule not found")

    # Update fields
    if request.name is not None:
        rule.name = request.name
    if request.trigger is not None:
        rule.trigger = request.trigger
    if request.conditions is not None:
        rule.conditions = [
            RuleCondition(field=c.field, operator=c.operator, value=c.value)
            for c in request.conditions
        ]
    if request.tag_ids is not None:
        rule.tag_ids = request.tag_ids
    if request.priority is not None:
        rule.priority = request.priority
    if request.is_enabled is not None:
        rule.is_enabled = request.is_enabled

    updated = manager.update_tag_rule(rule)
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update rule")

    return _rule_to_response(updated)


@router.delete(
    "/rules/{rule_id}",
    summary="Delete tag rule",
    description="Delete a tag rule.",
)
def delete_tag_rule(rule_id: int) -> dict[str, str]:
    """Delete a tag rule."""
    manager = get_tag_manager()

    if not manager.delete_tag_rule(rule_id):
        raise HTTPException(status_code=404, detail="Tag rule not found")

    return {"status": "deleted"}


# =============================================================================
# Tag Suggestions Endpoints
# =============================================================================


@router.post(
    "/suggestions",
    response_model=TagSuggestionsResponse,
    summary="Get tag suggestions",
    description="Get AI-powered tag suggestions for a conversation.",
)
def get_tag_suggestions(
    request: TagSuggestionsRequest,
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> TagSuggestionsResponse:
    """Get tag suggestions for a conversation."""
    # Get messages for the conversation
    messages = reader.get_messages(request.chat_id, limit=50)

    # Convert to dicts
    message_dicts = [
        {
            "text": m.text,
            "is_from_me": m.is_from_me,
            "date": m.date,
        }
        for m in messages
    ]

    # Get contact name via direct lookup (avoids N+1)
    conv = reader.get_conversation(request.chat_id)
    contact_name = conv.display_name if conv else None

    # Get suggestions
    auto_tagger = get_auto_tagger()
    suggestions = auto_tagger.suggest_tags(
        chat_id=request.chat_id,
        messages=message_dicts,
        contact_name=contact_name,
        limit=request.limit,
    )

    return TagSuggestionsResponse(
        chat_id=request.chat_id,
        suggestions=[
            TagSuggestionResponse(
                tag_id=s.tag_id,
                tag_name=s.tag_name,
                confidence=s.confidence,
                reason=s.reason,
                source=s.source,
            )
            for s in suggestions
        ],
    )


@router.post(
    "/suggestions/feedback",
    summary="Record suggestion feedback",
    description="Record whether a tag suggestion was accepted or rejected for learning.",
)
def record_suggestion_feedback(request: SuggestionFeedbackRequest) -> dict[str, str]:
    """Record feedback on a tag suggestion."""
    auto_tagger = get_auto_tagger()
    auto_tagger.record_suggestion_feedback(
        chat_id=request.chat_id,
        tag_id=request.tag_id,
        accepted=request.accepted,
    )

    return {"status": "recorded"}


# =============================================================================
# Statistics Endpoints
# =============================================================================


@router.get(
    "/statistics",
    response_model=TagStatisticsResponse,
    summary="Get tag statistics",
    description="Get overall statistics about tag usage.",
)
def get_tag_statistics() -> TagStatisticsResponse:
    """Get tag usage statistics."""
    manager = get_tag_manager()
    stats = manager.get_tag_statistics()

    return TagStatisticsResponse(**stats)


# =============================================================================
# Helper Functions
# =============================================================================


def _tag_to_response(tag: Any) -> TagResponse:
    """Convert a Tag model to TagResponse."""
    return TagResponse(
        id=tag.id,
        name=tag.name,
        color=tag.color,
        icon=tag.icon,
        description=tag.description,
        parent_id=tag.parent_id,
        aliases=tag.aliases,
        sort_order=tag.sort_order,
        is_system=tag.is_system,
        created_at=tag.created_at,
        updated_at=tag.updated_at,
    )


def _folder_to_response(folder: Any) -> SmartFolderResponse:
    """Convert a SmartFolder model to SmartFolderResponse."""
    rules = folder.rules
    return SmartFolderResponse(
        id=folder.id,
        name=folder.name,
        icon=folder.icon,
        color=folder.color,
        rules=SmartFolderRulesSchema(
            match=rules.match,
            conditions=[
                RuleConditionSchema(
                    field=c.field,
                    operator=c.operator,
                    value=c.value,
                )
                for c in rules.conditions
            ],
            sort_by=rules.sort_by,
            sort_order=rules.sort_order,
            limit=rules.limit,
        ),
        sort_order=folder.sort_order,
        is_default=folder.is_default,
        created_at=folder.created_at,
        updated_at=folder.updated_at,
    )


def _rule_to_response(rule: Any) -> TagRuleResponse:
    """Convert a TagRule model to TagRuleResponse."""
    return TagRuleResponse(
        id=rule.id,
        name=rule.name,
        trigger=rule.trigger,
        conditions=[
            RuleConditionSchema(
                field=c.field,
                operator=c.operator,
                value=c.value,
            )
            for c in rule.conditions
        ],
        tag_ids=rule.tag_ids,
        priority=rule.priority,
        is_enabled=rule.is_enabled,
        created_at=rule.created_at,
        last_triggered_at=rule.last_triggered_at,
        trigger_count=rule.trigger_count,
    )


def _schema_to_rules(schema: SmartFolderRulesSchema) -> SmartFolderRules:
    """Convert SmartFolderRulesSchema to SmartFolderRules model."""
    return SmartFolderRules(
        match=schema.match,
        conditions=[
            RuleCondition(
                field=c.field,
                operator=c.operator,
                value=c.value,
            )
            for c in schema.conditions
        ],
        sort_by=schema.sort_by,
        sort_order=schema.sort_order,
        limit=schema.limit,
    )


def _conversation_to_dict(conv: Any) -> dict[str, Any]:
    """Convert a Conversation object to a dict for rules evaluation."""
    return {
        "chat_id": conv.chat_id,
        "display_name": conv.display_name,
        "last_message_date": conv.last_message_date.isoformat() if conv.last_message_date else None,
        "message_count": conv.message_count,
        "is_group": conv.is_group,
        "last_message_text": getattr(conv, "last_message_text", None),
        "unread_count": getattr(conv, "unread_count", 0),
        "is_flagged": getattr(conv, "is_flagged", False),
        "relationship": getattr(conv, "relationship", None),
        "contact_name": conv.display_name,
    }
