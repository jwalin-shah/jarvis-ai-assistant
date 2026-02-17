"""Graph-based context extraction for reply generation.

Queries contact facts and interaction data to build relationship context
injected into the generation prompt's <relationships> section.

Usage:
    from jarvis.graph.context import get_graph_context

    context = get_graph_context(contact_id="chat123", chat_id="chat123")
    # "Sarah is your sister. Last messaged 2 days ago."
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from jarvis.infrastructure.cache import TTLCache

logger = logging.getLogger(__name__)

# Cache graph context for 60 seconds to avoid repeated slow queries
_context_cache = TTLCache(maxsize=128, ttl_seconds=60.0)


def get_graph_context(contact_id: str, chat_id: str) -> str:
    """Extract graph-based context relevant to this contact.

    Combines relationship facts, knowledge facts, interaction recency,
    and shared connections into a compact context string for prompt injection.

    Args:
        contact_id: Contact identifier (usually same as chat_id).
        chat_id: Chat identifier for message stats lookup.

    Returns:
        Compact context string, or empty string if unavailable.
    """
    cache_key = f"{contact_id}:{chat_id}"
    cached = _context_cache.get(cache_key)
    if cached is not None:
        return str(cached)

    parts: list[str] = []

    # Identify if this is a group chat
    is_group = "chat" in contact_id.lower() or "chat" in chat_id.lower()

    # 1. Get all fact summaries (relationships, location, work, preferences, etc.)
    # Skip for group chats as facts are per-person
    if not is_group:
        try:
            fact_summary = _get_fact_summary(contact_id)
            if fact_summary:
                parts.append(fact_summary)
        except Exception as e:
            logger.debug("Failed to get fact summary: %s", e)

    # 2. Get interaction recency
    try:
        recency = _get_interaction_recency(chat_id)
        if recency:
            parts.append(recency)
    except Exception as e:
        logger.debug("Failed to get interaction recency: %s", e)

    # 3. Get shared connections (from linked contacts)
    # Skip for group chats
    if not is_group:
        try:
            connections = _get_shared_connections(contact_id)
            if connections:
                parts.append(connections)
        except Exception as e:
            logger.debug("Failed to get shared connections: %s", e)

    result = " ".join(parts)
    _context_cache.set(cache_key, result)
    return result


def _get_fact_summary(contact_id: str) -> str:
    """Build a comprehensive fact summary across all categories.

    Queries all fact types (relationship, location, work, preference, health,
    personal) and formats them as grouped natural language descriptions.
    """
    from jarvis.db import get_db

    db = get_db()

    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT category, subject, predicate, value
            FROM contact_facts
            WHERE contact_id = ?
            ORDER BY confidence DESC
            LIMIT 15
            """,
            (contact_id,),
        ).fetchall()

    if not rows:
        return ""

    # Map predicates to natural language templates
    predicate_templates: dict[str, str] = {
        # Relationship
        "is_family_of": "{subject} is your {value}",
        "is_friend_of": "{subject} is your friend",
        "is_partner_of": "{subject} is your partner",
        "is_coworker_of": "{subject} is your coworker",
        "is_associated_with": "{subject} is your {value}",
        # Location
        "lives_in": "They live in {subject}",
        "lived_in": "They used to live in {subject}",
        "moving_to": "They're moving to {subject}",
        "from": "They're from {subject}",
        # Work
        "works_at": "They work at {subject}",
        "worked_at": "They used to work at {subject}",
        "job_title": "Their job title is {subject}",
        # Preference
        "likes": "They like {subject}",
        "dislikes": "They dislike {subject}",
        "likes_food": "They like {subject}",
        "dislikes_food": "They dislike {subject}",
        "enjoys": "They enjoy {subject}",
        # Health
        "allergic_to": "They're allergic to {subject}",
        "has_condition": "They have {subject}",
        "dietary": "They have a dietary restriction: {subject}",
        # Personal
        "attends": "They attend {subject}",
        "birthday_is": "Their birthday is {subject}",
        "has_pet": "They have a pet named {subject}",
    }

    # Group descriptions by category for readability
    by_category: dict[str, list[str]] = {}
    for row in rows:
        predicate = row["predicate"]
        subject = row["subject"]
        value = row["value"] or ""
        category = row["category"]

        template = predicate_templates.get(predicate)
        if template:
            # Relationship predicates use value for the role type
            if predicate in ("is_family_of", "is_associated_with") and value:
                desc = f"{subject} is your {value}"
            else:
                desc = template.format(subject=subject, value=value)
            by_category.setdefault(category, []).append(desc)

    if not by_category:
        return ""

    # Format: "Relationship: X. Y. Location: Z."
    parts: list[str] = []
    # Order categories for consistent output
    category_order = ["relationship", "location", "work", "preference", "health", "personal"]
    category_labels = {
        "relationship": "Relationship",
        "location": "Location",
        "work": "Work",
        "preference": "Preferences",
        "health": "Health",
        "personal": "Personal",
    }

    for cat in category_order:
        descs = by_category.get(cat)
        if descs:
            label = category_labels.get(cat, cat.title())
            parts.append(f"{label}: {'. '.join(descs)}.")

    # Include any categories not in the predefined order
    for cat, descs in by_category.items():
        if cat not in category_order:
            parts.append(f"{cat.title()}: {'. '.join(descs)}.")

    return " ".join(parts)


def _get_interaction_recency(chat_id: str) -> str:
    """Get human-readable interaction recency.

    Queries the most recent message timestamp for this chat from iMessage DB.
    """
    try:
        from integrations.imessage import ChatDBReader

        reader = ChatDBReader()
        messages = reader.get_messages(chat_id=chat_id, limit=1)
        if not messages:
            return ""

        last_msg = messages[0]
        if not hasattr(last_msg, "date") or last_msg.date is None:
            return ""

        now = datetime.now(UTC)
        msg_time = last_msg.date
        if msg_time.tzinfo is None:
            msg_time = msg_time.replace(tzinfo=UTC)

        delta = now - msg_time
        days = delta.days

        if days == 0:
            hours = delta.seconds // 3600
            if hours == 0:
                return "Last messaged just now."
            elif hours == 1:
                return "Last messaged 1 hour ago."
            else:
                return f"Last messaged {hours} hours ago."
        elif days == 1:
            return "Last messaged yesterday."
        elif days < 7:
            return f"Last messaged {days} days ago."
        elif days < 30:
            weeks = days // 7
            return f"Last messaged {weeks} week{'s' if weeks > 1 else ''} ago."
        else:
            months = days // 30
            return f"Last messaged {months} month{'s' if months > 1 else ''} ago."
    except Exception as e:
        logger.debug("Could not determine interaction recency: %s", e)
        return ""


def _get_shared_connections(contact_id: str) -> str:
    """Find shared connections via linked_contact_id in facts.

    Looks for facts that reference other contacts (e.g., "Sarah's friend John").
    """
    from jarvis.db import get_db

    db = get_db()

    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT cf.subject, cf.predicate, c.display_name
            FROM contact_facts cf
            JOIN contacts c ON cf.linked_contact_id = c.chat_id
            WHERE cf.contact_id = ?
            AND cf.linked_contact_id IS NOT NULL
            AND cf.linked_contact_id != ''
            LIMIT 5
            """,
            (contact_id,),
        ).fetchall()

    if not rows:
        return ""

    connections = [row["display_name"] for row in rows if row["display_name"]]
    if not connections:
        return ""

    unique = list(dict.fromkeys(connections))  # Dedupe preserving order
    if len(unique) == 1:
        return f"Connected to: {unique[0]}."
    return f"Connected to: {', '.join(unique)}."
