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
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def get_graph_context(contact_id: str, chat_id: str) -> str:
    """Extract graph-based context relevant to this contact.

    Combines relationship facts with interaction recency to produce
    a compact context string for prompt injection.

    Args:
        contact_id: Contact identifier (usually same as chat_id).
        chat_id: Chat identifier for message stats lookup.

    Returns:
        Compact relationship context string, or empty string if unavailable.
    """
    parts: list[str] = []

    # 1. Get relationship facts for this contact
    try:
        relationship_desc = _get_relationship_description(contact_id)
        if relationship_desc:
            parts.append(relationship_desc)
    except Exception as e:
        logger.debug("Failed to get relationship facts: %s", e)

    # 2. Get interaction recency
    try:
        recency = _get_interaction_recency(chat_id)
        if recency:
            parts.append(recency)
    except Exception as e:
        logger.debug("Failed to get interaction recency: %s", e)

    # 3. Get shared connections (from linked contacts)
    try:
        connections = _get_shared_connections(contact_id)
        if connections:
            parts.append(connections)
    except Exception as e:
        logger.debug("Failed to get shared connections: %s", e)

    return " ".join(parts)


def _get_relationship_description(contact_id: str) -> str:
    """Build relationship description from contact facts.

    Looks for relationship-type facts (is_family_of, is_friend_of, etc.)
    and formats them as natural language.
    """
    from jarvis.db import get_db

    db = get_db()

    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT subject, predicate, value
            FROM contact_facts
            WHERE contact_id = ? AND category = 'relationship'
            ORDER BY confidence DESC
            LIMIT 5
            """,
            (contact_id,),
        ).fetchall()

    if not rows:
        return ""

    # Map predicates to natural language
    predicate_templates = {
        "is_family_of": "{subject} is your {value}" if "{value}" else "{subject} is family",
        "is_friend_of": "{subject} is your friend",
        "is_partner_of": "{subject} is your partner",
        "is_coworker_of": "{subject} is your coworker",
    }

    descriptions: list[str] = []
    for row in rows:
        predicate = row["predicate"]
        subject = row["subject"]
        value = row["value"] or ""

        if predicate in predicate_templates:
            if value:
                desc = f"{subject} is your {value}"
            else:
                template = predicate_templates[predicate]
                desc = template.format(subject=subject, value=value)
            descriptions.append(desc)

    if not descriptions:
        return ""

    return ". ".join(descriptions) + "."


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

        now = datetime.now(timezone.utc)
        msg_time = last_msg.date
        if msg_time.tzinfo is None:
            msg_time = msg_time.replace(tzinfo=timezone.utc)

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
