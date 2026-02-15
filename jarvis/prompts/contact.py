from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.contacts.contact_profile import Fact


def format_facts_for_prompt(facts: list[Fact], max_facts: int = 10) -> str:
    """Format contact facts compactly for prompt injection."""
    if not facts:
        return ""

    qualified = [f for f in facts if f.confidence >= 0.5][:max_facts]
    if not qualified:
        return ""

    parts: list[str] = []
    for fact in qualified:
        if fact.predicate == "note" and len(fact.value) > 100:
            continue
        if "maximizing shareholder value" in fact.value.lower():
            continue

        pred = fact.predicate.replace("_", " ")
        subject = fact.subject

        if subject == "Contact":
            subject = "They"
        elif subject == "Jwalin":
            subject = "You"

        if fact.predicate == "note":
            entry = f"- {fact.value}"
        elif fact.predicate in ["lives_in", "lived_in", "from"]:
            entry = f"- {subject} lives in {fact.value}"
        elif fact.predicate == "moving_to":
            entry = f"- {subject} is moving to {fact.value}"
        elif fact.predicate in ["works_at", "worked_at"]:
            entry = f"- {subject} works at {fact.value}"
        elif fact.predicate == "job_title":
            entry = f"- {subject} is a {fact.value}"
        elif fact.predicate == "studies_at":
            entry = f"- {subject} studies at {fact.value}"
        elif fact.predicate in ["likes", "enjoys"]:
            entry = f"- {subject} likes {fact.value}"
        elif fact.predicate == "dislikes":
            entry = f"- {subject} dislikes {fact.value}"
        elif fact.predicate == "has_condition":
            entry = f"- {subject} has {fact.value}"
        elif fact.predicate == "allergic_to":
            entry = f"- {subject} is allergic to {fact.value}"
        elif fact.predicate == "is_family_of":
            entry = f"- {subject} is related to {fact.value}"
        elif fact.predicate == "is_friend_of":
            entry = f"- {subject} is friends with {fact.value}"
        elif fact.predicate == "is_partner_of":
            entry = f"- {subject} is dating {fact.value}"
        elif fact.predicate == "birthday_is":
            entry = f"- {subject}'s birthday is {fact.value}"
        elif fact.predicate == "age_is":
            entry = f"- {subject} is {fact.value}"
        elif fact.predicate == "has_pet":
            entry = f"- {subject} has a {fact.value}"
        else:
            entry = f"- {subject} {pred} {fact.value}"

        parts.append(entry)

    return "\n".join(parts) if parts else ""
