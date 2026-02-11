"""Adapter to convert regex-based Fact objects to span dicts for bakeoff evaluation.

Maps (category, predicate) -> (span_label, fact_type) so the regex FactExtractor
can be evaluated against the same goldset as GLiNER.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.contacts.contact_profile import Fact

# Food-related words for disambiguating preference spans (food_item vs activity).
# Intentionally small: only unambiguous food terms. Default to "activity" when unsure.
FOOD_WORDS = frozenset(
    {
        "sushi",
        "pizza",
        "pasta",
        "tacos",
        "ramen",
        "burgers",
        "steak",
        "chicken",
        "salmon",
        "rice",
        "noodles",
        "bread",
        "cheese",
        "chocolate",
        "ice cream",
        "coffee",
        "tea",
        "beer",
        "wine",
        "curry",
        "soup",
        "salad",
        "fries",
        "bacon",
        "eggs",
        "fruit",
        "cake",
        "pie",
        "cookies",
        "donuts",
        "seafood",
        "shrimp",
        "lobster",
        "crab",
        "pho",
        "dim sum",
        "bbq",
        "barbecue",
        "cilantro",
        "avocado",
        "tofu",
        "hummus",
        "boba",
        "matcha",
        "smoothie",
        "burrito",
        "quesadilla",
        "sandwich",
        "wings",
        "ribs",
        "pork",
        "lamb",
        "vegetables",
        "veggies",
        "broccoli",
        "spinach",
        "kale",
        "mushrooms",
        "garlic",
        "onions",
        "peppers",
        "tomatoes",
        "potatoes",
        "corn",
        "mango",
        "banana",
        "strawberry",
        "blueberry",
        "apple",
        "orange",
        "peach",
        "watermelon",
        "grapes",
        "pineapple",
        "coconut",
        "pancakes",
        "waffles",
        "oatmeal",
        "cereal",
        "yogurt",
        "food",
        "meal",
        "snack",
        "dessert",
        "breakfast",
        "lunch",
        "dinner",
    }
)

# Static mapping: (category, predicate) -> (span_label, fact_type)
# Covers all regex extractor output combinations.
_CATEGORY_PREDICATE_MAP: dict[tuple[str, str], tuple[str, str]] = {
    # Relationships
    ("relationship", "is_family_of"): ("family_member", "relationship.family"),
    ("relationship", "is_friend_of"): ("person_name", "relationship.friend"),
    ("relationship", "is_associated_with"): ("person_name", "relationship.friend"),
    ("relationship", "mentioned_person"): ("person_name", "relationship.friend"),
    # Locations
    ("location", "lives_in"): ("place", "location.current"),
    ("location", "moving_to"): ("place", "location.future"),
    ("location", "lived_in"): ("place", "location.past"),
    ("location", "mentioned_location"): ("place", "location.current"),
    # Work
    ("work", "works_at"): ("org", "work.employer"),
    ("work", "mentioned_org"): ("org", "work.employer"),
    # Preferences (span_label depends on subject content - resolved at runtime)
    # See _preference_span_label() below.
}


def _is_food_subject(subject: str) -> bool:
    """Check if subject looks like a food item."""
    tokens = set(subject.lower().split())
    return bool(tokens & FOOD_WORDS)


def _preference_span_label(subject: str, predicate: str) -> tuple[str, str]:
    """Resolve preference span_label based on subject content."""
    if _is_food_subject(subject):
        if predicate == "dislikes":
            return "food_item", "preference.food_dislike"
        return "food_item", "preference.food_like"
    # Default: activity
    return "activity", "preference.activity"


def fact_to_span(fact: Fact, source_text: str = "") -> dict | None:
    """Convert a single Fact to a span dict compatible with bakeoff evaluation.

    Returns None if the fact category/predicate combination is unmapped.
    """
    key = (fact.category, fact.predicate)

    if fact.category == "preference":
        span_label, fact_type = _preference_span_label(fact.subject, fact.predicate)
    elif key in _CATEGORY_PREDICATE_MAP:
        span_label, fact_type = _CATEGORY_PREDICATE_MAP[key]
    else:
        return None

    return {
        "span_text": fact.subject,
        "span_label": span_label,
        "fact_type": fact_type,
        "score": fact.confidence,
    }


def facts_to_spans(facts: list[Fact], source_text: str = "") -> list[dict]:
    """Convert a list of Facts to span dicts, dropping unmapped facts."""
    spans = []
    for f in facts:
        span = fact_to_span(f, source_text)
        if span is not None:
            spans.append(span)
    return spans
