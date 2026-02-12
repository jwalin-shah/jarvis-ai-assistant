"""Shared evaluation utilities for entity span matching.

Common functions used across evaluation and dataset-building scripts
(run_extractor_bakeoff, eval_gliner_candidates, build_fact_filter_dataset).
"""

from __future__ import annotations

# Label aliasing: generic GLiNER predictions -> fine-grained gold labels.
# E.g. GLiNER predicts "place" but goldset has "current_location".
DEFAULT_LABEL_ALIASES: dict[str, set[str]] = {
    "place": {"current_location", "future_location", "past_location", "place", "hometown"},
    "org": {"employer", "org", "school"},
    "person_name": {"friend_name", "partner_name", "person_name", "family_member"},
    "family_member": {"family_member", "person_name"},
    "health_condition": {"allergy", "health_condition", "dietary"},
    "activity": {"activity", "hobby"},
    "job_role": {"job_role", "job_title"},
    "food_item": {"food_item", "food_like", "food_dislike"},
    "email": {"email"},
    "phone_number": {"phone_number", "phone"},
}


def jaccard_tokens(a: str, b: str) -> float:
    """Token-level Jaccard similarity (case-insensitive)."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def spans_match(
    pred_text: str,
    pred_label: str,
    gold_text: str,
    gold_label: str,
    label_aliases: dict[str, set[str]] | None = None,
) -> bool:
    """Check if a predicted span matches a gold span.

    Match criteria:
      - span_label matches (exact, or via label_aliases if provided)
      - AND text overlap: Jaccard token >= 0.5 OR substring containment (case-insensitive)

    Args:
        pred_text: Predicted span text.
        pred_label: Predicted span label.
        gold_text: Gold span text.
        gold_label: Gold span label.
        label_aliases: Optional mapping from predicted label to set of acceptable gold labels.
            When None, requires exact label match.
    """
    if label_aliases is not None:
        allowed = label_aliases.get(pred_label, {pred_label})
        if gold_label not in allowed:
            return False
    else:
        if pred_label != gold_label:
            return False

    pl = pred_text.lower().strip()
    gl = gold_text.lower().strip()
    if pl in gl or gl in pl:
        return True
    if jaccard_tokens(pred_text, gold_text) >= 0.5:
        return True
    return False
