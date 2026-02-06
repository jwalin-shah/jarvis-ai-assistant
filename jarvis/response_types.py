"""Response option types and trigger->response mappings.

This module intentionally contains ONLY the lightweight enums/constants that are
used across the runtime (retrieval, multi-option generation, API schemas).

Historically these lived in `jarvis.response_classifier_v2`, but that module was
an implementation detail of a legacy classifier pipeline. The *labels* remain
stable because they are persisted in the DB (`pairs.response_da_type`) and used
by retrieval for filtering.
"""

from __future__ import annotations

from enum import Enum


class ResponseType(str, Enum):
    """Dialogue-act-ish response option labels stored in the DB.

    Note: These labels are *not* the same as `jarvis.classifiers.ResponseType`
    (which describes response pressure categories). These are the discrete
    response options like AGREE/DECLINE/DEFER used by retrieval and UI.
    """

    # Commitment options
    AGREE = "AGREE"
    DECLINE = "DECLINE"
    DEFER = "DEFER"

    # Question / information
    ANSWER = "ANSWER"
    YES = "YES"
    NO = "NO"
    QUESTION = "QUESTION"

    # Statements / acknowledgments
    STATEMENT = "STATEMENT"
    ACKNOWLEDGE = "ACKNOWLEDGE"

    # Reactions
    REACT_POSITIVE = "REACT_POSITIVE"
    REACT_SYMPATHY = "REACT_SYMPATHY"

    # Social
    GREETING = "GREETING"


# Common group used by multi-option generation.
COMMITMENT_RESPONSE_TYPES: tuple[ResponseType, ...] = (
    ResponseType.AGREE,
    ResponseType.DECLINE,
    ResponseType.DEFER,
)


# Mapping from trigger DA label -> valid response option types.
# Trigger labels are the values of `jarvis.trigger_classifier.TriggerType`.
TRIGGER_TO_VALID_RESPONSES: dict[str | None, list[ResponseType]] = {
    # New trigger classifier labels
    "commitment": list(COMMITMENT_RESPONSE_TYPES),
    "question": [ResponseType.ANSWER, ResponseType.YES, ResponseType.NO],
    "reaction": [
        ResponseType.REACT_POSITIVE,
        ResponseType.REACT_SYMPATHY,
        ResponseType.ACKNOWLEDGE,
    ],
    "social": [ResponseType.GREETING, ResponseType.ACKNOWLEDGE],
    "statement": [ResponseType.ACKNOWLEDGE, ResponseType.STATEMENT],
    "unknown": [ResponseType.ACKNOWLEDGE],
    None: list(COMMITMENT_RESPONSE_TYPES),
    # Legacy DA labels that may still exist in older DB rows
    "INVITATION": list(COMMITMENT_RESPONSE_TYPES),
    "REQUEST": list(COMMITMENT_RESPONSE_TYPES),
    "YN_QUESTION": [ResponseType.YES, ResponseType.NO],
    "WH_QUESTION": [ResponseType.ANSWER],
    "INFO_STATEMENT": [ResponseType.ACKNOWLEDGE, ResponseType.STATEMENT],
}


__all__ = [
    "ResponseType",
    "COMMITMENT_RESPONSE_TYPES",
    "TRIGGER_TO_VALID_RESPONSES",
]
