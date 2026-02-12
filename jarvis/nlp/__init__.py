"""NLP utilities - text processing for iMessage normalization.

Submodules:
    coref_resolver: Coreference resolution via FastCoref
    patterns: Shared regex patterns and word sets
    slang: Slang/abbreviation expansion
    ner_client: Named entity recognition via Unix socket
    validity_gate: Three-layer validation for candidate exchanges
"""

from jarvis.nlp.coref_resolver import CorefResolver, get_coref_resolver, reset_coref_resolver
from jarvis.nlp.ner_client import (
    Entity,
    extract_locations,
    extract_organizations,
    extract_person_names,
    get_entities,
    get_entities_batch,
    get_pid,
    is_service_running,
)
from jarvis.nlp.slang import SLANG_MAP, expand_slang, get_slang_map
from jarvis.nlp.validity_gate import (
    GateConfig,
    GateResult,
    ValidityGate,
    load_nli_model,
)

__all__ = [
    "CorefResolver",
    "Entity",
    "GateConfig",
    "GateResult",
    "SLANG_MAP",
    "ValidityGate",
    "expand_slang",
    "extract_locations",
    "extract_organizations",
    "extract_person_names",
    "get_coref_resolver",
    "get_entities",
    "get_entities_batch",
    "get_pid",
    "get_slang_map",
    "is_service_running",
    "load_nli_model",
    "reset_coref_resolver",
]
