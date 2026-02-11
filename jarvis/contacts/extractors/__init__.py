"""Extractor adapters for fact candidate extraction.

This module provides a common interface for multiple extraction backends:
- GLiNER (baseline)
- GLiNER2 (improved architecture)
- NuExtract (schema-driven LLM-based)
- spaCy (high-precision standard NER)
- Ensemble (spaCy + GLiNER hybrid)

Usage:
    from jarvis.contacts.extractors import create_extractor, list_extractors

    # List available extractors
    print(list_extractors())  # ['ensemble', 'gliner', 'gliner2', 'nuextract', 'spacy']

    # Create an extractor
    extractor = create_extractor("gliner", config={"threshold": 0.35})

    # Extract from a message
    candidates = extractor.extract_from_text(
        text="I live in Austin and work at Google",
        message_id=123,
    )

    # Batch extraction
    results = extractor.extract_batch(messages, batch_size=32)
"""

from __future__ import annotations

# Import and register adapters
# This triggers registration of all built-in adapters
from jarvis.contacts.extractors import (
    ensemble_adapter,
    gliner2_adapter,
    gliner_adapter,
    lfm_adapter,
    nuextract_adapter,
    regex_adapter,
    spacy_adapter,
)

# Import base classes
from jarvis.contacts.extractors.base import (
    ExtractedCandidate,
    ExtractionResult,
    ExtractorAdapter,
    create_extractor,
    get_extractor_class,
    list_extractors,
    register_extractor,
)

__all__ = [
    # Base classes
    "ExtractedCandidate",
    "ExtractionResult",
    "ExtractorAdapter",
    # Factory functions
    "create_extractor",
    "get_extractor_class",
    "list_extractors",
    "register_extractor",
    # Adapter classes (for direct use)
    "GLiNERAdapter",
    "GLiNER2Adapter",
    "LFMAdapter",
    "NuExtractAdapter",
    "SpaCyAdapter",
    "EnsembleAdapter",
    "RegexAdapter",
]

# Re-export adapter classes for direct instantiation
GLiNERAdapter = gliner_adapter.GLiNERAdapter
GLiNER2Adapter = gliner2_adapter.GLiNER2Adapter
LFMAdapter = lfm_adapter.LFMAdapter
NuExtractAdapter = nuextract_adapter.NuExtractAdapter
SpaCyAdapter = spacy_adapter.SpaCyAdapter
EnsembleAdapter = ensemble_adapter.EnsembleAdapter
RegexAdapter = regex_adapter.RegexAdapter
