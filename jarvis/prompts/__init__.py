"""Prompt templates and builders for iMessage reply generation.

Provides well-engineered prompts optimized for small local LLMs (Qwen2.5-0.5B/1.5B)
with clear structure, few-shot examples, and tone-aware generation.

This module is the SINGLE SOURCE OF TRUTH for all prompts in the JARVIS system.
Import prompts from here, not from other modules.

Submodules:
    examples   - Few-shot example data and metadata
    templates  - Prompt template definitions
    tone       - Tone detection and style analysis
    builders   - Core prompt construction functions
    rag        - RAG-enhanced prompt builders
    categories - Category config, mapping, and utilities
    optimized  - DSPy-optimized per-category programs
    registry   - Centralized prompt management
"""

# Re-export everything for backward compatibility.
# All imports that previously worked as `from jarvis.prompts import X`
# continue to work after the package conversion.

# --- examples ---
# --- builders ---
from jarvis.prompts.builders import (
    build_reply_prompt,
    build_search_answer_prompt,
    build_summary_prompt,
    build_threaded_reply_prompt,
    format_facts_for_prompt,
    get_thread_max_tokens,
)

# --- categories ---
from jarvis.prompts.categories import (
    ACKNOWLEDGE_TEMPLATES,
    API_REPLY_EXAMPLES,
    API_REPLY_EXAMPLES_METADATA,
    API_SUMMARY_EXAMPLES,
    API_SUMMARY_EXAMPLES_METADATA,
    CATEGORY_CONFIGS,
    CATEGORY_MAP,
    CHAT_SYSTEM_PROMPT,
    CLOSING_TEMPLATES,
    REPLY_EXAMPLES,
    SUMMARY_EXAMPLES,
    CategoryConfig,
    estimate_tokens,
    get_category_config,
    is_within_token_limit,
    resolve_category,
)
from jarvis.prompts.examples import (
    CASUAL_REPLY_EXAMPLES,
    CATCHING_UP_THREAD_EXAMPLES,
    EMOTIONAL_SUPPORT_THREAD_EXAMPLES,
    LOGISTICS_THREAD_EXAMPLES,
    PLANNING_THREAD_EXAMPLES,
    PROFESSIONAL_REPLY_EXAMPLES,
    PROMPT_LAST_UPDATED,
    PROMPT_VERSION,
    QUICK_EXCHANGE_THREAD_EXAMPLES,
    SEARCH_ANSWER_EXAMPLES,
    SUMMARIZATION_EXAMPLES,
    THREAD_EXAMPLES,
    FewShotExample,
    PromptMetadata,
)

# --- optimized ---
from jarvis.prompts.optimized import (
    CATCHING_UP_EXAMPLES,
    EMOTIONAL_SUPPORT_EXAMPLES,
    QUICK_EXCHANGE_EXAMPLES,
    OptimizedCategoryProgram,
    get_optimized_examples,
    get_optimized_instruction,
    get_optimized_program,
    reset_optimized_programs,
)

# --- rag ---
from jarvis.prompts.rag import (
    build_prompt_from_request,
    build_rag_reply_prompt,
    build_rag_reply_prompt_from_embeddings,
)

# --- registry ---
from jarvis.prompts.registry import (
    PromptRegistry,
    get_prompt_registry,
    reset_prompt_registry,
)

# --- templates ---
from jarvis.prompts.templates import (
    MAX_CONTEXT_CHARS,
    MAX_PROMPT_TOKENS,
    RAG_REPLY_PROMPT,
    REPLY_PROMPT,
    SEARCH_PROMPT,
    SUMMARY_PROMPT,
    SYSTEM_PREFIX,
    THREADED_REPLY_PROMPT,
    PromptTemplate,
)

# --- tone ---
from jarvis.prompts.tone import (
    CASUAL_INDICATORS,
    EMOJI_PATTERN,
    PROFESSIONAL_INDICATORS,
    TEXT_ABBREVIATIONS,
    UserStyleAnalysis,
    analyze_user_style,
    build_style_instructions,
    detect_tone,
)

__all__ = [
    # examples
    "CASUAL_REPLY_EXAMPLES",
    "CATCHING_UP_THREAD_EXAMPLES",
    "EMOTIONAL_SUPPORT_THREAD_EXAMPLES",
    "FewShotExample",
    "LOGISTICS_THREAD_EXAMPLES",
    "PLANNING_THREAD_EXAMPLES",
    "PROFESSIONAL_REPLY_EXAMPLES",
    "PROMPT_LAST_UPDATED",
    "PROMPT_VERSION",
    "PromptMetadata",
    "QUICK_EXCHANGE_THREAD_EXAMPLES",
    "SEARCH_ANSWER_EXAMPLES",
    "SUMMARIZATION_EXAMPLES",
    "THREAD_EXAMPLES",
    # templates
    "MAX_CONTEXT_CHARS",
    "MAX_PROMPT_TOKENS",
    "PromptTemplate",
    "RAG_REPLY_PROMPT",
    "REPLY_PROMPT",
    "SEARCH_PROMPT",
    "SUMMARY_PROMPT",
    "SYSTEM_PREFIX",
    "THREADED_REPLY_PROMPT",
    # tone
    "CASUAL_INDICATORS",
    "EMOJI_PATTERN",
    "PROFESSIONAL_INDICATORS",
    "TEXT_ABBREVIATIONS",
    "UserStyleAnalysis",
    "analyze_user_style",
    "build_style_instructions",
    "detect_tone",
    # builders
    "build_reply_prompt",
    "build_search_answer_prompt",
    "build_summary_prompt",
    "build_threaded_reply_prompt",
    "format_facts_for_prompt",
    "get_thread_max_tokens",
    # rag
    "build_prompt_from_request",
    "build_rag_reply_prompt",
    "build_rag_reply_prompt_from_embeddings",
    # categories
    "ACKNOWLEDGE_TEMPLATES",
    "API_REPLY_EXAMPLES",
    "API_REPLY_EXAMPLES_METADATA",
    "API_SUMMARY_EXAMPLES",
    "API_SUMMARY_EXAMPLES_METADATA",
    "CATEGORY_CONFIGS",
    "CATEGORY_MAP",
    "CHAT_SYSTEM_PROMPT",
    "CLOSING_TEMPLATES",
    "CategoryConfig",
    "REPLY_EXAMPLES",
    "SUMMARY_EXAMPLES",
    "estimate_tokens",
    "get_category_config",
    "is_within_token_limit",
    "resolve_category",
    # optimized
    "CATCHING_UP_EXAMPLES",
    "EMOTIONAL_SUPPORT_EXAMPLES",
    "OptimizedCategoryProgram",
    "QUICK_EXCHANGE_EXAMPLES",
    "get_optimized_examples",
    "get_optimized_instruction",
    "get_optimized_program",
    "reset_optimized_programs",
    # registry
    "PromptRegistry",
    "get_prompt_registry",
    "reset_prompt_registry",
]
