"""Prompt templates and builders for iMessage reply generation.

Provides well-engineered prompts optimized for small local LLMs (Qwen2.5-0.5B/1.5B)
with clear structure, few-shot examples, and tone-aware generation.

This module is the SINGLE SOURCE OF TRUTH for all prompts in the JARVIS system.
Import prompts from here, not from other modules.

All public names are re-exported here for backward compatibility.
"""

from jarvis.prompts.classify import get_category_config, resolve_category
from jarvis.prompts.constants import (
    ACKNOWLEDGE_TEMPLATES,
    CASUAL_INDICATORS,
    CATEGORY_CONFIGS,
    CATEGORY_MAP,
    CHAT_SYSTEM_PROMPT,
    CLOSING_TEMPLATES,
    EMOJI_PATTERN,
    MAX_CONTEXT_CHARS,
    MAX_PROMPT_TOKENS,
    PROFESSIONAL_INDICATORS,
    PROMPT_LAST_UPDATED,
    PROMPT_VERSION,
    RAG_REPLY_PROMPT,
    REPLY_PROMPT,
    SEARCH_PROMPT,
    SUMMARY_PROMPT,
    SYSTEM_PREFIX,
    TEXT_ABBREVIATIONS,
    THREADED_REPLY_PROMPT,
    CategoryConfig,
    FewShotExample,
    PromptMetadata,
    PromptTemplate,
    UserStyleAnalysis,
)
from jarvis.prompts.contact import format_facts_for_prompt

# --- examples.py ---
from jarvis.prompts.examples import (
    API_REPLY_EXAMPLES,
    API_REPLY_EXAMPLES_METADATA,
    API_SUMMARY_EXAMPLES,
    API_SUMMARY_EXAMPLES_METADATA,
    CASUAL_REPLY_EXAMPLES,
    CATCHING_UP_EXAMPLES,
    CATCHING_UP_THREAD_EXAMPLES,
    EMOTIONAL_SUPPORT_EXAMPLES,
    EMOTIONAL_SUPPORT_THREAD_EXAMPLES,
    LOGISTICS_THREAD_EXAMPLES,
    PLANNING_THREAD_EXAMPLES,
    PROFESSIONAL_REPLY_EXAMPLES,
    QUICK_EXCHANGE_EXAMPLES,
    QUICK_EXCHANGE_THREAD_EXAMPLES,
    REPLY_EXAMPLES,
    SEARCH_ANSWER_EXAMPLES,
    SUMMARIZATION_EXAMPLES,
    SUMMARY_EXAMPLES,
    THREAD_EXAMPLES,
)
from jarvis.prompts.rag import (
    build_prompt_from_request,
    build_rag_reply_prompt,
    build_rag_reply_prompt_from_embeddings,
)

# --- registry.py ---
from jarvis.prompts.registry import (
    OptimizedCategoryProgram,
    PromptRegistry,
    get_optimized_examples,
    get_optimized_instruction,
    get_optimized_program,
    get_prompt_registry,
    reset_optimized_programs,
    reset_prompt_registry,
)
from jarvis.prompts.reply import (
    build_reply_prompt,
    build_threaded_reply_prompt,
    get_thread_max_tokens,
)
from jarvis.prompts.search import build_search_prompt as build_search_answer_prompt
from jarvis.prompts.summary import build_summary_prompt
from jarvis.prompts.tone import analyze_user_style, build_style_instructions, detect_tone
from jarvis.prompts.utils import (
    estimate_tokens,
    format_examples,
    format_search_examples,
    format_summary_examples,
    is_within_token_limit,
    truncate_context,
)

__all__ = [
    # Constants
    "ACKNOWLEDGE_TEMPLATES",
    "CASUAL_INDICATORS",
    "CATEGORY_CONFIGS",
    "CATEGORY_MAP",
    "CHAT_SYSTEM_PROMPT",
    "CLOSING_TEMPLATES",
    "EMOJI_PATTERN",
    "MAX_CONTEXT_CHARS",
    "MAX_PROMPT_TOKENS",
    "PROFESSIONAL_INDICATORS",
    "PROMPT_LAST_UPDATED",
    "PROMPT_VERSION",
    "RAG_REPLY_PROMPT",
    "REPLY_PROMPT",
    "SEARCH_PROMPT",
    "SUMMARY_PROMPT",
    "SYSTEM_PREFIX",
    "TEXT_ABBREVIATIONS",
    "THREADED_REPLY_PROMPT",
    # Dataclasses
    "CategoryConfig",
    "FewShotExample",
    "OptimizedCategoryProgram",
    "PromptMetadata",
    "PromptRegistry",
    "PromptTemplate",
    "UserStyleAnalysis",
    # Examples
    "API_REPLY_EXAMPLES",
    "API_REPLY_EXAMPLES_METADATA",
    "API_SUMMARY_EXAMPLES",
    "API_SUMMARY_EXAMPLES_METADATA",
    "CASUAL_REPLY_EXAMPLES",
    "CATCHING_UP_EXAMPLES",
    "CATCHING_UP_THREAD_EXAMPLES",
    "EMOTIONAL_SUPPORT_EXAMPLES",
    "EMOTIONAL_SUPPORT_THREAD_EXAMPLES",
    "LOGISTICS_THREAD_EXAMPLES",
    "PLANNING_THREAD_EXAMPLES",
    "PROFESSIONAL_REPLY_EXAMPLES",
    "QUICK_EXCHANGE_EXAMPLES",
    "QUICK_EXCHANGE_THREAD_EXAMPLES",
    "REPLY_EXAMPLES",
    "SEARCH_ANSWER_EXAMPLES",
    "SUMMARIZATION_EXAMPLES",
    "SUMMARY_EXAMPLES",
    "THREAD_EXAMPLES",
    # Builder functions
    "analyze_user_style",
    "build_prompt_from_request",
    "build_rag_reply_prompt",
    "build_rag_reply_prompt_from_embeddings",
    "build_reply_prompt",
    "build_search_answer_prompt",
    "build_style_instructions",
    "build_summary_prompt",
    "build_threaded_reply_prompt",
    "detect_tone",
    "estimate_tokens",
    "format_examples",
    "format_facts_for_prompt",
    "format_search_examples",
    "format_summary_examples",
    "get_category_config",
    "get_thread_max_tokens",
    "is_within_token_limit",
    "resolve_category",
    "truncate_context",
    # Registry
    "get_optimized_examples",
    "get_optimized_instruction",
    "get_optimized_program",
    "get_prompt_registry",
    "reset_optimized_programs",
    "reset_prompt_registry",
]
