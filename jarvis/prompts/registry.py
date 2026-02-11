"""PromptRegistry class and DSPy-optimized program loading.

This module contains:
- OptimizedCategoryProgram for DSPy-compiled programs
- Loading and caching of optimized programs from disk
- PromptRegistry class for centralized prompt management
- Global registry instance management
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from jarvis.prompts.constants import (
    PROMPT_LAST_UPDATED,
    PROMPT_VERSION,
    REPLY_PROMPT,
    SEARCH_PROMPT,
    SUMMARY_PROMPT,
    THREADED_REPLY_PROMPT,
    FewShotExample,
    PromptMetadata,
    PromptTemplate,
)
from jarvis.prompts.examples import (
    API_REPLY_EXAMPLES,
    API_REPLY_EXAMPLES_METADATA,
    API_SUMMARY_EXAMPLES,
    API_SUMMARY_EXAMPLES_METADATA,
    CATCHING_UP_THREAD_EXAMPLES,
    EMOTIONAL_SUPPORT_THREAD_EXAMPLES,
    LOGISTICS_THREAD_EXAMPLES,
    PLANNING_THREAD_EXAMPLES,
    PROFESSIONAL_REPLY_EXAMPLES,
    QUICK_EXCHANGE_THREAD_EXAMPLES,
    REPLY_EXAMPLES,
    SEARCH_ANSWER_EXAMPLES,
    SUMMARIZATION_EXAMPLES,
    THREAD_EXAMPLES,
)

# =============================================================================
# DSPy-Optimized Per-Category Prompt Loader
# =============================================================================


@dataclass
class OptimizedCategoryProgram:
    """Loaded optimized program for a specific category.

    Attributes:
        category: The response category name
        instruction: Optimized instruction text from MIPRO v2
        demos: Few-shot demo examples [(context, reply), ...]
    """

    category: str
    instruction: str
    demos: list[tuple[str, str]]


# Cache of loaded programs
_optimized_programs: dict[str, OptimizedCategoryProgram] | None = None
_optimized_programs_lock = threading.Lock()

# Path to per-category compiled programs
_CATEGORY_DIR = Path(__file__).parent.parent.parent / "evals" / "optimized_categories"


def _load_optimized_programs() -> dict[str, OptimizedCategoryProgram]:
    """Load all per-category optimized programs from disk.

    Reads DSPy compiled JSON files and extracts the instruction text
    and few-shot demonstrations for each category.

    Returns:
        Dict mapping category name -> OptimizedCategoryProgram
    """
    programs: dict[str, OptimizedCategoryProgram] = {}

    if not _CATEGORY_DIR.exists():
        return programs

    import json as _json

    for path in _CATEGORY_DIR.glob("optimized_*.json"):
        category = path.stem.replace("optimized_", "")
        try:
            data = _json.loads(path.read_text())

            # Extract instruction from DSPy compiled program format
            # DSPy saves programs as {"generate": {"lm": null, "traces": [...], ...}}
            instruction = ""
            demos: list[tuple[str, str]] = []

            # Navigate DSPy's saved format to find instruction and demos
            generate_data = data.get("generate", data)

            # Try to extract instruction from extended_signature or signature
            if "extended_signature_instructions" in generate_data:
                instruction = generate_data["extended_signature_instructions"]
            elif "signature_instructions" in generate_data:
                instruction = generate_data["signature_instructions"]

            # Extract demos from the compiled program
            for demo in generate_data.get("demos", []):
                ctx = demo.get("context", "")
                reply = demo.get("reply", demo.get("output", ""))
                if ctx and reply:
                    demos.append((ctx, reply))

            programs[category] = OptimizedCategoryProgram(
                category=category,
                instruction=instruction,
                demos=demos,
            )
        except Exception:
            # Skip malformed files silently
            continue

    return programs


def get_optimized_program(category: str) -> OptimizedCategoryProgram | None:
    """Get the optimized program for a category, loading from disk on first call.

    Args:
        category: Category name (quick_exchange, emotional_support, etc.)

    Returns:
        OptimizedCategoryProgram if found, None otherwise
    """
    global _optimized_programs
    if _optimized_programs is None:
        with _optimized_programs_lock:
            if _optimized_programs is None:
                _optimized_programs = _load_optimized_programs()
    return _optimized_programs.get(category)


def get_optimized_examples(category: str) -> list[FewShotExample]:
    """Get DSPy-optimized examples for a category, falling back to static ones.

    If a compiled program exists for the category, returns its optimized demos.
    Otherwise returns the static THREAD_EXAMPLES for that category.

    Args:
        category: Category name matching THREAD_EXAMPLES keys

    Returns:
        List of FewShotExample objects
    """
    program = get_optimized_program(category)
    if program and program.demos:
        return [FewShotExample(context=ctx, output=reply) for ctx, reply in program.demos]
    # Fallback to static examples
    return THREAD_EXAMPLES.get(category, CATCHING_UP_THREAD_EXAMPLES)


def get_optimized_instruction(category: str) -> str | None:
    """Get the MIPRO v2-optimized instruction prefix for a category.

    Args:
        category: Category name

    Returns:
        Optimized instruction string, or None if no compiled program exists
    """
    program = get_optimized_program(category)
    if program and program.instruction:
        return program.instruction
    return None


def reset_optimized_programs() -> None:
    """Reset the cached optimized programs (for testing)."""
    global _optimized_programs
    _optimized_programs = None


# =============================================================================
# Prompt Registry
# =============================================================================


class PromptRegistry:
    """Registry for dynamic prompt management.

    Provides centralized access to all prompts, examples, and templates
    with metadata tracking and versioning support.

    Example:
        >>> registry = PromptRegistry()
        >>> examples = registry.get_examples("casual_reply")
        >>> template = registry.get_template("reply_generation")
        >>> metadata = registry.get_metadata("casual_reply")
    """

    def __init__(self) -> None:
        """Initialize the prompt registry with all registered prompts."""
        self._examples: dict[str, list[tuple[str, str]]] = {
            "casual_reply": REPLY_EXAMPLES,
            "professional_reply": [(ex.context, ex.output) for ex in PROFESSIONAL_REPLY_EXAMPLES],
            "summarization": SUMMARIZATION_EXAMPLES,
            "search_answer": [
                (f"Messages:\n{msgs}\nQuestion: {q}", a) for msgs, q, a in SEARCH_ANSWER_EXAMPLES
            ],
            "api_reply": API_REPLY_EXAMPLES,
            "api_summary": API_SUMMARY_EXAMPLES,
            "thread_logistics": [(ex.context, ex.output) for ex in LOGISTICS_THREAD_EXAMPLES],
            "thread_emotional_support": [
                (ex.context, ex.output) for ex in EMOTIONAL_SUPPORT_THREAD_EXAMPLES
            ],
            "thread_planning": [(ex.context, ex.output) for ex in PLANNING_THREAD_EXAMPLES],
            "thread_catching_up": [(ex.context, ex.output) for ex in CATCHING_UP_THREAD_EXAMPLES],
            "thread_quick_exchange": [
                (ex.context, ex.output) for ex in QUICK_EXCHANGE_THREAD_EXAMPLES
            ],
        }

        self._templates: dict[str, PromptTemplate] = {
            "reply_generation": REPLY_PROMPT,
            "conversation_summary": SUMMARY_PROMPT,
            "search_answer": SEARCH_PROMPT,
            "threaded_reply": THREADED_REPLY_PROMPT,
        }

        self._metadata: dict[str, PromptMetadata] = {
            "casual_reply": PromptMetadata(
                name="casual_reply",
                description="Few-shot examples for casual iMessage replies",
            ),
            "professional_reply": PromptMetadata(
                name="professional_reply",
                description="Few-shot examples for professional iMessage replies",
            ),
            "summarization": PromptMetadata(
                name="summarization",
                description="Few-shot examples for conversation summarization",
            ),
            "search_answer": PromptMetadata(
                name="search_answer",
                description="Few-shot examples for question answering over messages",
            ),
            "api_reply": API_REPLY_EXAMPLES_METADATA,
            "api_summary": API_SUMMARY_EXAMPLES_METADATA,
            "reply_generation": PromptMetadata(
                name="reply_generation",
                description="Template for generating iMessage replies",
            ),
            "conversation_summary": PromptMetadata(
                name="conversation_summary",
                description="Template for summarizing conversations",
            ),
            "search_answer_template": PromptMetadata(
                name="search_answer_template",
                description="Template for answering questions about conversations",
            ),
            "threaded_reply": PromptMetadata(
                name="threaded_reply",
                description="Template for thread-aware reply generation",
            ),
            "thread_logistics": PromptMetadata(
                name="thread_logistics",
                description="Examples for logistics/coordination thread replies",
            ),
            "thread_emotional_support": PromptMetadata(
                name="thread_emotional_support",
                description="Examples for emotional support thread replies",
            ),
            "thread_planning": PromptMetadata(
                name="thread_planning",
                description="Examples for planning thread replies with action items",
            ),
            "thread_catching_up": PromptMetadata(
                name="thread_catching_up",
                description="Examples for catching up/casual thread replies",
            ),
            "thread_quick_exchange": PromptMetadata(
                name="thread_quick_exchange",
                description="Examples for quick exchange thread replies",
            ),
        }

    def get_examples(self, name: str) -> list[tuple[str, str]]:
        """Get few-shot examples by name.

        Args:
            name: The example set name (e.g., "casual_reply", "api_reply")

        Returns:
            List of (input, output) tuples for few-shot prompting

        Raises:
            KeyError: If the example set doesn't exist
        """
        if name not in self._examples:
            available = ", ".join(sorted(self._examples.keys()))
            raise KeyError(f"Unknown example set '{name}'. Available: {available}")
        return self._examples[name]

    def get_template(self, name: str) -> PromptTemplate:
        """Get a prompt template by name.

        Args:
            name: The template name (e.g., "reply_generation")

        Returns:
            The PromptTemplate instance

        Raises:
            KeyError: If the template doesn't exist
        """
        if name not in self._templates:
            available = ", ".join(sorted(self._templates.keys()))
            raise KeyError(f"Unknown template '{name}'. Available: {available}")
        return self._templates[name]

    def get_metadata(self, name: str) -> PromptMetadata:
        """Get metadata for a prompt or example set.

        Args:
            name: The prompt/example set name

        Returns:
            The PromptMetadata instance

        Raises:
            KeyError: If the metadata doesn't exist
        """
        if name not in self._metadata:
            available = ", ".join(sorted(self._metadata.keys()))
            raise KeyError(f"Unknown prompt '{name}'. Available: {available}")
        return self._metadata[name]

    def list_examples(self) -> list[str]:
        """List all available example set names.

        Returns:
            Sorted list of example set names
        """
        return sorted(self._examples.keys())

    def list_templates(self) -> list[str]:
        """List all available template names.

        Returns:
            Sorted list of template names
        """
        return sorted(self._templates.keys())

    def register_examples(
        self,
        name: str,
        examples: list[tuple[str, str]],
        metadata: PromptMetadata | None = None,
    ) -> None:
        """Register a new example set.

        Args:
            name: Unique name for the example set
            examples: List of (input, output) tuples
            metadata: Optional metadata for the example set
        """
        self._examples[name] = examples
        if metadata:
            self._metadata[name] = metadata
        else:
            self._metadata[name] = PromptMetadata(
                name=name,
                description=f"Custom example set: {name}",
            )

    def register_template(
        self,
        template: PromptTemplate,
        metadata: PromptMetadata | None = None,
    ) -> None:
        """Register a new prompt template.

        Args:
            template: The PromptTemplate to register
            metadata: Optional metadata for the template
        """
        self._templates[template.name] = template
        if metadata:
            self._metadata[template.name] = metadata
        else:
            self._metadata[template.name] = PromptMetadata(
                name=template.name,
                description=f"Custom template: {template.name}",
            )

    @property
    def version(self) -> str:
        """Get the prompt system version."""
        return PROMPT_VERSION

    @property
    def last_updated(self) -> str:
        """Get the last update date."""
        return PROMPT_LAST_UPDATED


# Global registry instance
_registry: PromptRegistry | None = None


def get_prompt_registry() -> PromptRegistry:
    """Get the global PromptRegistry instance.

    Returns:
        The shared PromptRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry


def reset_prompt_registry() -> None:
    """Reset the global PromptRegistry instance.

    Useful for testing or when prompts need to be reloaded.
    """
    global _registry
    _registry = None
