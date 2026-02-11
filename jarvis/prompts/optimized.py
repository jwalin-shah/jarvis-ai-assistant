"""DSPy-optimized per-category prompt loader.

Loads MIPRO v2 compiled programs from disk and provides optimized
instructions and few-shot demos for each message category.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from jarvis.prompts.examples import (
    CATCHING_UP_THREAD_EXAMPLES,
    EMOTIONAL_SUPPORT_THREAD_EXAMPLES,
    QUICK_EXCHANGE_THREAD_EXAMPLES,
    THREAD_EXAMPLES,
    FewShotExample,
)


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


# Legacy aliases for backward compatibility
EMOTIONAL_SUPPORT_EXAMPLES = EMOTIONAL_SUPPORT_THREAD_EXAMPLES
CATCHING_UP_EXAMPLES = CATCHING_UP_THREAD_EXAMPLES
QUICK_EXCHANGE_EXAMPLES = QUICK_EXCHANGE_THREAD_EXAMPLES


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
