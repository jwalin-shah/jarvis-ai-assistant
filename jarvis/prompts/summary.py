from __future__ import annotations

from jarvis.prompts.constants import SUMMARY_PROMPT
from jarvis.prompts.examples import SUMMARIZATION_EXAMPLES
from jarvis.prompts.utils import truncate_context


def build_summary_prompt(
    context: str,
    focus: str | None = None,
) -> str:
    """Build a prompt for summarizing conversations."""
    focus_instruction = ""
    if focus:
        focus_instruction = f"- Focus especially on: {focus}"

    # Select a subset of examples
    # (In builders.py it was SUMMARIZATION_EXAMPLES[:2])
    
    truncated_context = truncate_context(context)

    prompt = SUMMARY_PROMPT.template.format(
        context=truncated_context,
        focus_instruction=focus_instruction,
    )

    return prompt
