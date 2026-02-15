from __future__ import annotations

from jarvis.prompts.constants import SEARCH_PROMPT
from jarvis.prompts.examples import SEARCH_ANSWER_EXAMPLES
from jarvis.prompts.utils import format_search_examples, truncate_context


def build_search_prompt(
    context: str,
    query: str,
) -> str:
    """Build a prompt for answering questions based on message history."""
    truncated_context = truncate_context(context)

    prompt = SEARCH_PROMPT.template.format(
        context=truncated_context,
        question=query,
        examples=format_search_examples(SEARCH_ANSWER_EXAMPLES[:2]),
    )

    return prompt
