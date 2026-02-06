#!/usr/bin/env python3
"""Promptfoo provider for JARVIS MLX model.

Supports multiple prompt strategies for A/B testing:
- "baseline": Current JARVIS build_reply_prompt()
- "minimal": Bare-bones prompt, no examples
- "examples_only": Just examples, minimal instructions
- "style_focus": Heavy emphasis on style matching

Usage:
    echo '{"vars": {"context": "...", "strategy": "baseline"}}' | python jarvis_provider.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Strategies - These are what we're A/B testing
# =============================================================================


def strategy_baseline(context: str, last_message: str, tone: str, user_style: str, **kwargs) -> str:
    """Current JARVIS system - build_reply_prompt()."""
    from jarvis.prompts import build_reply_prompt

    instruction = f"Match this style: {user_style}" if user_style else None
    return build_reply_prompt(
        context=context,
        last_message=last_message,
        instruction=instruction,
        tone=tone,
    )


def strategy_minimal(context: str, last_message: str, tone: str, user_style: str, **kwargs) -> str:
    """Minimal prompt - no examples, just core instruction."""
    return f"""Reply to this text message. Be {tone}, brief, and human.

Conversation:
{context}

Reply to: {last_message}

Your reply (1-15 words):"""


def strategy_examples_only(
    context: str, last_message: str, tone: str, user_style: str, **kwargs
) -> str:
    """Few-shot examples with minimal instruction."""
    if tone == "professional":
        examples = """Examples:
"Can you send the report?" → "On it, will have it by 3"
"Meeting moved to 2pm" → "Got it, thanks for the heads up"
"Quick call?" → "Give me 5 min"
"""
    else:
        examples = """Examples:
"Want to grab lunch?" → "Sure! When works?"
"Running late" → "No worries, take your time"
"Did you see the game?" → "Yes!! That ending was wild"
"Can you pick up milk?" → "Got it"
"""

    return f"""{examples}
Now reply to this:
{context}

"{last_message}" →"""


def strategy_style_focus(
    context: str, last_message: str, tone: str, user_style: str, **kwargs
) -> str:
    """Heavy emphasis on matching user's texting style."""
    return f"""You are mimicking someone's EXACT texting style. This is critical.

Their style: {user_style or "casual, brief"}

Rules:
- If they use "u" for "you", YOU use "u"
- If they skip periods, YOU skip periods
- If they're brief (1-5 words), YOU are brief
- Match their energy exactly
- NEVER sound like an AI or assistant

Conversation:
{context}

Reply to "{last_message}" in their style:"""


def strategy_anti_ai(context: str, last_message: str, tone: str, user_style: str, **kwargs) -> str:
    """Explicitly forbid AI-sounding phrases."""
    return f"""Reply to this text. You MUST sound human.

FORBIDDEN PHRASES (never use):
- "I'd be happy to"
- "That sounds"
- "Absolutely!"
- "I understand"
- "Great question"
- "I appreciate"
- "Let me"
- "I can help"

Conversation:
{context}

Reply to: {last_message}
Tone: {tone}

Your human reply:"""


STRATEGIES = {
    "baseline": strategy_baseline,
    "minimal": strategy_minimal,
    "examples_only": strategy_examples_only,
    "style_focus": strategy_style_focus,
    "anti_ai": strategy_anti_ai,
}


# =============================================================================
# Model Loading & Generation
# =============================================================================


def load_model():
    """Load the JARVIS MLX model."""
    from models.loader import get_model

    loader = get_model()
    if not loader.is_loaded():
        loader.load()
    return loader


def generate_reply(prompt: str, config: dict) -> dict:
    """Generate a reply using the JARVIS model."""
    try:
        loader = load_model()
        result = loader.generate_sync(
            prompt=prompt,
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 50),
            top_p=config.get("top_p", 0.1),
            top_k=config.get("top_k", 50),
            repetition_penalty=config.get("repetition_penalty", 1.05),
        )
        return {
            "output": result.text.strip(),
            "tokenUsage": {"completion": result.tokens_generated},
        }
    except Exception as e:
        logger.error("Generation failed: %s", e)
        return {"error": str(e)}


# =============================================================================
# Main
# =============================================================================


def main():
    input_data = sys.stdin.read()

    try:
        request = json.loads(input_data)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON: {e}"}))
        sys.exit(1)

    vars_dict = request.get("vars", {})
    config = request.get("config", {})

    # Extract test case variables
    context = vars_dict.get("context", "")
    last_message = vars_dict.get("last_message", "")
    tone = vars_dict.get("tone", "casual")
    user_style = vars_dict.get("user_style", "")
    strategy_name = vars_dict.get("strategy", "baseline")

    # Get the strategy function
    strategy_fn = STRATEGIES.get(strategy_name, strategy_baseline)

    # Build prompt using selected strategy
    try:
        prompt = strategy_fn(
            context=context,
            last_message=last_message,
            tone=tone,
            user_style=user_style,
        )
    except Exception as e:
        print(json.dumps({"error": f"Prompt build failed: {e}"}))
        sys.exit(1)

    # Generate and return
    result = generate_reply(prompt, config)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
