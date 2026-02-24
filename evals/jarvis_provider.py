#!/usr/bin/env python3
"""Promptfoo provider for JARVIS MLX model.

Supports 8 prompt strategies (structure x framing matrix) for A/B testing:

Structures (2):
- xml: XML-tagged format (<system>, <style>, <examples>, etc.)
- md: Markdown ###-header format

Framings (4):
- drafter: "You draft text message replies" (assistant framing)
- persona: "You ARE this person texting" (first-person identity)
- completer: "Complete the next message" (neutral completion)
- anti_helper: "You are NOT an AI assistant" (anti-AI framing)

Usage:
    echo '{"vars": {"context": "...", "strategy": "xml_drafter"}}' | python jarvis_provider.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Ensure model-relative paths resolve regardless of promptfoo basePath/cwd.
os.chdir(PROJECT_ROOT)

from jarvis.prompts.generation_config import DEFAULT_REPETITION_PENALTY

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


# =============================================================================
# Framing Definitions
# =============================================================================

FRAMINGS = {
    "drafter": ("You draft text message replies matching the sender's exact style."),
    "persona": (
        "You ARE this person texting. Continue the conversation as them. "
        "Write exactly how they write."
    ),
    "completer": (
        "Complete the next message in this text conversation. "
        "Output only the reply text, nothing else."
    ),
    "anti_helper": (
        "You are NOT an AI assistant. You are replying to a text message from your phone. "
        "Just text back. No helpfulness, no formality, no assistant behavior."
    ),
}

# Shared rules appended to all framings
SHARED_RULES = """Rules:
- Match their texting style exactly (length, formality, abbreviations, emoji, punctuation)
- Sound natural, never like an AI
- No phrases like "I hope this helps" or "Let me know"
- No formal greetings unless they use them
- If the message is unclear or you lack context to reply properly, respond with just "?"
"""


# =============================================================================
# XML Structure Strategies
# =============================================================================


def _xml_strategy(
    framing: str,
    context: str,
    last_message: str,
    tone: str,
    user_style: str,
    **kwargs,
) -> str:
    """Build an XML-tagged prompt with the given framing."""
    system_text = FRAMINGS[framing]
    style_line = f"Tone: {tone}. Style: {user_style}" if user_style else f"Tone: {tone}"

    return f"""<system>
{system_text}
{SHARED_RULES}</system>

<style>
{style_line}
</style>

<conversation>
{context}
</conversation>

<last_message>{last_message}</last_message>

<reply>"""


def strategy_xml_drafter(context: str, last_message: str, tone: str, user_style: str, **kw) -> str:
    """XML structure + drafter framing (current default)."""
    return _xml_strategy("drafter", context, last_message, tone, user_style, **kw)


def strategy_xml_persona(context: str, last_message: str, tone: str, user_style: str, **kw) -> str:
    """XML structure + persona framing."""
    return _xml_strategy("persona", context, last_message, tone, user_style, **kw)


def strategy_xml_completer(
    context: str, last_message: str, tone: str, user_style: str, **kw
) -> str:
    """XML structure + completer framing."""
    return _xml_strategy("completer", context, last_message, tone, user_style, **kw)


def strategy_xml_anti_helper(
    context: str, last_message: str, tone: str, user_style: str, **kw
) -> str:
    """XML structure + anti-helper framing."""
    return _xml_strategy("anti_helper", context, last_message, tone, user_style, **kw)


# =============================================================================
# Markdown Structure Strategies
# =============================================================================


def _md_strategy(
    framing: str,
    context: str,
    last_message: str,
    tone: str,
    user_style: str,
    **kwargs,
) -> str:
    """Build a markdown ###-header prompt with the given framing."""
    system_text = FRAMINGS[framing]
    style_line = f"Style: {user_style}" if user_style else ""

    return f"""{system_text}

### Conversation Context:
{context}

### Instructions:
Generate a reply that matches the user's texting style exactly:
- Match the tone of the conversation ({tone})
- Keep response length similar to user's typical messages
- Sound like the user wrote it, not an AI
- If the message is unclear or you lack context, respond with just "?"
{style_line}

### Last message to reply to:
{last_message}

### Your reply:"""


def strategy_md_drafter(context: str, last_message: str, tone: str, user_style: str, **kw) -> str:
    """Markdown structure + drafter framing."""
    return _md_strategy("drafter", context, last_message, tone, user_style, **kw)


def strategy_md_persona(context: str, last_message: str, tone: str, user_style: str, **kw) -> str:
    """Markdown structure + persona framing."""
    return _md_strategy("persona", context, last_message, tone, user_style, **kw)


def strategy_md_completer(context: str, last_message: str, tone: str, user_style: str, **kw) -> str:
    """Markdown structure + completer framing."""
    return _md_strategy("completer", context, last_message, tone, user_style, **kw)


def strategy_md_anti_helper(
    context: str, last_message: str, tone: str, user_style: str, **kw
) -> str:
    """Markdown structure + anti-helper framing."""
    return _md_strategy("anti_helper", context, last_message, tone, user_style, **kw)


# =============================================================================
# Strategy Registry
# =============================================================================

STRATEGIES = {
    "xml_drafter": strategy_xml_drafter,
    "xml_persona": strategy_xml_persona,
    "xml_completer": strategy_xml_completer,
    "xml_anti_helper": strategy_xml_anti_helper,
    "md_drafter": strategy_md_drafter,
    "md_persona": strategy_md_persona,
    "md_completer": strategy_md_completer,
    "md_anti_helper": strategy_md_anti_helper,
}


# =============================================================================
# Model Loading & Generation
# =============================================================================


def load_model():
    """Load the JARVIS MLX model."""
    # Set MLX memory limits before loading to prevent swap thrashing on 8GB systems
    from models.memory_config import apply_embedder_limits

    apply_embedder_limits()

    from models.loader import get_model

    loader = get_model()
    if not loader.is_loaded():
        loader.load()
    return loader


def generate_reply(prompt: str, config: dict) -> str:
    """Generate a reply using the JARVIS model and return plain text."""
    try:
        loader = load_model()
        result = loader.generate_sync(
            prompt=prompt,
            temperature=config.get("temperature", 0.1),
            max_tokens=config.get("max_tokens", 50),
            top_p=config.get("top_p", 0.1),
            top_k=config.get("top_k", 50),
            repetition_penalty=config.get(
                "repetition_penalty",
                DEFAULT_REPETITION_PENALTY,  # From generation_config
            ),
        )
        return result.text.strip()
    except Exception as e:
        logger.error("Generation failed: %s", e)
        return f"[ERROR] {e}"


# =============================================================================
# Main
# =============================================================================


def main():
    # promptfoo exec provider passes: argv[1]=prompt, argv[2]=config_json, argv[3]=context_json
    if len(sys.argv) < 4:
        print(json.dumps({"error": f"Expected 3 args, got {len(sys.argv) - 1}"}), flush=True)
        sys.exit(1)

    raw_prompt = sys.argv[1]  # e.g. "strategy:xml_drafter"
    try:
        config = json.loads(sys.argv[2]).get("config", {})
    except json.JSONDecodeError:
        config = {}
    try:
        context_obj = json.loads(sys.argv[3])
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid context JSON: {e}"}), flush=True)
        sys.exit(1)

    vars_dict = context_obj.get("vars", {})

    # Extract strategy from prompt string (format: "strategy:xml_drafter")
    if ":" in raw_prompt:
        strategy_name = raw_prompt.split(":", 1)[1]
    else:
        strategy_name = vars_dict.get("strategy", "xml_drafter")

    # Extract test case variables
    context = vars_dict.get("context", "")
    last_message = vars_dict.get("last_message", "")
    tone = vars_dict.get("tone", "casual")
    user_style = vars_dict.get("user_style", "")

    # Get the strategy function
    strategy_fn = STRATEGIES.get(strategy_name, strategy_xml_drafter)

    # Build prompt using selected strategy
    try:
        prompt = strategy_fn(
            context=context,
            last_message=last_message,
            tone=tone,
            user_style=user_style,
        )
    except Exception as e:
        print(json.dumps({"error": f"Prompt build failed: {e}"}), flush=True)
        sys.exit(1)

    # Generate and return
    result = generate_reply(prompt, config)
    print(result, flush=True)


if __name__ == "__main__":
    main()
