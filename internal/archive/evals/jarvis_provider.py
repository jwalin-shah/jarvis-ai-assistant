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
# Ensure model-relative paths resolve regardless of promptfoo basePath/cwd.  # noqa: E402
os.chdir(PROJECT_ROOT)  # noqa: E402
  # noqa: E402
from jarvis.prompts.generation_config import DEFAULT_REPETITION_PENALTY  # noqa: E402

  # noqa: E402
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s", stream=sys.stderr)  # noqa: E402
logger = logging.getLogger(__name__)  # noqa: E402
  # noqa: E402
  # noqa: E402
# =============================================================================  # noqa: E402
# Framing Definitions  # noqa: E402
# =============================================================================  # noqa: E402
  # noqa: E402
FRAMINGS = {  # noqa: E402
    "drafter": ("You draft text message replies matching the sender's exact style."),  # noqa: E402
    "persona": (  # noqa: E402
        "You ARE this person texting. Continue the conversation as them. "  # noqa: E402
        "Write exactly how they write."  # noqa: E402
    ),  # noqa: E402
    "completer": (  # noqa: E402
        "Complete the next message in this text conversation. "  # noqa: E402
        "Output only the reply text, nothing else."  # noqa: E402
    ),  # noqa: E402
    "anti_helper": (  # noqa: E402
        "You are NOT an AI assistant. You are replying to a text message from your phone. "  # noqa: E402
        "Just text back. No helpfulness, no formality, no assistant behavior."  # noqa: E402
    ),  # noqa: E402
}  # noqa: E402
  # noqa: E402
# Shared rules appended to all framings  # noqa: E402
SHARED_RULES = """Rules:  # noqa: E402
- Match their texting style exactly (length, formality, abbreviations, emoji, punctuation)  # noqa: E402
- Sound natural, never like an AI  # noqa: E402
- No phrases like "I hope this helps" or "Let me know"  # noqa: E402
- No formal greetings unless they use them  # noqa: E402
- If the message is unclear or you lack context to reply properly, respond with just "?"  # noqa: E402
"""  # noqa: E402
  # noqa: E402
  # noqa: E402
# =============================================================================  # noqa: E402
# XML Structure Strategies  # noqa: E402
# =============================================================================  # noqa: E402
  # noqa: E402
  # noqa: E402
def _xml_strategy(  # noqa: E402
    framing: str,  # noqa: E402
    context: str,  # noqa: E402
    last_message: str,  # noqa: E402
    tone: str,  # noqa: E402
    user_style: str,  # noqa: E402
    **kwargs,  # noqa: E402
) -> str:  # noqa: E402
    """Build an XML-tagged prompt with the given framing."""  # noqa: E402
    system_text = FRAMINGS[framing]  # noqa: E402
    style_line = f"Tone: {tone}. Style: {user_style}" if user_style else f"Tone: {tone}"  # noqa: E402
  # noqa: E402
    return f"""<system>  # noqa: E402
{system_text}  # noqa: E402
{SHARED_RULES}</system>  # noqa: E402
  # noqa: E402
<style>  # noqa: E402
{style_line}  # noqa: E402
</style>  # noqa: E402
  # noqa: E402
<conversation>  # noqa: E402
{context}  # noqa: E402
</conversation>  # noqa: E402
  # noqa: E402
<last_message>{last_message}</last_message>  # noqa: E402
  # noqa: E402
<reply>"""  # noqa: E402
  # noqa: E402
  # noqa: E402
def strategy_xml_drafter(context: str, last_message: str, tone: str, user_style: str, **kw) -> str:  # noqa: E402
    """XML structure + drafter framing (current default)."""  # noqa: E402
    return _xml_strategy("drafter", context, last_message, tone, user_style, **kw)  # noqa: E402
  # noqa: E402
  # noqa: E402
def strategy_xml_persona(context: str, last_message: str, tone: str, user_style: str, **kw) -> str:  # noqa: E402
    """XML structure + persona framing."""  # noqa: E402
    return _xml_strategy("persona", context, last_message, tone, user_style, **kw)  # noqa: E402
  # noqa: E402
  # noqa: E402
def strategy_xml_completer(  # noqa: E402
    context: str, last_message: str, tone: str, user_style: str, **kw  # noqa: E402
) -> str:  # noqa: E402
    """XML structure + completer framing."""  # noqa: E402
    return _xml_strategy("completer", context, last_message, tone, user_style, **kw)  # noqa: E402
  # noqa: E402
  # noqa: E402
def strategy_xml_anti_helper(  # noqa: E402
    context: str, last_message: str, tone: str, user_style: str, **kw  # noqa: E402
) -> str:  # noqa: E402
    """XML structure + anti-helper framing."""  # noqa: E402
    return _xml_strategy("anti_helper", context, last_message, tone, user_style, **kw)  # noqa: E402
  # noqa: E402
  # noqa: E402
# =============================================================================  # noqa: E402
# Markdown Structure Strategies  # noqa: E402
# =============================================================================  # noqa: E402
  # noqa: E402
  # noqa: E402
def _md_strategy(  # noqa: E402
    framing: str,  # noqa: E402
    context: str,  # noqa: E402
    last_message: str,  # noqa: E402
    tone: str,  # noqa: E402
    user_style: str,  # noqa: E402
    **kwargs,  # noqa: E402
) -> str:  # noqa: E402
    """Build a markdown ###-header prompt with the given framing."""  # noqa: E402
    system_text = FRAMINGS[framing]  # noqa: E402
    style_line = f"Style: {user_style}" if user_style else ""  # noqa: E402
  # noqa: E402
    return f"""{system_text}  # noqa: E402
  # noqa: E402
### Conversation Context:  # noqa: E402
{context}  # noqa: E402
  # noqa: E402
### Instructions:  # noqa: E402
Generate a reply that matches the user's texting style exactly:  # noqa: E402
- Match the tone of the conversation ({tone})  # noqa: E402
- Keep response length similar to user's typical messages  # noqa: E402
- Sound like the user wrote it, not an AI  # noqa: E402
- If the message is unclear or you lack context, respond with just "?"  # noqa: E402
{style_line}  # noqa: E402
  # noqa: E402
### Last message to reply to:  # noqa: E402
{last_message}  # noqa: E402
  # noqa: E402
### Your reply:"""  # noqa: E402
  # noqa: E402
  # noqa: E402
def strategy_md_drafter(context: str, last_message: str, tone: str, user_style: str, **kw) -> str:  # noqa: E402
    """Markdown structure + drafter framing."""  # noqa: E402
    return _md_strategy("drafter", context, last_message, tone, user_style, **kw)  # noqa: E402
  # noqa: E402
  # noqa: E402
def strategy_md_persona(context: str, last_message: str, tone: str, user_style: str, **kw) -> str:  # noqa: E402
    """Markdown structure + persona framing."""  # noqa: E402
    return _md_strategy("persona", context, last_message, tone, user_style, **kw)  # noqa: E402
  # noqa: E402
  # noqa: E402
def strategy_md_completer(context: str, last_message: str, tone: str, user_style: str, **kw) -> str:  # noqa: E402
    """Markdown structure + completer framing."""  # noqa: E402
    return _md_strategy("completer", context, last_message, tone, user_style, **kw)  # noqa: E402
  # noqa: E402
  # noqa: E402
def strategy_md_anti_helper(  # noqa: E402
    context: str, last_message: str, tone: str, user_style: str, **kw  # noqa: E402
) -> str:  # noqa: E402
    """Markdown structure + anti-helper framing."""  # noqa: E402
    return _md_strategy("anti_helper", context, last_message, tone, user_style, **kw)  # noqa: E402
  # noqa: E402
  # noqa: E402
# =============================================================================  # noqa: E402
# Strategy Registry  # noqa: E402
# =============================================================================  # noqa: E402
  # noqa: E402
STRATEGIES = {  # noqa: E402
    "xml_drafter": strategy_xml_drafter,  # noqa: E402
    "xml_persona": strategy_xml_persona,  # noqa: E402
    "xml_completer": strategy_xml_completer,  # noqa: E402
    "xml_anti_helper": strategy_xml_anti_helper,  # noqa: E402
    "md_drafter": strategy_md_drafter,  # noqa: E402
    "md_persona": strategy_md_persona,  # noqa: E402
    "md_completer": strategy_md_completer,  # noqa: E402
    "md_anti_helper": strategy_md_anti_helper,  # noqa: E402
}  # noqa: E402
  # noqa: E402
  # noqa: E402
# =============================================================================  # noqa: E402
# Model Loading & Generation  # noqa: E402
# =============================================================================  # noqa: E402
  # noqa: E402
  # noqa: E402
def load_model():  # noqa: E402
    """Load the JARVIS MLX model."""  # noqa: E402
    # Set MLX memory limits before loading to prevent swap thrashing on 8GB systems  # noqa: E402
    from models.memory_config import apply_embedder_limits  # noqa: E402
  # noqa: E402
    apply_embedder_limits()  # noqa: E402
  # noqa: E402
    from models.loader import get_model  # noqa: E402
  # noqa: E402
    loader = get_model()  # noqa: E402
    if not loader.is_loaded():  # noqa: E402
        loader.load()  # noqa: E402
    return loader  # noqa: E402
  # noqa: E402
  # noqa: E402
def generate_reply(prompt: str, config: dict) -> str:  # noqa: E402
    """Generate a reply using the JARVIS model and return plain text."""  # noqa: E402
    try:  # noqa: E402
        loader = load_model()  # noqa: E402
        result = loader.generate_sync(  # noqa: E402
            prompt=prompt,  # noqa: E402
            temperature=config.get("temperature", 0.1),  # noqa: E402
            max_tokens=config.get("max_tokens", 50),  # noqa: E402
            top_p=config.get("top_p", 0.1),  # noqa: E402
            top_k=config.get("top_k", 50),  # noqa: E402
            repetition_penalty=config.get(  # noqa: E402
                "repetition_penalty",  # noqa: E402
                DEFAULT_REPETITION_PENALTY,  # From generation_config  # noqa: E402
            ),  # noqa: E402
        )  # noqa: E402
        return result.text.strip()  # noqa: E402
    except Exception as e:  # noqa: E402
        logger.error("Generation failed: %s", e)  # noqa: E402
        return f"[ERROR] {e}"  # noqa: E402
  # noqa: E402
  # noqa: E402
# =============================================================================  # noqa: E402
# Main  # noqa: E402
# =============================================================================  # noqa: E402
  # noqa: E402
  # noqa: E402
def main():  # noqa: E402
    # promptfoo exec provider passes: argv[1]=prompt, argv[2]=config_json, argv[3]=context_json  # noqa: E402
    if len(sys.argv) < 4:  # noqa: E402
        print(json.dumps({"error": f"Expected 3 args, got {len(sys.argv) - 1}"}), flush=True)  # noqa: E402
        sys.exit(1)  # noqa: E402
  # noqa: E402
    raw_prompt = sys.argv[1]  # e.g. "strategy:xml_drafter"  # noqa: E402
    try:  # noqa: E402
        config = json.loads(sys.argv[2]).get("config", {})  # noqa: E402
    except json.JSONDecodeError:  # noqa: E402
        config = {}  # noqa: E402
    try:  # noqa: E402
        context_obj = json.loads(sys.argv[3])  # noqa: E402
    except json.JSONDecodeError as e:  # noqa: E402
        print(json.dumps({"error": f"Invalid context JSON: {e}"}), flush=True)  # noqa: E402
        sys.exit(1)  # noqa: E402
  # noqa: E402
    vars_dict = context_obj.get("vars", {})  # noqa: E402
  # noqa: E402
    # Extract strategy from prompt string (format: "strategy:xml_drafter")  # noqa: E402
    if ":" in raw_prompt:  # noqa: E402
        strategy_name = raw_prompt.split(":", 1)[1]  # noqa: E402
    else:  # noqa: E402
        strategy_name = vars_dict.get("strategy", "xml_drafter")  # noqa: E402
  # noqa: E402
    # Extract test case variables  # noqa: E402
    context = vars_dict.get("context", "")  # noqa: E402
    last_message = vars_dict.get("last_message", "")  # noqa: E402
    tone = vars_dict.get("tone", "casual")  # noqa: E402
    user_style = vars_dict.get("user_style", "")  # noqa: E402
  # noqa: E402
    # Get the strategy function  # noqa: E402
    strategy_fn = STRATEGIES.get(strategy_name, strategy_xml_drafter)  # noqa: E402
  # noqa: E402
    # Build prompt using selected strategy  # noqa: E402
    try:  # noqa: E402
        prompt = strategy_fn(  # noqa: E402
            context=context,  # noqa: E402
            last_message=last_message,  # noqa: E402
            tone=tone,  # noqa: E402
            user_style=user_style,  # noqa: E402
        )  # noqa: E402
    except Exception as e:  # noqa: E402
        print(json.dumps({"error": f"Prompt build failed: {e}"}), flush=True)  # noqa: E402
        sys.exit(1)  # noqa: E402
  # noqa: E402
    # Generate and return  # noqa: E402
    result = generate_reply(prompt, config)  # noqa: E402
    print(result, flush=True)  # noqa: E402
  # noqa: E402
  # noqa: E402
if __name__ == "__main__":  # noqa: E402
    main()  # noqa: E402
