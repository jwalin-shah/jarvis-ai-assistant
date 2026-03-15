#!/usr/bin/env python3  # noqa: E501
"""Promptfoo provider for JARVIS MLX model.  # noqa: E501
  # noqa: E501
Supports 8 prompt strategies (structure x framing matrix) for A/B testing:  # noqa: E501
  # noqa: E501
Structures (2):  # noqa: E501
- xml: XML-tagged format (<system>, <style>, <examples>, etc.)  # noqa: E501
- md: Markdown ###-header format  # noqa: E501
  # noqa: E501
Framings (4):  # noqa: E501
- drafter: "You draft text message replies" (assistant framing)  # noqa: E501
- persona: "You ARE this person texting" (first-person identity)  # noqa: E501
- completer: "Complete the next message" (neutral completion)  # noqa: E501
- anti_helper: "You are NOT an AI assistant" (anti-AI framing)  # noqa: E501
  # noqa: E501
Usage:  # noqa: E501
    echo '{"vars": {"context": "...", "strategy": "xml_drafter"}}' | python jarvis_provider.py  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from __future__ import annotations  # noqa: E402  # noqa: E501

# noqa: E501
import json  # noqa: E501
import logging  # noqa: E501
import os  # noqa: E501
import sys  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501

  # noqa: E501
PROJECT_ROOT = Path(__file__).parent.parent  # noqa: E501
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E501
# Ensure model-relative paths resolve regardless of promptfoo basePath/cwd.  # noqa: E501
os.chdir(PROJECT_ROOT)  # noqa: E501
  # noqa: E501
from jarvis.prompts.generation_config import DEFAULT_REPETITION_PENALTY  # noqa: E402  # noqa: E501

  # noqa: E501
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s", stream=sys.stderr)  # noqa: E501
logger = logging.getLogger(__name__)  # noqa: E501
  # noqa: E501
  # noqa: E501
# =============================================================================  # noqa: E501
# Framing Definitions  # noqa: E501
# =============================================================================  # noqa: E501
  # noqa: E501
FRAMINGS = {  # noqa: E501
    "drafter": ("You draft text message replies matching the sender's exact style."),  # noqa: E501
    "persona": (  # noqa: E501
        "You ARE this person texting. Continue the conversation as them. "  # noqa: E501
        "Write exactly how they write."  # noqa: E501
    ),  # noqa: E501
    "completer": (  # noqa: E501
        "Complete the next message in this text conversation. "  # noqa: E501
        "Output only the reply text, nothing else."  # noqa: E501
    ),  # noqa: E501
    "anti_helper": (  # noqa: E501
        "You are NOT an AI assistant. You are replying to a text message from your phone. "  # noqa: E501
        "Just text back. No helpfulness, no formality, no assistant behavior."  # noqa: E501
    ),  # noqa: E501
}  # noqa: E501
  # noqa: E501
# Shared rules appended to all framings  # noqa: E501
SHARED_RULES = """Rules:  # noqa: E501
- Match their texting style exactly (length, formality, abbreviations, emoji, punctuation)  # noqa: E501
- Sound natural, never like an AI  # noqa: E501
- No phrases like "I hope this helps" or "Let me know"  # noqa: E501
- No formal greetings unless they use them  # noqa: E501
- If the message is unclear or you lack context to reply properly, respond with just "?"  # noqa: E501
"""  # noqa: E501
  # noqa: E501
  # noqa: E501
# =============================================================================  # noqa: E501
# XML Structure Strategies  # noqa: E501
# =============================================================================  # noqa: E501
  # noqa: E501
  # noqa: E501
def _xml_strategy(  # noqa: E501
    framing: str,  # noqa: E501
    context: str,  # noqa: E501
    last_message: str,  # noqa: E501
    tone: str,  # noqa: E501
    user_style: str,  # noqa: E501
    **kwargs,  # noqa: E501
) -> str:  # noqa: E501
    """Build an XML-tagged prompt with the given framing."""  # noqa: E501
    system_text = FRAMINGS[framing]  # noqa: E501
    style_line = f"Tone: {tone}. Style: {user_style}" if user_style else f"Tone: {tone}"  # noqa: E501
  # noqa: E501
    return f"""<system>  # noqa: E501
{system_text}  # noqa: E501
{SHARED_RULES}</system>  # noqa: E501
  # noqa: E501
<style>  # noqa: E501
{style_line}  # noqa: E501
</style>  # noqa: E501
  # noqa: E501
<conversation>  # noqa: E501
{context}  # noqa: E501
</conversation>  # noqa: E501
  # noqa: E501
<last_message>{last_message}</last_message>  # noqa: E501
  # noqa: E501
<reply>"""  # noqa: E501
  # noqa: E501
  # noqa: E501
def strategy_xml_drafter(context: str, last_message: str, tone: str, user_style: str, **kw) -> str:  # noqa: E501
    """XML structure + drafter framing (current default)."""  # noqa: E501
    return _xml_strategy("drafter", context, last_message, tone, user_style, **kw)  # noqa: E501
  # noqa: E501
  # noqa: E501
def strategy_xml_persona(context: str, last_message: str, tone: str, user_style: str, **kw) -> str:  # noqa: E501
    """XML structure + persona framing."""  # noqa: E501
    return _xml_strategy("persona", context, last_message, tone, user_style, **kw)  # noqa: E501
  # noqa: E501
  # noqa: E501
def strategy_xml_completer(  # noqa: E501
    context: str, last_message: str, tone: str, user_style: str, **kw  # noqa: E501
) -> str:  # noqa: E501
    """XML structure + completer framing."""  # noqa: E501
    return _xml_strategy("completer", context, last_message, tone, user_style, **kw)  # noqa: E501
  # noqa: E501
  # noqa: E501
def strategy_xml_anti_helper(  # noqa: E501
    context: str, last_message: str, tone: str, user_style: str, **kw  # noqa: E501
) -> str:  # noqa: E501
    """XML structure + anti-helper framing."""  # noqa: E501
    return _xml_strategy("anti_helper", context, last_message, tone, user_style, **kw)  # noqa: E501
  # noqa: E501
  # noqa: E501
# =============================================================================  # noqa: E501
# Markdown Structure Strategies  # noqa: E501
# =============================================================================  # noqa: E501
  # noqa: E501
  # noqa: E501
def _md_strategy(  # noqa: E501
    framing: str,  # noqa: E501
    context: str,  # noqa: E501
    last_message: str,  # noqa: E501
    tone: str,  # noqa: E501
    user_style: str,  # noqa: E501
    **kwargs,  # noqa: E501
) -> str:  # noqa: E501
    """Build a markdown ###-header prompt with the given framing."""  # noqa: E501
    system_text = FRAMINGS[framing]  # noqa: E501
    style_line = f"Style: {user_style}" if user_style else ""  # noqa: E501
  # noqa: E501
    return f"""{system_text}  # noqa: E501
  # noqa: E501
### Conversation Context:  # noqa: E501
{context}  # noqa: E501
  # noqa: E501
### Instructions:  # noqa: E501
Generate a reply that matches the user's texting style exactly:  # noqa: E501
- Match the tone of the conversation ({tone})  # noqa: E501
- Keep response length similar to user's typical messages  # noqa: E501
- Sound like the user wrote it, not an AI  # noqa: E501
- If the message is unclear or you lack context, respond with just "?"  # noqa: E501
{style_line}  # noqa: E501
  # noqa: E501
### Last message to reply to:  # noqa: E501
{last_message}  # noqa: E501
  # noqa: E501
### Your reply:"""  # noqa: E501
  # noqa: E501
  # noqa: E501
def strategy_md_drafter(context: str, last_message: str, tone: str, user_style: str, **kw) -> str:  # noqa: E501
    """Markdown structure + drafter framing."""  # noqa: E501
    return _md_strategy("drafter", context, last_message, tone, user_style, **kw)  # noqa: E501
  # noqa: E501
  # noqa: E501
def strategy_md_persona(context: str, last_message: str, tone: str, user_style: str, **kw) -> str:  # noqa: E501
    """Markdown structure + persona framing."""  # noqa: E501
    return _md_strategy("persona", context, last_message, tone, user_style, **kw)  # noqa: E501
  # noqa: E501
  # noqa: E501
def strategy_md_completer(context: str, last_message: str, tone: str, user_style: str, **kw) -> str:  # noqa: E501
    """Markdown structure + completer framing."""  # noqa: E501
    return _md_strategy("completer", context, last_message, tone, user_style, **kw)  # noqa: E501
  # noqa: E501
  # noqa: E501
def strategy_md_anti_helper(  # noqa: E501
    context: str, last_message: str, tone: str, user_style: str, **kw  # noqa: E501
) -> str:  # noqa: E501
    """Markdown structure + anti-helper framing."""  # noqa: E501
    return _md_strategy("anti_helper", context, last_message, tone, user_style, **kw)  # noqa: E501
  # noqa: E501
  # noqa: E501
# =============================================================================  # noqa: E501
# Strategy Registry  # noqa: E501
# =============================================================================  # noqa: E501
  # noqa: E501
STRATEGIES = {  # noqa: E501
    "xml_drafter": strategy_xml_drafter,  # noqa: E501
    "xml_persona": strategy_xml_persona,  # noqa: E501
    "xml_completer": strategy_xml_completer,  # noqa: E501
    "xml_anti_helper": strategy_xml_anti_helper,  # noqa: E501
    "md_drafter": strategy_md_drafter,  # noqa: E501
    "md_persona": strategy_md_persona,  # noqa: E501
    "md_completer": strategy_md_completer,  # noqa: E501
    "md_anti_helper": strategy_md_anti_helper,  # noqa: E501
}  # noqa: E501
  # noqa: E501
  # noqa: E501
# =============================================================================  # noqa: E501
# Model Loading & Generation  # noqa: E501
# =============================================================================  # noqa: E501
  # noqa: E501
  # noqa: E501
def load_model():  # noqa: E501
    """Load the JARVIS MLX model."""  # noqa: E501
    # Set MLX memory limits before loading to prevent swap thrashing on 8GB systems  # noqa: E501
    from models.memory_config import apply_embedder_limits  # noqa: E501
  # noqa: E501
    apply_embedder_limits()  # noqa: E501
  # noqa: E501
    from models.loader import get_model  # noqa: E501
  # noqa: E501
    loader = get_model()  # noqa: E501
    if not loader.is_loaded():  # noqa: E501
        loader.load()  # noqa: E501
    return loader  # noqa: E501
  # noqa: E501
  # noqa: E501
def generate_reply(prompt: str, config: dict) -> str:  # noqa: E501
    """Generate a reply using the JARVIS model and return plain text."""  # noqa: E501
    try:  # noqa: E501
        loader = load_model()  # noqa: E501
        result = loader.generate_sync(  # noqa: E501
            prompt=prompt,  # noqa: E501
            temperature=config.get("temperature", 0.1),  # noqa: E501
            max_tokens=config.get("max_tokens", 50),  # noqa: E501
            top_p=config.get("top_p", 0.1),  # noqa: E501
            top_k=config.get("top_k", 50),  # noqa: E501
            repetition_penalty=config.get(  # noqa: E501
                "repetition_penalty",  # noqa: E501
                DEFAULT_REPETITION_PENALTY,  # From generation_config  # noqa: E501
            ),  # noqa: E501
        )  # noqa: E501
        return result.text.strip()  # noqa: E501
    except Exception as e:  # noqa: E501
        logger.error("Generation failed: %s", e)  # noqa: E501
        return f"[ERROR] {e}"  # noqa: E501
  # noqa: E501
  # noqa: E501
# =============================================================================  # noqa: E501
# Main  # noqa: E501
# =============================================================================  # noqa: E501
  # noqa: E501
  # noqa: E501
def main():  # noqa: E501
    # promptfoo exec provider passes: argv[1]=prompt, argv[2]=config_json, argv[3]=context_json  # noqa: E501
    if len(sys.argv) < 4:  # noqa: E501
        print(json.dumps({"error": f"Expected 3 args, got {len(sys.argv) - 1}"}), flush=True)  # noqa: E501
        sys.exit(1)  # noqa: E501
  # noqa: E501
    raw_prompt = sys.argv[1]  # e.g. "strategy:xml_drafter"  # noqa: E501
    try:  # noqa: E501
        config = json.loads(sys.argv[2]).get("config", {})  # noqa: E501
    except json.JSONDecodeError:  # noqa: E501
        config = {}  # noqa: E501
    try:  # noqa: E501
        context_obj = json.loads(sys.argv[3])  # noqa: E501
    except json.JSONDecodeError as e:  # noqa: E501
        print(json.dumps({"error": f"Invalid context JSON: {e}"}), flush=True)  # noqa: E501
        sys.exit(1)  # noqa: E501
  # noqa: E501
    vars_dict = context_obj.get("vars", {})  # noqa: E501
  # noqa: E501
    # Extract strategy from prompt string (format: "strategy:xml_drafter")  # noqa: E501
    if ":" in raw_prompt:  # noqa: E501
        strategy_name = raw_prompt.split(":", 1)[1]  # noqa: E501
    else:  # noqa: E501
        strategy_name = vars_dict.get("strategy", "xml_drafter")  # noqa: E501
  # noqa: E501
    # Extract test case variables  # noqa: E501
    context = vars_dict.get("context", "")  # noqa: E501
    last_message = vars_dict.get("last_message", "")  # noqa: E501
    tone = vars_dict.get("tone", "casual")  # noqa: E501
    user_style = vars_dict.get("user_style", "")  # noqa: E501
  # noqa: E501
    # Get the strategy function  # noqa: E501
    strategy_fn = STRATEGIES.get(strategy_name, strategy_xml_drafter)  # noqa: E501
  # noqa: E501
    # Build prompt using selected strategy  # noqa: E501
    try:  # noqa: E501
        prompt = strategy_fn(  # noqa: E501
            context=context,  # noqa: E501
            last_message=last_message,  # noqa: E501
            tone=tone,  # noqa: E501
            user_style=user_style,  # noqa: E501
        )  # noqa: E501
    except Exception as e:  # noqa: E501
        print(json.dumps({"error": f"Prompt build failed: {e}"}), flush=True)  # noqa: E501
        sys.exit(1)  # noqa: E501
  # noqa: E501
    # Generate and return  # noqa: E501
    result = generate_reply(prompt, config)  # noqa: E501
    print(result, flush=True)  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    main()  # noqa: E501
