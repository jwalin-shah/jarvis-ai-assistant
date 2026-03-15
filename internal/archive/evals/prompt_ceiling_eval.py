#!/usr/bin/env python3  # noqa: E501
"""Clean evaluation script for measuring prompting ceiling.  # noqa: E501
  # noqa: E501
Usage:  # noqa: E501
    # Test on real data with preprocessing  # noqa: E501
    uv run python evals/prompt_ceiling_eval.py --dataset real --prompt clean --limit 20  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from __future__ import annotations  # noqa: E402  # noqa: E501

# noqa: E501
import json  # noqa: E501
import logging  # noqa: E501
import os  # noqa: E501
import re  # noqa: E501
import sys  # noqa: E501
import time  # noqa: E501
from dataclasses import dataclass  # noqa: E402  # noqa: E501
from datetime import UTC, datetime  # noqa: E402  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501
from typing import Any  # noqa: E402  # noqa: E501

# noqa: E501
from tqdm import tqdm  # noqa: E402  # noqa: E501

  # noqa: E501
PROJECT_ROOT = Path(__file__).parent.parent  # noqa: E501
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E501
  # noqa: E501
_env_path = PROJECT_ROOT / ".env"  # noqa: E501
if _env_path.exists():  # noqa: E501
    for line in _env_path.read_text().splitlines():  # noqa: E501
        line = line.strip()  # noqa: E501
        if line and not line.startswith("#") and "=" in line:  # noqa: E501
            key, _, val = line.partition("=")  # noqa: E501
            os.environ.setdefault(key.strip(), val.strip())  # noqa: E501
  # noqa: E501
logging.basicConfig(level=logging.WARNING)  # noqa: E501
logger = logging.getLogger(__name__)  # noqa: E501
  # noqa: E501
EVAL_DATASET_PATH = PROJECT_ROOT / "evals" / "eval_dataset.jsonl"  # noqa: E501
RESULTS_DIR = PROJECT_ROOT / "results" / "prompt_ceiling"  # noqa: E501
  # noqa: E501
# Simplified prompts - no rules list, no abbreviation examples  # noqa: E501
CLEAN_PROMPT = """You are texting a friend. Reply briefly and casually."""  # noqa: E501
  # noqa: E501
ULTRA_CLEAN_PROMPT = """Reply:"""  # noqa: E501
  # noqa: E501
# Full system prompt from constants  # noqa: E501
DEFAULT_PROMPT = """You are Jwalin. Reply to text messages in your natural texting style.  # noqa: E501
Rules:  # noqa: E501
- Match your typical reply length (9 words avg)  # noqa: E501
- Use your abbreviations naturally: wanna, bc, gonna, kinda, btw  # noqa: E501
- No emoji usage  # noqa: E501
- Never sound like an AI assistant  # noqa: E501
- No formal greetings or sign-offs  # noqa: E501
- Just text back like you normally would  # noqa: E501
"""  # noqa: E501
  # noqa: E501
PROMPT_VARIANTS: dict[str, str] = {  # noqa: E501
    "clean": CLEAN_PROMPT,  # noqa: E501
    "ultra": ULTRA_CLEAN_PROMPT,  # noqa: E501
    "default": DEFAULT_PROMPT,  # noqa: E501
}  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class EvalExample:  # noqa: E501
    category: str  # noqa: E501
    context: list[str]  # noqa: E501
    last_message: str  # noqa: E501
    ideal_response: str  # noqa: E501
    contact_style: str  # noqa: E501
    notes: str  # noqa: E501
    contact_name: str = "Them"  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class EvalResult:  # noqa: E501
    example: EvalExample  # noqa: E501
    generated_response: str  # noqa: E501
    latency_ms: float  # noqa: E501
    judge_score: float | None = None  # noqa: E501
    judge_reasoning: str = ""  # noqa: E501
    prompt_variant: str = ""  # noqa: E501
  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# Input Preprocessing  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
  # noqa: E501
def clean_message(text: str) -> str:  # noqa: E501
    """Clean a single message."""  # noqa: E501
    if not text:  # noqa: E501
        return ""  # noqa: E501
  # noqa: E501
    # Remove attachment placeholders  # noqa: E501
    text = re.sub(r'\[?image\]?', '', text, flags=re.IGNORECASE)  # noqa: E501
    text = re.sub(r'\[?photo\]?', '', text, flags=re.IGNORECASE)  # noqa: E501
    text = re.sub(r'\[?link\]?', '', text, flags=re.IGNORECASE)  # noqa: E501
    text = re.sub(r'\[?video\]?', '', text, flags=re.IGNORECASE)  # noqa: E501
    text = re.sub(r'\[?voice memo\]?', '', text, flags=re.IGNORECASE)  # noqa: E501
    text = re.sub(r'\[?audio\]?', '', text, flags=re.IGNORECASE)  # noqa: E501
    text = re.sub(r'\[?Location\]?', '', text, flags=re.IGNORECASE)  # noqa: E501
  # noqa: E501
    # Remove unicode replacement character  # noqa: E501
    text = text.replace('\ufffc', '')  # noqa: E501
  # noqa: E501
    # Remove reaction prefixes  # noqa: E501
    text = re.sub(r'^(liked|loved|laughed at|emphasized|questioned)\s+["\']?', '', text, flags=re.IGNORECASE)  # noqa: E501
  # noqa: E501
    # Strip whitespace  # noqa: E501
    text = text.strip()  # noqa: E501
  # noqa: E501
    return text  # noqa: E501
  # noqa: E501
  # noqa: E501
def strip_phone_numbers(text: str) -> str:  # noqa: E501
    """Replace phone numbers with 'Them'."""  # noqa: E501
    # Pattern for +14025551234 style numbers  # noqa: E501
    return re.sub(r'\+\d{10,15}', 'Them', text)  # noqa: E501
  # noqa: E501
  # noqa: E501
def preprocess_conversation(context: list[str], last_message: str) -> tuple[list[str], str]:  # noqa: E501
    """Preprocess conversation for model input.  # noqa: E501
  # noqa: E501
    IMPORTANT: Keep the "Name: message" format so the model knows who's speaking.  # noqa: E501
    Just replace phone numbers with "Them" for consistency.  # noqa: E501
    """  # noqa: E501
    # Clean messages but keep name prefixes  # noqa: E501
    clean_context = []  # noqa: E501
    for m in context:  # noqa: E501
        m = clean_message(m)  # noqa: E501
        if m:  # noqa: E501
            # Replace phone numbers with "Them" but keep the prefix  # noqa: E501
            m = strip_phone_numbers(m)  # noqa: E501
            clean_context.append(m)  # noqa: E501
  # noqa: E501
    clean_last = clean_message(last_message)  # noqa: E501
    clean_last = strip_phone_numbers(clean_last)  # noqa: E501
  # noqa: E501
    # Keep up to 6 lines of context (enough for 3 turns back and forth)  # noqa: E501
    # Don't go too short - model needs context to understand the conversation  # noqa: E501
    if len(clean_context) > 6:  # noqa: E501
        clean_context = clean_context[-6:]  # noqa: E501
  # noqa: E501
    return clean_context, clean_last  # noqa: E501
  # noqa: E501
  # noqa: E501
def build_simple_prompt(system_prompt: str, context: list[str], last_message: str) -> str:  # noqa: E501
    """Build minimal ChatML prompt.  # noqa: E501
  # noqa: E501
    Keep the "Name: message" format so model understands conversation flow.  # noqa: E501
    """  # noqa: E501
    # Keep the "Name: message" format from context  # noqa: E501
    # For last_message, it should already have "Them: " prefix if we kept it  # noqa: E501
    conversation_lines = list(context)  # noqa: E501
  # noqa: E501
    # Add last message - if it doesn't have a prefix, add "Them:"  # noqa: E501
    if last_message and not last_message.startswith(("Them:", "Jwalin:")):  # noqa: E501
        last_message = "Them: " + last_message  # noqa: E501
    conversation_lines.append(last_message)  # noqa: E501
  # noqa: E501
    conversation = "\n".join(conversation_lines)  # noqa: E501
  # noqa: E501
    return f"""<|im_start|>system  # noqa: E501
{system_prompt}<|im_end|>  # noqa: E501
<|im_start|>user  # noqa: E501
{conversation}<|im_end|>  # noqa: E501
<|im_start|>assistant  # noqa: E501
Jwalin: """  # noqa: E501
  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# Output Post-processing  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
  # noqa: E501
def clean_output(text: str) -> str:  # noqa: E501
    """Clean generated output."""  # noqa: E501
    if not text:  # noqa: E501
        return ""  # noqa: E501
  # noqa: E501
    original = text  # noqa: E501
  # noqa: E501
    # Strip contact name prefixes (Them:, +123:, Jwalin:, etc.)  # noqa: E501
    text = re.sub(r'^(?:Them|Jwalin|[+]?\d{10,15}|\w+)\s*:\s*', '', text)  # noqa: E501
  # noqa: E501
    # Strip quoted content  # noqa: E501
    text = re.sub(r'^"([^"]*)"\s*', r'\1', text)  # noqa: E501
  # noqa: E501
    # Strip markdown artifacts  # noqa: E501
    text = re.sub(r'\*\*', '', text)  # noqa: E501
    text = re.sub(r'\*\s*', '', text)  # noqa: E501
  # noqa: E501
    # Strip stage directions  # noqa: E501
    text = re.sub(r'\([^)]*\)', '', text)  # noqa: E501
  # noqa: E501
    # Strip meta-commentary patterns  # noqa: E501
    text = re.sub(r'is a phrase that means.*', '', text, flags=re.IGNORECASE)  # noqa: E501
    text = re.sub(r'seems like.*', '', text, flags=re.IGNORECASE)  # noqa: E501
    text = re.sub(r'it sounds like.*', '', text, flags=re.IGNORECASE)  # noqa: E501
    text = re.sub(r'sounds like.*', '', text, flags=re.IGNORECASE)  # noqa: E501
  # noqa: E501
    # Strip emojis  # noqa: E501
    emoji_pattern = re.compile(  # noqa: E501
        '[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'  # noqa: E501
        '\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+'  # noqa: E501
    )  # noqa: E501
    text = emoji_pattern.sub('', text)  # noqa: E501
  # noqa: E501
    # Normalize whitespace  # noqa: E501
    text = re.sub(r'\s+', ' ', text).strip()  # noqa: E501
  # noqa: E501
    # Remove trailing fragments  # noqa: E501
    text = re.sub(r'[,;\s]+$', '', text)  # noqa: E501
  # noqa: E501
    if text != original:  # noqa: E501
        logger.debug(f"Cleaned: '{original[:50]}...' -> '{text[:50]}...'")  # noqa: E501
  # noqa: E501
    return text if text else original.strip()  # noqa: E501
  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# Dataset Loading  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
  # noqa: E501
def load_training_format_dataset(path: Path) -> list[EvalExample]:  # noqa: E501
    """Load training-format dataset (messages format).  # noqa: E501
  # noqa: E501
    Keep the "Name: message" format throughout.  # noqa: E501
    """  # noqa: E501
    examples = []  # noqa: E501
    for line in path.read_text().splitlines():  # noqa: E501
        line = line.strip()  # noqa: E501
        if not line:  # noqa: E501
            continue  # noqa: E501
        data = json.loads(line)  # noqa: E501
  # noqa: E501
        messages = data.get("messages", [])  # noqa: E501
        if len(messages) < 3:  # noqa: E501
            continue  # noqa: E501
  # noqa: E501
        user_content = messages[1].get("content", "") if len(messages) > 1 else ""  # noqa: E501
        ideal_response = messages[2].get("content", "") if len(messages) > 2 else ""  # noqa: E501
  # noqa: E501
        # Parse conversation - keep the "Name: message" format  # noqa: E501
        lines = user_content.split("\n")  # noqa: E501
  # noqa: E501
        # All lines except last are context  # noqa: E501
        # Last line is what we need to reply to  # noqa: E501
        if len(lines) >= 2:  # noqa: E501
            context = lines[:-1]  # noqa: E501
            last_message = lines[-1]  # noqa: E501
        elif len(lines) == 1:  # noqa: E501
            context = []  # noqa: E501
            last_message = lines[0]  # noqa: E501
        else:  # noqa: E501
            continue  # noqa: E501
  # noqa: E501
        examples.append(  # noqa: E501
            EvalExample(  # noqa: E501
                category="statement",  # noqa: E501
                context=context,  # noqa: E501
                last_message=last_message,  # noqa: E501
                ideal_response=ideal_response,  # noqa: E501
                contact_style="casual",  # noqa: E501
                notes="real_message",  # noqa: E501
                contact_name="Them",  # noqa: E501
            )  # noqa: E501
        )  # noqa: E501
    return examples  # noqa: E501
  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# Generation  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
  # noqa: E501
def load_generator(model_id: str | None = None):  # noqa: E501
    """Load the MLX generator."""  # noqa: E501
    from models.generator import MLXGenerator  # noqa: E501
    from models.loader import MLXModelLoader, ModelConfig  # noqa: E501
  # noqa: E501
    config = ModelConfig(model_id=model_id)  # noqa: E501
    loader = MLXModelLoader(config)  # noqa: E501
    generator = MLXGenerator(loader=loader, config=config, skip_templates=True)  # noqa: E501
    return generator  # noqa: E501
  # noqa: E501
  # noqa: E501
def generate_reply(  # noqa: E501
    generator,  # noqa: E501
    system_prompt: str,  # noqa: E501
    context: list[str],  # noqa: E501
    last_message: str,  # noqa: E501
) -> tuple[str, float]:  # noqa: E501
    """Generate a reply with preprocessing and constraints."""  # noqa: E501
    from jarvis.contracts.models import GenerationRequest  # noqa: E501
  # noqa: E501
    # Preprocess input  # noqa: E501
    clean_context, clean_last = preprocess_conversation(context, last_message)  # noqa: E501
  # noqa: E501
    # Build prompt  # noqa: E501
    prompt = build_simple_prompt(system_prompt, clean_context, clean_last)  # noqa: E501
  # noqa: E501
    # Create generation request with constraints  # noqa: E501
    # Note: GenerationRequest doesn't support logit_bias directly in this version  # noqa: E501
    # We'll handle constraints via stop_sequences and post-processing  # noqa: E501
    request = GenerationRequest(  # noqa: E501
        prompt=prompt,  # noqa: E501
        max_tokens=20,  # Shorter to prevent run-on  # noqa: E501
        temperature=0.4,  # noqa: E501
        repetition_penalty=1.1,  # Lower as suggested  # noqa: E501
        stop_sequences=[  # noqa: E501
            "<|im_end|>",  # noqa: E501
            "<|im_start|>",  # noqa: E501
            "\n",  # noqa: E501
            "Them:",  # Prevent contact name echo  # noqa: E501
            ":",  # Prevent "Name:" patterns  # noqa: E501
        ],  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    start_time = time.perf_counter()  # noqa: E501
    response = generator.generate(request)  # noqa: E501
    latency_ms = (time.perf_counter() - start_time) * 1000  # noqa: E501
  # noqa: E501
    # Post-process output  # noqa: E501
    cleaned_text = clean_output(response.text)  # noqa: E501
  # noqa: E501
    return cleaned_text, latency_ms  # noqa: E501
  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# Judge  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
  # noqa: E501
def _extract_json_blob(text: str) -> str:  # noqa: E501
    """Extract JSON object from text."""  # noqa: E501
    text = text.strip()  # noqa: E501
    if text.startswith("```"):  # noqa: E501
        parts = text.split("```")  # noqa: E501
        if len(parts) >= 2:  # noqa: E501
            text = parts[1]  # noqa: E501
        if text.startswith("json"):  # noqa: E501
            text = text[4:]  # noqa: E501
  # noqa: E501
    start = text.find("{")  # noqa: E501
    end = text.rfind("}")  # noqa: E501
    if start != -1 and end != -1 and end > start:  # noqa: E501
        return text[start:end+1]  # noqa: E501
    return text  # noqa: E501
  # noqa: E501
  # noqa: E501
def judge_reply(  # noqa: E501
    judge_client,  # noqa: E501
    judge_model: str,  # noqa: E501
    example: EvalExample,  # noqa: E501
    generated: str,  # noqa: E501
) -> tuple[float | None, str]:  # noqa: E501
    """Judge a single reply."""  # noqa: E501
    try:  # noqa: E501
        # Build conversation history for judge  # noqa: E501
        context_str = chr(10).join(example.context[-6:] + [example.last_message])  # noqa: E501
  # noqa: E501
        prompt = (  # noqa: E501
            "You are an expert evaluator for a text message reply generator.\n\n"  # noqa: E501
            f"CONVERSATION:\n{context_str}\n\n"  # noqa: E501
            f"IDEAL RESPONSE:\n{example.ideal_response}\n\n"  # noqa: E501
            f"GENERATED REPLY:\n{generated}\n\n"  # noqa: E501
            f"CATEGORY: {example.category}\n"  # noqa: E501
            f"NOTES: {example.notes}\n\n"  # noqa: E501
            "Score the generated reply from 0-10. Consider:\n"  # noqa: E501
            "- Does it match the tone and intent of the ideal response?\n"  # noqa: E501
            "- Does it sound like a real person texting (not an AI)?\n"  # noqa: E501
            "- Is it appropriate for the category?\n"  # noqa: E501
            "- Is the length appropriate?\n\n"  # noqa: E501
            'Respond in JSON: {"score": <0-10>, "reasoning": "<1-2 sentences>"}'  # noqa: E501
        )  # noqa: E501
  # noqa: E501
        resp = judge_client.chat.completions.create(  # noqa: E501
            model=judge_model,  # noqa: E501
            messages=[{"role": "user", "content": prompt}],  # noqa: E501
            temperature=0.0,  # noqa: E501
            max_tokens=150,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
        content = resp.choices[0].message.content or ""  # noqa: E501
        payload = json.loads(_extract_json_blob(content))  # noqa: E501
  # noqa: E501
        score = float(payload["score"])  # noqa: E501
        score = max(0.0, min(10.0, score))  # noqa: E501
  # noqa: E501
        return score, str(payload.get("reasoning", ""))  # noqa: E501
    except Exception as e:  # noqa: E501
        logger.error(f"Judge error: {e}")  # noqa: E501
        return None, f"judge error: {e}"  # noqa: E501
  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# Evaluation  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
  # noqa: E501
def run_evaluation(  # noqa: E501
    prompt_variant: str,  # noqa: E501
    examples: list[EvalExample],  # noqa: E501
    use_judge: bool = True,  # noqa: E501
    model_id: str | None = None,  # noqa: E501
) -> list[EvalResult]:  # noqa: E501
    """Run evaluation with specified prompt variant."""  # noqa: E501
  # noqa: E501
    print(f"Loading generator (model: {model_id or 'default'})...")  # noqa: E501
    generator = load_generator(model_id=model_id)  # noqa: E501
  # noqa: E501
    system_prompt = PROMPT_VARIANTS.get(prompt_variant, CLEAN_PROMPT)  # noqa: E501
  # noqa: E501
    judge_client = None  # noqa: E501
    judge_model = None  # noqa: E501
    if use_judge:  # noqa: E501
        from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E501
        judge_client = get_judge_client()  # noqa: E501
        judge_model = JUDGE_MODEL  # noqa: E501
        if judge_client:  # noqa: E501
            print(f"Judge ready: {judge_model}")  # noqa: E501
        else:  # noqa: E501
            print("WARNING: Judge client not available")  # noqa: E501
  # noqa: E501
    results = []  # noqa: E501
  # noqa: E501
    for ex in tqdm(examples, desc=f"Evaluating ({prompt_variant})"):  # noqa: E501
        try:  # noqa: E501
            generated, latency_ms = generate_reply(  # noqa: E501
                generator=generator,  # noqa: E501
                system_prompt=system_prompt,  # noqa: E501
                context=ex.context,  # noqa: E501
                last_message=ex.last_message,  # noqa: E501
            )  # noqa: E501
        except Exception as e:  # noqa: E501
            logger.error(f"Generation error: {e}")  # noqa: E501
            generated = f"[ERROR: {e}]"  # noqa: E501
            latency_ms = 0.0  # noqa: E501
  # noqa: E501
        judge_score = None  # noqa: E501
        judge_reasoning = ""  # noqa: E501
        if judge_client and use_judge and not generated.startswith("[ERROR"):  # noqa: E501
            judge_score, judge_reasoning = judge_reply(  # noqa: E501
                judge_client, judge_model, ex, generated  # noqa: E501
            )  # noqa: E501
  # noqa: E501
        result = EvalResult(  # noqa: E501
            example=ex,  # noqa: E501
            generated_response=generated,  # noqa: E501
            latency_ms=latency_ms,  # noqa: E501
            judge_score=judge_score,  # noqa: E501
            judge_reasoning=judge_reasoning,  # noqa: E501
            prompt_variant=prompt_variant,  # noqa: E501
        )  # noqa: E501
        results.append(result)  # noqa: E501
  # noqa: E501
        print(f"[{ex.category}] {ex.last_message[:40]}...")  # noqa: E501
        print(f"  Generated: {generated[:60]}")  # noqa: E501
        if judge_score is not None:  # noqa: E501
            print(f"  Judge: {judge_score:.1f}/10")  # noqa: E501
        print()  # noqa: E501
  # noqa: E501
    return results  # noqa: E501
  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# Reporting  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
  # noqa: E501
def print_summary(results: list[EvalResult], variant: str) -> dict[str, Any]:  # noqa: E501
    """Print and return summary statistics."""  # noqa: E501
    print("=" * 70)  # noqa: E501
    print(f"SUMMARY: {variant}")  # noqa: E501
    print("=" * 70)  # noqa: E501
  # noqa: E501
    n = len(results)  # noqa: E501
    if n == 0:  # noqa: E501
        print("No results.")  # noqa: E501
        return {}  # noqa: E501
  # noqa: E501
    latencies = [r.latency_ms for r in results]  # noqa: E501
    avg_latency = sum(latencies) / n  # noqa: E501
  # noqa: E501
    scored = [r for r in results if r.judge_score is not None]  # noqa: E501
    if scored:  # noqa: E501
        scores = [r.judge_score for r in scored]  # noqa: E501
        avg_score = sum(scores) / len(scores)  # noqa: E501
        pass_7 = sum(1 for s in scores if s >= 7)  # noqa: E501
  # noqa: E501
        print(f"Examples:           {n}")  # noqa: E501
        print(f"Avg latency:        {avg_latency:.0f}ms")  # noqa: E501
        print(f"Judge avg:          {avg_score:.2f}/10")  # noqa: E501
        print(f"Judge pass (>=7):   {pass_7}/{len(scored)} ({pass_7/len(scored)*100:.0f}%)")  # noqa: E501
        print()  # noqa: E501
  # noqa: E501
        low = sorted(scored, key=lambda r: r.judge_score)[:3]  # noqa: E501
        print("Lowest scores:")  # noqa: E501
        for r in low:  # noqa: E501
            print(f"  [{r.judge_score:.0f}/10] {r.example.last_message[:40]}...")  # noqa: E501
            print(f"           Generated: {r.generated_response[:50]}")  # noqa: E501
        print()  # noqa: E501
  # noqa: E501
        return {  # noqa: E501
            "variant": variant,  # noqa: E501
            "n_examples": n,  # noqa: E501
            "avg_latency_ms": round(avg_latency, 1),  # noqa: E501
            "judge_avg": round(avg_score, 2),  # noqa: E501
            "judge_pass_rate": round(pass_7 / len(scored), 3),  # noqa: E501
        }  # noqa: E501
    else:  # noqa: E501
        print(f"Examples:    {n}")  # noqa: E501
        print(f"Avg latency: {avg_latency:.0f}ms")  # noqa: E501
        return {  # noqa: E501
            "variant": variant,  # noqa: E501
            "n_examples": n,  # noqa: E501
            "avg_latency_ms": round(avg_latency, 1),  # noqa: E501
        }  # noqa: E501
  # noqa: E501
  # noqa: E501
def save_results(results: list[EvalResult], variant: str) -> Path:  # noqa: E501
    """Save results to file."""  # noqa: E501
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # noqa: E501
  # noqa: E501
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")  # noqa: E501
    filename = f"{variant}_{timestamp}.json"  # noqa: E501
    output_path = RESULTS_DIR / filename  # noqa: E501
  # noqa: E501
    output_data = {  # noqa: E501
        "timestamp": timestamp,  # noqa: E501
        "prompt_variant": variant,  # noqa: E501
        "system_prompt": PROMPT_VARIANTS.get(variant, ""),  # noqa: E501
        "results": [  # noqa: E501
            {  # noqa: E501
                "category": r.example.category,  # noqa: E501
                "context": r.example.context,  # noqa: E501
                "last_message": r.example.last_message,  # noqa: E501
                "ideal_response": r.example.ideal_response,  # noqa: E501
                "generated_response": r.generated_response,  # noqa: E501
                "latency_ms": round(r.latency_ms, 1),  # noqa: E501
                "judge_score": r.judge_score,  # noqa: E501
                "judge_reasoning": r.judge_reasoning,  # noqa: E501
            }  # noqa: E501
            for r in results  # noqa: E501
        ],  # noqa: E501
    }  # noqa: E501
  # noqa: E501
    scored = [r for r in results if r.judge_score is not None]  # noqa: E501
    if scored:  # noqa: E501
        output_data["summary"] = {  # noqa: E501
            "n_examples": len(results),  # noqa: E501
            "avg_latency_ms": round(sum(r.latency_ms for r in results) / len(results), 1),  # noqa: E501
            "judge_avg": round(sum(r.judge_score for r in scored) / len(scored), 2),  # noqa: E501
            "judge_pass_rate": round(sum(1 for r in scored if r.judge_score >= 7) / len(scored), 3),  # noqa: E501
        }  # noqa: E501
  # noqa: E501
    output_path.write_text(json.dumps(output_data, indent=2))  # noqa: E501
    print(f"Results saved to: {output_path}")  # noqa: E501
    return output_path  # noqa: E501
  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# Main  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
  # noqa: E501
def main() -> int:  # noqa: E501
    import argparse  # noqa: E501
  # noqa: E501
    parser = argparse.ArgumentParser(description="Prompt Ceiling Evaluation")  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--prompt",  # noqa: E501
        choices=["clean", "ultra", "default", "all"],  # noqa: E501
        default="clean",  # noqa: E501
        help="Prompt variant to test",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--no-judge",  # noqa: E501
        action="store_true",  # noqa: E501
        help="Skip judge scoring",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--limit",  # noqa: E501
        type=int,  # noqa: E501
        help="Limit number of examples",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--dataset",  # noqa: E501
        choices=["synthetic", "real"],  # noqa: E501
        default="real",  # noqa: E501
        help="Dataset to use",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--model",  # noqa: E501
        type=str,  # noqa: E501
        default="lfm-1.2b",  # noqa: E501
        help="Model ID to use",  # noqa: E501
    )  # noqa: E501
    args = parser.parse_args()  # noqa: E501
  # noqa: E501
    # Load dataset  # noqa: E501
    if args.dataset == "real":  # noqa: E501
        real_test_path = PROJECT_ROOT / "data" / "personal" / "raw_style_variable" / "test.jsonl"  # noqa: E501
        if not real_test_path.exists():  # noqa: E501
            print(f"ERROR: Real test set not found at {real_test_path}", file=sys.stderr)  # noqa: E501
            return 1  # noqa: E501
        print(f"Loading real test set: {real_test_path}")  # noqa: E501
        examples = load_training_format_dataset(real_test_path)  # noqa: E501
    else:  # noqa: E501
        print("ERROR: Synthetic dataset not supported in this version", file=sys.stderr)  # noqa: E501
        return 1  # noqa: E501
  # noqa: E501
    if args.limit:  # noqa: E501
        examples = examples[:args.limit]  # noqa: E501
  # noqa: E501
    print("=" * 70)  # noqa: E501
    print("PROMPT CEILING EVALUATION - CLEAN VERSION")  # noqa: E501
    print("=" * 70)  # noqa: E501
    print(f"Model:     {args.model}")  # noqa: E501
    print(f"Dataset:   {args.dataset} ({len(examples)} examples)")  # noqa: E501
    print(f"Judge:     {'disabled' if args.no_judge else 'enabled'}")  # noqa: E501
    print()  # noqa: E501
  # noqa: E501
    variants = list(PROMPT_VARIANTS.keys()) if args.prompt == "all" else [args.prompt]  # noqa: E501
  # noqa: E501
    all_summaries = []  # noqa: E501
  # noqa: E501
    for variant in variants:  # noqa: E501
        print(f"\n{'='*70}")  # noqa: E501
        print(f"Testing prompt variant: {variant}")  # noqa: E501
        print(f"{'='*70}\n")  # noqa: E501
  # noqa: E501
        results = run_evaluation(  # noqa: E501
            prompt_variant=variant,  # noqa: E501
            examples=examples,  # noqa: E501
            use_judge=not args.no_judge,  # noqa: E501
            model_id=args.model,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
        summary = print_summary(results, variant)  # noqa: E501
        all_summaries.append(summary)  # noqa: E501
  # noqa: E501
        save_results(results, variant)  # noqa: E501
  # noqa: E501
    if len(all_summaries) > 1:  # noqa: E501
        print("\n" + "=" * 70)  # noqa: E501
        print("COMPARISON")  # noqa: E501
        print("=" * 70)  # noqa: E501
        print(f"{'Variant':<15} {'Judge Avg':<12} {'Pass Rate':<12} {'Latency':<10}")  # noqa: E501
        print("-" * 70)  # noqa: E501
        for s in all_summaries:  # noqa: E501
            if "judge_avg" in s:  # noqa: E501
                print(  # noqa: E501
                    f"{s['variant']:<15} "  # noqa: E501
                    f"{s['judge_avg']:<12.2f} "  # noqa: E501
                    f"{s['judge_pass_rate']:<12.1%} "  # noqa: E501
                    f"{s['avg_latency_ms']:<10.0f}ms"  # noqa: E501
                )  # noqa: E501
        print()  # noqa: E501
  # noqa: E501
    return 0  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    sys.exit(main())  # noqa: E501
