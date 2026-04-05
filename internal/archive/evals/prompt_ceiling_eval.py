#!/usr/bin/env python3
"""Clean evaluation script for measuring prompting ceiling.

Usage:
    # Test on real data with preprocessing
    uv run python evals/prompt_ceiling_eval.py --dataset real --prompt clean --limit 20
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
  # noqa: E402
_env_path = PROJECT_ROOT / ".env"  # noqa: E402
if _env_path.exists():  # noqa: E402
    for line in _env_path.read_text().splitlines():  # noqa: E402
        line = line.strip()  # noqa: E402
        if line and not line.startswith("#") and "=" in line:  # noqa: E402
            key, _, val = line.partition("=")  # noqa: E402
            os.environ.setdefault(key.strip(), val.strip())  # noqa: E402
  # noqa: E402
logging.basicConfig(level=logging.WARNING)  # noqa: E402
logger = logging.getLogger(__name__)  # noqa: E402
  # noqa: E402
EVAL_DATASET_PATH = PROJECT_ROOT / "evals" / "eval_dataset.jsonl"  # noqa: E402
RESULTS_DIR = PROJECT_ROOT / "results" / "prompt_ceiling"  # noqa: E402
  # noqa: E402
# Simplified prompts - no rules list, no abbreviation examples  # noqa: E402
CLEAN_PROMPT = """You are texting a friend. Reply briefly and casually."""  # noqa: E402
  # noqa: E402
ULTRA_CLEAN_PROMPT = """Reply:"""  # noqa: E402
  # noqa: E402
# Full system prompt from constants  # noqa: E402
DEFAULT_PROMPT = """You are Jwalin. Reply to text messages in your natural texting style.  # noqa: E402
Rules:  # noqa: E402
- Match your typical reply length (9 words avg)  # noqa: E402
- Use your abbreviations naturally: wanna, bc, gonna, kinda, btw  # noqa: E402
- No emoji usage  # noqa: E402
- Never sound like an AI assistant  # noqa: E402
- No formal greetings or sign-offs  # noqa: E402
- Just text back like you normally would  # noqa: E402
"""  # noqa: E402
  # noqa: E402
PROMPT_VARIANTS: dict[str, str] = {  # noqa: E402
    "clean": CLEAN_PROMPT,  # noqa: E402
    "ultra": ULTRA_CLEAN_PROMPT,  # noqa: E402
    "default": DEFAULT_PROMPT,  # noqa: E402
}  # noqa: E402
  # noqa: E402
  # noqa: E402
@dataclass  # noqa: E402
class EvalExample:  # noqa: E402
    category: str  # noqa: E402
    context: list[str]  # noqa: E402
    last_message: str  # noqa: E402
    ideal_response: str  # noqa: E402
    contact_style: str  # noqa: E402
    notes: str  # noqa: E402
    contact_name: str = "Them"  # noqa: E402
  # noqa: E402
  # noqa: E402
@dataclass  # noqa: E402
class EvalResult:  # noqa: E402
    example: EvalExample  # noqa: E402
    generated_response: str  # noqa: E402
    latency_ms: float  # noqa: E402
    judge_score: float | None = None  # noqa: E402
    judge_reasoning: str = ""  # noqa: E402
    prompt_variant: str = ""  # noqa: E402
  # noqa: E402
  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
# Input Preprocessing  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
  # noqa: E402
def clean_message(text: str) -> str:  # noqa: E402
    """Clean a single message."""  # noqa: E402
    if not text:  # noqa: E402
        return ""  # noqa: E402
  # noqa: E402
    # Remove attachment placeholders  # noqa: E402
    text = re.sub(r'\[?image\]?', '', text, flags=re.IGNORECASE)  # noqa: E402
    text = re.sub(r'\[?photo\]?', '', text, flags=re.IGNORECASE)  # noqa: E402
    text = re.sub(r'\[?link\]?', '', text, flags=re.IGNORECASE)  # noqa: E402
    text = re.sub(r'\[?video\]?', '', text, flags=re.IGNORECASE)  # noqa: E402
    text = re.sub(r'\[?voice memo\]?', '', text, flags=re.IGNORECASE)  # noqa: E402
    text = re.sub(r'\[?audio\]?', '', text, flags=re.IGNORECASE)  # noqa: E402
    text = re.sub(r'\[?Location\]?', '', text, flags=re.IGNORECASE)  # noqa: E402
  # noqa: E402
    # Remove unicode replacement character  # noqa: E402
    text = text.replace('\ufffc', '')  # noqa: E402
  # noqa: E402
    # Remove reaction prefixes  # noqa: E402
    text = re.sub(r'^(liked|loved|laughed at|emphasized|questioned)\s+["\']?', '', text, flags=re.IGNORECASE)  # noqa: E402
  # noqa: E402
    # Strip whitespace  # noqa: E402
    text = text.strip()  # noqa: E402
  # noqa: E402
    return text  # noqa: E402
  # noqa: E402
  # noqa: E402
def strip_phone_numbers(text: str) -> str:  # noqa: E402
    """Replace phone numbers with 'Them'."""  # noqa: E402
    # Pattern for +14025551234 style numbers  # noqa: E402
    return re.sub(r'\+\d{10,15}', 'Them', text)  # noqa: E402
  # noqa: E402
  # noqa: E402
def preprocess_conversation(context: list[str], last_message: str) -> tuple[list[str], str]:  # noqa: E402
    """Preprocess conversation for model input.  # noqa: E402
  # noqa: E402
    IMPORTANT: Keep the "Name: message" format so the model knows who's speaking.  # noqa: E402
    Just replace phone numbers with "Them" for consistency.  # noqa: E402
    """  # noqa: E402
    # Clean messages but keep name prefixes  # noqa: E402
    clean_context = []  # noqa: E402
    for m in context:  # noqa: E402
        m = clean_message(m)  # noqa: E402
        if m:  # noqa: E402
            # Replace phone numbers with "Them" but keep the prefix  # noqa: E402
            m = strip_phone_numbers(m)  # noqa: E402
            clean_context.append(m)  # noqa: E402
  # noqa: E402
    clean_last = clean_message(last_message)  # noqa: E402
    clean_last = strip_phone_numbers(clean_last)  # noqa: E402
  # noqa: E402
    # Keep up to 6 lines of context (enough for 3 turns back and forth)  # noqa: E402
    # Don't go too short - model needs context to understand the conversation  # noqa: E402
    if len(clean_context) > 6:  # noqa: E402
        clean_context = clean_context[-6:]  # noqa: E402
  # noqa: E402
    return clean_context, clean_last  # noqa: E402
  # noqa: E402
  # noqa: E402
def build_simple_prompt(system_prompt: str, context: list[str], last_message: str) -> str:  # noqa: E402
    """Build minimal ChatML prompt.  # noqa: E402
  # noqa: E402
    Keep the "Name: message" format so model understands conversation flow.  # noqa: E402
    """  # noqa: E402
    # Keep the "Name: message" format from context  # noqa: E402
    # For last_message, it should already have "Them: " prefix if we kept it  # noqa: E402
    conversation_lines = list(context)  # noqa: E402
  # noqa: E402
    # Add last message - if it doesn't have a prefix, add "Them:"  # noqa: E402
    if last_message and not last_message.startswith(("Them:", "Jwalin:")):  # noqa: E402
        last_message = "Them: " + last_message  # noqa: E402
    conversation_lines.append(last_message)  # noqa: E402
  # noqa: E402
    conversation = "\n".join(conversation_lines)  # noqa: E402
  # noqa: E402
    return f"""<|im_start|>system  # noqa: E402
{system_prompt}<|im_end|>  # noqa: E402
<|im_start|>user  # noqa: E402
{conversation}<|im_end|>  # noqa: E402
<|im_start|>assistant  # noqa: E402
Jwalin: """  # noqa: E402
  # noqa: E402
  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
# Output Post-processing  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
  # noqa: E402
def clean_output(text: str) -> str:  # noqa: E402
    """Clean generated output."""  # noqa: E402
    if not text:  # noqa: E402
        return ""  # noqa: E402
  # noqa: E402
    original = text  # noqa: E402
  # noqa: E402
    # Strip contact name prefixes (Them:, +123:, Jwalin:, etc.)  # noqa: E402
    text = re.sub(r'^(?:Them|Jwalin|[+]?\d{10,15}|\w+)\s*:\s*', '', text)  # noqa: E402
  # noqa: E402
    # Strip quoted content  # noqa: E402
    text = re.sub(r'^"([^"]*)"\s*', r'\1', text)  # noqa: E402
  # noqa: E402
    # Strip markdown artifacts  # noqa: E402
    text = re.sub(r'\*\*', '', text)  # noqa: E402
    text = re.sub(r'\*\s*', '', text)  # noqa: E402
  # noqa: E402
    # Strip stage directions  # noqa: E402
    text = re.sub(r'\([^)]*\)', '', text)  # noqa: E402
  # noqa: E402
    # Strip meta-commentary patterns  # noqa: E402
    text = re.sub(r'is a phrase that means.*', '', text, flags=re.IGNORECASE)  # noqa: E402
    text = re.sub(r'seems like.*', '', text, flags=re.IGNORECASE)  # noqa: E402
    text = re.sub(r'it sounds like.*', '', text, flags=re.IGNORECASE)  # noqa: E402
    text = re.sub(r'sounds like.*', '', text, flags=re.IGNORECASE)  # noqa: E402
  # noqa: E402
    # Strip emojis  # noqa: E402
    emoji_pattern = re.compile(  # noqa: E402
        '[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'  # noqa: E402
        '\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+'  # noqa: E402
    )  # noqa: E402
    text = emoji_pattern.sub('', text)  # noqa: E402
  # noqa: E402
    # Normalize whitespace  # noqa: E402
    text = re.sub(r'\s+', ' ', text).strip()  # noqa: E402
  # noqa: E402
    # Remove trailing fragments  # noqa: E402
    text = re.sub(r'[,;\s]+$', '', text)  # noqa: E402
  # noqa: E402
    if text != original:  # noqa: E402
        logger.debug(f"Cleaned: '{original[:50]}...' -> '{text[:50]}...'")  # noqa: E402
  # noqa: E402
    return text if text else original.strip()  # noqa: E402
  # noqa: E402
  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
# Dataset Loading  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
  # noqa: E402
def load_training_format_dataset(path: Path) -> list[EvalExample]:  # noqa: E402
    """Load training-format dataset (messages format).  # noqa: E402
  # noqa: E402
    Keep the "Name: message" format throughout.  # noqa: E402
    """  # noqa: E402
    examples = []  # noqa: E402
    for line in path.read_text().splitlines():  # noqa: E402
        line = line.strip()  # noqa: E402
        if not line:  # noqa: E402
            continue  # noqa: E402
        data = json.loads(line)  # noqa: E402
  # noqa: E402
        messages = data.get("messages", [])  # noqa: E402
        if len(messages) < 3:  # noqa: E402
            continue  # noqa: E402
  # noqa: E402
        user_content = messages[1].get("content", "") if len(messages) > 1 else ""  # noqa: E402
        ideal_response = messages[2].get("content", "") if len(messages) > 2 else ""  # noqa: E402
  # noqa: E402
        # Parse conversation - keep the "Name: message" format  # noqa: E402
        lines = user_content.split("\n")  # noqa: E402
  # noqa: E402
        # All lines except last are context  # noqa: E402
        # Last line is what we need to reply to  # noqa: E402
        if len(lines) >= 2:  # noqa: E402
            context = lines[:-1]  # noqa: E402
            last_message = lines[-1]  # noqa: E402
        elif len(lines) == 1:  # noqa: E402
            context = []  # noqa: E402
            last_message = lines[0]  # noqa: E402
        else:  # noqa: E402
            continue  # noqa: E402
  # noqa: E402
        examples.append(  # noqa: E402
            EvalExample(  # noqa: E402
                category="statement",  # noqa: E402
                context=context,  # noqa: E402
                last_message=last_message,  # noqa: E402
                ideal_response=ideal_response,  # noqa: E402
                contact_style="casual",  # noqa: E402
                notes="real_message",  # noqa: E402
                contact_name="Them",  # noqa: E402
            )  # noqa: E402
        )  # noqa: E402
    return examples  # noqa: E402
  # noqa: E402
  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
# Generation  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
  # noqa: E402
def load_generator(model_id: str | None = None):  # noqa: E402
    """Load the MLX generator."""  # noqa: E402
    from models.generator import MLXGenerator  # noqa: E402
    from models.loader import MLXModelLoader, ModelConfig  # noqa: E402
  # noqa: E402
    config = ModelConfig(model_id=model_id)  # noqa: E402
    loader = MLXModelLoader(config)  # noqa: E402
    generator = MLXGenerator(loader=loader, config=config, skip_templates=True)  # noqa: E402
    return generator  # noqa: E402
  # noqa: E402
  # noqa: E402
def generate_reply(  # noqa: E402
    generator,  # noqa: E402
    system_prompt: str,  # noqa: E402
    context: list[str],  # noqa: E402
    last_message: str,  # noqa: E402
) -> tuple[str, float]:  # noqa: E402
    """Generate a reply with preprocessing and constraints."""  # noqa: E402
    from jarvis.contracts.models import GenerationRequest  # noqa: E402
  # noqa: E402
    # Preprocess input  # noqa: E402
    clean_context, clean_last = preprocess_conversation(context, last_message)  # noqa: E402
  # noqa: E402
    # Build prompt  # noqa: E402
    prompt = build_simple_prompt(system_prompt, clean_context, clean_last)  # noqa: E402
  # noqa: E402
    # Create generation request with constraints  # noqa: E402
    # Note: GenerationRequest doesn't support logit_bias directly in this version  # noqa: E402
    # We'll handle constraints via stop_sequences and post-processing  # noqa: E402
    request = GenerationRequest(  # noqa: E402
        prompt=prompt,  # noqa: E402
        max_tokens=20,  # Shorter to prevent run-on  # noqa: E402
        temperature=0.4,  # noqa: E402
        repetition_penalty=1.1,  # Lower as suggested  # noqa: E402
        stop_sequences=[  # noqa: E402
            "<|im_end|>",  # noqa: E402
            "<|im_start|>",  # noqa: E402
            "\n",  # noqa: E402
            "Them:",  # Prevent contact name echo  # noqa: E402
            ":",  # Prevent "Name:" patterns  # noqa: E402
        ],  # noqa: E402
    )  # noqa: E402
  # noqa: E402
    start_time = time.perf_counter()  # noqa: E402
    response = generator.generate(request)  # noqa: E402
    latency_ms = (time.perf_counter() - start_time) * 1000  # noqa: E402
  # noqa: E402
    # Post-process output  # noqa: E402
    cleaned_text = clean_output(response.text)  # noqa: E402
  # noqa: E402
    return cleaned_text, latency_ms  # noqa: E402
  # noqa: E402
  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
# Judge  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
  # noqa: E402
def _extract_json_blob(text: str) -> str:  # noqa: E402
    """Extract JSON object from text."""  # noqa: E402
    text = text.strip()  # noqa: E402
    if text.startswith("```"):  # noqa: E402
        parts = text.split("```")  # noqa: E402
        if len(parts) >= 2:  # noqa: E402
            text = parts[1]  # noqa: E402
        if text.startswith("json"):  # noqa: E402
            text = text[4:]  # noqa: E402
  # noqa: E402
    start = text.find("{")  # noqa: E402
    end = text.rfind("}")  # noqa: E402
    if start != -1 and end != -1 and end > start:  # noqa: E402
        return text[start:end+1]  # noqa: E402
    return text  # noqa: E402
  # noqa: E402
  # noqa: E402
def judge_reply(  # noqa: E402
    judge_client,  # noqa: E402
    judge_model: str,  # noqa: E402
    example: EvalExample,  # noqa: E402
    generated: str,  # noqa: E402
) -> tuple[float | None, str]:  # noqa: E402
    """Judge a single reply."""  # noqa: E402
    try:  # noqa: E402
        # Build conversation history for judge  # noqa: E402
        context_str = chr(10).join(example.context[-6:] + [example.last_message])  # noqa: E402
  # noqa: E402
        prompt = (  # noqa: E402
            "You are an expert evaluator for a text message reply generator.\n\n"  # noqa: E402
            f"CONVERSATION:\n{context_str}\n\n"  # noqa: E402
            f"IDEAL RESPONSE:\n{example.ideal_response}\n\n"  # noqa: E402
            f"GENERATED REPLY:\n{generated}\n\n"  # noqa: E402
            f"CATEGORY: {example.category}\n"  # noqa: E402
            f"NOTES: {example.notes}\n\n"  # noqa: E402
            "Score the generated reply from 0-10. Consider:\n"  # noqa: E402
            "- Does it match the tone and intent of the ideal response?\n"  # noqa: E402
            "- Does it sound like a real person texting (not an AI)?\n"  # noqa: E402
            "- Is it appropriate for the category?\n"  # noqa: E402
            "- Is the length appropriate?\n\n"  # noqa: E402
            'Respond in JSON: {"score": <0-10>, "reasoning": "<1-2 sentences>"}'  # noqa: E402
        )  # noqa: E402
  # noqa: E402
        resp = judge_client.chat.completions.create(  # noqa: E402
            model=judge_model,  # noqa: E402
            messages=[{"role": "user", "content": prompt}],  # noqa: E402
            temperature=0.0,  # noqa: E402
            max_tokens=150,  # noqa: E402
        )  # noqa: E402
  # noqa: E402
        content = resp.choices[0].message.content or ""  # noqa: E402
        payload = json.loads(_extract_json_blob(content))  # noqa: E402
  # noqa: E402
        score = float(payload["score"])  # noqa: E402
        score = max(0.0, min(10.0, score))  # noqa: E402
  # noqa: E402
        return score, str(payload.get("reasoning", ""))  # noqa: E402
    except Exception as e:  # noqa: E402
        logger.error(f"Judge error: {e}")  # noqa: E402
        return None, f"judge error: {e}"  # noqa: E402
  # noqa: E402
  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
# Evaluation  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
  # noqa: E402
def run_evaluation(  # noqa: E402
    prompt_variant: str,  # noqa: E402
    examples: list[EvalExample],  # noqa: E402
    use_judge: bool = True,  # noqa: E402
    model_id: str | None = None,  # noqa: E402
) -> list[EvalResult]:  # noqa: E402
    """Run evaluation with specified prompt variant."""  # noqa: E402
  # noqa: E402
    print(f"Loading generator (model: {model_id or 'default'})...")  # noqa: E402
    generator = load_generator(model_id=model_id)  # noqa: E402
  # noqa: E402
    system_prompt = PROMPT_VARIANTS.get(prompt_variant, CLEAN_PROMPT)  # noqa: E402
  # noqa: E402
    judge_client = None  # noqa: E402
    judge_model = None  # noqa: E402
    if use_judge:  # noqa: E402
        from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402
        judge_client = get_judge_client()  # noqa: E402
        judge_model = JUDGE_MODEL  # noqa: E402
        if judge_client:  # noqa: E402
            print(f"Judge ready: {judge_model}")  # noqa: E402
        else:  # noqa: E402
            print("WARNING: Judge client not available")  # noqa: E402
  # noqa: E402
    results = []  # noqa: E402
  # noqa: E402
    for ex in tqdm(examples, desc=f"Evaluating ({prompt_variant})"):  # noqa: E402
        try:  # noqa: E402
            generated, latency_ms = generate_reply(  # noqa: E402
                generator=generator,  # noqa: E402
                system_prompt=system_prompt,  # noqa: E402
                context=ex.context,  # noqa: E402
                last_message=ex.last_message,  # noqa: E402
            )  # noqa: E402
        except Exception as e:  # noqa: E402
            logger.error(f"Generation error: {e}")  # noqa: E402
            generated = f"[ERROR: {e}]"  # noqa: E402
            latency_ms = 0.0  # noqa: E402
  # noqa: E402
        judge_score = None  # noqa: E402
        judge_reasoning = ""  # noqa: E402
        if judge_client and use_judge and not generated.startswith("[ERROR"):  # noqa: E402
            judge_score, judge_reasoning = judge_reply(  # noqa: E402
                judge_client, judge_model, ex, generated  # noqa: E402
            )  # noqa: E402
  # noqa: E402
        result = EvalResult(  # noqa: E402
            example=ex,  # noqa: E402
            generated_response=generated,  # noqa: E402
            latency_ms=latency_ms,  # noqa: E402
            judge_score=judge_score,  # noqa: E402
            judge_reasoning=judge_reasoning,  # noqa: E402
            prompt_variant=prompt_variant,  # noqa: E402
        )  # noqa: E402
        results.append(result)  # noqa: E402
  # noqa: E402
        print(f"[{ex.category}] {ex.last_message[:40]}...")  # noqa: E402
        print(f"  Generated: {generated[:60]}")  # noqa: E402
        if judge_score is not None:  # noqa: E402
            print(f"  Judge: {judge_score:.1f}/10")  # noqa: E402
        print()  # noqa: E402
  # noqa: E402
    return results  # noqa: E402
  # noqa: E402
  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
# Reporting  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
  # noqa: E402
def print_summary(results: list[EvalResult], variant: str) -> dict[str, Any]:  # noqa: E402
    """Print and return summary statistics."""  # noqa: E402
    print("=" * 70)  # noqa: E402
    print(f"SUMMARY: {variant}")  # noqa: E402
    print("=" * 70)  # noqa: E402
  # noqa: E402
    n = len(results)  # noqa: E402
    if n == 0:  # noqa: E402
        print("No results.")  # noqa: E402
        return {}  # noqa: E402
  # noqa: E402
    latencies = [r.latency_ms for r in results]  # noqa: E402
    avg_latency = sum(latencies) / n  # noqa: E402
  # noqa: E402
    scored = [r for r in results if r.judge_score is not None]  # noqa: E402
    if scored:  # noqa: E402
        scores = [r.judge_score for r in scored]  # noqa: E402
        avg_score = sum(scores) / len(scores)  # noqa: E402
        pass_7 = sum(1 for s in scores if s >= 7)  # noqa: E402
  # noqa: E402
        print(f"Examples:           {n}")  # noqa: E402
        print(f"Avg latency:        {avg_latency:.0f}ms")  # noqa: E402
        print(f"Judge avg:          {avg_score:.2f}/10")  # noqa: E402
        print(f"Judge pass (>=7):   {pass_7}/{len(scored)} ({pass_7/len(scored)*100:.0f}%)")  # noqa: E402
        print()  # noqa: E402
  # noqa: E402
        low = sorted(scored, key=lambda r: r.judge_score)[:3]  # noqa: E402
        print("Lowest scores:")  # noqa: E402
        for r in low:  # noqa: E402
            print(f"  [{r.judge_score:.0f}/10] {r.example.last_message[:40]}...")  # noqa: E402
            print(f"           Generated: {r.generated_response[:50]}")  # noqa: E402
        print()  # noqa: E402
  # noqa: E402
        return {  # noqa: E402
            "variant": variant,  # noqa: E402
            "n_examples": n,  # noqa: E402
            "avg_latency_ms": round(avg_latency, 1),  # noqa: E402
            "judge_avg": round(avg_score, 2),  # noqa: E402
            "judge_pass_rate": round(pass_7 / len(scored), 3),  # noqa: E402
        }  # noqa: E402
    else:  # noqa: E402
        print(f"Examples:    {n}")  # noqa: E402
        print(f"Avg latency: {avg_latency:.0f}ms")  # noqa: E402
        return {  # noqa: E402
            "variant": variant,  # noqa: E402
            "n_examples": n,  # noqa: E402
            "avg_latency_ms": round(avg_latency, 1),  # noqa: E402
        }  # noqa: E402
  # noqa: E402
  # noqa: E402
def save_results(results: list[EvalResult], variant: str) -> Path:  # noqa: E402
    """Save results to file."""  # noqa: E402
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # noqa: E402
  # noqa: E402
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")  # noqa: E402
    filename = f"{variant}_{timestamp}.json"  # noqa: E402
    output_path = RESULTS_DIR / filename  # noqa: E402
  # noqa: E402
    output_data = {  # noqa: E402
        "timestamp": timestamp,  # noqa: E402
        "prompt_variant": variant,  # noqa: E402
        "system_prompt": PROMPT_VARIANTS.get(variant, ""),  # noqa: E402
        "results": [  # noqa: E402
            {  # noqa: E402
                "category": r.example.category,  # noqa: E402
                "context": r.example.context,  # noqa: E402
                "last_message": r.example.last_message,  # noqa: E402
                "ideal_response": r.example.ideal_response,  # noqa: E402
                "generated_response": r.generated_response,  # noqa: E402
                "latency_ms": round(r.latency_ms, 1),  # noqa: E402
                "judge_score": r.judge_score,  # noqa: E402
                "judge_reasoning": r.judge_reasoning,  # noqa: E402
            }  # noqa: E402
            for r in results  # noqa: E402
        ],  # noqa: E402
    }  # noqa: E402
  # noqa: E402
    scored = [r for r in results if r.judge_score is not None]  # noqa: E402
    if scored:  # noqa: E402
        output_data["summary"] = {  # noqa: E402
            "n_examples": len(results),  # noqa: E402
            "avg_latency_ms": round(sum(r.latency_ms for r in results) / len(results), 1),  # noqa: E402
            "judge_avg": round(sum(r.judge_score for r in scored) / len(scored), 2),  # noqa: E402
            "judge_pass_rate": round(sum(1 for r in scored if r.judge_score >= 7) / len(scored), 3),  # noqa: E402
        }  # noqa: E402
  # noqa: E402
    output_path.write_text(json.dumps(output_data, indent=2))  # noqa: E402
    print(f"Results saved to: {output_path}")  # noqa: E402
    return output_path  # noqa: E402
  # noqa: E402
  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
# Main  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
  # noqa: E402
def main() -> int:  # noqa: E402
    import argparse  # noqa: E402
  # noqa: E402
    parser = argparse.ArgumentParser(description="Prompt Ceiling Evaluation")  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--prompt",  # noqa: E402
        choices=["clean", "ultra", "default", "all"],  # noqa: E402
        default="clean",  # noqa: E402
        help="Prompt variant to test",  # noqa: E402
    )  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--no-judge",  # noqa: E402
        action="store_true",  # noqa: E402
        help="Skip judge scoring",  # noqa: E402
    )  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--limit",  # noqa: E402
        type=int,  # noqa: E402
        help="Limit number of examples",  # noqa: E402
    )  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--dataset",  # noqa: E402
        choices=["synthetic", "real"],  # noqa: E402
        default="real",  # noqa: E402
        help="Dataset to use",  # noqa: E402
    )  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--model",  # noqa: E402
        type=str,  # noqa: E402
        default="lfm-1.2b",  # noqa: E402
        help="Model ID to use",  # noqa: E402
    )  # noqa: E402
    args = parser.parse_args()  # noqa: E402
  # noqa: E402
    # Load dataset  # noqa: E402
    if args.dataset == "real":  # noqa: E402
        real_test_path = PROJECT_ROOT / "data" / "personal" / "raw_style_variable" / "test.jsonl"  # noqa: E402
        if not real_test_path.exists():  # noqa: E402
            print(f"ERROR: Real test set not found at {real_test_path}", file=sys.stderr)  # noqa: E402
            return 1  # noqa: E402
        print(f"Loading real test set: {real_test_path}")  # noqa: E402
        examples = load_training_format_dataset(real_test_path)  # noqa: E402
    else:  # noqa: E402
        print("ERROR: Synthetic dataset not supported in this version", file=sys.stderr)  # noqa: E402
        return 1  # noqa: E402
  # noqa: E402
    if args.limit:  # noqa: E402
        examples = examples[:args.limit]  # noqa: E402
  # noqa: E402
    print("=" * 70)  # noqa: E402
    print("PROMPT CEILING EVALUATION - CLEAN VERSION")  # noqa: E402
    print("=" * 70)  # noqa: E402
    print(f"Model:     {args.model}")  # noqa: E402
    print(f"Dataset:   {args.dataset} ({len(examples)} examples)")  # noqa: E402
    print(f"Judge:     {'disabled' if args.no_judge else 'enabled'}")  # noqa: E402
    print()  # noqa: E402
  # noqa: E402
    variants = list(PROMPT_VARIANTS.keys()) if args.prompt == "all" else [args.prompt]  # noqa: E402
  # noqa: E402
    all_summaries = []  # noqa: E402
  # noqa: E402
    for variant in variants:  # noqa: E402
        print(f"\n{'='*70}")  # noqa: E402
        print(f"Testing prompt variant: {variant}")  # noqa: E402
        print(f"{'='*70}\n")  # noqa: E402
  # noqa: E402
        results = run_evaluation(  # noqa: E402
            prompt_variant=variant,  # noqa: E402
            examples=examples,  # noqa: E402
            use_judge=not args.no_judge,  # noqa: E402
            model_id=args.model,  # noqa: E402
        )  # noqa: E402
  # noqa: E402
        summary = print_summary(results, variant)  # noqa: E402
        all_summaries.append(summary)  # noqa: E402
  # noqa: E402
        save_results(results, variant)  # noqa: E402
  # noqa: E402
    if len(all_summaries) > 1:  # noqa: E402
        print("\n" + "=" * 70)  # noqa: E402
        print("COMPARISON")  # noqa: E402
        print("=" * 70)  # noqa: E402
        print(f"{'Variant':<15} {'Judge Avg':<12} {'Pass Rate':<12} {'Latency':<10}")  # noqa: E402
        print("-" * 70)  # noqa: E402
        for s in all_summaries:  # noqa: E402
            if "judge_avg" in s:  # noqa: E402
                print(  # noqa: E402
                    f"{s['variant']:<15} "  # noqa: E402
                    f"{s['judge_avg']:<12.2f} "  # noqa: E402
                    f"{s['judge_pass_rate']:<12.1%} "  # noqa: E402
                    f"{s['avg_latency_ms']:<10.0f}ms"  # noqa: E402
                )  # noqa: E402
        print()  # noqa: E402
  # noqa: E402
    return 0  # noqa: E402
  # noqa: E402
  # noqa: E402
if __name__ == "__main__":  # noqa: E402
    sys.exit(main())  # noqa: E402
