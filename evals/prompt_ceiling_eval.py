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

_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

EVAL_DATASET_PATH = PROJECT_ROOT / "evals" / "eval_dataset.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results" / "prompt_ceiling"

# Simplified prompts - no rules list, no abbreviation examples
CLEAN_PROMPT = """You are texting a friend. Reply briefly and casually."""

ULTRA_CLEAN_PROMPT = """Reply:"""

# Full system prompt from constants
DEFAULT_PROMPT = """You are Jwalin. Reply to text messages in your natural texting style.
Rules:
- Match your typical reply length (9 words avg)
- Use your abbreviations naturally: wanna, bc, gonna, kinda, btw
- No emoji usage
- Never sound like an AI assistant
- No formal greetings or sign-offs
- Just text back like you normally would
"""

PROMPT_VARIANTS: dict[str, str] = {
    "clean": CLEAN_PROMPT,
    "ultra": ULTRA_CLEAN_PROMPT,
    "default": DEFAULT_PROMPT,
}


@dataclass
class EvalExample:
    category: str
    context: list[str]
    last_message: str
    ideal_response: str
    contact_style: str
    notes: str
    contact_name: str = "Them"


@dataclass
class EvalResult:
    example: EvalExample
    generated_response: str
    latency_ms: float
    judge_score: float | None = None
    judge_reasoning: str = ""
    prompt_variant: str = ""


# ---------------------------------------------------------------------------
# Input Preprocessing
# ---------------------------------------------------------------------------

def clean_message(text: str) -> str:
    """Clean a single message."""
    if not text:
        return ""
    
    # Remove attachment placeholders
    text = re.sub(r'\[?image\]?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[?photo\]?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[?link\]?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[?video\]?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[?voice memo\]?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[?audio\]?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[?Location\]?', '', text, flags=re.IGNORECASE)
    
    # Remove unicode replacement character
    text = text.replace('\ufffc', '')
    
    # Remove reaction prefixes
    text = re.sub(r'^(liked|loved|laughed at|emphasized|questioned)\s+["\']?', '', text, flags=re.IGNORECASE)
    
    # Strip whitespace
    text = text.strip()
    
    return text


def strip_phone_numbers(text: str) -> str:
    """Replace phone numbers with 'Them'."""
    # Pattern for +14025551234 style numbers
    return re.sub(r'\+\d{10,15}', 'Them', text)


def preprocess_conversation(context: list[str], last_message: str) -> tuple[list[str], str]:
    """Preprocess conversation for model input.
    
    IMPORTANT: Keep the "Name: message" format so the model knows who's speaking.
    Just replace phone numbers with "Them" for consistency.
    """
    # Clean messages but keep name prefixes
    clean_context = []
    for m in context:
        m = clean_message(m)
        if m:
            # Replace phone numbers with "Them" but keep the prefix
            m = strip_phone_numbers(m)
            clean_context.append(m)
    
    clean_last = clean_message(last_message)
    clean_last = strip_phone_numbers(clean_last)
    
    # Keep up to 6 lines of context (enough for 3 turns back and forth)
    # Don't go too short - model needs context to understand the conversation
    if len(clean_context) > 6:
        clean_context = clean_context[-6:]
    
    return clean_context, clean_last


def build_simple_prompt(system_prompt: str, context: list[str], last_message: str) -> str:
    """Build minimal ChatML prompt.
    
    Keep the "Name: message" format so model understands conversation flow.
    """
    # Keep the "Name: message" format from context
    # For last_message, it should already have "Them: " prefix if we kept it
    conversation_lines = list(context)
    
    # Add last message - if it doesn't have a prefix, add "Them:"
    if last_message and not last_message.startswith(("Them:", "Jwalin:")):
        last_message = "Them: " + last_message
    conversation_lines.append(last_message)
    
    conversation = "\n".join(conversation_lines)
    
    return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{conversation}<|im_end|>
<|im_start|>assistant
Jwalin: """


# ---------------------------------------------------------------------------
# Output Post-processing
# ---------------------------------------------------------------------------

def clean_output(text: str) -> str:
    """Clean generated output."""
    if not text:
        return ""
    
    original = text
    
    # Strip contact name prefixes (Them:, +123:, Jwalin:, etc.)
    text = re.sub(r'^(?:Them|Jwalin|[+]?\d{10,15}|\w+)\s*:\s*', '', text)
    
    # Strip quoted content
    text = re.sub(r'^"([^"]*)"\s*', r'\1', text)
    
    # Strip markdown artifacts
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'\*\s*', '', text)
    
    # Strip stage directions
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Strip meta-commentary patterns
    text = re.sub(r'is a phrase that means.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'seems like.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'it sounds like.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'sounds like.*', '', text, flags=re.IGNORECASE)
    
    # Strip emojis
    emoji_pattern = re.compile(
        '[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
        '\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+'
    )
    text = emoji_pattern.sub('', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove trailing fragments
    text = re.sub(r'[,;\s]+$', '', text)
    
    if text != original:
        logger.debug(f"Cleaned: '{original[:50]}...' -> '{text[:50]}...'")
    
    return text if text else original.strip()


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_training_format_dataset(path: Path) -> list[EvalExample]:
    """Load training-format dataset (messages format).
    
    Keep the "Name: message" format throughout.
    """
    examples = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        
        messages = data.get("messages", [])
        if len(messages) < 3:
            continue
        
        user_content = messages[1].get("content", "") if len(messages) > 1 else ""
        ideal_response = messages[2].get("content", "") if len(messages) > 2 else ""
        
        # Parse conversation - keep the "Name: message" format
        lines = user_content.split("\n")
        
        # All lines except last are context
        # Last line is what we need to reply to
        if len(lines) >= 2:
            context = lines[:-1]
            last_message = lines[-1]
        elif len(lines) == 1:
            context = []
            last_message = lines[0]
        else:
            continue
        
        examples.append(
            EvalExample(
                category="statement",
                context=context,
                last_message=last_message,
                ideal_response=ideal_response,
                contact_style="casual",
                notes="real_message",
                contact_name="Them",
            )
        )
    return examples


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def load_generator(model_id: str | None = None):
    """Load the MLX generator."""
    from models.generator import MLXGenerator
    from models.loader import MLXModelLoader, ModelConfig
    
    config = ModelConfig(model_id=model_id)
    loader = MLXModelLoader(config)
    generator = MLXGenerator(loader=loader, config=config, skip_templates=True)
    return generator


def generate_reply(
    generator,
    system_prompt: str,
    context: list[str],
    last_message: str,
) -> tuple[str, float]:
    """Generate a reply with preprocessing and constraints."""
    from contracts.models import GenerationRequest
    
    # Preprocess input
    clean_context, clean_last = preprocess_conversation(context, last_message)
    
    # Build prompt
    prompt = build_simple_prompt(system_prompt, clean_context, clean_last)
    
    # Create generation request with constraints
    # Note: GenerationRequest doesn't support logit_bias directly in this version
    # We'll handle constraints via stop_sequences and post-processing
    request = GenerationRequest(
        prompt=prompt,
        max_tokens=20,  # Shorter to prevent run-on
        temperature=0.4,
        repetition_penalty=1.1,  # Lower as suggested
        stop_sequences=[
            "<|im_end|>",
            "<|im_start|>",
            "\n",
            "Them:",  # Prevent contact name echo
            ":",  # Prevent "Name:" patterns
        ],
    )
    
    start_time = time.perf_counter()
    response = generator.generate(request)
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    # Post-process output
    cleaned_text = clean_output(response.text)
    
    return cleaned_text, latency_ms


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def _extract_json_blob(text: str) -> str:
    """Extract JSON object from text."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
        if text.startswith("json"):
            text = text[4:]
    
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return text


def judge_reply(
    judge_client,
    judge_model: str,
    example: EvalExample,
    generated: str,
) -> tuple[float | None, str]:
    """Judge a single reply."""
    try:
        # Build conversation history for judge
        context_str = chr(10).join(example.context[-6:] + [example.last_message])
        
        prompt = (
            "You are an expert evaluator for a text message reply generator.\n\n"
            f"CONVERSATION:\n{context_str}\n\n"
            f"IDEAL RESPONSE:\n{example.ideal_response}\n\n"
            f"GENERATED REPLY:\n{generated}\n\n"
            f"CATEGORY: {example.category}\n"
            f"NOTES: {example.notes}\n\n"
            "Score the generated reply from 0-10. Consider:\n"
            "- Does it match the tone and intent of the ideal response?\n"
            "- Does it sound like a real person texting (not an AI)?\n"
            "- Is it appropriate for the category?\n"
            "- Is the length appropriate?\n\n"
            'Respond in JSON: {"score": <0-10>, "reasoning": "<1-2 sentences>"}'
        )
        
        resp = judge_client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        )
        
        content = resp.choices[0].message.content or ""
        payload = json.loads(_extract_json_blob(content))
        
        score = float(payload["score"])
        score = max(0.0, min(10.0, score))
        
        return score, str(payload.get("reasoning", ""))
    except Exception as e:
        logger.error(f"Judge error: {e}")
        return None, f"judge error: {e}"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    prompt_variant: str,
    examples: list[EvalExample],
    use_judge: bool = True,
    model_id: str | None = None,
) -> list[EvalResult]:
    """Run evaluation with specified prompt variant."""
    
    print(f"Loading generator (model: {model_id or 'default'})...")
    generator = load_generator(model_id=model_id)
    
    system_prompt = PROMPT_VARIANTS.get(prompt_variant, CLEAN_PROMPT)
    
    judge_client = None
    judge_model = None
    if use_judge:
        from evals.judge_config import JUDGE_MODEL, get_judge_client
        judge_client = get_judge_client()
        judge_model = JUDGE_MODEL
        if judge_client:
            print(f"Judge ready: {judge_model}")
        else:
            print("WARNING: Judge client not available")
    
    results = []
    
    for ex in tqdm(examples, desc=f"Evaluating ({prompt_variant})"):
        try:
            generated, latency_ms = generate_reply(
                generator=generator,
                system_prompt=system_prompt,
                context=ex.context,
                last_message=ex.last_message,
            )
        except Exception as e:
            logger.error(f"Generation error: {e}")
            generated = f"[ERROR: {e}]"
            latency_ms = 0.0
        
        judge_score = None
        judge_reasoning = ""
        if judge_client and use_judge and not generated.startswith("[ERROR"):
            judge_score, judge_reasoning = judge_reply(
                judge_client, judge_model, ex, generated
            )
        
        result = EvalResult(
            example=ex,
            generated_response=generated,
            latency_ms=latency_ms,
            judge_score=judge_score,
            judge_reasoning=judge_reasoning,
            prompt_variant=prompt_variant,
        )
        results.append(result)
        
        print(f"[{ex.category}] {ex.last_message[:40]}...")
        print(f"  Generated: {generated[:60]}")
        if judge_score is not None:
            print(f"  Judge: {judge_score:.1f}/10")
        print()
    
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(results: list[EvalResult], variant: str) -> dict[str, Any]:
    """Print and return summary statistics."""
    print("=" * 70)
    print(f"SUMMARY: {variant}")
    print("=" * 70)
    
    n = len(results)
    if n == 0:
        print("No results.")
        return {}
    
    latencies = [r.latency_ms for r in results]
    avg_latency = sum(latencies) / n
    
    scored = [r for r in results if r.judge_score is not None]
    if scored:
        scores = [r.judge_score for r in scored]
        avg_score = sum(scores) / len(scores)
        pass_7 = sum(1 for s in scores if s >= 7)
        
        print(f"Examples:           {n}")
        print(f"Avg latency:        {avg_latency:.0f}ms")
        print(f"Judge avg:          {avg_score:.2f}/10")
        print(f"Judge pass (>=7):   {pass_7}/{len(scored)} ({pass_7/len(scored)*100:.0f}%)")
        print()
        
        low = sorted(scored, key=lambda r: r.judge_score)[:3]
        print("Lowest scores:")
        for r in low:
            print(f"  [{r.judge_score:.0f}/10] {r.example.last_message[:40]}...")
            print(f"           Generated: {r.generated_response[:50]}")
        print()
        
        return {
            "variant": variant,
            "n_examples": n,
            "avg_latency_ms": round(avg_latency, 1),
            "judge_avg": round(avg_score, 2),
            "judge_pass_rate": round(pass_7 / len(scored), 3),
        }
    else:
        print(f"Examples:    {n}")
        print(f"Avg latency: {avg_latency:.0f}ms")
        return {
            "variant": variant,
            "n_examples": n,
            "avg_latency_ms": round(avg_latency, 1),
        }


def save_results(results: list[EvalResult], variant: str) -> Path:
    """Save results to file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"{variant}_{timestamp}.json"
    output_path = RESULTS_DIR / filename
    
    output_data = {
        "timestamp": timestamp,
        "prompt_variant": variant,
        "system_prompt": PROMPT_VARIANTS.get(variant, ""),
        "results": [
            {
                "category": r.example.category,
                "context": r.example.context,
                "last_message": r.example.last_message,
                "ideal_response": r.example.ideal_response,
                "generated_response": r.generated_response,
                "latency_ms": round(r.latency_ms, 1),
                "judge_score": r.judge_score,
                "judge_reasoning": r.judge_reasoning,
            }
            for r in results
        ],
    }
    
    scored = [r for r in results if r.judge_score is not None]
    if scored:
        output_data["summary"] = {
            "n_examples": len(results),
            "avg_latency_ms": round(sum(r.latency_ms for r in results) / len(results), 1),
            "judge_avg": round(sum(r.judge_score for r in scored) / len(scored), 2),
            "judge_pass_rate": round(sum(1 for r in scored if r.judge_score >= 7) / len(scored), 3),
        }
    
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"Results saved to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    
    parser = argparse.ArgumentParser(description="Prompt Ceiling Evaluation")
    parser.add_argument(
        "--prompt",
        choices=["clean", "ultra", "default", "all"],
        default="clean",
        help="Prompt variant to test",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip judge scoring",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of examples",
    )
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "real"],
        default="real",
        help="Dataset to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lfm-1.2b",
        help="Model ID to use",
    )
    args = parser.parse_args()
    
    # Load dataset
    if args.dataset == "real":
        real_test_path = PROJECT_ROOT / "data" / "personal" / "raw_style_variable" / "test.jsonl"
        if not real_test_path.exists():
            print(f"ERROR: Real test set not found at {real_test_path}", file=sys.stderr)
            return 1
        print(f"Loading real test set: {real_test_path}")
        examples = load_training_format_dataset(real_test_path)
    else:
        print(f"ERROR: Synthetic dataset not supported in this version", file=sys.stderr)
        return 1
    
    if args.limit:
        examples = examples[:args.limit]
    
    print("=" * 70)
    print("PROMPT CEILING EVALUATION - CLEAN VERSION")
    print("=" * 70)
    print(f"Model:     {args.model}")
    print(f"Dataset:   {args.dataset} ({len(examples)} examples)")
    print(f"Judge:     {'disabled' if args.no_judge else 'enabled'}")
    print()
    
    variants = list(PROMPT_VARIANTS.keys()) if args.prompt == "all" else [args.prompt]
    
    all_summaries = []
    
    for variant in variants:
        print(f"\n{'='*70}")
        print(f"Testing prompt variant: {variant}")
        print(f"{'='*70}\n")
        
        results = run_evaluation(
            prompt_variant=variant,
            examples=examples,
            use_judge=not args.no_judge,
            model_id=args.model,
        )
        
        summary = print_summary(results, variant)
        all_summaries.append(summary)
        
        save_results(results, variant)
    
    if len(all_summaries) > 1:
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)
        print(f"{'Variant':<15} {'Judge Avg':<12} {'Pass Rate':<12} {'Latency':<10}")
        print("-" * 70)
        for s in all_summaries:
            if "judge_avg" in s:
                print(
                    f"{s['variant']:<15} "
                    f"{s['judge_avg']:<12.2f} "
                    f"{s['judge_pass_rate']:<12.1%} "
                    f"{s['avg_latency_ms']:<10.0f}ms"
                )
        print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
