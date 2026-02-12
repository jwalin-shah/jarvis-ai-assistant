#!/usr/bin/env python3
"""Bakeoff: Test 20+ prompting strategies x multiple local models for fact extraction.

Tests every combination of model + prompt strategy on real iMessages,
saves all outputs for human review.

Usage:
    uv run python scripts/extraction_bakeoff.py
    uv run python scripts/extraction_bakeoff.py --models lfm-1.2b-extract
    uv run python scripts/extraction_bakeoff.py --strategies 1,5,10
    uv run python scripts/extraction_bakeoff.py --message-limit 10
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, ".")

# ─── Model Definitions ──────────────────────────────────────────────────────

MODELS = {
    "lfm-1.2b-extract": {
        "path": "models/lfm2-1.2b-extract-mlx-4bit",
        "chat_template": True,  # has im_start/im_end tokens
        "description": "LiquidAI LFM2.5 1.2B Extract (4-bit MLX)",
    },
    "lfm-350m-extract": {
        "path": "models/lfm2-350m-extract-mlx-4bit",
        "chat_template": True,
        "description": "LiquidAI LFM2.5 350M Extract (4-bit MLX)",
    },
    "qwen2.5-0.5b": {
        "path": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "chat_template": True,
        "description": "Qwen2.5 0.5B Instruct (4-bit MLX)",
    },
    "qwen2.5-1.5b": {
        "path": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "chat_template": True,
        "description": "Qwen2.5 1.5B Instruct (4-bit MLX)",
    },
    "qwen2.5-3b": {
        "path": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "chat_template": True,
        "description": "Qwen2.5 3B Instruct (4-bit MLX)",
    },
    "smollm2-1.7b": {
        "path": "mlx-community/SmolLM2-1.7B-Instruct-4bit",
        "chat_template": True,
        "description": "SmolLM2 1.7B Instruct (4-bit MLX)",
    },
    "smollm2-360m": {
        "path": "mlx-community/SmolLM2-360M-Instruct-4bit",
        "chat_template": True,
        "description": "SmolLM2 360M Instruct (4-bit MLX)",
    },
    "gemma2-2b": {
        "path": "mlx-community/gemma-2-2b-it-4bit",
        "chat_template": True,
        "description": "Gemma 2 2B Instruct (4-bit MLX)",
    },
    "llama3.2-1b": {
        "path": "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "chat_template": True,
        "description": "LLaMA 3.2 1B Instruct (4-bit MLX)",
    },
    "llama3.2-3b": {
        "path": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "chat_template": True,
        "description": "LLaMA 3.2 3B Instruct (4-bit MLX)",
    },
}

# ─── Prompt Strategies ───────────────────────────────────────────────────────
# Each strategy is a function(message_text) -> (system_prompt, user_prompt)
# system_prompt can be None for models that don't support it


def strategy_direct_json(text: str) -> tuple[str | None, str]:
    """Strategy 1: Direct JSON extraction"""
    system = "You extract facts from text messages. Output valid JSON only."
    user = f"""Extract all personal facts from this message as JSON.

Message: "{text}"

Output format: {{"facts": [{{"type": "...", "subject": "...", "value": "..."}}]}}
If no facts, output: {{"facts": []}}"""
    return system, user


def strategy_tag_based(text: str) -> tuple[str | None, str]:
    """Strategy 2: Tag-based extraction with delimiters"""
    system = None
    user = f"""Extract facts from this text message. Use this exact format for each fact:
<fact>TYPE: value</fact>

Types: location, person, preference, activity, relationship, job, school, health

Message: "{text}"

Facts (or "none" if no facts):"""
    return system, user


def strategy_one_category_at_a_time(text: str) -> tuple[str | None, str]:
    """Strategy 3: Ask for each category separately"""
    system = None
    user = f"""Message: "{text}"

For this message, answer each question. Write "none" if not mentioned.
Location mentioned:
Person mentioned:
Preference or opinion:
Activity or event:
Relationship:
Job or school:"""
    return system, user


def strategy_yes_no_gate(text: str) -> tuple[str | None, str]:
    """Strategy 4: Yes/No gate before extraction"""
    system = None
    user = f"""Message: "{text}"

Step 1: Does this message contain any personal facts? (yes/no)
Step 2: If yes, list each fact as "- type: value"
"""
    return system, user


def strategy_few_shot_3(text: str) -> tuple[str | None, str]:
    """Strategy 5: 3-shot examples"""
    system = None
    user = f"""Extract personal facts from text messages.

Example 1:
Message: "I just moved to Austin last week"
Facts: location: Austin (current city)

Example 2:
Message: "haha yeah"
Facts: none

Example 3:
Message: "My sister works at Google"
Facts: relationship: has a sister; employer: sister works at Google

Now extract facts from:
Message: "{text}"
Facts:"""
    return system, user


def strategy_few_shot_5(text: str) -> tuple[str | None, str]:
    """Strategy 6: 5-shot examples with more variety"""
    system = None
    user = f"""Extract personal facts from text messages. Only extract what is explicitly stated.

Example 1:
Message: "I just moved to Austin last week"
Facts: location: Austin (current city)

Example 2:
Message: "haha yeah"
Facts: none

Example 3:
Message: "My sister works at Google"
Facts: relationship: has a sister | employer: sister works at Google

Example 4:
Message: "I'm allergic to peanuts so I can't eat that"
Facts: health: allergic to peanuts

Example 5:
Message: "gonna watch the Lakers game tonight"
Facts: interest: Lakers fan

Now extract facts from:
Message: "{text}"
Facts:"""
    return system, user


def strategy_role_ner_system(text: str) -> tuple[str | None, str]:
    """Strategy 7: Role prompt as NER system"""
    system = (
        "You are a Named Entity Recognition system specialized in extracting "
        "personal facts from informal text messages. You ONLY output entities "
        "found in the text. You NEVER generate information not present in the input."
    )
    user = f"""Input: "{text}"
Entities:"""
    return system, user


def strategy_chain_of_thought(text: str) -> tuple[str | None, str]:
    """Strategy 8: Chain of thought"""
    system = None
    user = f"""Message: "{text}"

Let me analyze this message for personal facts step by step:
1. What is the topic?
2. Are any people mentioned?
3. Are any places mentioned?
4. Are any preferences, activities, or personal details mentioned?
5. Final facts list:"""
    return system, user


def strategy_negative_examples(text: str) -> tuple[str | None, str]:
    """Strategy 9: Show what NOT to extract"""
    system = None
    user = f"""Extract personal facts from this message. Be precise.

DO extract: names, locations, jobs, schools, relationships, preferences, health info
DO NOT extract: greetings, opinions about weather, generic statements, emotions

DO NOT hallucinate facts not in the message.

Message: "{text}"
Facts (or "none"):"""
    return system, user


def strategy_minimal(text: str) -> tuple[str | None, str]:
    """Strategy 10: Minimal prompt"""
    system = None
    user = f"""Facts in "{text}":"""
    return system, user


def strategy_question_format(text: str) -> tuple[str | None, str]:
    """Strategy 11: Question format"""
    system = None
    user = f"""What personal facts can be extracted from this message?

Message: "{text}"

Answer:"""
    return system, user


def strategy_bullet_list(text: str) -> tuple[str | None, str]:
    """Strategy 12: Bullet list format"""
    system = "Extract facts as a bullet list. Write 'none' if no facts."
    user = f"""Message: "{text}"

Facts:
-"""
    return system, user


def strategy_key_value(text: str) -> tuple[str | None, str]:
    """Strategy 13: Key-value pairs"""
    system = None
    user = f"""Extract facts as key-value pairs from this message.

Message: "{text}"

person_name:
location:
job:
school:
relationship:
preference:
health:
activity:"""
    return system, user


def strategy_pipe_delimited(text: str) -> tuple[str | None, str]:
    """Strategy 14: Pipe-delimited for easy parsing"""
    system = "Output facts as: TYPE|SUBJECT|VALUE (one per line). Output NONE if no facts."
    user = f"""Message: "{text}"
Facts:"""
    return system, user


def strategy_xml_structured(text: str) -> tuple[str | None, str]:
    """Strategy 15: XML-structured output"""
    system = None
    user = f"""Extract facts from this message using XML tags.

Message: "{text}"

<extraction>"""
    return system, user


def strategy_sentence_completion(text: str) -> tuple[str | None, str]:
    """Strategy 16: Sentence completion"""
    system = None
    user = f"""Complete the sentence based ONLY on what's in the message.

Message: "{text}"

The personal facts in this message are:"""
    return system, user


def strategy_classification_first(text: str) -> tuple[str | None, str]:
    """Strategy 17: Classify then extract"""
    system = None
    user = f"""Message: "{text}"

Category (personal_info/no_info):
If personal_info, extract:"""
    return system, user


def strategy_context_aware(text: str) -> tuple[str | None, str]:
    """Strategy 18: Include message metadata context"""
    system = (
        "You analyze text messages to build a knowledge graph about the sender. "
        "Extract ONLY facts explicitly stated. Never infer or guess."
    )
    user = f"""This is a text message from a conversation:
"{text}"

What facts about the sender or people they mention can we add to their profile?
Format: fact_type: detail"""
    return system, user


def strategy_constrained_categories(text: str) -> tuple[str | None, str]:
    """Strategy 19: Strict category list"""
    system = None
    user = f"""From the message below, extract ONLY these fact types:
- current_location (city/state/country they live in or are at)
- hometown (where they're from)
- employer (company they work at)
- school (school they attend/attended)
- family (family members mentioned)
- friend (friends mentioned by name)
- preference (food, music, hobby preferences)
- allergy (allergies or dietary restrictions)
- pet (pets they have)

Message: "{text}"

Extracted (or "none"):"""
    return system, user


def strategy_two_pass_entities_then_relations(text: str) -> tuple[str | None, str]:
    """Strategy 20: Two-pass - entities first, then relationships"""
    system = None
    user = f"""Message: "{text}"

Pass 1 - Named entities (people, places, organizations):

Pass 2 - Facts connecting entities to the sender:"""
    return system, user


def strategy_fill_template(text: str) -> tuple[str | None, str]:
    """Strategy 21: Fill-in-the-blank template (NuExtract style)"""
    system = None
    user = f"""<|input|>
### Template:
{{"location": "", "person": "", "job": "", "school": "", "relationship": "", "preference": "", "health": ""}}

### Text:
{text}

<|output|>"""
    return system, user


def strategy_extractive_only(text: str) -> tuple[str | None, str]:
    """Strategy 22: Emphasize extractive (copy from text, don't generate)"""
    system = (
        "You are an extractive system. You may ONLY output words that appear "
        "in the input text. Never generate new words."
    )
    user = f"""Input text: "{text}"

Copy out any personal facts (names, places, jobs, preferences) using ONLY words from the text above:"""
    return system, user


def strategy_json_schema_strict(text: str) -> tuple[str | None, str]:
    """Strategy 23: Strict JSON schema with types"""
    system = "Respond with valid JSON only. No other text."
    user = f"""Extract facts from: "{text}"

Schema:
{{"has_facts": true/false, "facts": [{{"category": "location|person|job|school|preference|health|relationship", "value": "exact text from message", "confidence": 0.0-1.0}}]}}"""
    return system, user


def strategy_sparse_triplets(text: str) -> tuple[str | None, str]:
    """Strategy 24: Knowledge graph triplets"""
    system = None
    user = f"""Extract knowledge graph triplets from this message.
Format: (subject, relation, object)

Message: "{text}"

Triplets:"""
    return system, user


# Collect all strategies
STRATEGIES = {
    "01_direct_json": strategy_direct_json,
    "02_tag_based": strategy_tag_based,
    "03_one_category": strategy_one_category_at_a_time,
    "04_yes_no_gate": strategy_yes_no_gate,
    "05_few_shot_3": strategy_few_shot_3,
    "06_few_shot_5": strategy_few_shot_5,
    "07_role_ner": strategy_role_ner_system,
    "08_chain_of_thought": strategy_chain_of_thought,
    "09_negative_examples": strategy_negative_examples,
    "10_minimal": strategy_minimal,
    "11_question_format": strategy_question_format,
    "12_bullet_list": strategy_bullet_list,
    "13_key_value": strategy_key_value,
    "14_pipe_delimited": strategy_pipe_delimited,
    "15_xml_structured": strategy_xml_structured,
    "16_sentence_completion": strategy_sentence_completion,
    "17_classify_first": strategy_classification_first,
    "18_context_aware": strategy_context_aware,
    "19_constrained_categories": strategy_constrained_categories,
    "20_two_pass": strategy_two_pass_entities_then_relations,
    "21_fill_template": strategy_fill_template,
    "22_extractive_only": strategy_extractive_only,
    "23_json_schema_strict": strategy_json_schema_strict,
    "24_kg_triplets": strategy_sparse_triplets,
}


# ─── Model Loading ───────────────────────────────────────────────────────────


def load_model(model_path: str):
    """Load an MLX model, returns (model, tokenizer)."""
    import mlx.core as mx
    from mlx_lm import load

    # Memory limits for 8GB system
    mx.set_memory_limit(1 * 1024 * 1024 * 1024)  # 1GB
    mx.set_cache_limit(512 * 1024 * 1024)  # 512MB

    print(f"  Loading {model_path}...", flush=True)
    t0 = time.time()
    model, tokenizer = load(model_path)
    elapsed = time.time() - t0
    print(f"  Loaded in {elapsed:.1f}s", flush=True)
    return model, tokenizer


def unload_model(model, tokenizer):
    """Unload model and free memory."""
    import gc
    import mlx.core as mx

    del model
    del tokenizer
    gc.collect()
    mx.clear_cache()


def generate_response(
    model, tokenizer, system_prompt: str | None, user_prompt: str,
    max_tokens: int = 200, temperature: float = 0.0,
) -> tuple[str, float, int]:
    """Generate a response, returns (text, time_ms, token_count)."""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_repetition_penalty, make_sampler

    # Build chat messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    # Try chat template first, fall back to raw prompt
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Model doesn't support chat template, use raw prompt
        prompt = user_prompt

    sampler = make_sampler(temp=temperature)
    repetition_penalty = make_repetition_penalty(1.05)

    t0 = time.time()
    response = generate(
        model, tokenizer, prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=[repetition_penalty],
    )
    elapsed_ms = (time.time() - t0) * 1000

    # Count approximate tokens
    token_count = len(tokenizer.encode(response)) if response else 0

    return response, elapsed_ms, token_count


# ─── Main Bakeoff Runner ────────────────────────────────────────────────────


@dataclass
class ExtractionResult:
    model: str
    strategy: str
    message_id: int
    message_text: str
    expected_has_facts: bool
    system_prompt: str | None
    user_prompt: str
    response: str
    time_ms: float
    tokens: int
    error: str | None = None


def run_bakeoff(
    model_names: list[str],
    strategy_names: list[str],
    messages: list[dict],
    max_tokens: int = 200,
    output_dir: str = "results/extraction_bakeoff",
) -> None:
    """Run the full bakeoff."""
    os.makedirs(output_dir, exist_ok=True)

    total_combos = len(model_names) * len(strategy_names) * len(messages)
    print(f"\n{'='*70}", flush=True)
    print(f"EXTRACTION BAKEOFF", flush=True)
    print(f"  Models: {len(model_names)}", flush=True)
    print(f"  Strategies: {len(strategy_names)}", flush=True)
    print(f"  Messages: {len(messages)}", flush=True)
    print(f"  Total combinations: {total_combos}", flush=True)
    print(f"  Output: {output_dir}/", flush=True)
    print(f"{'='*70}\n", flush=True)

    all_results: list[dict] = []
    combo_idx = 0

    for model_name in model_names:
        model_info = MODELS[model_name]
        print(f"\n{'─'*50}", flush=True)
        print(f"MODEL: {model_name} ({model_info['description']})", flush=True)
        print(f"{'─'*50}", flush=True)

        try:
            model, tokenizer = load_model(model_info["path"])
        except Exception as e:
            print(f"  FAILED to load: {e}", flush=True)
            # Record failures for all combos with this model
            for strat_name in strategy_names:
                for msg in messages:
                    combo_idx += 1
                    all_results.append(asdict(ExtractionResult(
                        model=model_name, strategy=strat_name,
                        message_id=msg["id"], message_text=msg["text"],
                        expected_has_facts=msg["expected_has_facts"],
                        system_prompt=None, user_prompt="",
                        response="", time_ms=0, tokens=0,
                        error=f"Model load failed: {e}",
                    )))
            continue

        model_results = []
        for strat_name in strategy_names:
            strategy_fn = STRATEGIES[strat_name]
            print(f"\n  Strategy: {strat_name}", flush=True)

            for i, msg in enumerate(messages):
                combo_idx += 1
                progress = f"[{combo_idx}/{total_combos}]"
                msg_preview = msg["text"][:40].replace("\n", " ")
                print(f"    {progress} msg {msg['id']}: {msg_preview}...", end="", flush=True)

                try:
                    system_prompt, user_prompt = strategy_fn(msg["text"])
                    response, time_ms, tokens = generate_response(
                        model, tokenizer, system_prompt, user_prompt,
                        max_tokens=max_tokens,
                    )
                    result = ExtractionResult(
                        model=model_name, strategy=strat_name,
                        message_id=msg["id"], message_text=msg["text"],
                        expected_has_facts=msg["expected_has_facts"],
                        system_prompt=system_prompt, user_prompt=user_prompt,
                        response=response, time_ms=time_ms, tokens=tokens,
                    )
                    print(f" {time_ms:.0f}ms, {tokens}tok", flush=True)
                except Exception as e:
                    result = ExtractionResult(
                        model=model_name, strategy=strat_name,
                        message_id=msg["id"], message_text=msg["text"],
                        expected_has_facts=msg["expected_has_facts"],
                        system_prompt=None, user_prompt="",
                        response="", time_ms=0, tokens=0,
                        error=str(e),
                    )
                    print(f" ERROR: {e}", flush=True)

                model_results.append(asdict(result))

            # Save incrementally per model+strategy
            all_results.extend(model_results[-len(messages):])
            _save_results(all_results, output_dir)

        # Unload model before loading next
        print(f"\n  Unloading {model_name}...", flush=True)
        unload_model(model, tokenizer)
        model_results.clear()

    # Final save and summary
    _save_results(all_results, output_dir)
    _print_summary(all_results, output_dir)


def _save_results(results: list[dict], output_dir: str) -> None:
    """Save results incrementally."""
    path = os.path.join(output_dir, "all_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def _print_summary(results: list[dict], output_dir: str) -> None:
    """Print a summary table of results."""
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    # Group by model + strategy
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        key = (r["model"], r["strategy"])
        groups[key].append(r)

    print(f"\n{'Model':<20} {'Strategy':<25} {'Avg ms':>8} {'Errors':>7}", flush=True)
    print("-" * 65, flush=True)

    for (model, strategy), group in sorted(groups.items()):
        times = [r["time_ms"] for r in group if not r.get("error")]
        errors = sum(1 for r in group if r.get("error"))
        avg_ms = sum(times) / len(times) if times else 0
        print(f"{model:<20} {strategy:<25} {avg_ms:>7.0f}ms {errors:>6}err", flush=True)

    # Save a human-readable review file
    review_path = os.path.join(output_dir, "review.md")
    with open(review_path, "w") as f:
        f.write("# Extraction Bakeoff Results\n\n")
        f.write(f"Total results: {len(results)}\n\n")

        for (model, strategy), group in sorted(groups.items()):
            f.write(f"\n## {model} / {strategy}\n\n")
            for r in group:
                has_facts = "FACT" if r["expected_has_facts"] else "NO-FACT"
                msg_preview = r["message_text"][:60].replace("\n", " ")
                f.write(f"### [{has_facts}] msg {r['message_id']}: {msg_preview}\n")
                if r.get("error"):
                    f.write(f"**ERROR**: {r['error']}\n\n")
                else:
                    f.write(f"**Response** ({r['time_ms']:.0f}ms):\n```\n{r['response']}\n```\n\n")

    print(f"\nResults saved to: {output_dir}/all_results.json", flush=True)
    print(f"Human review: {output_dir}/review.md", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Extraction prompting bakeoff")
    parser.add_argument(
        "--models", type=str, default=None,
        help=f"Comma-separated model names. Available: {','.join(MODELS.keys())}"
    )
    parser.add_argument(
        "--strategies", type=str, default=None,
        help="Comma-separated strategy numbers (e.g. 1,5,10) or names"
    )
    parser.add_argument(
        "--messages", type=str, default="results/sample_messages.json",
        help="Path to messages JSON file"
    )
    parser.add_argument("--message-limit", type=int, default=None, help="Limit messages to test")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per generation")
    parser.add_argument("--output", type=str, default="results/extraction_bakeoff")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--list-strategies", action="store_true", help="List available strategies")
    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable models:")
        for name, info in MODELS.items():
            print(f"  {name:<20} {info['description']}")
        return

    if args.list_strategies:
        print("\nAvailable strategies:")
        for name, fn in STRATEGIES.items():
            print(f"  {name:<30} {fn.__doc__}")
        return

    # Load messages
    msg_path = Path(args.messages)
    if not msg_path.exists():
        print(f"Messages file not found: {msg_path}", flush=True)
        print("Run: uv run python scripts/sample_messages.py", flush=True)
        sys.exit(1)

    with open(msg_path) as f:
        messages = json.load(f)

    if args.message_limit:
        # Take a balanced sample
        fact_msgs = [m for m in messages if m["expected_has_facts"]]
        no_fact_msgs = [m for m in messages if not m["expected_has_facts"]]
        half = args.message_limit // 2
        messages = fact_msgs[:half] + no_fact_msgs[:half]

    # Select models
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
        for m in model_names:
            if m not in MODELS:
                print(f"Unknown model: {m}. Use --list-models to see options.", flush=True)
                sys.exit(1)
    else:
        # Default: just test the models we already have locally
        model_names = ["lfm-1.2b-extract", "lfm-350m-extract"]

    # Select strategies
    if args.strategies:
        parts = [s.strip() for s in args.strategies.split(",")]
        strategy_names = []
        for p in parts:
            # Support both "01" and "01_direct_json"
            matches = [s for s in STRATEGIES if s.startswith(p.zfill(2))]
            if matches:
                strategy_names.extend(matches)
            elif p in STRATEGIES:
                strategy_names.append(p)
            else:
                print(f"Unknown strategy: {p}. Use --list-strategies to see options.", flush=True)
                sys.exit(1)
    else:
        strategy_names = list(STRATEGIES.keys())

    run_bakeoff(
        model_names=model_names,
        strategy_names=strategy_names,
        messages=messages,
        max_tokens=args.max_tokens,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
