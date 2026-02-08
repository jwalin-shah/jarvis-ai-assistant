#!/usr/bin/env python3
"""Generate ORPO preference pairs using Gemini 2.5 Flash as oracle.

For each SOC conversation context:
1. Generate reply with fine-tuned LFM 1.2B -> rejected candidate
2. Generate gold reply with Gemini 2.5 Flash -> chosen candidate
3. Save as preference pairs for ORPO training

Output: data/soc_orpo/{train,valid}.jsonl

Usage:
    uv run python scripts/generate_preference_pairs.py
    uv run python scripts/generate_preference_pairs.py --sample 500
    uv run python scripts/generate_preference_pairs.py --model-path models/lfm-1.2b-soc-fused
    uv run python scripts/generate_preference_pairs.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

from evals.judge_config import (  # noqa: E402
    JUDGE_BASE_URL,
    JUDGE_MODEL as GEMINI_MODEL,
    get_judge_api_key,
)


def get_gemini_client():
    """Create OpenAI-compatible client for the judge/oracle model."""
    from openai import OpenAI

    return OpenAI(base_url=JUDGE_BASE_URL, api_key=get_judge_api_key())


def generate_gold_reply(client, context: str, last_message: str, style_guide: str) -> str | None:
    """Generate a gold-standard reply using Gemini 2.5 Flash.

    Returns the reply text, or None on failure.
    """
    prompt = (
        "You are generating a training example for a text message reply model.\n\n"
        "Generate a SINGLE natural text message reply. Requirements:\n"
        "- Sound like a real person texting, NOT an AI\n"
        "- Match the conversation's tone and energy\n"
        "- Keep it brief (1-2 short sentences max, often just a few words)\n"
        "- Use natural texting patterns: abbreviations, lowercase, casual punctuation\n"
        "- NO AI phrases: 'I hope', 'Let me know', 'Certainly', 'Of course!'\n"
        "- NO excessive punctuation or emoji spam\n"
    )
    if style_guide:
        prompt += f"\nStyle guide: {style_guide}\n"

    prompt += (
        f"\nCONVERSATION:\n{context}\n\n"
        f"LAST MESSAGE: {last_message}\n\n"
        "Reply with ONLY the text message reply, nothing else:"
    )

    try:
        resp = client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100,
        )
        return resp.choices[0].message.content.strip().strip('"').strip("'")
    except Exception as e:
        print(f"  Gemini error: {e}")
        return None


def score_reply(client, context: str, last_message: str, reply: str) -> float:
    """Score a reply 0-10 using Gemini as judge.

    Returns float score, or -1.0 on failure.
    """
    prompt = (
        "Score this text message reply from 0-10.\n\n"
        f"CONVERSATION:\n{context}\n\n"
        f"LAST MESSAGE: {last_message}\n\n"
        f"REPLY: {reply}\n\n"
        "Criteria:\n"
        "- Natural (sounds like a real person texting, not AI)\n"
        "- Appropriate tone and length\n"
        "- Relevant to the conversation\n"
        "- Brief and to the point\n\n"
        'Respond in JSON: {"score": <0-10>, "reasoning": "<1 sentence>"}'
    )

    try:
        resp = client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)
        return float(data["score"])
    except Exception:
        return -1.0


def generate_local_reply(loader, system_msg: str, user_msg: str) -> str:
    """Generate a reply using the local fine-tuned model."""
    prompt = f"{system_msg}\n\n{user_msg}\n\n<reply>"
    result = loader.generate_sync(
        prompt=prompt,
        temperature=0.1,
        max_tokens=50,
        top_p=0.1,
    )
    return result.text.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ORPO preference pairs")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Sample N examples from SFT data (0=all, default: 0)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/lfm-1.2b-soc-fused",
        help="Path to fine-tuned model for rejected candidates",
    )
    parser.add_argument(
        "--min-score-delta",
        type=float,
        default=1.0,
        help="Min score difference between chosen and rejected (default: 1.0)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load SFT test data (use as source for preference pair contexts)
    sft_dir = PROJECT_ROOT / "data" / "soc_sft"
    train_path = sft_dir / "train.jsonl"
    if not train_path.exists():
        print(f"ERROR: SFT data not found at {train_path}")
        print("       Run: uv run python scripts/prepare_soc_data.py")
        return 1

    print("Loading SFT training data...")
    examples = []
    with open(train_path) as f:
        for line in f:
            examples.append(json.loads(line))

    if args.sample > 0 and args.sample < len(examples):
        examples = random.sample(examples, args.sample)

    print(f"Processing {len(examples)} examples")

    if args.dry_run:
        print("Dry run - would generate preference pairs for these examples.")
        return 0

    # Init Gemini client
    client = get_gemini_client()
    print(f"Gemini judge: {GEMINI_MODEL} via DeepInfra")

    # Init local model for rejected candidates
    print(f"Loading local model from {args.model_path}...")
    model_path = PROJECT_ROOT / args.model_path
    if model_path.exists():
        # Local fused model
        os.environ["JARVIS_MODEL_PATH"] = str(model_path)

    from models.loader import get_model

    loader = get_model()
    if not loader.is_loaded():
        loader.load()
    print("Local model loaded")

    # Generate preference pairs
    pairs: list[dict] = []
    skipped = 0
    errors = 0

    for i, ex in enumerate(examples):
        msgs = ex["messages"]
        system_msg = msgs[0]["content"]
        user_msg = msgs[1]["content"]
        original_reply = msgs[2]["content"]

        # Extract context and last_message from user message
        context = ""
        last_message = ""
        if "<conversation>" in user_msg:
            parts = user_msg.split("<conversation>")
            if len(parts) > 1:
                context = parts[1].split("</conversation>")[0].strip()
        if "<last_message>" in user_msg:
            lm_parts = user_msg.split("<last_message>")
            if len(lm_parts) > 1:
                last_message = lm_parts[1].split("</last_message>")[0].strip()

        if not last_message:
            skipped += 1
            continue

        # Extract style guide from system message
        style_guide = ""
        if "Style guide:" in system_msg:
            sg_parts = system_msg.split("Style guide:")
            if len(sg_parts) > 1:
                style_guide = sg_parts[1].split("</system>")[0].strip()

        # Generate rejected: local model output
        try:
            rejected = generate_local_reply(loader, system_msg, user_msg)
        except Exception as e:
            print(f"  [{i + 1}] Local model error: {e}")
            errors += 1
            continue

        # Generate chosen: Gemini gold reply
        chosen = generate_gold_reply(client, context, last_message, style_guide)
        if not chosen:
            errors += 1
            continue

        # Score both
        chosen_score = score_reply(client, context, last_message, chosen)
        rejected_score = score_reply(client, context, last_message, rejected)

        # Also score the original SOC reply
        original_score = score_reply(client, context, last_message, original_reply)

        # Pick best as chosen: Gemini gold or high-scoring original
        if original_score > chosen_score and original_score >= 7:
            final_chosen = original_reply
            final_chosen_score = original_score
        else:
            final_chosen = chosen
            final_chosen_score = chosen_score

        # Ensure meaningful score delta
        if final_chosen_score - rejected_score < args.min_score_delta:
            skipped += 1
            continue

        # Format for ORPO training (mlx-lm-lora DPO/ORPO format)
        pair = {
            "prompt": f"{system_msg}\n\n{user_msg}",
            "chosen": final_chosen,
            "rejected": rejected,
        }
        pairs.append(pair)

        if (i + 1) % 50 == 0:
            print(
                f"  [{i + 1}/{len(examples)}] {len(pairs)} pairs, "
                f"{skipped} skipped, {errors} errors"
            )

        # Rate limiting (DeepInfra)
        time.sleep(0.2)

    print(f"\nTotal preference pairs: {len(pairs)}")
    print(f"Skipped: {skipped}, Errors: {errors}")

    if not pairs:
        print("No pairs generated. Check model and API connectivity.")
        return 1

    # Split into train/valid (90/10)
    random.shuffle(pairs)
    n_valid = max(1, len(pairs) // 10)
    train_pairs = pairs[n_valid:]
    valid_pairs = pairs[:n_valid]

    # Save
    output_dir = PROJECT_ROOT / "data" / "soc_orpo"
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_pairs in [("train", train_pairs), ("valid", valid_pairs)]:
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for pair in split_pairs:
                f.write(json.dumps(pair) + "\n")
        print(f"Saved {len(split_pairs)} pairs to {path}")

    # Save metadata
    meta = {
        "total_pairs": len(pairs),
        "train": len(train_pairs),
        "valid": len(valid_pairs),
        "skipped": skipped,
        "errors": errors,
        "source_examples": len(examples),
        "gemini_model": GEMINI_MODEL,
        "local_model": args.model_path,
        "min_score_delta": args.min_score_delta,
    }
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Metadata saved to {meta_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
