"""Prepare personal iMessage data for fine-tuning.

Reads raw_pairs.jsonl and produces training data in two variants:
  A) category-aware: includes category label in system prompt
  B) raw-style: auto-detects user's texting style, bakes into system prompt

Output: data/personal/{category_aware,raw_style}/{train,valid,test}.jsonl

Usage:
    uv run python scripts/prepare_personal_data.py --both
    uv run python scripts/prepare_personal_data.py --variant category_aware
    uv run python scripts/prepare_personal_data.py --variant raw_style
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("prepare_personal.log")],
)
log = logging.getLogger(__name__)

# --- System prompt templates ---

SYSTEM_TEMPLATE_CATAWARE = """<system>
You are NOT an AI assistant. You are replying to a text message from your phone.
Just text back. No helpfulness, no formality, no assistant behavior.
Category: {category}
{style_section}
</system>"""

SYSTEM_TEMPLATE_RAW = """<system>
You are NOT an AI assistant. You are replying to a text message from your phone.
Just text back. No helpfulness, no formality, no assistant behavior.
{style_section}
</system>"""


def analyze_style(replies: list[str]) -> str:
    """Analyze user's texting style from their replies and return a style section."""
    if not replies:
        return ""

    # Average length
    lengths = [len(r) for r in replies]
    avg_len = np.mean(lengths)

    # Emoji frequency
    emoji_pat = re.compile(
        "["
        "\U0001f600-\U0001f64f"
        "\U0001f300-\U0001f5ff"
        "\U0001f680-\U0001f6ff"
        "\U0001f1e0-\U0001f1ff"
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "]+"
    )
    emoji_counts = [len(emoji_pat.findall(r)) for r in replies]
    emoji_per_msg = np.mean(emoji_counts)

    # Abbreviation detection
    abbrevs = {
        "u", "ur", "r", "y", "k", "kk", "bc", "cuz", "gonna", "wanna",
        "gotta", "thx", "ty", "pls", "plz", "idk", "nvm", "brb", "ttyl",
        "omw", "lol", "lmao", "omg", "tbh", "imo", "ikr", "rn", "atm", "btw",
    }
    all_words = set()
    for r in replies:
        all_words.update(re.findall(r"\b\w+\b", r.lower()))
    found_abbrevs = all_words & abbrevs

    # Capitalization
    starts_lower = sum(1 for r in replies if r and r[0].islower()) / max(len(replies), 1)

    # Build style description
    parts = []
    if avg_len < 15:
        parts.append("Very short messages (1-5 words)")
    elif avg_len < 30:
        parts.append("Brief messages (1 sentence)")
    elif avg_len < 60:
        parts.append("Moderate length (1-2 sentences)")
    else:
        parts.append("Longer messages (2-3 sentences)")

    if emoji_per_msg < 0.1:
        parts.append("Rarely uses emoji")
    elif emoji_per_msg > 0.5:
        parts.append("Frequently uses emoji")

    if found_abbrevs:
        top = sorted(found_abbrevs)[:5]
        parts.append(f"Uses abbreviations: {', '.join(top)}")

    if starts_lower > 0.7:
        parts.append("Mostly lowercase")

    return "Style: " + ". ".join(parts) + "."


def format_context(context_msgs: list[dict]) -> str:
    """Format context messages into conversation string."""
    lines = []
    for msg in context_msgs:
        lines.append(f"{msg['sender']}: {msg['text']}")
    return "\n".join(lines)


def build_training_example(
    pair: dict,
    system_prompt: str,
) -> dict:
    """Build a single training example in chat format."""
    context_str = format_context(pair["context"])
    last_msg = pair["context"][-1] if pair["context"] else {"sender": "Unknown", "text": ""}

    user_content = f"<conversation>\n{context_str}\n</conversation>\n<last_message>{last_msg['sender']}: {last_msg['text']}</last_message>"

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": pair["reply"]},
        ]
    }


def split_by_contact(
    pairs: list[dict],
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split pairs into train/valid/test, stratified by contact.

    Ensures each contact's messages appear proportionally in all splits.
    """
    rng = np.random.default_rng(seed)

    # Group by contact
    by_contact: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_contact[p["contact_name"]].append(p)

    train, valid, test = [], [], []

    for contact, contact_pairs in by_contact.items():
        rng.shuffle(contact_pairs)
        n = len(contact_pairs)
        n_train = max(1, int(n * train_ratio))
        n_valid = max(0, int(n * valid_ratio))

        train.extend(contact_pairs[:n_train])
        valid.extend(contact_pairs[n_train : n_train + n_valid])
        test.extend(contact_pairs[n_train + n_valid :])

    # Shuffle each split
    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)

    return train, valid, test


def prepare_category_aware(pairs: list[dict], output_dir: Path) -> None:
    """Prepare category-aware training data."""
    log.info("Preparing category-aware variant...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to import category classifier
    try:
        from jarvis.classifiers.category_classifier import classify_category

        has_classifier = True
        log.info("Category classifier loaded successfully")
    except Exception as e:
        has_classifier = False
        log.warning("Category classifier unavailable (%s), using 'statement' as default", e)

    # Analyze overall style
    all_replies = [p["reply"] for p in pairs]
    style_section = analyze_style(all_replies)

    # Split
    train, valid, test = split_by_contact(pairs)
    log.info("Split: %d train, %d valid, %d test", len(train), len(valid), len(test))

    category_counts: Counter = Counter()

    for split_name, split_pairs in [("train", train), ("valid", valid), ("test", test)]:
        examples = []
        for i, pair in enumerate(split_pairs):
            if (i + 1) % 500 == 0:
                log.info("  %s: processing %d/%d", split_name, i + 1, len(split_pairs))

            # Classify
            if has_classifier:
                try:
                    context_texts = [m["text"] for m in pair["context"]]
                    result = classify_category(pair["reply"], context=context_texts)
                    category = result.category
                except Exception:
                    category = "statement"
            else:
                category = "statement"

            category_counts[category] += 1

            system_prompt = SYSTEM_TEMPLATE_CATAWARE.format(
                category=category, style_section=style_section
            )
            examples.append(build_training_example(pair, system_prompt))

        # Write
        out_path = output_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        log.info("  Wrote %d examples to %s", len(examples), out_path)

    log.info("Category distribution: %s", dict(category_counts.most_common()))


def prepare_raw_style(pairs: list[dict], output_dir: Path) -> None:
    """Prepare raw-style training data."""
    log.info("Preparing raw-style variant...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze overall style
    all_replies = [p["reply"] for p in pairs]
    style_section = analyze_style(all_replies)

    system_prompt = SYSTEM_TEMPLATE_RAW.format(style_section=style_section)

    # Split
    train, valid, test = split_by_contact(pairs)
    log.info("Split: %d train, %d valid, %d test", len(train), len(valid), len(test))

    for split_name, split_pairs in [("train", train), ("valid", valid), ("test", test)]:
        examples = [build_training_example(p, system_prompt) for p in split_pairs]

        out_path = output_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        log.info("  Wrote %d examples to %s", len(examples), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare personal fine-tuning data")
    parser.add_argument(
        "--variant",
        choices=["category_aware", "raw_style"],
        help="Which variant to prepare",
    )
    parser.add_argument(
        "--both", action="store_true", help="Prepare both variants"
    )
    parser.add_argument(
        "--input", default="data/personal/quality_pairs.jsonl", help="Input JSONL path"
    )
    args = parser.parse_args()

    if not args.both and not args.variant:
        parser.error("Specify --variant or --both")

    input_path = Path(args.input)
    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        log.error("Run scripts/extract_personal_data.py first")
        sys.exit(1)

    # Load pairs
    pairs = []
    with open(input_path) as f:
        for line in f:
            pairs.append(json.loads(line))
    log.info("Loaded %d pairs from %s", len(pairs), input_path)

    if args.both or args.variant == "category_aware":
        prepare_category_aware(pairs, Path("data/personal/category_aware"))

    if args.both or args.variant == "raw_style":
        prepare_raw_style(pairs, Path("data/personal/raw_style"))

    log.info("Done!")


if __name__ == "__main__":
    main()
