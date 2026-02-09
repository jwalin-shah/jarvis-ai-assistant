"""Filter raw_pairs.jsonl to high-quality training pairs.

Filters:
- Min reply length 10 chars
- No single-word replies
- No pure emoji/reaction replies
- No duplicate replies
- Diverse contact sampling (cap per contact)
- Prioritize replies with actual conversational substance

Output: data/personal/quality_pairs.jsonl
"""

from __future__ import annotations

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
    handlers=[logging.StreamHandler(), logging.FileHandler("filter_quality.log")],
)
log = logging.getLogger(__name__)

MIN_REPLY_CHARS = 10
MIN_REPLY_WORDS = 3
MAX_PER_CONTACT = 200  # Cap per contact to avoid domination
TARGET_TOTAL = 7500  # Target ~7.5k pairs
SEED = 42

# Low-signal replies to skip
LOW_SIGNAL = {
    "ok",
    "okay",
    "k",
    "kk",
    "lol",
    "haha",
    "hahaha",
    "lmao",
    "yes",
    "no",
    "yeah",
    "yep",
    "nope",
    "yup",
    "sure",
    "thanks",
    "thank you",
    "thx",
    "ty",
    "np",
    "cool",
    "nice",
    "bet",
    "word",
    "facts",
    "true",
    "same",
    "fr",
    "good",
    "great",
    "perfect",
    "awesome",
    "sounds good",
    "got it",
    "alright",
    "bye",
    "later",
    "ttyl",
    "gn",
    "gm",
    "hey",
    "hi",
    "hello",
    "yo",
    "sup",
    "omg",
    "wow",
    "damn",
    "bruh",
    "bro",
    "ugh",
    "smh",
    "idk",
    "nvm",
}

# Emoji-only pattern
EMOJI_ONLY = re.compile(
    r"^["
    r"\U0001f600-\U0001f64f"
    r"\U0001f300-\U0001f5ff"
    r"\U0001f680-\U0001f6ff"
    r"\U0001f1e0-\U0001f1ff"
    r"\U00002702-\U000027b0"
    r"\U000024c2-\U0001f251"
    r"\s"
    r"]+$"
)


def is_quality_reply(reply: str) -> bool:
    """Check if a reply is high quality for training."""
    text = reply.strip()

    # Length checks
    if len(text) < MIN_REPLY_CHARS:
        return False
    if len(text.split()) < MIN_REPLY_WORDS:
        return False

    # Low-signal check
    if text.lower().strip("!?.") in LOW_SIGNAL:
        return False

    # Emoji-only
    if EMOJI_ONLY.match(text):
        return False

    # URL-only
    if re.match(r"^https?://\S+$", text):
        return False

    return True


def filter_pairs(input_path: Path, output_path: Path) -> None:
    """Filter raw pairs to quality pairs."""
    # Load all pairs
    pairs = []
    with open(input_path) as f:
        for line in f:
            pairs.append(json.loads(line))
    log.info("Loaded %d raw pairs", len(pairs))

    # Phase 1: Quality filter
    quality = [p for p in pairs if is_quality_reply(p["reply"])]
    log.info(
        "After quality filter: %d pairs (%.1f%%)", len(quality), 100 * len(quality) / len(pairs)
    )

    # Phase 2: Deduplicate by reply text
    seen_replies: set[str] = set()
    deduped = []
    for p in quality:
        reply_key = p["reply"].lower().strip()
        if reply_key not in seen_replies:
            seen_replies.add(reply_key)
            deduped.append(p)
    log.info("After dedup: %d pairs", len(deduped))

    # Phase 3: Cap per contact
    rng = np.random.default_rng(SEED)
    by_contact: dict[str, list[dict]] = defaultdict(list)
    for p in deduped:
        by_contact[p["contact_name"]].append(p)

    capped = []
    for contact, contact_pairs in by_contact.items():
        if len(contact_pairs) > MAX_PER_CONTACT:
            indices = rng.choice(len(contact_pairs), MAX_PER_CONTACT, replace=False)
            contact_pairs = [contact_pairs[i] for i in indices]
        capped.extend(contact_pairs)
    log.info("After per-contact cap (%d/contact): %d pairs", MAX_PER_CONTACT, len(capped))

    # Phase 4: If still over target, sample down
    if len(capped) > TARGET_TOTAL:
        indices = rng.choice(len(capped), TARGET_TOTAL, replace=False)
        capped = [capped[i] for i in indices]
        log.info("Sampled down to target: %d pairs", len(capped))

    # Shuffle
    rng.shuffle(capped)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in capped:
            f.write(json.dumps(p) + "\n")

    log.info("Wrote %d quality pairs to %s", len(capped), output_path)

    # Stats
    contacts = Counter(p["contact_name"] for p in capped)
    reply_lens = [len(p["reply"]) for p in capped]
    log.info("--- Quality Pairs Summary ---")
    log.info("Total: %d", len(capped))
    log.info("Unique contacts: %d", len(contacts))
    log.info("Avg reply length: %.1f chars", np.mean(reply_lens))
    log.info("Median reply length: %.0f chars", np.median(reply_lens))
    log.info("Top 10 contacts:")
    for name, count in contacts.most_common(10):
        log.info("  %s: %d pairs", name, count)


def main() -> None:
    input_path = Path("data/personal/raw_pairs.jsonl")
    output_path = Path("data/personal/quality_pairs.jsonl")

    if not input_path.exists():
        log.error("Input not found: %s. Run extract_personal_data.py first.", input_path)
        sys.exit(1)

    filter_pairs(input_path, output_path)


if __name__ == "__main__":
    main()
