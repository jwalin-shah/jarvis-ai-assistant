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

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _build_log_handlers() -> list[logging.Handler]:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        handlers.append(logging.FileHandler("filter_quality.log"))
    except OSError as exc:
        print(f"Warning: could not open filter_quality.log for writing: {exc}", flush=True)
    return handlers


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=_build_log_handlers(),
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


def iter_with_progress(
    items: Sequence,
    label: str,
    progress_every: int = 500,
) -> Iterator:
    """Yield items with visible progress for larger collections."""
    total = len(items)
    if total <= 10:
        yield from items
        return

    try:
        from tqdm import tqdm

        yield from tqdm(items, desc=label, unit="item")
        return
    except ImportError:
        pass

    print(f"{label}: processing {total} items...", flush=True)
    for idx, item in enumerate(items, 1):
        if progress_every > 0 and (idx % progress_every == 0 or idx == total):
            print(f"{label}: {idx}/{total}", flush=True)
        yield item


def filter_pairs(
    input_path: Path,
    output_path: Path,
    max_per_contact: int = MAX_PER_CONTACT,
    target_total: int = TARGET_TOTAL,
    seed: int = SEED,
    progress_every: int = 500,
) -> None:
    """Filter raw pairs to quality pairs."""
    import numpy as np

    # Load all pairs
    try:
        with input_path.open() as f:
            lines = f.readlines()
    except OSError as exc:
        log.error("Failed to read input file %s: %s", input_path, exc)
        raise SystemExit(1) from exc

    pairs = []
    for line in iter_with_progress(lines, "Loading pairs", progress_every):
        pairs.append(json.loads(line))
    log.info("Loaded %d raw pairs", len(pairs))

    # Phase 1: Quality filter
    quality = []
    for pair in iter_with_progress(pairs, "Quality filter", progress_every):
        if is_quality_reply(pair["reply"]):
            quality.append(pair)
    log.info(
        "After quality filter: %d pairs (%.1f%%)", len(quality), 100 * len(quality) / len(pairs)
    )

    # Phase 2: Deduplicate by reply text
    seen_replies: set[str] = set()
    deduped = []
    for p in iter_with_progress(quality, "Deduplication", progress_every):
        reply_key = p["reply"].lower().strip()
        if reply_key not in seen_replies:
            seen_replies.add(reply_key)
            deduped.append(p)
    log.info("After dedup: %d pairs", len(deduped))

    # Phase 3: Cap per contact
    rng = np.random.default_rng(seed)
    by_contact: dict[str, list[dict]] = defaultdict(list)
    for p in iter_with_progress(deduped, "Grouping by contact", progress_every):
        by_contact[p["contact_name"]].append(p)

    capped = []
    contact_items = list(by_contact.items())
    for _contact, contact_pairs in iter_with_progress(
        contact_items, "Per-contact capping", progress_every
    ):
        if len(contact_pairs) > max_per_contact:
            indices = rng.choice(len(contact_pairs), max_per_contact, replace=False)
            contact_pairs = [contact_pairs[i] for i in indices]
        capped.extend(contact_pairs)
    log.info("After per-contact cap (%d/contact): %d pairs", max_per_contact, len(capped))

    # Phase 4: If still over target, sample down
    if len(capped) > target_total:
        indices = rng.choice(len(capped), target_total, replace=False)
        capped = [capped[i] for i in indices]
        log.info("Sampled down to target: %d pairs", len(capped))

    # Shuffle
    rng.shuffle(capped)

    # Write output
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.error("Failed to create output directory %s: %s", output_path.parent, exc)
        raise SystemExit(1) from exc

    try:
        with output_path.open("w") as f:
            for p in iter_with_progress(capped, "Writing output", progress_every):
                f.write(json.dumps(p) + "\n")
    except OSError as exc:
        log.error("Failed to write output file %s: %s", output_path, exc)
        raise SystemExit(1) from exc

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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/personal/raw_pairs.jsonl"),
        help="Input raw JSONL path (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/personal/quality_pairs.jsonl"),
        help="Output quality JSONL path (default: %(default)s).",
    )
    parser.add_argument(
        "--max-per-contact",
        type=int,
        default=MAX_PER_CONTACT,
        help="Maximum examples retained per contact (default: %(default)s).",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=TARGET_TOTAL,
        help="Maximum total output pairs after sampling (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for sampling and shuffling (default: %(default)s).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Fallback progress print interval when tqdm is unavailable (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = args.input
    output_path = args.output

    if not input_path.exists():
        log.error("Input not found: %s. Run extract_personal_data.py first.", input_path)
        sys.exit(1)

    filter_pairs(
        input_path,
        output_path,
        max_per_contact=args.max_per_contact,
        target_total=args.target_total,
        seed=args.seed,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
