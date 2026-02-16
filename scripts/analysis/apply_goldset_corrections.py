#!/usr/bin/env python3
"""Apply manual corrections to 58 flagged goldset samples where annotators disagreed.

Each correction was reviewed against the extraction guidelines:
- family_member: lasting family relationships only
- person_name: friends/acquaintances only, NOT celebrities/public figures
- place: where someone LIVES/is FROM/MOVING TO, NOT travel/events
- org: where someone WORKS/STUDIES, NOT casual business mentions
- food_item: lasting preferences only, NOT one-time meals
- activity: lasting hobbies only, NOT one-time events
- health_condition: allergies, dietary restrictions, medical conditions

Usage:
    uv run python scripts/apply_goldset_corrections.py
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict

GOLDSET_DIR = "training_data/goldset_v6"
MERGED_PATH = os.path.join(GOLDSET_DIR, "goldset_v6_merged.json")

# ---------------------------------------------------------------------------
# Corrections: sample_id -> correct expected_candidates
# Each entry is the FULL replacement for expected_candidates.
# ---------------------------------------------------------------------------

def _c(span_text: str, span_label: str, agreement: int = 3, confidence: str = "high") -> dict:
    """Shorthand for building a candidate dict."""
    return {
        "span_text": span_text,
        "span_label": span_label,
        "agreement": agreement,
        "confidence": confidence,
    }


CORRECTIONS: dict[str, list[dict]] = {
    # Celebrity travel mention, not personal
    "v6_0001": [],
    # "Ro" is a friend nickname; Ryanair is casual airline mention
    "v6_0004": [_c("Ro", "person_name", 1, "low")],
    # Transient bagel purchase, not lasting preference
    "v6_0005": [],
    # Past trip question, not where anyone lives
    "v6_0007": [],
    # Restaurant location description
    "v6_0008": [],
    # Restaurant recommendation
    "v6_0009": [],
    # Dinner plan, transient
    "v6_0011": [],
    # Reveals family/friends are in bay area and SoCal (lasting social geography)
    "v6_0013": [_c("bay", "place", 2, "medium"), _c("socal", "place", 2, "medium")],
    # Activity suggestion, not residence
    "v6_0015": [],
    # One-time activity suggestion, not lasting hobbies
    "v6_0016": [],
    # Casual parking garage mention, not employer
    "v6_0021": [],
    # Wineries visited, not residence
    "v6_0022": [],
    # "I used to live in Toronto" = lasting. "my parents" = lasting family.
    "v6_0024": [_c("Toronto", "place", 3, "high"), _c("parents", "family_member", 1, "low")],
    # Travel plan to NOLA
    "v6_0033": [],
    # Event venue
    "v6_0034": [],
    # Travel itinerary
    "v6_0035": [],
    # "France" not even in message; temporary situation about someone else
    "v6_0036": [],
    # Meetup location
    "v6_0039": [],
    # "my parents" = lasting family relationship. Costco = store visit.
    "v6_0042": [_c("parents", "family_member", 1, "low")],
    # Someone moving to Dallas = lasting location change
    "v6_0044": [_c("dallas", "place", 2, "medium")],
    # Travel plan
    "v6_0045": [],
    # Too vague, no living context
    "v6_0050": [],
    # "Parents are out" reveals sender lives with parents
    "v6_0054": [_c("Parents", "family_member", 1, "low")],
    # One-time meal, not lasting preference
    "v6_0057": [],
    # Restaurant review from a trip
    "v6_0060": [],
    # "my brother" = lasting family. Portland/Screen Door = restaurant rec.
    "v6_0062": [_c("brother", "family_member", 2, "medium")],
    # "Chicago" not in message; deep dish is trip food
    "v6_0075": [],
    # Transit map discussion
    "v6_0078": [],
    # Too vague, no living context
    "v6_0080": [],
    # Generic statement about regional cuisine
    "v6_0082": [],
    # Travel planning with flight prices
    "v6_0094": [],
    # Username/handle, not a person name fact
    "v6_0095": [],
    # Discussing news about fires
    "v6_0097": [],
    # Generic "someone" at Google, not a specific contact's employer
    "v6_0098": [],
    # Games they play = recurring hobbies
    "v6_0099": [_c("haxball", "activity", 1, "low"), _c("catan", "activity", 1, "low")],
    # Vague "roommate" reference with no name, transient event
    "v6_0103": [],
    # dad and mom are high-agreement lasting family. "parents" is redundant.
    "v6_0105": [_c("dad", "family_member", 3, "high"), _c("mom", "family_member", 3, "high")],
    # List of friend names
    "v6_0107": [
        _c("Rachit", "person_name", 2, "medium"),
        _c("Tejas", "person_name", 2, "medium"),
        _c("Jwalin", "person_name", 1, "low"),
    ],
    # "Big Fremont guy" = from/lives in Fremont. Angela from context, not message.
    "v6_0110": [_c("Fremont", "place", 3, "high")],
    # Street intersection for meetup
    "v6_0118": [],
    # kishan = friend. Grayson Allen = NBA player (public figure).
    "v6_0122": [_c("kishan", "person_name", 2, "medium")],
    # Ambiguous city name
    "v6_0129": [],
    # Car preference, doesn't fit taxonomy
    "v6_0130": [],
    # Requesting photos, not a lasting preference
    "v6_0134": [],
    # Talking about ride arrival time at a location
    "v6_0135": [],
    # "my mom" = lasting family relationship
    "v6_0136": [_c("mom", "family_member", 1, "low")],
    # "ohlone" not in message text
    "v6_0153": [],
    # "rishi" in context_prev, not this message
    "v6_0154": [],
    # General knowledge about Chicago events
    "v6_0156": [],
    # "mahi" = likely friend name/nickname
    "v6_0158": [_c("mahi", "person_name", 1, "low")],
    # Names from context, not this message
    "v6_0159": [],
    # Commenting on someone else's activity, not own lasting hobby
    "v6_0162": [],
    # Dallas/sangati from context, not message
    "v6_0171": [],
    # All entities from context, not the message
    "v6_0174": [],
    # "mahi" = friend. "jwallys" = joke/slang.
    "v6_0175": [_c("mahi", "person_name", 1, "low")],
    # Glass Animals = music band, not a hobby/activity
    "v6_0178": [],
    # yash and rachit already 3/3 high. "rishi" not in this message.
    "v6_0179": [_c("yash", "person_name", 3, "high"), _c("rachit", "person_name", 3, "high")],
    # ALL are NBA players (public figures)
    "v6_0185": [],
}


# ---------------------------------------------------------------------------
# Stratified train/dev/test split (copied from compute_goldset_iaa.py)
# ---------------------------------------------------------------------------

def stratified_split(
    samples: list[dict],
    train_frac: float = 0.6,
    dev_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split samples into train/dev/test stratified by slice and has_candidates."""
    rng = random.Random(seed)

    strata: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    for s in samples:
        has_cands = len([
            c for c in s.get("expected_candidates", [])
            if c.get("agreement", 0) >= 2
        ]) > 0
        key = (s.get("slice", "unknown"), has_cands)
        strata[key].append(s)

    train, dev, test = [], [], []

    for key, group in sorted(strata.items()):
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * train_frac))
        n_dev = max(1, round(n * dev_frac)) if n > 1 else 0
        if n_train + n_dev >= n:
            n_dev = max(0, n - n_train)
        n_test = n - n_train - n_dev

        train.extend(group[:n_train])
        dev.extend(group[n_train:n_train + n_dev])
        test.extend(group[n_train + n_dev:])

        print(
            f"  Stratum {key}: {n} samples -> "
            f"train={n_train}, dev={n_dev}, test={n_test}",
            flush=True,
        )

    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)

    return train, dev, test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading merged goldset from {MERGED_PATH}...", flush=True)
    with open(MERGED_PATH) as f:
        merged = json.load(f)

    print(f"Loaded {len(merged)} samples.", flush=True)

    # Index by sample_id for fast lookup
    by_id = {s["sample_id"]: s for s in merged}

    applied = 0
    missing = []
    for sample_id, correct_candidates in CORRECTIONS.items():
        if sample_id not in by_id:
            missing.append(sample_id)
            continue

        sample = by_id[sample_id]
        sample["expected_candidates"] = correct_candidates
        sample["review_needed"] = False
        applied += 1

    if missing:
        print(f"WARNING: {len(missing)} sample_ids not found: {missing}", flush=True)

    print(f"Applied {applied} corrections.", flush=True)

    # Stats
    still_flagged = sum(1 for s in merged if s.get("review_needed"))
    total_empty = sum(1 for s in merged if len(s.get("expected_candidates", [])) == 0)
    total_with = sum(1 for s in merged if len(s.get("expected_candidates", [])) > 0)
    print(
        f"After corrections: {total_with} samples with candidates, "
        f"{total_empty} empty, {still_flagged} still flagged for review.",
        flush=True,
    )

    # Save merged
    with open(MERGED_PATH, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"Saved merged goldset: {MERGED_PATH}", flush=True)

    # Re-split
    print("\nRe-splitting into train/dev/test (60/20/20)...", flush=True)
    train, dev, test = stratified_split(merged, seed=42)

    for name, split_data in [("train", train), ("dev", dev), ("test", test)]:
        path = os.path.join(GOLDSET_DIR, f"{name}.json")
        with open(path, "w") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"  Wrote {path}: {len(split_data)} samples", flush=True)

    print(
        f"\nDone. Total: {len(train)} train / {len(dev)} dev / {len(test)} test",
        flush=True,
    )


if __name__ == "__main__":
    main()
