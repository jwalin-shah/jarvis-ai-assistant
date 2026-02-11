#!/usr/bin/env python3
"""Smoke test: CandidateExtractor with entailment gate on tricky messages.

Tests attribution (self vs third-party), false positives, and true positives.

Usage:
    uv run python scripts/smoke_test_entailment.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis.contacts.candidate_extractor import CandidateExtractor

# Test messages grouped by expected behavior
TESTS: list[dict] = [
    # === TRUE POSITIVES (should extract) ===
    {"text": "I live in Austin", "expect": "extract", "note": "clear self-location"},
    {"text": "I work at Google", "expect": "extract", "note": "clear self-employer"},
    {"text": "I'm allergic to peanuts", "expect": "extract", "note": "clear health fact"},
    {"text": "I love sushi so much", "expect": "extract", "note": "food preference"},
    {"text": "My birthday is March 15th", "expect": "filter", "note": "GLiNER misses dates"},
    {"text": "I just started at Netflix", "expect": "extract", "note": "new employer"},
    {
        "text": "I moved to Denver last month",
        "expect": "filter",
        "note": "GLiNER misses this phrasing",
    },
    # === ATTRIBUTION CASES ===
    # GLiNER extracts the *relationship* (user has a sister), not the third-party fact.
    # That's correct - "My sister lives in Austin" tells us the user has a sister.
    {
        "text": "My sister lives in Austin",
        "expect": "extract",
        "note": "sister relationship extracted",
    },
    {"text": "My mom works at Google", "expect": "extract", "note": "mom relationship extracted"},
    {
        "text": "My friend is allergic to cats",
        "expect": "filter",
        "note": "friend not extracted (GLiNER)",
    },
    {
        "text": "My brother just moved to Seattle",
        "expect": "extract",
        "note": "brother relationship extracted",
    },
    {"text": "My dad loves fishing", "expect": "extract", "note": "dad relationship extracted"},
    # === FALSE POSITIVE TRAPS (no real fact content) ===
    {"text": "my mom says hi", "expect": "extract", "note": "mom relationship (borderline)"},
    {"text": "ok sounds good", "expect": "filter", "note": "acknowledgment"},
    {"text": "see you at 5pm", "expect": "filter", "note": "logistics"},
    {"text": "lol that's so funny", "expect": "filter", "note": "reaction"},
    {"text": "Can you pick up some milk?", "expect": "filter", "note": "request"},
    {"text": "Have you been to Austin?", "expect": "filter", "note": "question about place"},
    # === EDGE CASES ===
    {"text": "We're both moving to Portland next year", "expect": "extract", "note": "shared move"},
    {"text": "I used to work at Amazon before Google", "expect": "extract", "note": "past+current"},
    {"text": "Sarah is my sister", "expect": "extract", "note": "relationship declaration"},
]


def main() -> None:
    print("Loading CandidateExtractor with entailment gate...", flush=True)
    t0 = time.time()

    # With entailment enabled (uses default threshold from CandidateExtractor)
    ext_with = CandidateExtractor(use_entailment=True)

    # Without entailment for comparison
    ext_without = CandidateExtractor(use_entailment=False)

    print(f"Extractors created in {time.time() - t0:.1f}s\n", flush=True)

    correct = 0
    total = len(TESTS)
    results = []

    for i, test in enumerate(TESTS):
        text = test["text"]
        expect = test["expect"]
        note = test["note"]

        # Run both
        cands_with = ext_with.extract_candidates(text, message_id=i, use_gate=False)
        cands_without = ext_without.extract_candidates(text, message_id=i, use_gate=False)

        got_with = len(cands_with) > 0
        got_without = len(cands_without) > 0

        if expect == "extract":
            passed = got_with
        else:
            passed = not got_with

        correct += int(passed)
        status = "PASS" if passed else "FAIL"

        # Show details
        entailment_effect = ""
        if got_without and not got_with:
            entailment_effect = " [entailment filtered]"
        elif not got_without and not got_with:
            entailment_effect = " [GLiNER filtered]"

        print(f"[{status}] {note}", flush=True)
        print(f'       text: "{text}"', flush=True)
        print(
            f"       expect={expect}, got={'extract' if got_with else 'filter'}{entailment_effect}",
            flush=True,
        )

        if cands_with:
            for c in cands_with:
                print(
                    f"       -> {c.span_text} ({c.span_label}) "
                    f"type={c.fact_type} score={c.gliner_score:.2f}",
                    flush=True,
                )
        if not got_with and cands_without:
            print("       (without entailment would have extracted:)", flush=True)
            for c in cands_without:
                print(
                    f"       -> {c.span_text} ({c.span_label}) "
                    f"type={c.fact_type} score={c.gliner_score:.2f}",
                    flush=True,
                )
        print(flush=True)

    print(f"{'=' * 60}", flush=True)
    print(f"Score: {correct}/{total} ({correct / total * 100:.0f}%)", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
