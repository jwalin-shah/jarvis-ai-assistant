#!/usr/bin/env python3
"""Prompt Bakeoff: Compare different extraction strategies side-by-side.

Evaluates how different system prompts affect the quality of fact extraction
using the LFM-0.7b model.

Usage:
    uv run python scripts/prompt_bakeoff.py
"""

import sys
import time

sys.path.insert(0, ".")

from jarvis.contacts.instruction_extractor import _EXTRACTION_SYSTEM_PROMPT
from models.loader import MLXModelLoader, ModelConfig

# --- TEST CASES ---
# A mix of 1-on-1, group chats, slang, and complex nuance.
TEST_CASES = [
    {
        "id": "ochos_group",
        "chat_text": """
Jwalin: I'm good with mid summer to late summer! Maybe 4 or 5 days pto paired with a long weekend or just the weekend potentially!
Fantasy Ball Chat: I'm down for whatever.
Bob: Yeah same.
Jwalin: Also I finally got that promotion at Google!
Alice: Congrats!!
        """,
        "context": "Group chat 'Fantasy Ball Chat' with Jwalin, Bob, Alice.",
    },
    {
        "id": "slang_preference",
        "chat_text": """
Jwalin: bro that sushi place was mid tbh.
Contact: really? i thought it was fire.
Jwalin: nah, too much rice. i hate when they do that.
        """,
        "context": "1-on-1 chat.",
    },
    {
        "id": "logistics_vs_fact",
        "chat_text": """
Jwalin: I'm driving to SF right now.
Contact: Drive safe!
Jwalin: I'm moving to SF next month though, for real.
        """,
        "context": "1-on-1 chat. Distinguish logistics vs enduring fact.",
    },
    {
        "id": "complex_inference",
        "chat_text": """
Jwalin: My sister is visiting from Chicago next week.
Contact: Oh nice, is she staying with you?
Jwalin: Yeah, my apartment is tiny but we'll make it work.
        """,
        "context": "1-on-1 chat. Infer sister's location and Jwalin's living situation.",
    },
]

# --- PROMPT VARIANTS ---

# Variant 1: Strict Triples (The "Old" Way)
PROMPT_STRICT = """You are a chat analyzer. Extract facts about Jwalin or Contact.
OUTPUT FORMAT: [Subject] | [Predicate] | [Object]
PREDICATES: lives_in, works_at, likes, dislikes, is_related_to
STRICT RULES: Use ONLY these predicates. No sentences."""

# Variant 2: Sentence-First + Open Predicates (The "New" Way)
PROMPT_SENTENCE_FIRST = _EXTRACTION_SYSTEM_PROMPT

# Variant 3: The "Biographer" (Experiment)
PROMPT_BIOGRAPHER = """You are a Biographer compiling a dossier on Jwalin and his network.
Read the chat and extract every permanent detail you learn about their lives.
Format: Bullet points with rich detail. Capture the vibe."""

VARIANTS = {
    "Strict Triples": PROMPT_STRICT,
    "Sentence-First (Current)": PROMPT_SENTENCE_FIRST,
    "Biographer (Experimental)": PROMPT_BIOGRAPHER,
}


def main() -> None:
    print("Loading Model (LFM-0.7b)...")
    config = ModelConfig(model_id="lfm-0.7b")
    loader = MLXModelLoader(config)
    loader.load()
    print("Model Loaded.\n")

    for case in TEST_CASES:
        print(f"\n{'=' * 60}")
        print(f"TEST CASE: {case['id']}")
        print(f"Context: {case['context']}")
        print("-" * 60)
        print(case["chat_text"].strip())
        print("-" * 60)

        for name, system_prompt in VARIANTS.items():
            print(f"\n>> Variant: {name}")

            # Simple ChatML format
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Chat:\n{case['chat_text']}\n\nFacts:"},
            ]
            prompt = loader._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            start = time.perf_counter()
            res = loader.generate_sync(
                prompt=prompt, max_tokens=150, temperature=0.0, stop_sequences=["<|im_end|>"]
            )
            elapsed = (time.perf_counter() - start) * 1000

            print(f"Output ({elapsed:.0f}ms):\n{res.text.strip()}")


if __name__ == "__main__":
    main()
