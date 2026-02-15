#!/usr/bin/env python3
"""Diagnostic: capture raw pass-1 LLM output to understand parse failures.

Shows exactly what the model generates for each conversation so we can
diagnose why _parse_pass1_json_lines finds no claims.

Usage:
    uv run python scripts/diagnose_pass1.py --limit 10 --tier 0.7b
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, ".")

from integrations.imessage import ChatDBReader
from jarvis.contacts.instruction_extractor import (
    NEGATIVE_CONSTRAINTS,
    InstructionFactExtractor,
    _build_extraction_system_prompt,
    _is_transactional_message,
    _parse_pass1_json_lines,
)
from jarvis.contacts.junk_filters import is_junk_message
from jarvis.text_normalizer import normalize_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--tier", default="0.7b", choices=["0.7b", "1.2b", "350m"])
    args = parser.parse_args()

    print(f"Loading model ({args.tier})...")
    extractor = InstructionFactExtractor(model_tier=args.tier)
    if not extractor.load():
        print("Failed to load")
        sys.exit(1)
    loader = extractor._loader
    print("Loaded.\n")

    reader = ChatDBReader()
    convos = reader.get_conversations(limit=max(50, args.limit * 10))
    user_name = reader.get_user_name()
    active = [
        c
        for c in convos
        if c.message_count >= 5
        and ("iMessage" in c.chat_id or "RCS" in c.chat_id or "SMS" in c.chat_id)
    ][: args.limit]

    for idx, conv in enumerate(active, 1):
        chat_id = conv.chat_id
        contact_name = conv.display_name or "Contact"
        messages = reader.get_messages(chat_id, limit=100)
        messages.reverse()
        if not messages:
            continue

        # Build prompt text (same as production pipeline)
        prompt_lines = []
        current_label = None
        current_block = []
        for m in messages[:25]:
            if m.is_from_me:
                label = user_name
            else:
                label = getattr(m, "sender_name", None) or contact_name
            raw = " ".join((m.text or "").splitlines()).strip()
            if not raw:
                continue
            clean = normalize_text(
                raw, filter_garbage=True, filter_attributed_artifacts=True, strip_signatures=True
            )
            if not clean:
                continue
            if is_junk_message(clean, chat_id):
                continue
            if _is_transactional_message(clean):
                continue
            if label == current_label:
                current_block.append(clean)
            else:
                if current_block and current_label:
                    prompt_lines.append(f"{current_label}: {' '.join(current_block)}")
                current_label = label
                current_block = [clean]
        if current_block and current_label:
            prompt_lines.append(f"{current_label}: {' '.join(current_block)}")

        batch_text = "\n".join(prompt_lines)
        if not batch_text.strip():
            print(f"  [{idx}/{args.limit}] {contact_name} — EMPTY (skipped)")
            print()
            continue

        # Build pass-1 prompt
        p1_system = _build_extraction_system_prompt(
            user_name=user_name, contact_name=contact_name, segment_count=1
        )
        p1_user = f"Conversation:\n{batch_text}\n\nReturn JSONL durable claims now (or NONE):\n"

        msgs = [
            {"role": "system", "content": p1_system},
            {"role": "user", "content": p1_user},
        ]
        formatted = loader._tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        if not formatted.endswith("\n"):
            formatted += "\n"

        t0 = time.perf_counter()
        res = loader.generate_sync(
            prompt=formatted,
            max_tokens=240,
            temperature=0.0,
            stop_sequences=["<|im_end|>", "###"],
            pre_formatted=True,
            negative_constraints=NEGATIVE_CONSTRAINTS,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        raw_output = res.text.strip()

        # Parse
        claims, canonical = _parse_pass1_json_lines(raw_output, 1)
        claim_count = sum(len(c) for c in claims)

        turns = len(prompt_lines)

        print(f"{'=' * 80}")
        print(
            f"[{idx}/{args.limit}] {contact_name} | {turns} turns | {claim_count} parsed claims | {elapsed_ms:.0f}ms"
        )
        print(f"{'=' * 80}")
        print(f"INPUT ({turns} turns):")
        for line in prompt_lines[:5]:
            print(f"  {line[:100]}")
        if turns > 5:
            print(f"  ... ({turns - 5} more turns)")
        print("\nRAW OUTPUT:")
        for line in raw_output.splitlines():
            print(f"  > {line}")
        print(f"\nPARSED: {canonical}")
        print()


if __name__ == "__main__":
    main()
