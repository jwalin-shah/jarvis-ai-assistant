#!/usr/bin/env python3
"""Dump full chat inputs and JSON-vs-summary outputs for manual review."""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")

from integrations.imessage import ChatDBReader
from jarvis.contacts.instruction_extractor import (
    NEGATIVE_CONSTRAINTS,
    _build_extraction_system_prompt,
    _is_transactional_message,
    _parse_pass1_json_lines,
    get_instruction_extractor,
)
from jarvis.contacts.junk_filters import is_junk_message
from jarvis.text_normalizer import normalize_text
from scripts.compare_json_vs_summary import _build_segments, _parse_summary_output


def _build_batch_text(segments, contact_id: str, contact_name: str, user_name: str) -> str:
    segment_texts: list[str] = []
    for i, segment in enumerate(segments):
        messages = getattr(segment, "messages", [])
        prompt_lines: list[str] = []
        current_label = None
        current_block: list[str] = []
        for m in messages:
            label = user_name if m.is_from_me else (getattr(m, "sender_name", None) or contact_name)
            raw = " ".join((m.text or "").splitlines()).strip()
            if not raw:
                continue
            clean = normalize_text(
                raw,
                filter_garbage=True,
                filter_attributed_artifacts=True,
                strip_signatures=True,
            )
            if not clean:
                continue
            if is_junk_message(clean, contact_id):
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

        seg_text = "\n".join(prompt_lines)
        if len(segments) == 1:
            segment_texts.append(seg_text)
        else:
            segment_texts.append(f"[Segment {i}]\n{seg_text}")
    return "\n\n".join(segment_texts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump strict-JSON vs summary outputs per chat")
    parser.add_argument("--limit", type=int, default=20, help="Number of chats to dump")
    parser.add_argument("--window", type=int, default=100, help="Messages per chat")
    parser.add_argument("--tier", default="0.7b", help="Extractor tier")
    parser.add_argument(
        "--output",
        default="/Users/jwalinshah/projects/jarvis-ai-assistant/results/json_vs_summary_dump.txt",
        help="Output file path",
    )
    parser.add_argument("--summary-max-tokens", type=int, default=240, help="Summary generation max tokens")
    args = parser.parse_args()

    extractor = get_instruction_extractor(args.tier)
    if not extractor.is_loaded():
        extractor.load()

    summary_system = (
        "Extract durable personal facts about people from chat text.\n"
        "Return concise lines only in format: <Person> | <durable fact>.\n"
        "No JSON. No commentary. No inference. If none, return NONE."
    )

    with ChatDBReader() as reader:
        datasets, user_name = _build_segments(reader, args.limit, args.window)

    lines: list[str] = []
    for idx, (chat_id, contact_name, segments) in enumerate(datasets, start=1):
        batch_text = _build_batch_text(segments, chat_id, contact_name, user_name)

        p1_system = _build_extraction_system_prompt(user_name, contact_name, len(segments))
        p1_user = f"Conversation:\n{batch_text}\n\nReturn JSONL durable claims now (or NONE):\n"
        json_msgs = [
            {"role": "system", "content": p1_system},
            {"role": "user", "content": p1_user},
        ]
        formatted_json = extractor._loader._tokenizer.apply_chat_template(
            json_msgs, tokenize=False, add_generation_prompt=True
        )
        if not formatted_json.endswith("\n"):
            formatted_json += "\n"
        json_res = extractor._loader.generate_sync(
            prompt=formatted_json,
            max_tokens=240,
            temperature=0.0,
            stop_sequences=["<|im_end|>", "###"],
            pre_formatted=True,
            negative_constraints=NEGATIVE_CONSTRAINTS,
        )
        parsed_json, _ = _parse_pass1_json_lines(json_res.text.strip(), len(segments))

        summary_user = f"Conversation:\n{batch_text}\n\nPeople facts:\n"
        sum_msgs = [
            {"role": "system", "content": summary_system},
            {"role": "user", "content": summary_user},
        ]
        formatted_sum = extractor._loader._tokenizer.apply_chat_template(
            sum_msgs, tokenize=False, add_generation_prompt=True
        )
        if not formatted_sum.endswith("\n"):
            formatted_sum += "\n"
        sum_res = extractor._loader.generate_sync(
            prompt=formatted_sum,
            max_tokens=args.summary_max_tokens,
            temperature=0.0,
            stop_sequences=["<|im_end|>", "###"],
            pre_formatted=True,
            negative_constraints=NEGATIVE_CONSTRAINTS,
        )
        parsed_sum = _parse_summary_output(sum_res.text.strip(), len(segments))

        lines.append("=" * 120)
        lines.append(
            f"EXAMPLE {idx} | chat_id={chat_id} | contact={contact_name} | segments={len(segments)}"
        )
        lines.append("=" * 120)
        lines.append("FULL CHAT TEXT PASSED TO LLM (after normalization+filters):")
        lines.append("-" * 120)
        lines.append(batch_text if batch_text.strip() else "<EMPTY_AFTER_FILTERS>")

        lines.append("")
        lines.append("STRICT JSON RAW OUTPUT:")
        lines.append("-" * 120)
        lines.append(json_res.text.strip() or "<EMPTY>")

        lines.append("")
        lines.append("STRICT JSON PARSED CLAIMS:")
        lines.append("-" * 120)
        any_json = False
        for sidx, claims in enumerate(parsed_json):
            for claim in claims:
                any_json = True
                lines.append(f"[Segment {sidx}] {claim}")
        if not any_json:
            lines.append("<NO_PARSED_CLAIMS>")

        lines.append("")
        lines.append("SUMMARY RAW OUTPUT:")
        lines.append("-" * 120)
        lines.append(sum_res.text.strip() or "<EMPTY>")

        lines.append("")
        lines.append("SUMMARY PARSED CLAIMS:")
        lines.append("-" * 120)
        any_sum = False
        for sidx, claims in enumerate(parsed_sum):
            for claim in claims:
                any_sum = True
                lines.append(f"[Segment {sidx}] {claim}")
        if not any_sum:
            lines.append("<NO_PARSED_CLAIMS>")
        lines.append("")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {len(datasets)} chat dumps to: {args.output}")


if __name__ == "__main__":
    main()
