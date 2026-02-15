#!/usr/bin/env python3
"""Prompt Bakeoff v3: Zero-shot conversation-level fact extraction.

Tests 5 zero-shot prompt variants against real iMessage conversations,
scoring on hallucination grounding, empty-input safety, parse success,
and claim volume.

Usage:
    uv run python scripts/prompt_bakeoff_v3.py --limit 20 --tier 0.7b
    uv run python scripts/prompt_bakeoff_v3.py --limit 10 --tier 1.2b
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, ".")

from integrations.imessage import ChatDBReader
from jarvis.contacts.instruction_extractor import (
    NEGATIVE_CONSTRAINTS,
    _is_transactional_message,
    _parse_pass1_json_lines,
)
from jarvis.contacts.junk_filters import is_junk_message
from jarvis.text_normalizer import normalize_text
from jarvis.topics.topic_segmenter import TopicSegment
from models.loader import MLXModelLoader, ModelConfig

# ─── Output ──────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("results/prompt_bakeoff_v3")

# ─── Prompt Variants ─────────────────────────────────────────────────────────


def _prompt_current_baseline(user_name: str, contact_name: str, seg_count: int) -> str:
    """Variant 1: Current production prompt — unchanged control."""
    base = f"""You extract durable personal facts from chat turns.

Task:
- Return ONLY stable personal claims that are useful for a long-lived profile.

Allowed claim types:
- identity/relationship (family, partner, close friend)
- work/school/role/company
- home/current location (not temporary travel)
- durable preferences (likes/dislikes/habits)
- stable schedule preference (e.g., "usually free after 6")

Do NOT extract:
- one-off logistics (meetups, ETAs, "on my way", "can we call", orders, deliveries)
- sports/news chatter, jokes, reactions, tapbacks
- meta speech ("X said", "X asked", "X mentioned")
- facts about group chats/platforms/companies unless clearly about a person
- speculative/uncertain claims

Output format:
- Output JSONL only: one JSON object per line, no markdown.
- If multiple [Segment N] blocks are provided, each line MUST use this schema:
  {{"segment_id": <int>, "subject": "<person>", "speaker": "<speaker>", "claim": "<durable fact>", "evidence_quote": "<exact quote from conversation>"}}
- If only one segment is provided, `segment_id` is optional:
  {{"subject": "<person>", "speaker": "<speaker>", "claim": "<durable fact>", "evidence_quote": "<exact quote from conversation>"}}
- Max 3 claims per segment.
- If no claims at all, output exactly: NONE

Hard rules:
- Only explicit claims from text (no inference).
- Use 3rd-person wording.
- Subject must be a person, never a group/chat.
- `evidence_quote` must be verbatim text copied from the conversation.
- Do not use placeholders, brackets, or variables (no "[City]", "[Job Title]", "<unknown>").
- No headings, markdown, commentary, or extra labels."""
    if seg_count > 1:
        base += "\n\nBatch constraint:\n- Multiple segments are present.\n- `segment_id` is REQUIRED on every JSON line."
    return base


def _prompt_extractive_grounded(user_name: str, contact_name: str, seg_count: int) -> str:
    """Variant 2: Forces verbatim grounding — kills hallucinated quotes."""
    base = f"""You extract durable personal facts from chat messages.

CRITICAL RULE — VERBATIM ONLY:
- The "verbatim_quote" field MUST be an EXACT copy-paste from the conversation text.
- If you cannot find an exact substring to quote, DO NOT output a claim for it.
- NEVER paraphrase, summarize, or invent text for the quote field.

Allowed claims: identity, relationship, work, school, location, durable preferences.
Do NOT extract: logistics, jokes, reactions, greetings, speculative claims.

Output JSONL — one JSON object per line, no markdown:
{{"segment_id": <int>, "subject": "<person>", "claim": "<durable fact in 3rd person>", "verbatim_quote": "<EXACT text from conversation>"}}

Max 3 claims per segment. If no durable facts exist, output exactly: NONE"""
    if seg_count > 1:
        base += "\n\n`segment_id` is REQUIRED on every line."
    return base


def _prompt_empty_safe(user_name: str, contact_name: str, seg_count: int) -> str:
    """Variant 3: Hardened against empty/trivial inputs."""
    base = f"""You extract durable personal facts from chat turns.

FIRST RULE — EMPTY INPUT HANDLING:
- If the conversation text is empty, blank, or contains only greetings/reactions with no personal facts, output EXACTLY the word: NONE
- Do NOT generate example facts. Do NOT output fictional people. Do NOT use placeholder data.

Allowed claims: identity, relationship, work, school, location, durable preferences/habits.
Do NOT extract: logistics, meetups, jokes, reactions, tapbacks, news, speculative claims.

Output format — JSONL only (one JSON object per line, no markdown):
{{"segment_id": <int>, "subject": "<person>", "speaker": "<who said it>", "claim": "<durable fact>", "evidence_quote": "<exact quote>"}}

Max 3 claims per segment. Subject must be a person's name, never a group.
Only extract facts EXPLICITLY stated. No inference. Use 3rd-person wording.
evidence_quote must be verbatim text from the conversation — never fabricate."""
    if seg_count > 1:
        base += "\n\n`segment_id` is REQUIRED on every line (multiple segments present)."
    return base


def _prompt_role_anchored(user_name: str, contact_name: str, seg_count: int) -> str:
    """Variant 4: Explicit role anchoring to prevent role confusion."""
    base = f"""You extract durable personal facts from chat messages.

PARTICIPANT ROLES:
- "{user_name}" is the phone owner (lines starting with "{user_name}:").
- "{contact_name}" is the other participant (lines starting with "{contact_name}:" or other names).
- NEVER attribute {user_name}'s statements to {contact_name} or vice versa.
- NEVER confuse who is the professional vs. the client, the sender vs. receiver.

Allowed claims: identity, relationship, work, school, location, preferences, habits.
Do NOT extract: logistics, meetups, jokes, reactions, greetings, speculative claims.

Output JSONL — one JSON per line, no markdown:
{{"segment_id": <int>, "subject": "<EXACT person name>", "speaker": "<who actually said it>", "claim": "<durable fact in 3rd person>", "evidence_quote": "<exact quote from conversation>"}}

Hard rules:
- Max 3 claims per segment. If none, output: NONE
- subject must be a specific person's name.
- evidence_quote must be verbatim from the conversation.
- No placeholders, no markdown, no commentary."""
    if seg_count > 1:
        base += "\n\n`segment_id` is REQUIRED (multiple segments)."
    return base


def _prompt_combined_v3(user_name: str, contact_name: str, seg_count: int) -> str:
    """Variant 5: Combined best-of-all — grounding + empty safety + role anchoring."""
    base = f"""Extract durable personal facts from a chat conversation.

PARTICIPANTS:
- "{user_name}" = phone owner. "{contact_name}" = other participant(s).
- Never confuse who said what. Check the speaker label on each line.

EMPTY INPUT: If chat is empty or has no personal facts, output EXACTLY: NONE
Do NOT generate fictional examples, placeholder names, or made-up quotes.

WHAT TO EXTRACT: stable facts useful for a long-lived profile:
- Identity/relationship (family, partner, friend by name)
- Work/school/role/company
- Home location (not travel)
- Durable preferences (likes, dislikes, habits)

WHAT TO SKIP: logistics, meetups, jokes, reactions, tapbacks, greetings, speculation.

OUTPUT FORMAT: JSONL only — one JSON object per line. No markdown, no commentary.
{{"segment_id": <int>, "subject": "<person name>", "speaker": "<who said it>", "claim": "<durable fact, 3rd person>", "verbatim_quote": "<EXACT substring copied from the conversation>"}}

HARD RULES:
1. Max 3 claims per segment.
2. verbatim_quote MUST be an exact substring from the chat — never paraphrase or invent.
3. If you cannot find a verbatim quote for a claim, DO NOT output that claim.
4. Subject must be a person, never a group/chat name.
5. No inference — only what is explicitly stated."""
    if seg_count > 1:
        base += "\n\n`segment_id` is REQUIRED on every JSON line."
    return base


def _prompt_v3_primed_json(user_name: str, contact_name: str, seg_count: int) -> str:
    """Variant 6: combined_v3 prompt + JSON output priming with '{' appended."""
    # Same prompt as combined_v3 — the improvement is in the output priming
    return _prompt_combined_v3(user_name, contact_name, seg_count)


def _prompt_v3_slim_schema(user_name: str, contact_name: str, seg_count: int) -> str:
    """Variant 7: Simplified 3-field JSON schema (drop speaker + segment_id)."""
    base = f"""Extract durable personal facts from a chat conversation.

PARTICIPANTS:
- "{user_name}" = phone owner. "{contact_name}" = other participant(s).
- Never confuse who said what. Check the speaker label on each line.

EMPTY INPUT: If chat is empty or has no personal facts, output EXACTLY: NONE
Do NOT generate fictional examples, placeholder names, or made-up quotes.

WHAT TO EXTRACT: stable facts useful for a long-lived profile:
- Identity/relationship (family, partner, friend by name)
- Work/school/role/company
- Home location (not travel)
- Durable preferences (likes, dislikes, habits)

WHAT TO SKIP: logistics, meetups, jokes, reactions, tapbacks, greetings, speculation.

OUTPUT FORMAT: JSONL only — one JSON object per line. No markdown, no commentary.
{{"subject": "<person name>", "claim": "<durable fact, 3rd person>", "quote": "<EXACT substring from the conversation>"}}

HARD RULES:
1. Max 3 claims per segment.
2. quote MUST be an exact substring from the chat — never paraphrase or invent.
3. If you cannot find a verbatim quote for a claim, DO NOT output that claim.
4. Subject must be a person, never a group/chat name.
5. No inference — only what is explicitly stated."""
    return base


def _prompt_v3_all_optimized(user_name: str, contact_name: str, seg_count: int) -> str:
    """Variant 8: All optimizations stacked — slim schema + negative example + priming."""
    base = f"""Extract durable personal facts from a chat conversation.

PARTICIPANTS:
- "{user_name}" = phone owner. "{contact_name}" = other participant(s).
- Never confuse who said what. Check the speaker label on each line.

EMPTY INPUT: If chat is empty or has no personal facts, output EXACTLY: NONE
WRONG (never do this): {{"subject": "Alice", "claim": "enjoys hiking", "quote": "I love hiking!"}}
Do NOT generate fictional names, placeholder quotes, or made-up facts.

WHAT TO EXTRACT: stable facts useful for a long-lived profile:
- Identity/relationship (family, partner, friend by name)
- Work/school/role/company
- Home location (not travel)
- Durable preferences (likes, dislikes, habits)

WHAT TO SKIP: logistics, meetups, jokes, reactions, tapbacks, greetings, speculation.

OUTPUT FORMAT: JSONL only — one JSON object per line. No markdown, no commentary.
{{"subject": "<person name>", "claim": "<durable fact, 3rd person>", "quote": "<EXACT substring from the conversation>"}}

HARD RULES:
1. Max 3 claims per segment.
2. quote MUST be an exact substring from the chat — never paraphrase or invent.
3. If you cannot find a verbatim quote for a claim, DO NOT output that claim.
4. Subject must be a person, never a group/chat name.
5. No inference — only what is explicitly stated."""
    return base


# Expanded negative constraints: original + hallucination patterns from v3/v3.1
EXPANDED_NEGATIVE_CONSTRAINTS = NEGATIVE_CONSTRAINTS + [
    "Alice", "Bob", "Charlie", "example", "Durable",
    "[Subject", "[name", "[phone", "[preference",
    "placeholder", "fictional",
]

# Strategy registry. "format" controls parsing, priming, and user prompt.
STRATEGIES: dict[str, dict] = {
    "current_baseline": {"fn": _prompt_current_baseline, "format": "json"},
    "combined_v3": {"fn": _prompt_combined_v3, "format": "json"},
    "v3_primed_json": {"fn": _prompt_v3_primed_json, "format": "json_primed"},
    "v3_slim_schema": {"fn": _prompt_v3_slim_schema, "format": "json"},
    "v3_all_optimized": {"fn": _prompt_v3_all_optimized, "format": "json_primed"},
}


# ─── Scoring ──────────────────────────────────────────────────────────────────


def is_grounded(quote: str, source_text: str) -> bool:
    """Check if evidence/verbatim quote appears (fuzzy) in source text."""
    if not quote or not source_text:
        return False
    q_lower = quote.lower().strip()
    s_lower = source_text.lower()
    # Exact substring
    if q_lower in s_lower:
        return True
    # Fuzzy: 60%+ of words present in source
    words = [w for w in q_lower.split() if len(w) > 2]
    if not words:
        return False
    found = sum(1 for w in words if w in s_lower)
    return found / len(words) >= 0.6


@dataclass
class StrategyMetrics:
    """Aggregate metrics for one prompt strategy."""
    name: str
    total_claims: int = 0
    grounded_claims: int = 0
    ungrounded_claims: int = 0
    empty_correct: int = 0     # correctly output NONE on empty input
    empty_hallucinated: int = 0  # hallucinated on empty input
    empty_total: int = 0       # total empty-input examples
    parse_success: int = 0
    parse_fail: int = 0
    total_examples: int = 0
    total_ms: float = 0.0
    per_example: list = field(default_factory=list)

    @property
    def grounding_rate(self) -> float:
        return self.grounded_claims / self.total_claims if self.total_claims else 1.0

    @property
    def hallucination_rate(self) -> float:
        return self.ungrounded_claims / self.total_claims if self.total_claims else 0.0

    @property
    def parse_rate(self) -> float:
        total = self.parse_success + self.parse_fail
        return self.parse_success / total if total else 0.0

    @property
    def empty_safety_rate(self) -> float:
        return self.empty_correct / self.empty_total if self.empty_total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total_examples if self.total_examples else 0.0


# ─── Conversation Builder (same as dump_json_vs_summary_examples.py) ─────────


def _build_segments(reader: ChatDBReader, limit: int, window: int):
    """Build conversation segments from real iMessage data."""
    convos = reader.get_conversations(limit=max(50, min(1000, limit * 10)))
    user_name = reader.get_user_name()
    active = [
        c for c in convos
        if c.message_count >= 5
        and ("iMessage" in c.chat_id or "RCS" in c.chat_id or "SMS" in c.chat_id)
    ][:limit]

    out = []
    for conv in active:
        chat_id = conv.chat_id
        contact_name = conv.display_name or "Contact"
        messages = reader.get_messages(chat_id, limit=window)
        messages.reverse()
        if not messages:
            continue
        segments = []
        for j in range(0, len(messages), 20):
            window_msgs = messages[j:j + 25]
            if not window_msgs:
                break
            segments.append(
                TopicSegment(
                    chat_id=chat_id,
                    contact_id=chat_id,
                    messages=window_msgs,
                    start_time=window_msgs[0].date,
                    end_time=window_msgs[-1].date,
                    message_count=len(window_msgs),
                    segment_id=f"bakeoff_{chat_id}_{j}",
                    text="\n".join((m.text or "") for m in window_msgs),
                )
            )
        if segments:
            out.append((chat_id, contact_name, segments))
    return out, user_name


def _build_batch_text(segments, contact_id: str, contact_name: str, user_name: str) -> str:
    """Format segments into the chat text the LLM sees."""
    segment_texts = []
    for i, segment in enumerate(segments):
        messages = getattr(segment, "messages", [])
        prompt_lines = []
        current_label = None
        current_block = []
        for m in messages:
            label = user_name if m.is_from_me else (getattr(m, "sender_name", None) or contact_name)
            raw = " ".join((m.text or "").splitlines()).strip()
            if not raw:
                continue
            clean = normalize_text(raw, filter_garbage=True, filter_attributed_artifacts=True, strip_signatures=True)
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


def _parse_raw_jsonl(raw: str, seg_count: int) -> list[dict]:
    """Parse JSONL output, tolerating markdown fences and partial output."""
    # Strip markdown fences
    raw = re.sub(r"```(?:jsonl?|json)?\s*", "", raw)
    raw = re.sub(r"```\s*$", "", raw)

    items = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.upper() == "NONE":
            continue
        line = re.sub(r"^[\s\-\*\d\.]+\s*", "", line).strip()
        if not line:
            continue
        # Try to parse as JSON
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                items.append(obj)
        except json.JSONDecodeError:
            pass
    return items


def _apply_grounding_filter(items: list[dict], source_text: str) -> list[dict]:
    """Post-processing: reject claims whose quote is not grounded in source."""
    if not source_text:
        return items
    grounded = []
    for item in items:
        quote = item.get("verbatim_quote", "") or item.get("quote", "")
        if not quote:
            # No quote provided — let it through (scored as ungrounded later)
            grounded.append(item)
        elif is_grounded(quote, source_text):
            grounded.append(item)
        # else: silently drop ungrounded claim
    return grounded


# ─── Main Runner ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt Bakeoff v3")
    parser.add_argument("--limit", type=int, default=20, help="Number of chats")
    parser.add_argument("--window", type=int, default=100, help="Messages per chat")
    parser.add_argument("--tier", default="0.7b", help="Model tier")
    parser.add_argument("--max-tokens", type=int, default=300, help="Max generation tokens")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    from jarvis.contacts.instruction_extractor import MODELS
    model_path = MODELS.get(args.tier, MODELS.get("0.7b"))
    config = ModelConfig(model_path=model_path, default_temperature=0.1)
    loader = MLXModelLoader(config)
    print(f"Loading model ({args.tier})...")
    loader.load()
    print("Model loaded.\n")

    # Load conversations
    print("Reading iMessage conversations...")
    with ChatDBReader() as reader:
        datasets, user_name = _build_segments(reader, args.limit, args.window)
    print(f"Loaded {len(datasets)} conversations as {user_name}.\n")

    # Init metrics
    all_metrics: dict[str, StrategyMetrics] = {
        name: StrategyMetrics(name=name) for name in STRATEGIES
    }

    comparison_lines: list[str] = []
    comparison_lines.append(f"PROMPT BAKEOFF V3.2 — {len(datasets)} conversations, tier={args.tier}")
    comparison_lines.append(f"{'=' * 100}\n")

    for idx, (chat_id, contact_name, segments) in enumerate(datasets, 1):
        seg_count = len(segments)
        # Limit to max 5 segments per chat to keep experiment manageable
        segments = segments[:5]
        seg_count = len(segments)

        batch_text = _build_batch_text(segments, chat_id, contact_name, user_name)
        is_empty = not batch_text.strip()

        comparison_lines.append(f"\n{'=' * 100}")
        comparison_lines.append(f"CHAT {idx}/{len(datasets)} | {contact_name} | segments={seg_count} | empty={is_empty}")
        comparison_lines.append(f"{'=' * 100}")
        if is_empty:
            comparison_lines.append("INPUT: <EMPTY_AFTER_FILTERS>")
        else:
            # Show first 500 chars of input
            preview = batch_text[:500] + ("..." if len(batch_text) > 500 else "")
            comparison_lines.append(f"INPUT PREVIEW:\n{preview}")
        comparison_lines.append("")

        for strat_name, strat_info in STRATEGIES.items():
            strat_fn = strat_info["fn"]
            strat_format = strat_info["format"]
            metrics = all_metrics[strat_name]
            metrics.total_examples += 1

            system_prompt = strat_fn(user_name, contact_name, seg_count)
            user_prompt = f"Conversation:\n{batch_text}\n\nReturn JSONL durable claims now (or NONE):\n"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            formatted = loader._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Output priming: append '{' for json_primed to force JSON start
            if strat_format == "json_primed":
                formatted = formatted.rstrip() + '\n{'
            else:
                if not formatted.endswith("\n"):
                    formatted += "\n"

            # Use expanded constraints for optimized variants
            constraints = (
                EXPANDED_NEGATIVE_CONSTRAINTS
                if strat_name in ("v3_all_optimized", "v3_primed_json")
                else NEGATIVE_CONSTRAINTS
            )

            t0 = time.perf_counter()
            try:
                res = loader.generate_sync(
                    prompt=formatted,
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                    stop_sequences=["<|im_end|>", "###"],
                    pre_formatted=True,
                    negative_constraints=constraints,
                )
                raw_output = res.text.strip()
                # For primed format, prepend the '{' we used to prime
                if strat_format == "json_primed":
                    raw_output = "{" + raw_output
            except Exception as e:
                raw_output = f"<ERROR: {e}>"
            elapsed_ms = (time.perf_counter() - t0) * 1000
            metrics.total_ms += elapsed_ms

            # Parse JSON output
            parsed_items = _parse_raw_jsonl(raw_output, seg_count)

            # Post-processing grounding filter for all_optimized
            if strat_name == "v3_all_optimized" and not is_empty:
                parsed_items = _apply_grounding_filter(parsed_items, batch_text)
            is_none = raw_output.upper().strip() == "NONE" or not raw_output.strip()

            if is_empty:
                # Empty input scoring
                metrics.empty_total += 1
                if is_none or len(parsed_items) == 0:
                    metrics.empty_correct += 1
                else:
                    metrics.empty_hallucinated += 1

            if parsed_items:
                metrics.parse_success += 1
            elif not is_none and not is_empty:
                metrics.parse_fail += 1
            else:
                metrics.parse_success += 1  # NONE is a valid parse

            # Score grounding for each claim
            example_claims = []
            for item in parsed_items:
                quote = item.get("evidence_quote", "") or item.get("verbatim_quote", "")
                claim = item.get("claim", "")
                subject = item.get("subject", "")
                grounded = is_grounded(quote, batch_text) if quote else False
                metrics.total_claims += 1
                if grounded:
                    metrics.grounded_claims += 1
                else:
                    metrics.ungrounded_claims += 1
                example_claims.append({
                    "subject": subject,
                    "claim": claim,
                    "quote": quote[:80],
                    "grounded": grounded,
                })

            # Write comparison
            comparison_lines.append(f"  >> {strat_name} ({elapsed_ms:.0f}ms) — {len(parsed_items)} claims")
            if is_none and is_empty:
                comparison_lines.append(f"     ✅ Correctly output NONE for empty input")
            elif is_empty and parsed_items:
                comparison_lines.append(f"     ❌ HALLUCINATED {len(parsed_items)} claims on empty input!")
            for c in example_claims:
                icon = "✅" if c["grounded"] else "❌"
                comparison_lines.append(f"     {icon} [{c['subject']}] {c['claim'][:60]}")
                if not c["grounded"] and c["quote"]:
                    comparison_lines.append(f"        ungrounded quote: \"{c['quote']}\"")
            comparison_lines.append("")

        # Progress
        print(f"  [{idx}/{len(datasets)}] {contact_name} — done", flush=True)

    # Summary
    comparison_lines.append(f"\n{'=' * 100}")
    comparison_lines.append("FINAL SUMMARY")
    comparison_lines.append(f"{'=' * 100}\n")

    summary_data = {}
    for name, m in all_metrics.items():
        summary_data[name] = {
            "total_claims": m.total_claims,
            "grounding_rate": round(m.grounding_rate, 3),
            "hallucination_rate": round(m.hallucination_rate, 3),
            "grounded": m.grounded_claims,
            "ungrounded": m.ungrounded_claims,
            "parse_rate": round(m.parse_rate, 3),
            "empty_safety_rate": round(m.empty_safety_rate, 3),
            "empty_correct": m.empty_correct,
            "empty_hallucinated": m.empty_hallucinated,
            "avg_ms": round(m.avg_ms, 1),
            "total_examples": m.total_examples,
        }
        comparison_lines.append(
            f"  {name:25s}  "
            f"claims={m.total_claims:3d}  "
            f"grounding={m.grounding_rate:.1%}  "
            f"halluc={m.hallucination_rate:.1%}  "
            f"parse={m.parse_rate:.1%}  "
            f"empty_safe={m.empty_safety_rate:.1%}  "
            f"avg={m.avg_ms:.0f}ms"
        )

    # Determine winner
    best_name = max(
        all_metrics,
        key=lambda n: (
            all_metrics[n].grounding_rate * 0.5
            + all_metrics[n].empty_safety_rate * 0.3
            + all_metrics[n].parse_rate * 0.2
        ),
    )
    comparison_lines.append(f"\n  🏆 RECOMMENDED: {best_name}")

    # Save
    comp_path = OUTPUT_DIR / "comparison.txt"
    with open(comp_path, "w") as f:
        f.write("\n".join(comparison_lines))
    print(f"\nComparison written to: {comp_path}")

    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Metrics written to: {metrics_path}")

    # Print summary
    print(f"\n{'=' * 80}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 80}")
    for name, m in all_metrics.items():
        print(
            f"  {name:25s}  "
            f"claims={m.total_claims:3d}  "
            f"ground={m.grounding_rate:.1%}  "
            f"halluc={m.hallucination_rate:.1%}  "
            f"empty_safe={m.empty_safety_rate:.1%}  "
            f"parse={m.parse_rate:.1%}"
        )
    print(f"\n  🏆 RECOMMENDED: {best_name}")


if __name__ == "__main__":
    main()
