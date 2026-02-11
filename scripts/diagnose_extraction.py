"""Diagnose the GLiNER extraction pipeline on real iMessage conversations.

Runs the full candidate extraction pipeline step-by-step, logging where
entities are lost at each stage (threshold, vague filter, fact_type mapping,
entailment). Produces per-message traces and an aggregate funnel summary.

Usage:
    uv run python scripts/diagnose_extraction.py --max-contacts 3 --messages 100
    uv run python scripts/diagnose_extraction.py --chat-id "chat12345"
    uv run python scripts/diagnose_extraction.py --max-contacts 3 -o results/diagnosis.txt
    uv run python scripts/diagnose_extraction.py --max-contacts 3 --with-entailment
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Funnel counters
# ---------------------------------------------------------------------------


class FunnelStats:
    """Track entity counts through each pipeline stage."""

    def __init__(self) -> None:
        self.messages_total = 0
        self.messages_with_text = 0
        self.messages_junk_filtered = 0
        self.messages_gate_filtered = 0
        self.messages_sent_to_gliner = 0

        self.raw_entities = 0
        self.after_threshold = 0
        self.after_vague = 0
        self.after_dedup = 0
        self.after_fact_type = 0  # survived other_personal_fact drop
        self.after_entailment = 0

        # Detailed breakdown
        self.threshold_drops: Counter = Counter()  # label -> count
        self.vague_drops: Counter = Counter()  # label -> count
        self.dedup_drops: Counter = Counter()
        self.fact_type_drops: Counter = Counter()  # label -> count (mapped to other_personal_fact)
        self.entailment_drops: Counter = Counter()  # fact_type -> count

        self.fact_type_kept: Counter = Counter()  # fact_type -> count
        self.label_raw: Counter = Counter()  # label -> count (all raw entities)

    def print_summary(self, out=sys.stdout) -> None:
        w = out.write

        w("\n" + "=" * 60 + "\n")
        w("PIPELINE FUNNEL\n")
        w("=" * 60 + "\n\n")

        w("--- Messages ---\n")
        w(f"  Total messages:          {self.messages_total}\n")
        w(f"  With text:               {self.messages_with_text}\n")
        w(f"  Junk-filtered:           {self.messages_junk_filtered}\n")
        w(f"  Gate-filtered:           {self.messages_gate_filtered}\n")
        w(f"  Sent to GLiNER:          {self.messages_sent_to_gliner}\n")

        w("\n--- Entities ---\n")
        w(f"  Raw GLiNER entities:     {self.raw_entities}\n")
        pct = lambda n, d: f"({n}/{d} dropped - {n * 100 / d:.0f}%)" if d else ""
        thresh_drop = self.raw_entities - self.after_threshold
        w(
            f"  After threshold filter:  {self.after_threshold}  "
            f"{pct(thresh_drop, self.raw_entities)}\n"
        )
        vague_drop = self.after_threshold - self.after_vague
        w(
            f"  After vague filter:      {self.after_vague}  "
            f"{pct(vague_drop, self.after_threshold)}\n"
        )
        dedup_drop = self.after_vague - self.after_dedup
        w(f"  After dedup:             {self.after_dedup}  {pct(dedup_drop, self.after_vague)}\n")
        ft_drop = self.after_dedup - self.after_fact_type
        w(f"  After fact_type mapping: {self.after_fact_type}  {pct(ft_drop, self.after_dedup)}\n")
        if self.after_entailment > 0 or self.entailment_drops:
            ent_drop = self.after_fact_type - self.after_entailment
            w(
                f"  After entailment:        {self.after_entailment}  "
                f"{pct(ent_drop, self.after_fact_type)}\n"
            )

        w(
            f"\n  Final candidates:        "
            f"{self.after_entailment if self.entailment_drops else self.after_fact_type}\n"
        )

        # Breakdown: drops by label at fact_type stage
        if self.fact_type_drops:
            w("\n--- DROPS: No fact_type rule matched (-> other_personal_fact) ---\n")
            for label, count in self.fact_type_drops.most_common():
                w(f"  {label:20s}  {count}\n")

        # Breakdown: drops by label at threshold stage
        if self.threshold_drops:
            w("\n--- DROPS: Below per-label threshold ---\n")
            for label, count in self.threshold_drops.most_common():
                w(f"  {label:20s}  {count}\n")

        # Breakdown: drops at vague stage
        if self.vague_drops:
            w("\n--- DROPS: Vague word filter ---\n")
            for label, count in self.vague_drops.most_common():
                w(f"  {label:20s}  {count}\n")

        # Entailment drops
        if self.entailment_drops:
            w("\n--- DROPS: Entailment gate ---\n")
            for ft, count in self.entailment_drops.most_common():
                w(f"  {ft:30s}  {count}\n")

        # Kept fact_types
        if self.fact_type_kept:
            w("\n--- KEPT: Final fact_type distribution ---\n")
            for ft, count in self.fact_type_kept.most_common():
                w(f"  {ft:30s}  {count}\n")

        # Raw label distribution
        if self.label_raw:
            w("\n--- RAW: GLiNER label distribution ---\n")
            for label, count in self.label_raw.most_common():
                w(f"  {label:20s}  {count}\n")

        w("\n")


# ---------------------------------------------------------------------------
# Per-message diagnosis
# ---------------------------------------------------------------------------


def diagnose_message(
    extractor,
    text: str,
    message_id: int,
    is_from_me: bool,
    prev_messages: list[str] | None,
    next_messages: list[str] | None,
    stats: FunnelStats,
    out=sys.stdout,
    verbose: bool = True,
    skip_gate: bool = False,
) -> list:
    """Run pipeline stages manually on a single message, logging each decision."""
    from jarvis.contacts.candidate_extractor import (
        VAGUE,
    )
    from jarvis.contacts.fact_filter import is_fact_likely
    from jarvis.contacts.junk_filters import is_junk_message

    # Stage 0: junk filter
    if is_junk_message(text):
        stats.messages_junk_filtered += 1
        return []

    # Stage 0b: gate filter
    if not skip_gate and not is_fact_likely(text, is_from_me=is_from_me):
        stats.messages_gate_filtered += 1
        return []

    stats.messages_sent_to_gliner += 1

    # Stage 1: raw GLiNER prediction
    raw_ents = extractor.predict_raw_entities(
        text,
        threshold=extractor._global_threshold,
        prev_messages=prev_messages,
        next_messages=next_messages,
    )

    if not raw_ents:
        return []

    stats.raw_entities += len(raw_ents)
    for e in raw_ents:
        stats.label_raw[e.get("label", "?")] += 1

    if verbose and raw_ents:
        direction = "FROM_ME" if is_from_me else "FROM_THEM"
        out.write(f'\nMSG {message_id}: "{text[:80]}" ({direction})\n')

    candidates = []
    seen: set[tuple[str, str]] = set()

    for e in raw_ents:
        span = str(e.get("text", "")).strip()
        label = str(e.get("label", ""))
        score = float(e.get("score", 0.0))

        # Stage 2: per-label threshold
        min_score = extractor._per_label_min.get(label, extractor._default_min)
        if score < min_score:
            stats.threshold_drops[label] += 1
            if verbose:
                out.write(
                    f'  "{span}" {label} score={score:.2f} '
                    f"-> DROPPED (below threshold {min_score:.2f})\n"
                )
            continue
        stats.after_threshold += 1

        # Stage 3: vague filter
        if span.casefold() in VAGUE or len(span) < 2:
            stats.vague_drops[label] += 1
            if verbose:
                out.write(f'  "{span}" {label} score={score:.2f} -> DROPPED (vague word)\n')
            continue
        stats.after_vague += 1

        # Stage 4: dedup
        dedup_key = (span.casefold(), label)
        if dedup_key in seen:
            stats.dedup_drops[label] += 1
            if verbose:
                out.write(f'  "{span}" {label} score={score:.2f} -> DROPPED (duplicate)\n')
            continue
        seen.add(dedup_key)
        stats.after_dedup += 1

        # Stage 5: fact_type resolution
        fact_type = extractor._resolve_fact_type(text, span, label)

        if fact_type == "other_personal_fact":
            stats.fact_type_drops[label] += 1
            # Show WHY it failed: check rules
            if verbose:
                reason = _explain_fact_type_miss(text, span, label)
                out.write(
                    f'  "{span}" {label} score={score:.2f} -> DROPPED (no fact_type: {reason})\n'
                )
            continue

        stats.after_fact_type += 1
        stats.fact_type_kept[fact_type] += 1

        if verbose:
            # Show which rule matched
            rule_match = _explain_fact_type_hit(text, span, label, fact_type)
            out.write(
                f'  "{span}" {label} score={score:.2f} -> {fact_type} ({rule_match}) -> KEPT\n'
            )

        candidates.append(
            {
                "span": span,
                "label": label,
                "score": score,
                "fact_type": fact_type,
                "source_text": text,
                "message_id": message_id,
            }
        )

    return candidates


def _explain_fact_type_miss(text: str, span: str, label: str) -> str:
    """Explain why _resolve_fact_type returned other_personal_fact."""
    from jarvis.contacts.candidate_extractor import DIRECT_LABEL_MAP, FACT_TYPE_RULES

    # Check if label is in DIRECT_LABEL_MAP
    if label in DIRECT_LABEL_MAP:
        return "BUG: label IS in DIRECT_LABEL_MAP but still got other_personal_fact"

    # Check which rules could have matched on label
    matching_rules = [
        (pattern, ft) for pattern, label_set, ft in FACT_TYPE_RULES if label in label_set
    ]
    if not matching_rules:
        return f"label '{label}' not in DIRECT_LABEL_MAP and not in any FACT_TYPE_RULES label_set"

    # Rules exist for this label, but pattern didn't match
    tried = []
    for pattern, ft in matching_rules:
        tried.append(f"/{pattern}/ -> {ft}")
    return f"label '{label}' has {len(matching_rules)} rules but none matched: {'; '.join(tried)}"


def _explain_fact_type_hit(text: str, span: str, label: str, fact_type: str) -> str:
    """Explain which rule/map entry produced the fact_type."""
    from jarvis.contacts.candidate_extractor import DIRECT_LABEL_MAP, FACT_TYPE_RULES

    # Check pattern rules first (they have priority)
    for pattern, label_set, ft in FACT_TYPE_RULES:
        if label in label_set and re.search(pattern, text, re.IGNORECASE) and ft == fact_type:
            return f"rule: /{pattern}/"

    if label in DIRECT_LABEL_MAP and DIRECT_LABEL_MAP[label] == fact_type:
        return "direct_label_map"

    return "unknown"


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run_diagnosis(
    max_contacts: int = 3,
    messages_per_contact: int = 100,
    chat_id: str | None = None,
    with_entailment: bool = False,
    output_file: str | None = None,
    verbose: bool = True,
    label_profile: str = "high_recall",
    skip_gate: bool = False,
) -> None:
    """Run the full diagnosis on real iMessage data."""

    out = sys.stdout
    out_fh = None
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        out_fh = open(output_file, "w")
        out = out_fh

    try:
        _run_diagnosis_inner(
            max_contacts=max_contacts,
            messages_per_contact=messages_per_contact,
            chat_id=chat_id,
            with_entailment=with_entailment,
            verbose=verbose,
            label_profile=label_profile,
            skip_gate=skip_gate,
            out=out,
        )
    finally:
        if out_fh:
            out_fh.close()
            print(f"Results written to {output_file}", flush=True)


def _run_diagnosis_inner(
    max_contacts: int,
    messages_per_contact: int,
    chat_id: str | None,
    with_entailment: bool,
    verbose: bool,
    label_profile: str,
    skip_gate: bool,
    out,
) -> None:
    from integrations.imessage import ChatDBReader
    from jarvis.contacts.candidate_extractor import CandidateExtractor

    reader = ChatDBReader()
    extractor = CandidateExtractor(
        use_entailment=False,  # we handle entailment separately
        label_profile=label_profile,
    )

    stats = FunnelStats()

    # Gather conversations
    if chat_id:
        chat_ids = [chat_id]
    else:
        conversations = reader.get_conversations(limit=max_contacts * 3)
        # Prefer 1:1 chats, sorted by message count
        conversations.sort(key=lambda c: (len(c.participants), -c.message_count))
        chat_ids = []
        for conv in conversations:
            if conv.chat_id and len(conv.participants) <= 2:
                chat_ids.append(conv.chat_id)
                if len(chat_ids) >= max_contacts:
                    break

    out.write(
        f"Diagnosing {len(chat_ids)} conversation(s), up to {messages_per_contact} messages each\n"
    )
    out.write(f"Label profile: {label_profile}\n")
    out.write(f"Message gate: {'OFF (--no-gate)' if skip_gate else 'ON'}\n")
    out.write(f"Entailment: {'ON' if with_entailment else 'OFF'}\n")
    out.write("=" * 60 + "\n")

    all_candidates = []
    start = time.time()

    for ci, cid in enumerate(chat_ids):
        messages = reader.get_messages(cid, limit=messages_per_contact)
        stats.messages_total += len(messages)

        # Filter to messages with text
        msgs = [m for m in messages if m.text]
        stats.messages_with_text += len(msgs)

        # Sort oldest first for context windowing
        msgs.sort(key=lambda m: m.date)

        elapsed = time.time() - start
        print(
            f"[{ci + 1}/{len(chat_ids)}] {cid[:40]:40s} "
            f"({len(msgs)} msgs with text) elapsed={elapsed:.1f}s",
            flush=True,
        )

        out.write(f"\n{'=' * 60}\n")
        out.write(f"CONVERSATION: {cid}\n")
        out.write(f"Messages: {len(msgs)} with text / {len(messages)} total\n")
        out.write(f"{'=' * 60}\n")

        for mi, msg in enumerate(msgs):
            # Build context: 2 messages before, 1 after
            prev = [msgs[j].text for j in range(max(0, mi - 2), mi) if msgs[j].text]
            nxt = [msgs[j].text for j in range(mi + 1, min(len(msgs), mi + 2)) if msgs[j].text]

            candidates = diagnose_message(
                extractor=extractor,
                text=msg.text,
                message_id=msg.id,
                is_from_me=msg.is_from_me,
                prev_messages=prev or None,
                next_messages=nxt or None,
                stats=stats,
                out=out,
                verbose=verbose,
                skip_gate=skip_gate,
            )
            all_candidates.extend(candidates)

    # Optional entailment pass
    if with_entailment and all_candidates:
        out.write(f"\n{'=' * 60}\n")
        out.write("ENTAILMENT GATE\n")
        out.write(f"{'=' * 60}\n")

        from jarvis.contacts.candidate_extractor import FactCandidate

        # Build FactCandidate objects for entailment
        fc_list = []
        for c in all_candidates:
            fc = FactCandidate(
                message_id=c["message_id"],
                span_text=c["span"],
                span_label=c["label"],
                gliner_score=c["score"],
                fact_type=c["fact_type"],
                start_char=0,
                end_char=len(c["span"]),
                source_text=c["source_text"],
            )
            fc_list.append(fc)

        # Run entailment
        verified = extractor._verify_entailment(fc_list)
        stats.after_entailment = len(verified)

        # Figure out which were dropped
        verified_set = {(fc.message_id, fc.span_text, fc.fact_type) for fc in verified}
        for fc in fc_list:
            key = (fc.message_id, fc.span_text, fc.fact_type)
            if key not in verified_set:
                stats.entailment_drops[fc.fact_type] += 1
                if verbose:
                    out.write(
                        f'  REJECTED: "{fc.span_text}" {fc.fact_type} (msg {fc.message_id})\n'
                    )

        out.write(f"\nEntailment: {len(fc_list)} -> {len(verified)} candidates\n")

    elapsed = time.time() - start
    out.write(f"\nTotal time: {elapsed:.1f}s\n")
    stats.print_summary(out)

    # Also print summary to stdout if writing to file
    if out is not sys.stdout:
        stats.print_summary(sys.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose GLiNER extraction pipeline on real iMessage data",
    )
    parser.add_argument(
        "--max-contacts",
        type=int,
        default=3,
        help="Number of conversations to analyze (default: 3)",
    )
    parser.add_argument(
        "--messages",
        type=int,
        default=100,
        help="Messages per conversation (default: 100)",
    )
    parser.add_argument(
        "--chat-id",
        type=str,
        default=None,
        help="Specific chat_id to diagnose",
    )
    parser.add_argument(
        "--with-entailment",
        action="store_true",
        help="Run entailment gate (requires NLI model, slow)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Write detailed output to file",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Skip the is_fact_likely message gate (see all GLiNER results)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show summary, not per-message traces",
    )
    parser.add_argument(
        "--label-profile",
        type=str,
        default="high_recall",
        choices=["high_recall", "balanced", "high_precision"],
        help="Label profile to use (default: high_recall)",
    )
    args = parser.parse_args()

    run_diagnosis(
        max_contacts=args.max_contacts,
        messages_per_contact=args.messages,
        chat_id=args.chat_id,
        with_entailment=args.with_entailment,
        output_file=args.output,
        verbose=not args.quiet,
        label_profile=args.label_profile,
        skip_gate=args.no_gate,
    )


if __name__ == "__main__":
    main()
