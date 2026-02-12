#!/usr/bin/env python3
"""Combined extraction evaluation: GliNER + spaCy ensemble with LLM filtering.

Architecture:
  1. GliNER + spaCy extract candidate entities (high recall)
  2. LLM evaluates all candidates per message in one batch call:
     "Which of these are lasting personal facts about the sender's contact?"
  3. Metrics computed against gold set

No hardcoded entity lists. The LLM handles all contextual filtering.

Usage:
    uv run python scripts/eval_combined_extraction.py \
        --gold training_data/goldset_v6/goldset_v6_merged.json

    # Skip GliNER (slow), spaCy + LLM filter only:
    uv run python scripts/eval_combined_extraction.py \
        --gold training_data/goldset_v6/goldset_v6_merged.json --spacy-only

    # Skip LLM filter, raw extractor output:
    uv run python scripts/eval_combined_extraction.py \
        --gold training_data/goldset_v6/goldset_v6_merged.json --no-llm

    # Verbose per-message output:
    uv run python scripts/eval_combined_extraction.py \
        --gold training_data/goldset_v6/goldset_v6_merged.json --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from eval_shared import DEFAULT_LABEL_ALIASES, spans_match
from gliner_shared import enforce_runtime_stack

log = logging.getLogger(__name__)

GOLD_PATH = Path("training_data/goldset_v6/goldset_v6_merged.json")
METRICS_PATH = Path("results/combined_extraction/combined_metrics.json")
ERRORS_PATH = Path("results/combined_extraction/errors.json")

# Labels that map to our goldset schema
VALID_LABELS = {
    "family_member", "activity", "health_condition", "job_role", "org",
    "place", "food_item", "current_location", "future_location",
    "past_location", "friend_name", "person_name",
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def support(self) -> int:
        return self.tp + self.fn

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "support": self.support,
        }


def compute_metrics(
    gold_records: list[dict],
    predictions: dict[str, list[dict]],
) -> dict:
    """Compute span-level P/R/F1 with label aliasing."""
    overall = Metrics()
    per_label: dict[str, Metrics] = defaultdict(Metrics)
    per_slice: dict[str, Metrics] = defaultdict(Metrics)
    errors: list[dict] = []

    for rec in gold_records:
        sid = rec["sample_id"]
        gold_cands = rec.get("expected_candidates") or []
        pred_cands = predictions.get(sid, [])
        slc = rec.get("slice", "unknown")

        gold_matched = [False] * len(gold_cands)
        pred_matched = [False] * len(pred_cands)

        for gi, gc in enumerate(gold_cands):
            for pi, pc in enumerate(pred_cands):
                if pred_matched[pi]:
                    continue
                if spans_match(
                    pc.get("span_text", ""),
                    pc.get("span_label", ""),
                    gc.get("span_text", ""),
                    gc.get("span_label", ""),
                    label_aliases=DEFAULT_LABEL_ALIASES,
                ):
                    gold_matched[gi] = True
                    pred_matched[pi] = True
                    overall.tp += 1
                    per_label[gc["span_label"]].tp += 1
                    per_slice[slc].tp += 1
                    break

        for gi, gc in enumerate(gold_cands):
            if not gold_matched[gi]:
                overall.fn += 1
                per_label[gc["span_label"]].fn += 1
                per_slice[slc].fn += 1
                errors.append({
                    "type": "fn",
                    "sample_id": sid,
                    "slice": slc,
                    "message_text": rec["message_text"][:120],
                    "gold_span": gc["span_text"],
                    "gold_label": gc["span_label"],
                })

        for pi, pc in enumerate(pred_cands):
            if not pred_matched[pi]:
                overall.fp += 1
                label = pc.get("span_label", "unknown")
                per_label[label].fp += 1
                per_slice[slc].fp += 1
                errors.append({
                    "type": "fp",
                    "sample_id": sid,
                    "slice": slc,
                    "message_text": rec["message_text"][:120],
                    "pred_span": pc.get("span_text", ""),
                    "pred_label": label,
                    "source": pc.get("source", "unknown"),
                })

    return {
        "overall": overall.to_dict(),
        "per_label": {k: v.to_dict() for k, v in sorted(per_label.items())},
        "per_slice": {k: v.to_dict() for k, v in sorted(per_slice.items())},
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# spaCy extraction
# ---------------------------------------------------------------------------

_NLP = None


def _get_spacy():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


_SPACY_LABEL_MAP = {
    "PERSON": "person_name",
    "GPE": "place",
    "LOC": "place",
    "ORG": "org",
}


def extract_spacy(text: str) -> list[dict]:
    """Extract entities using spaCy NER."""
    nlp = _get_spacy()
    doc = nlp(text)
    spans = []
    for ent in doc.ents:
        label = _SPACY_LABEL_MAP.get(ent.label_)
        if label is None:
            continue
        spans.append({
            "span_text": ent.text.strip(),
            "span_label": label,
            "source": "spacy",
        })
    return spans


# ---------------------------------------------------------------------------
# GliNER extraction
# ---------------------------------------------------------------------------

_GLINER_EXTRACTOR = None


def _get_gliner():
    global _GLINER_EXTRACTOR
    if _GLINER_EXTRACTOR is None:
        from jarvis.contacts.candidate_extractor import CandidateExtractor
        _GLINER_EXTRACTOR = CandidateExtractor(label_profile="balanced")
        _GLINER_EXTRACTOR._load_model()
    return _GLINER_EXTRACTOR


def extract_gliner(
    text: str,
    message_id: int = 0,
    is_from_me: bool | None = None,
) -> list[dict]:
    """Extract entities using GliNER."""
    extractor = _get_gliner()
    candidates = extractor.extract_candidates(
        text=text,
        message_id=message_id,
        is_from_me=is_from_me,
        apply_label_thresholds=True,
    )
    return [
        {
            "span_text": c.span_text,
            "span_label": c.span_label,
            "gliner_score": c.gliner_score,
            "source": "gliner",
        }
        for c in candidates
    ]


# ---------------------------------------------------------------------------
# Ensemble: union + dedup
# ---------------------------------------------------------------------------


def _normalize_text(t: str) -> str:
    return t.lower().strip().rstrip(".,!?")


def merge_spans(
    gliner_spans: list[dict],
    spacy_spans: list[dict],
) -> list[dict]:
    """Union GliNER and spaCy predictions, deduplicating overlapping spans."""
    merged = list(gliner_spans)
    existing = set()
    for s in merged:
        existing.add((_normalize_text(s["span_text"]), s["span_label"]))

    for s in spacy_spans:
        if s["span_label"] not in ("person_name", "place", "org"):
            continue
        key = (_normalize_text(s["span_text"]), s["span_label"])
        if key in existing:
            continue
        norm = _normalize_text(s["span_text"])
        overlap = False
        for ex_text, ex_label in existing:
            if ex_label == s["span_label"] and (norm in ex_text or ex_text in norm):
                overlap = True
                break
        if not overlap:
            existing.add(key)
            merged.append(s)

    return merged


# ---------------------------------------------------------------------------
# LLM-based validation filter
# ---------------------------------------------------------------------------

_LLM_LOADER = None

FILTER_SYSTEM_PROMPT = """\
You are a personal fact validator. Given a chat message and a list of \
candidate entities extracted from it, decide which ones represent \
LASTING personal facts about the people in the conversation.

KEEP entities that reveal:
- Ongoing relationships (family members, close friends by name)
- Lasting preferences (foods they love/hate, hobbies, activities)
- Stable life facts (where they live, work, go to school)
- Health conditions, allergies
- Job roles, employers, schools

REMOVE entities that are:
- Temporary/one-time mentions (ordering food, scheduling, passing references)
- Generic words that aren't real entities (pronouns, common verbs, filler)
- Part of a brand/company name when the person name was extracted (e.g. "Ryan" from "Ryanair")
- Places from itineraries/restaurant lists (not where someone lives/is from)
- Celebrities or public figures (not personal contacts)

For each candidate, output ONLY the ones you keep. Return JSON:
{"kept": [{"text": "...", "label": "..."}]}

If none should be kept, return: {"kept": []}"""

FILTER_FEW_SHOT = [
    # Positive: lasting facts
    (
        'Message: "my brother bakes and I just eat whatever he makes"\n'
        'Candidates: [{"text": "brother", "label": "family_member"}, '
        '{"text": "bakes", "label": "activity"}, '
        '{"text": "whatever he makes", "label": "food_item"}]',
        '{"kept": [{"text": "brother", "label": "family_member"}, '
        '{"text": "bakes", "label": "activity"}]}'
    ),
    # Negative: transient family mention
    (
        'Message: "Yeah that\'s fine I\'ll leave as soon as my mom gets home at 4"\n'
        'Candidates: [{"text": "mom", "label": "family_member"}]',
        '{"kept": []}'
    ),
    # Mixed: org is real, person is part of brand
    (
        'Message: "Yea Ro remember how surprised you were with Ryan air flight prices"\n'
        'Candidates: [{"text": "Ro", "label": "person_name"}, '
        '{"text": "Ryan", "label": "person_name"}, '
        '{"text": "Ryan air", "label": "org"}]',
        '{"kept": [{"text": "Ro", "label": "person_name"}]}'
    ),
    # Preference context
    (
        'Message: "you love bagels so much hahaha"\n'
        'Candidates: [{"text": "bagels", "label": "food_item"}]',
        '{"kept": [{"text": "bagels", "label": "food_item"}]}'
    ),
    # List/itinerary - no personal context
    (
        'Message: "Glazed \\nApple fritter \\nOld fashioned glazed\\nMaple"\n'
        'Candidates: [{"text": "Apple", "label": "org"}, '
        '{"text": "Maple", "label": "place"}]',
        '{"kept": []}'
    ),
    # Empty message
    (
        'Message: "helloooo"\n'
        'Candidates: []',
        '{"kept": []}'
    ),
]

FILTER_USER_TEMPLATE = 'Message: "{message}"\nCandidates: {candidates}'


def _load_llm(model_id: str = "lfm-1.2b"):
    global _LLM_LOADER
    if _LLM_LOADER is None:
        from models.loader import MLXModelLoader, ModelConfig
        config = ModelConfig(model_id=model_id)
        config.memory_buffer_multiplier = 0.0
        loader = MLXModelLoader(config)
        loader.load()
        _LLM_LOADER = loader
    return _LLM_LOADER


def _strip_emojis(text: str) -> str:
    """Strip emoji characters that confuse the model."""
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
        "\U00002600-\U000026FF\U0000FE00-\U0000FE0F\U0000200D]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text).strip()


def _parse_filter_response(raw: str) -> list[dict]:
    """Parse LLM filter response, robustly handling malformed JSON."""
    raw = raw.strip()

    # Strip markdown code fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    # Try direct parse
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "kept" in obj:
            return obj["kept"]
        if isinstance(obj, list):
            return obj
        return []
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the output
    match = re.search(r'\{[^{}]*"kept"\s*:\s*\[.*?\]\s*\}', raw, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            return obj.get("kept", [])
        except json.JSONDecodeError:
            pass

    # Try to find just the array
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group())
            if isinstance(arr, list):
                return arr
        except json.JSONDecodeError:
            pass

    return []


def llm_filter_candidates(
    loader,
    message_text: str,
    candidates: list[dict],
) -> list[dict]:
    """Use the LLM to filter candidates to only lasting personal facts.

    Args:
        loader: MLXModelLoader instance
        message_text: Original message text
        candidates: List of candidate span dicts from GliNER/spaCy

    Returns:
        Filtered list of span dicts the LLM deemed valid
    """
    if not candidates:
        return []

    clean_text = _strip_emojis(message_text)
    if not clean_text:
        return []

    # Build candidate list for the prompt
    cand_list = [{"text": c["span_text"], "label": c["span_label"]} for c in candidates]
    cand_json = json.dumps(cand_list)

    # Build multi-turn conversation
    messages = [{"role": "system", "content": FILTER_SYSTEM_PROMPT}]
    for user_msg, assistant_resp in FILTER_FEW_SHOT:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_resp})

    user_prompt = FILTER_USER_TEMPLATE.format(message=clean_text, candidates=cand_json)
    messages.append({"role": "user", "content": user_prompt})

    formatted = loader._tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Scale tokens based on candidate count
    max_tok = min(50 + len(candidates) * 30, 400)

    result = loader.generate_sync(
        formatted,
        max_tokens=max_tok,
        temperature=0.0,
        top_p=0.1,
        repetition_penalty=1.0,
        pre_formatted=True,
    )

    kept = _parse_filter_response(result.text)

    # Map kept items back to original candidates (preserving source metadata)
    kept_keys = set()
    for k in kept:
        if isinstance(k, dict) and "text" in k:
            kept_keys.add((_normalize_text(k["text"]), k.get("label", "")))

    filtered = []
    for c in candidates:
        key = (_normalize_text(c["span_text"]), c["span_label"])
        if key in kept_keys:
            filtered.append(c)

    return filtered


# ---------------------------------------------------------------------------
# Lightweight pre-filter (removes obvious junk before LLM call)
# ---------------------------------------------------------------------------

# iMessage reaction prefix
_REACTION_RE = re.compile(
    r'^(Loved|Liked|Laughed at|Emphasized|Disliked)\s+\u201c'
)


def pre_filter(spans: list[dict], message_text: str) -> list[dict]:
    """Fast pre-filter to remove obvious non-entities before LLM call.

    Only removes things that are unambiguously wrong (saves LLM tokens).
    """
    if _REACTION_RE.match(message_text):
        return []

    msg_lower = message_text.lower()
    msg_len = len(message_text)
    result = []

    for span in spans:
        text = span["span_text"].strip()
        text_lower = text.lower()

        # Skip empty/single-char
        if len(text) < 2:
            continue

        # Skip spans longer than message (hallucinated)
        if len(text) > msg_len * 0.8 and msg_len > 10:
            continue

        # Skip if span not in message at all
        if text_lower not in msg_lower:
            words = text_lower.split()
            matching = [w for w in words if w in msg_lower and len(w) > 2]
            if len(matching) < max(1, len(words) * 0.5):
                continue

        # Skip English stop words (pronouns, determiners) - universal
        stop_words = {
            "i", "me", "my", "you", "your", "he", "him", "his", "she", "her",
            "it", "its", "we", "us", "our", "they", "them", "their",
            "the", "a", "an", "this", "that", "these", "those",
            "ok", "okay", "yeah", "yes", "no", "nah", "lol", "haha", "omg",
        }
        if text_lower in stop_words:
            continue

        result.append(span)

    # Deduplicate
    seen = set()
    deduped = []
    for s in result:
        key = (_normalize_text(s["span_text"]), s["span_label"])
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    return deduped


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_combined_extraction(
    gold_records: list[dict],
    *,
    use_gliner: bool = True,
    use_spacy: bool = True,
    use_llm: bool = True,
    model_id: str = "lfm-1.2b",
    verbose: bool = False,
) -> dict[str, list[dict]]:
    """Run the combined extraction pipeline on all gold records."""
    predictions: dict[str, list[dict]] = {}
    t0 = time.time()
    total = len(gold_records)

    loader = None
    if use_llm:
        print("Loading LLM for validation...", flush=True)
        loader = _load_llm(model_id)
        print("LLM loaded.", flush=True)

    for i, rec in enumerate(gold_records):
        if (i + 1) % 10 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i + 1}/{total}] {elapsed:.1f}s elapsed, ETA {eta:.0f}s",
                flush=True,
            )

        sid = rec["sample_id"]
        text = rec["message_text"]

        # Step 1: Extract candidates from each system
        gliner_spans = []
        spacy_spans = []

        if use_gliner:
            gliner_spans = extract_gliner(
                text,
                message_id=rec.get("message_id", 0),
                is_from_me=rec.get("is_from_me"),
            )

        if use_spacy:
            spacy_spans = extract_spacy(text)

        # Step 2: Merge
        merged = merge_spans(gliner_spans, spacy_spans)

        # Step 3: Pre-filter obvious junk
        merged = pre_filter(merged, text)

        # Step 4: LLM validation
        if use_llm and loader and merged:
            filtered = llm_filter_candidates(loader, text, merged)
        else:
            filtered = merged

        predictions[sid] = filtered

        if verbose:
            gold = rec.get("expected_candidates") or []
            gold_str = ", ".join(f"{g['span_text']}({g['span_label']})" for g in gold)
            merged_str = ", ".join(
                f"{p['span_text']}({p['span_label']},{p.get('source','?')})" for p in merged
            )
            filt_str = ", ".join(
                f"{p['span_text']}({p['span_label']})" for p in filtered
            )
            if gold or merged:
                print(f"    {sid}:", flush=True)
                if gold:
                    print(f"      gold: [{gold_str}]", flush=True)
                if merged:
                    print(f"      cand: [{merged_str}]", flush=True)
                if filtered:
                    print(f"      kept: [{filt_str}]", flush=True)

    elapsed = time.time() - t0
    total_preds = sum(len(v) for v in predictions.values())
    print(
        f"\nExtraction complete: {total_preds} predictions "
        f"in {elapsed:.1f}s ({elapsed / total * 1000:.1f}ms/msg)",
        flush=True,
    )
    return predictions


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def print_report(metrics: dict) -> None:
    """Print evaluation report."""
    ov = metrics["overall"]
    print("\n" + "=" * 60, flush=True)
    print("Combined Extraction Evaluation", flush=True)
    print("=" * 60, flush=True)
    print(
        f"\nOverall:  P={ov['precision']:.3f}  R={ov['recall']:.3f}  "
        f"F1={ov['f1']:.3f}  (TP={ov['tp']} FP={ov['fp']} FN={ov['fn']})",
        flush=True,
    )

    print(
        f"\n{'Label':<20} {'P':>6} {'R':>6} {'F1':>6} "
        f"{'TP':>4} {'FP':>4} {'FN':>4} {'Sup':>5}",
        flush=True,
    )
    print("-" * 60, flush=True)
    for label, m in sorted(metrics["per_label"].items(), key=lambda x: -x[1]["support"]):
        print(
            f"{label:<20} {m['precision']:>6.3f} {m['recall']:>6.3f} "
            f"{m['f1']:>6.3f} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['support']:>5}",
            flush=True,
        )

    print(
        f"\n{'Slice':<20} {'P':>6} {'R':>6} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4}",
        flush=True,
    )
    print("-" * 55, flush=True)
    for slc, m in sorted(metrics["per_slice"].items()):
        print(
            f"{slc:<20} {m['precision']:>6.3f} {m['recall']:>6.3f} "
            f"{m['f1']:>6.3f} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4}",
            flush=True,
        )

    errors = metrics.get("errors", [])
    fps = [e for e in errors if e["type"] == "fp"]
    fns = [e for e in errors if e["type"] == "fn"]

    if fps:
        print(f"\nFalse Positives ({len(fps)} total):", flush=True)
        fp_by_label: dict[str, list] = defaultdict(list)
        for e in fps:
            fp_by_label[e.get("pred_label", "?")].append(e)
        for label, examples in sorted(fp_by_label.items(), key=lambda x: -len(x[1])):
            print(f"  {label} ({len(examples)} FPs):", flush=True)
            for e in examples[:5]:
                src = e.get("source", "?")
                print(
                    f"    [{src}] \"{e.get('pred_span', '')}\" "
                    f"in \"{e['message_text'][:70]}...\"",
                    flush=True,
                )

    if fns:
        print(f"\nFalse Negatives ({len(fns)} total):", flush=True)
        fn_by_label: dict[str, list] = defaultdict(list)
        for e in fns:
            fn_by_label[e["gold_label"]].append(e)
        for label, examples in sorted(fn_by_label.items(), key=lambda x: -len(x[1])):
            print(f"  {label} ({len(examples)} FNs):", flush=True)
            for e in examples[:5]:
                print(
                    f"    missed \"{e['gold_span']}\" "
                    f"in \"{e['message_text'][:70]}...\"",
                    flush=True,
                )

    print("\n" + "=" * 60, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("eval_combined.log", mode="w"),
        ],
    )

    parser = argparse.ArgumentParser(description="Combined extraction evaluation")
    parser.add_argument("--gold", type=Path, default=GOLD_PATH, help="Path to gold set JSON")
    parser.add_argument("--spacy-only", action="store_true", help="Skip GliNER, use only spaCy")
    parser.add_argument("--gliner-only", action="store_true", help="Skip spaCy, use only GliNER")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM filter (raw output)")
    parser.add_argument("--model", type=str, default="lfm-1.2b", help="LLM model ID")
    parser.add_argument("--verbose", action="store_true", help="Print per-message predictions")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N records")
    parser.add_argument(
        "--allow-unstable-stack", action="store_true",
        help="Allow running with incompatible transformers version",
    )
    args = parser.parse_args()

    if not args.gold.exists():
        log.error(f"Gold set not found: {args.gold}")
        sys.exit(1)

    print(f"Loading gold set from {args.gold}...", flush=True)
    with open(args.gold) as f:
        gold_records = json.load(f)

    if args.limit:
        gold_records = gold_records[:args.limit]

    print(f"Loaded {len(gold_records)} records", flush=True)
    total_gold = sum(len(r.get("expected_candidates") or []) for r in gold_records)
    print(f"Total gold entities: {total_gold}", flush=True)

    use_gliner = not args.spacy_only
    use_spacy = not args.gliner_only
    use_llm = not args.no_llm

    if use_gliner and not args.allow_unstable_stack:
        enforce_runtime_stack(False)

    config_str = []
    if use_gliner:
        config_str.append("GliNER")
    if use_spacy:
        config_str.append("spaCy")
    if use_llm:
        config_str.append(f"LLM-filter({args.model})")
    else:
        config_str.append("no-filter")
    print(f"Config: {' + '.join(config_str)}", flush=True)

    print("\nRunning combined extraction...", flush=True)

    predictions = run_combined_extraction(
        gold_records,
        use_gliner=use_gliner,
        use_spacy=use_spacy,
        use_llm=use_llm,
        model_id=args.model,
        verbose=args.verbose,
    )

    metrics = compute_metrics(gold_records, predictions)
    print_report(metrics)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "gold_path": str(args.gold),
        "num_records": len(gold_records),
        "total_gold_entities": total_gold,
        "config": config_str,
        "overall": metrics["overall"],
        "per_label": metrics["per_label"],
        "per_slice": metrics["per_slice"],
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nMetrics saved to {METRICS_PATH}", flush=True)

    with open(ERRORS_PATH, "w") as f:
        json.dump(metrics["errors"], f, indent=2)
    print(f"Errors saved to {ERRORS_PATH}", flush=True)


if __name__ == "__main__":
    main()
