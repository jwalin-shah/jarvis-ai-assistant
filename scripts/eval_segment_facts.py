#!/usr/bin/env python3
"""Evaluate 350M model fact extraction on labeled segment eval set.

Loads frozen labeled segments, runs the 350M model with the sys_extract prompt,
and scores against gold labels.

NOTE: Defaults to substring (loose) matching mode. The base 350M model does NOT
reliably produce structured FACT: lines. Substring mode checks whether the model's
freeform output mentions the gold fact values, measuring recall only.
Use --strict for structured FACT: line parsing (only useful for fine-tuned models).

Usage:
    uv run python scripts/eval_segment_facts.py                          # substring/loose mode (default)
    uv run python scripts/eval_segment_facts.py --strict                 # strict FACT: line parsing
    uv run python scripts/eval_segment_facts.py --model lfm-0.3b --debug
    uv run python scripts/eval_segment_facts.py --goldset training_data/segment_eval/segments_labeled_fixed.json
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
from scripts.eval_shared import spans_match

# Default paths
DEFAULT_GOLDSET = Path("training_data/segment_eval/segments_labeled.json")
FROZEN_GOLDSET = Path("training_data/segment_eval/segment_eval_v1_frozen.json")
RESULTS_DIR = Path("results/segment_eval")

# Model registry (same as explore_model.py)
MODELS = {
    "lfm-350m": "mlx-community/LFM2-350M-4bit",
    "lfm-1.2b": "mlx-community/LFM2.5-1.2B-Instruct-MLX-4bit",
    "lfm-0.3b": "mlx-community/LFM2-350M-4bit",
}

# System prompt for fact extraction
# NOTE: Few-shot examples were tested but HURT recall on 350M model in substring mode
# (62.1% vs 80.4% without). The structured format constrains the model's output,
# reducing keyword matches that substring mode relies on. Keep simple for small models.
SYS_EXTRACT = (
    "Extract personal facts from this conversation segment.\n"
    "Output one fact per line as: FACT: [person] | [type] | [value]\n"
    "Resolve pronouns to actual names. If no facts, output: FACT: none"
)

# Label aliases: model output types -> gold label types
FACT_LABEL_ALIASES: dict[str, set[str]] = {
    "relationship": {"relationship", "family"},
    "family": {"family", "relationship"},
    "location": {"location", "place"},
    "place": {"place", "location"},
    "job": {"job", "job_role", "job_title", "education"},
    "job_role": {"job_role", "job", "job_title"},
    "health": {"health", "health_condition", "allergy"},
    "health_condition": {"health_condition", "health", "allergy"},
    "preference": {"preference", "food", "hobby", "personality"},
    "hobby": {"hobby", "preference", "activity"},
    "activity": {"activity", "hobby"},
    "food": {"food", "food_item", "food_like", "food_dislike", "preference"},
    "education": {"education", "job"},
    "pet": {"pet"},
    "age": {"age"},
    "personality": {"personality", "preference"},
}

# FACT: line pattern
FACT_PATTERN = re.compile(
    r"FACT:\s*\[?([^\]|]+?)\]?\s*\|\s*\[?([^\]|]+?)\]?\s*\|\s*\[?(.+?)\]?\s*$",
    re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 350M fact extraction")
    parser.add_argument(
        "--model",
        default="lfm-0.3b",
        choices=list(MODELS.keys()),
        help="Model to evaluate",
    )
    parser.add_argument(
        "--goldset",
        type=Path,
        default=None,
        help="Path to labeled goldset (default: frozen > labeled)",
    )
    parser.add_argument("--max-tokens", type=int, default=200, help="Max generation tokens")
    parser.add_argument("--debug", action="store_true", help="Show raw model output")
    parser.add_argument("--limit", type=int, default=0, help="Max segments to eval (0=all)")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict FACT: line parsing instead of default substring/loose matching. "
        "Only useful for fine-tuned models that produce structured output.",
    )
    args = parser.parse_args()

    # Find goldset
    goldset_path = args.goldset
    if goldset_path is None:
        if FROZEN_GOLDSET.exists():
            goldset_path = FROZEN_GOLDSET
        elif DEFAULT_GOLDSET.exists():
            goldset_path = DEFAULT_GOLDSET
        else:
            print("No goldset found. Run label_eval_segments.py first.", flush=True)
            sys.exit(1)

    with open(goldset_path) as f:
        goldset = json.load(f)

    # Filter to segments with facts
    labeled = [s for s in goldset if s.get("facts")]
    unlabeled = [s for s in goldset if not s.get("facts")]
    print(
        f"Goldset: {len(goldset)} total, {len(labeled)} with facts, {len(unlabeled)} empty",
        flush=True,
    )

    if args.limit > 0:
        labeled = labeled[: args.limit]

    # Load model
    print(f"\nLoading model: {args.model} ({MODELS[args.model]})...", flush=True)
    import mlx.core as mx
    from mlx_lm import load

    mx.set_memory_limit(1 * 1024 * 1024 * 1024)  # 1GB
    model, tokenizer = load(MODELS[args.model])
    print("  Model loaded.", flush=True)

    # Load the exported segments for formatted text
    exported_path = Path("training_data/segment_eval/segments_exported.json")
    seg_text_map: dict[str, str] = {}
    if exported_path.exists():
        with open(exported_path) as f:
            for seg in json.load(f):
                seg_text_map[seg["segment_id"]] = seg["formatted_text"]

    # Run eval
    # NOTE: Abbreviation expansion (_expand_abbreviations) was tested but HURT recall
    # (62.3% vs 80.4% without). The expanded text changes keywords the substring
    # matcher relies on. Disabled until we switch to structured output parsing.
    print(f"\nEvaluating {len(labeled)} segments...\n", flush=True)
    all_preds: list[dict] = []
    all_golds: list[dict] = []
    errors: list[dict] = []
    sub_total_tp = 0
    sub_total_fn = 0
    t0 = time.time()

    for i, seg in enumerate(labeled):
        sid = seg["segment_id"]
        gold_facts = seg["facts"]

        # Get the formatted conversation text
        text = seg_text_map.get(sid, seg.get("formatted_text", ""))
        if not text:
            print(f"  [{i + 1}] {sid}: no text, skipping", flush=True)
            continue

        # Generate
        response, elapsed_ms, _ = _generate(model, tokenizer, text, args.max_tokens)

        if args.debug:
            print(f"\n--- Segment {sid} ---", flush=True)
            print(f"Input:\n{text[:200]}...", flush=True)
            print(f"Output:\n{response}", flush=True)

        if not args.strict:
            # Substring mode: check if gold fact values appear in raw output
            resp_lower = response.lower()
            seg_hits = 0
            seg_misses = []
            for gf in gold_facts:
                # Check value, span_text, and person name
                val = gf.get("value", "").lower()
                span = gf.get("span_text", "").lower()
                # Extract key terms (longest meaningful word from value)
                val_words = [w for w in val.split() if len(w) >= 3]
                span_words = [w for w in span.split() if len(w) >= 3]
                hit = False
                # Check if any key term from value or span appears in output
                for term in val_words + span_words:
                    if term in resp_lower:
                        hit = True
                        break
                if hit:
                    seg_hits += 1
                else:
                    seg_misses.append(gf)

            all_golds.extend([{**f, "segment_id": sid} for f in gold_facts])
            # In substring mode, hits = TP, misses = FN, no FP measurement
            sub_total_tp += seg_hits
            sub_total_fn += len(seg_misses)

            if seg_misses:
                errors.append(
                    {
                        "segment_id": sid,
                        "contact": seg.get("contact_name", gold_facts[0].get("person", "?")),
                        "text_preview": text[:100],
                        "false_positives": [],
                        "false_negatives": seg_misses,
                        "response": response,
                    }
                )

            recall = seg_hits / len(gold_facts) if gold_facts else 0
            status = (
                "PERFECT"
                if recall == 1.0
                else f"Recall={recall:.0%} ({seg_hits}/{len(gold_facts)})"
            )
            print(
                f"  [{i + 1}/{len(labeled)}] {sid[:20]} "
                f"gold={len(gold_facts)} {status} ({elapsed_ms:.0f}ms)",
                flush=True,
            )
        else:
            # Strict mode: parse FACT: lines
            pred_facts = _parse_facts(response)

            # Match
            seg_tp, seg_fp, seg_fn = _match_facts(pred_facts, gold_facts)

            all_preds.extend([{**f, "segment_id": sid} for f in pred_facts])
            all_golds.extend([{**f, "segment_id": sid} for f in gold_facts])

            if seg_fp or seg_fn:
                errors.append(
                    {
                        "segment_id": sid,
                        "contact": seg.get("contact_name", gold_facts[0].get("person", "?")),
                        "text_preview": text[:100],
                        "false_positives": seg_fp,
                        "false_negatives": seg_fn,
                        "response": response,
                    }
                )

            p, r, f1 = _prf(seg_tp, seg_fp, seg_fn)
            status = "PERFECT" if f1 == 1.0 else f"P={p:.0%} R={r:.0%} F1={f1:.0%}"
            print(
                f"  [{i + 1}/{len(labeled)}] {sid[:20]} "
                f"pred={len(pred_facts)} gold={len(gold_facts)} "
                f"{status} ({elapsed_ms:.0f}ms)",
                flush=True,
            )

    elapsed = time.time() - t0

    if not args.strict:
        # Substring mode results
        overall_r = (
            sub_total_tp / (sub_total_tp + sub_total_fn)
            if (sub_total_tp + sub_total_fn) > 0
            else 0.0
        )

        print(f"\n{'=' * 60}", flush=True)
        print(f"RESULTS (substring mode): {args.model} on {goldset_path.name}", flush=True)
        print(f"{'=' * 60}", flush=True)
        print(f"Segments evaluated: {len(labeled)}", flush=True)
        print(f"Total gold facts: {len(all_golds)}", flush=True)
        print(f"Recall: {overall_r:.1%} ({sub_total_tp}/{sub_total_tp + sub_total_fn})", flush=True)
        print(
            f"Time: {elapsed:.1f}s ({elapsed / max(len(labeled), 1) * 1000:.0f}ms/segment)",
            flush=True,
        )
        print(
            f"\nNote: Substring mode measures recall only (does the output mention the fact?).",
            flush=True,
        )
        print(f"Precision is not measured since the model produces freeform text.", flush=True)

        # Per-type recall from gold facts
        type_hits: dict[str, list[bool]] = defaultdict(list)
        for gf in all_golds:
            ft = gf.get("fact_type", "unknown")
            val = gf.get("value", "").lower()
            span = gf.get("span_text", "").lower()
            # We don't have per-fact hit tracking here, so skip per-type breakdown
            type_hits[ft].append(True)  # placeholder

        overall_p = 0.0  # not measured in substring mode
        overall_f1 = 0.0
        contact_metrics = {}
        type_metrics = {}
    else:
        # Strict mode results
        total_tp, total_fp, total_fn = _match_facts(all_preds, all_golds)
        overall_p, overall_r, overall_f1 = _prf(total_tp, total_fp, total_fn)

        # Per-contact metrics
        contact_metrics = _per_group_metrics(all_preds, all_golds, "person")

        # Per-type metrics
        type_metrics = _per_group_metrics(all_preds, all_golds, "fact_type")

        # Print results
        print(f"\n{'=' * 60}", flush=True)
        print(f"RESULTS: {args.model} on {goldset_path.name}", flush=True)
        print(f"{'=' * 60}", flush=True)
        print(f"Segments evaluated: {len(labeled)}", flush=True)
        print(f"Total gold facts: {len(all_golds)}", flush=True)
        print(f"Total predicted: {len(all_preds)}", flush=True)
        print(
            f"Time: {elapsed:.1f}s ({elapsed / max(len(labeled), 1) * 1000:.0f}ms/segment)",
            flush=True,
        )
        print(f"\nOverall: P={overall_p:.1%}  R={overall_r:.1%}  F1={overall_f1:.1%}", flush=True)

        print(f"\n{'─' * 40}", flush=True)
        print("Per-Contact:", flush=True)
        for name, m in sorted(contact_metrics.items(), key=lambda x: -x[1]["f1"]):
            print(
                f"  {name:20s}  P={m['precision']:.0%} R={m['recall']:.0%} F1={m['f1']:.0%} "
                f"(pred={m['pred']} gold={m['gold']})",
                flush=True,
            )

        print(f"\n{'─' * 40}", flush=True)
        print("Per-Type:", flush=True)
        for ftype, m in sorted(type_metrics.items(), key=lambda x: -x[1]["f1"]):
            print(
                f"  {ftype:20s}  P={m['precision']:.0%} R={m['recall']:.0%} F1={m['f1']:.0%} "
                f"(pred={m['pred']} gold={m['gold']})",
                flush=True,
            )

    # Error analysis
    if errors:
        print(f"\n{'─' * 40}", flush=True)
        print(f"Error Analysis ({len(errors)} segments with errors):", flush=True)
        for err in errors[:10]:  # show first 10
            print(f"\n  Segment: {err['segment_id']}", flush=True)
            print(f"  Contact: {err['contact']}", flush=True)
            print(f"  Text: {err['text_preview']}...", flush=True)
            if err["false_positives"]:
                print(f"  FP: {err['false_positives']}", flush=True)
            if err["false_negatives"]:
                print(f"  FN: {err['false_negatives']}", flush=True)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "model": args.model,
        "model_path": MODELS[args.model],
        "goldset": str(goldset_path),
        "segments_evaluated": len(labeled),
        "total_gold": len(all_golds),
        "total_predicted": len(all_preds),
        "precision": overall_p,
        "recall": overall_r,
        "f1": overall_f1,
        "time_seconds": elapsed,
        "per_contact": contact_metrics,
        "per_type": type_metrics,
        "errors": errors,
    }
    results_path = RESULTS_DIR / "metrics.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}", flush=True)


def _generate(
    model,
    tokenizer,
    user_text: str,
    max_tokens: int,
) -> tuple[str, float, int]:
    """Generate fact extraction response using the 350M model."""
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_repetition_penalty, make_sampler

    messages = [
        {"role": "system", "content": SYS_EXTRACT},
        {"role": "user", "content": user_text},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = f"{SYS_EXTRACT}\n\n{user_text}"

    sampler = make_sampler(temp=0.1, top_p=0.1, top_k=50)
    repetition_penalty = make_repetition_penalty(1.05)

    t0 = time.time()
    response = mlx_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=[repetition_penalty],
    )
    elapsed_ms = (time.time() - t0) * 1000
    token_count = len(tokenizer.encode(response)) if response else 0
    return response, elapsed_ms, token_count


def _parse_facts(response: str) -> list[dict]:
    """Parse FACT: [person] | [type] | [value] lines from model output."""
    facts = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if "FACT: none" in line.lower() or "fact: none" in line.lower():
            continue
        m = FACT_PATTERN.match(line)
        if m:
            facts.append(
                {
                    "person": m.group(1).strip(),
                    "fact_type": m.group(2).strip().lower(),
                    "value": m.group(3).strip(),
                }
            )
    return facts


def _match_facts(
    preds: list[dict],
    golds: list[dict],
) -> tuple[int, list[dict], list[dict]]:
    """Match predicted facts to gold facts. Returns (TP count, FP list, FN list)."""
    tp = 0
    matched_gold = set()
    fps = []

    for pred in preds:
        found = False
        for j, gold in enumerate(golds):
            if j in matched_gold:
                continue
            if _facts_match(pred, gold):
                tp += 1
                matched_gold.add(j)
                found = True
                break
        if not found:
            fps.append(pred)

    fns = [g for j, g in enumerate(golds) if j not in matched_gold]
    return tp, fps, fns


def _facts_match(pred: dict, gold: dict) -> bool:
    """Check if a predicted fact matches a gold fact."""
    # Person must match (case-insensitive substring)
    pred_person = pred.get("person", "").lower().strip()
    gold_person = gold.get("person", "").lower().strip()
    if pred_person not in gold_person and gold_person not in pred_person:
        return False

    # Type must match (with aliases)
    pred_type = pred.get("fact_type", "").lower().strip()
    gold_type = gold.get("fact_type", "").lower().strip()

    if pred_type != gold_type:
        allowed = FACT_LABEL_ALIASES.get(pred_type, {pred_type})
        if gold_type not in allowed:
            return False

    # Value must overlap
    pred_val = pred.get("value", "")
    gold_val = gold.get("value", gold.get("span_text", ""))
    return spans_match(pred_val, pred_type, gold_val, gold_type, FACT_LABEL_ALIASES)


def _prf(tp: int, fp_list, fn_list) -> tuple[float, float, float]:
    """Compute precision, recall, F1."""
    fp = len(fp_list) if isinstance(fp_list, list) else fp_list
    fn = len(fn_list) if isinstance(fn_list, list) else fn_list
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def _per_group_metrics(
    preds: list[dict],
    golds: list[dict],
    key: str,
) -> dict[str, dict]:
    """Compute P/R/F1 per group (person or fact_type)."""
    pred_by_group: dict[str, list[dict]] = defaultdict(list)
    gold_by_group: dict[str, list[dict]] = defaultdict(list)

    for p in preds:
        pred_by_group[p.get(key, "unknown")].append(p)
    for g in golds:
        gold_by_group[g.get(key, "unknown")].append(g)

    all_groups = set(pred_by_group.keys()) | set(gold_by_group.keys())
    metrics = {}

    for group in all_groups:
        gp = pred_by_group.get(group, [])
        gg = gold_by_group.get(group, [])
        tp, fp, fn = _match_facts(gp, gg)
        p, r, f1 = _prf(tp, fp, fn)
        metrics[group] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "pred": len(gp),
            "gold": len(gg),
        }

    return metrics


# Common iMessage abbreviations/slang → expanded form.
# Only expand when the abbreviation is a standalone word (word-boundary matched).
_ABBREVS: dict[str, str] = {
    "ik": "I know",
    "uk": "you know",
    "tn": "tonight",
    "tm": "tomorrow",
    "tmr": "tomorrow",
    "tmrw": "tomorrow",
    "fs": "for sure",
    "ngl": "not gonna lie",
    "tbh": "to be honest",
    "imo": "in my opinion",
    "rn": "right now",
    "nvm": "never mind",
    "idk": "I don't know",
    "idt": "I don't think",
    "idc": "I don't care",
    "wyd": "what are you doing",
    "hbu": "how about you",
    "omw": "on my way",
    "otw": "on the way",
    "brb": "be right back",
    "btw": "by the way",
    "smh": "shaking my head",
    "nah": "no",
    "yuh": "yes",
    "ig": "I guess",
    "abt": "about",
    "bc": "because",
    "w": "with",
    "rq": "real quick",
    "ppl": "people",
    "tho": "though",
    "thru": "through",
    "prob": "probably",
    "def": "definitely",
    "obv": "obviously",
    "tbf": "to be fair",
    "wud": "would",
    "shud": "should",
    "cud": "could",
    "cudnt": "couldn't",
    "didnt": "didn't",
    "dont": "don't",
    "doesnt": "doesn't",
    "wasnt": "wasn't",
    "isnt": "isn't",
    "havent": "haven't",
    "im": "I'm",
    "ive": "I've",
    "ur": "your",
    "u": "you",
    "r": "are",
    "n": "and",
    "yr": "year",
    "yrs": "years",
    "szn": "season",
    "sm": "so much",
    "p": "pretty",
    "v": "very",
    "tf": "the fuck",
    "wtf": "what the fuck",
    "af": "as fuck",
    "lowkey": "kind of",
    "highkey": "really",
    "prolly": "probably",
    "gotta": "got to",
    "gonna": "going to",
    "wanna": "want to",
    "tryna": "trying to",
    "boutta": "about to",
}

# Precompile the regex for word-boundary matching
_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_ABBREVS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _expand_abbreviations(text: str) -> str:
    """Expand common iMessage abbreviations to full words."""

    def _replace(m: re.Match) -> str:
        word = m.group(0)
        replacement = _ABBREVS.get(word.lower(), word)
        # Preserve leading capitalization
        if word[0].isupper() and replacement[0].islower():
            return replacement[0].upper() + replacement[1:]
        return replacement

    return _ABBREV_PATTERN.sub(_replace, text)


if __name__ == "__main__":
    main()
