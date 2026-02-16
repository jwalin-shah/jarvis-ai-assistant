#!/usr/bin/env python3
"""Single experiment subprocess executor for the Extraction Lab.

Runs one extraction experiment and writes structured JSON results.
Designed to be called as a subprocess for memory isolation.

Supports extractors: spacy, llm, rules, hybrid
(GLiNER is handled by gliner_extract_standalone.py in compat venv)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

# Setup path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from eval_shared import DEFAULT_LABEL_ALIASES, spans_match  # noqa: E402


def load_goldset(goldset_dir: str, split: str) -> list[dict]:
    """Load goldset data for the given split."""
    goldset_dir = Path(goldset_dir)

    if split == "train+dev":
        with open(goldset_dir / "train.json") as f:
            train = json.load(f)
        with open(goldset_dir / "dev.json") as f:
            dev = json.load(f)
        return train + dev
    else:
        with open(goldset_dir / f"{split}.json") as f:
            return json.load(f)


def cross_validate(
    data: list[dict],
    extractor_fn,
    n_folds: int = 5,
) -> dict:
    """Run cross-validation and return aggregated metrics."""
    random.seed(42)
    indices = list(range(len(data)))
    random.shuffle(indices)

    fold_size = len(indices) // n_folds
    all_tp, all_fp, all_fn = 0, 0, 0
    per_label_tp: dict[str, int] = defaultdict(int)
    per_label_fp: dict[str, int] = defaultdict(int)
    per_label_fn: dict[str, int] = defaultdict(int)
    fp_examples: list[dict] = []
    fn_examples: list[dict] = []

    for fold in range(n_folds):
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < n_folds - 1 else len(indices)
        val_indices = set(indices[start_idx:end_idx])
        train_indices = [i for i in indices if i not in val_indices]

        val_data = [data[i] for i in sorted(val_indices)]
        train_data = [data[i] for i in train_indices]

        print(
            f"  Fold {fold+1}/{n_folds}: train={len(train_data)},"
            f" val={len(val_data)}",
            flush=True,
        )

        fold_result = evaluate_fold(val_data, extractor_fn, train_data)
        all_tp += fold_result["total_tp"]
        all_fp += fold_result["total_fp"]
        all_fn += fold_result["total_fn"]
        for label, counts in fold_result["per_label"].items():
            per_label_tp[label] += counts["tp"]
            per_label_fp[label] += counts["fp"]
            per_label_fn[label] += counts["fn"]
        fp_examples.extend(fold_result.get("fp_examples", []))
        fn_examples.extend(fold_result.get("fn_examples", []))

    return build_metrics(
        per_label_tp, per_label_fp, per_label_fn,
        all_tp, all_fp, all_fn,
        len(data), fp_examples, fn_examples,
    )


def evaluate_fold(
    val_data: list[dict],
    extractor_fn,
    train_data: list[dict] | None = None,
) -> dict:
    """Evaluate extractor on a single fold/split."""
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn_count: dict[str, int] = defaultdict(int)
    fp_examples: list[dict] = []
    fn_examples: list[dict] = []

    for msg in val_data:
        preds = extractor_fn(msg, train_data)

        # Normalize labels
        from eval_extraction import normalize_label
        for p in preds:
            p["span_label"] = normalize_label(p["span_label"])

        gold = msg.get("expected_candidates", [])
        for g in gold:
            g["span_label"] = normalize_label(g.get("span_label", ""))

        # Match preds to gold
        gold_matched = [False] * len(gold)
        pred_matched = [False] * len(preds)

        for pi, pred in enumerate(preds):
            for gi, g in enumerate(gold):
                if gold_matched[gi]:
                    continue
                if spans_match(
                    pred["span_text"], pred["span_label"],
                    g["span_text"], g["span_label"],
                    label_aliases=DEFAULT_LABEL_ALIASES,
                ):
                    gold_matched[gi] = True
                    pred_matched[pi] = True
                    tp[pred["span_label"]] += 1
                    break

        for pi, pred in enumerate(preds):
            if not pred_matched[pi]:
                fp[pred["span_label"]] += 1
                fp_examples.append({
                    "message_text": msg["message_text"],
                    "pred": pred,
                })

        for gi, g in enumerate(gold):
            if not gold_matched[gi]:
                fn_count[g["span_label"]] += 1
                fn_examples.append({
                    "message_text": msg["message_text"],
                    "gold": g,
                })

    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn_count.values())

    per_label = {}
    all_labels = sorted(set(tp) | set(fp) | set(fn_count))
    for label in all_labels:
        per_label[label] = {"tp": tp[label], "fp": fp[label], "fn": fn_count[label]}

    return {
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "per_label": per_label,
        "fp_examples": fp_examples[:20],
        "fn_examples": fn_examples[:20],
    }


def build_metrics(
    per_label_tp, per_label_fp, per_label_fn,
    total_tp, total_fp, total_fn,
    num_messages, fp_examples, fn_examples,
) -> dict:
    """Build structured metrics dict."""
    per_label = {}
    all_labels = sorted(set(per_label_tp) | set(per_label_fp) | set(per_label_fn))

    for label in all_labels:
        t = per_label_tp[label]
        f = per_label_fp[label]
        n = per_label_fn[label]
        precision = t / (t + f) if (t + f) > 0 else 0.0
        recall = t / (t + n) if (t + n) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        per_label[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "tp": t, "fp": f, "fn": n,
            "support": t + n,
        }

    micro_p = (
        total_tp / (total_tp + total_fp)
        if (total_tp + total_fp) > 0 else 0.0
    )
    micro_r = (
        total_tp / (total_tp + total_fn)
        if (total_tp + total_fn) > 0 else 0.0
    )
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r)
        if (micro_p + micro_r) > 0 else 0.0
    )

    labels_with_support = [l for l in all_labels if per_label[l]["support"] > 0]
    macro_f1 = (
        sum(per_label[l]["f1"] for l in labels_with_support) / len(labels_with_support)
        if labels_with_support else 0.0
    )

    return {
        "num_messages": num_messages,
        "micro_f1": round(micro_f1, 3),
        "micro_precision": round(micro_p, 3),
        "micro_recall": round(micro_r, 3),
        "macro_f1": round(macro_f1, 3),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "per_label": per_label,
        "fp_examples": fp_examples[:30],
        "fn_examples": fn_examples[:30],
    }


# ---------------------------------------------------------------------------
# Extractor implementations
# ---------------------------------------------------------------------------

def make_spacy_extractor(params: dict):
    """Create spaCy extraction function."""
    model_name = params.get("model", "en_core_web_sm")
    import spacy
    nlp = spacy.load(model_name)

    from eval_extraction import trim_span

    spacy_to_ours = {
        "PERSON": "person_name",
        "ORG": "org",
        "GPE": "place",
        "LOC": "place",
        "FAC": "place",
    }

    def extract(msg: dict, train_data: list[dict] | None = None) -> list[dict]:
        doc = nlp(msg["message_text"])
        candidates = []
        seen: set[tuple[str, str]] = set()
        for ent in doc.ents:
            our_label = spacy_to_ours.get(ent.label_)
            if our_label is None:
                continue
            span_text = trim_span(ent.text)
            if len(span_text) < 2:
                continue
            key = (span_text.lower(), our_label)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"span_text": span_text, "span_label": our_label})
        return candidates

    return extract


def make_llm_extractor(params: dict):
    """Create LLM extraction function."""
    model_path = params.get("model", "LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit")
    n_fewshot = params.get("n_fewshot", 0)
    temperature = params.get("temperature", 0.1)
    max_tokens = params.get("max_tokens", 512)
    prompt_variant = params.get("prompt_variant", "minimal")

    # Resolve model path to local dir if shorthand given
    model_map = {
        "lfm-1.2b": "LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit",
        "lfm2-1.2b-extract": "models/lfm2-1.2b-extract-mlx-4bit",
        "lfm2-350m-extract": "models/lfm2-350m-extract-mlx-4bit",
    }
    model_path = model_map.get(model_path, model_path)

    print(f"  Loading LLM: {model_path}...", flush=True)
    from mlx_lm import generate as mlx_generate
    from mlx_lm import load
    model, tokenizer = load(model_path)
    print("  LLM loaded.", flush=True)

    import re

    from eval_extraction import normalize_label, trim_span

    def _build_system_prompt(few_shot: list[dict]) -> str:
        base = (
            "You are a personal fact extractor. "
            "Given an iMessage, extract lasting personal facts "
            "as structured spans.\n\n"
            "Labels: family_member, person_name, place, org, "
            "job_role, food_item, activity, health_condition\n\n"
            "Rules:\n"
            "- Only extract LASTING personal facts "
            "(not transient events)\n"
            "- Extract minimal spans "
            "(just the entity, not surrounding words)\n"
            "- Skip vague references (it, that, stuff)\n"
            "- Skip bot/spam messages\n"
            '- Output JSON array of '
            '{"span_text": "...", "span_label": "..."} '
            "or empty array []"
        )

        if prompt_variant == "cot":
            base += (
                "\n\nThink step by step: first identify "
                "potential entities, then check if each "
                "is a lasting fact."
            )
        elif prompt_variant == "negative":
            base += (
                '\n\nExamples of things NOT to extract: '
                '"my mom called" (transient event), '
                '"that place" (vague), "lol" (filler)'
            )

        if few_shot:
            base += "\n\n## Examples\n\n"
            for ex in few_shot[:n_fewshot or 5]:
                msg_text = ex.get("message_text", "")
                cands = ex.get("expected_candidates", [])
                base += f'Message: "{msg_text}"\nOutput: {json.dumps(cands)}\n\n'

        return base

    def _parse_response(response: str) -> list[dict]:
        response = response.strip()
        # Try direct JSON
        try:
            result = json.loads(response)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
        # Try extracting from code block
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group(1))
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass
        # Try finding array
        m = re.search(r"\[.*\]", response, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group(0))
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass
        return []

    def extract(msg: dict, train_data: list[dict] | None = None) -> list[dict]:
        # Build few-shot from train data
        few_shot = []
        if train_data and n_fewshot > 0:
            positive = [m for m in train_data if m.get("expected_candidates")]
            few_shot = positive[:n_fewshot]

        system_prompt = _build_system_prompt(few_shot)
        user_prompt = (
            f'Message: "{msg["message_text"]}"\n'
            f"Extract lasting personal facts as JSON array."
        )

        # Format for chat
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt_text = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"

        response = mlx_generate(
            model, tokenizer, prompt=prompt_text,
            max_tokens=max_tokens, temp=temperature, verbose=False,
        )

        raw_spans = _parse_response(response)
        validated = []
        for s in raw_spans:
            if not isinstance(s, dict):
                continue
            text = s.get("span_text", "").strip()
            label = s.get("span_label", "").strip()
            if text and label:
                validated.append({
                    "span_text": trim_span(text),
                    "span_label": normalize_label(label),
                })
        return validated

    return extract


def make_rules_extractor(params: dict):
    """Create rules-based extraction function (existing FactExtractor patterns)."""
    from jarvis.contacts.fact_extractor import FactExtractor
    extractor = FactExtractor()

    # Map Fact.category to span_label
    category_to_label = {
        "relationship": "family_member",
        "person": "person_name",
        "location": "place",
        "work": "org",
        "preference": "food_item",  # approximation; may be activity too
        "health": "health_condition",
    }

    def extract(msg: dict, train_data: list[dict] | None = None) -> list[dict]:
        text = msg["message_text"]
        # Wrap text in the dict format FactExtractor expects
        messages = [{"text": text, "id": msg.get("message_id", 0)}]
        facts = extractor.extract_facts(messages)

        candidates = []
        for fact in facts:
            # Convert Fact(category, subject, predicate) to span format
            span_text = fact.subject
            span_label = category_to_label.get(fact.category, fact.category)

            # Refine label from predicate if possible
            if fact.predicate in ("lives_in", "is_from", "moved_to"):
                span_label = "place"
            elif fact.predicate in ("works_at", "studies_at"):
                span_label = "org"
            elif fact.predicate in ("has_job",):
                span_label = "job_role"
            elif fact.predicate in ("likes_food", "dislikes_food"):
                span_label = "food_item"
            elif fact.predicate in ("has_hobby", "plays_sport"):
                span_label = "activity"
            elif fact.predicate in ("has_allergy", "has_condition"):
                span_label = "health_condition"
            elif fact.predicate in ("is_family_of", "has_sibling", "has_parent"):
                span_label = "family_member"
            elif fact.predicate in ("knows_person", "is_friend_of"):
                span_label = "person_name"

            if span_text and len(span_text) >= 2:
                candidates.append({
                    "span_text": span_text,
                    "span_label": span_label,
                })
        return candidates

    return extract


def _run_gliner_subprocess(
    messages: list[dict], params: dict, goldset_dir: str,
) -> dict[str, list[dict]]:
    """Run GLiNER as subprocess and return extractions keyed by sample_id."""
    import tempfile

    params.get("threshold", 0.35)

    # Write messages to temp file for GLiNER subprocess
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(messages, tmp)
        tmp_path = tmp.name

    # Build a minimal payload to run GLiNER extraction (not full eval)
    {
        "name": "_hybrid_gliner_sub",
        "extractor": "gliner",
        "params": params,
        "split": "dev",  # unused, we pass data directly
        "cv_folds": 0,
        "goldset_dir": goldset_dir,
        "output_path": tmp_path + ".out.json",
    }

    # We can't easily use the standalone script for just extraction,
    # so for hybrid we run GLiNER inline via the compat venv with a simpler call
    # For now, just skip GLiNER in hybrid and warn
    print(
        "  WARNING: GLiNER in hybrid requires compat venv."
        " Skipping GLiNER sub-extractor.",
        flush=True,
    )
    print("  Use extraction_lab.py to run GLiNER separately, then combine results.", flush=True)

    os.unlink(tmp_path)
    return {}


def make_hybrid_extractor(params: dict):
    """Create hybrid extraction function combining multiple extractors.

    NOTE: GLiNER sub-extractors are not supported in hybrid mode because GLiNER
    requires the compat venv. Run GLiNER separately and combine results manually,
    or use extraction_lab.py to orchestrate.
    """
    sub_configs = params.get("extractors", [])
    merge = params.get("merge", "union")  # union, intersection, voting
    vote_threshold = params.get("vote_threshold", 2)

    # Build sub-extractors (skip gliner with warning)
    sub_extractors = []
    for sub in sub_configs:
        if sub["extractor"] == "gliner":
            print("  WARNING: Skipping GLiNER in hybrid (needs compat venv). "
                  "Run GLiNER separately.", flush=True)
            continue
        ext = build_extractor(sub["extractor"], sub.get("params", {}))
        sub_extractors.append(ext)

    from eval_shared import jaccard_tokens

    def extract(msg: dict, train_data: list[dict] | None = None) -> list[dict]:
        all_results = [ext(msg, train_data) for ext in sub_extractors]

        if merge == "union":
            # Deduplicated union
            merged = []
            seen_keys: set[tuple[str, str]] = set()
            for results in all_results:
                for c in results:
                    key = (c["span_text"].lower(), c["span_label"])
                    if key not in seen_keys:
                        # Check fuzzy overlap
                        overlaps = False
                        for existing in merged:
                            if (
                                c["span_label"] == existing["span_label"]
                                and jaccard_tokens(c["span_text"], existing["span_text"]) > 0.5
                            ):
                                overlaps = True
                                break
                        if not overlaps:
                            seen_keys.add(key)
                            merged.append(c)
            return merged

        elif merge == "intersection":
            # Only keep spans found by all extractors
            if not all_results:
                return []
            base = all_results[0]
            result = []
            for c in base:
                found_in_all = True
                for other_results in all_results[1:]:
                    found = False
                    for oc in other_results:
                        if (
                            c["span_label"] == oc["span_label"]
                            and (
                                c["span_text"].lower() == oc["span_text"].lower()
                                or jaccard_tokens(c["span_text"], oc["span_text"]) > 0.5
                            )
                        ):
                            found = True
                            break
                    if not found:
                        found_in_all = False
                        break
                if found_in_all:
                    result.append(c)
            return result

        elif merge == "voting":
            # Keep spans with >= vote_threshold votes
            from collections import Counter
            span_votes: dict[tuple[str, str], int] = Counter()
            span_texts: dict[tuple[str, str], str] = {}
            for results in all_results:
                for c in results:
                    key = (c["span_text"].lower(), c["span_label"])
                    span_votes[key] += 1
                    span_texts[key] = c["span_text"]
            result = []
            for key, votes in span_votes.items():
                if votes >= vote_threshold:
                    result.append({"span_text": span_texts[key], "span_label": key[1]})
            return result

        return []

    return extract


def build_extractor(extractor_type: str, params: dict):
    """Build an extractor function from type and params."""
    builders = {
        "spacy": make_spacy_extractor,
        "llm": make_llm_extractor,
        "rules": make_rules_extractor,
        "hybrid": make_hybrid_extractor,
    }
    builder = builders.get(extractor_type)
    if builder is None:
        raise ValueError(
            f"Unknown extractor: {extractor_type}. "
            f"Available: {list(builders.keys())}"
        )
    return builder(params)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single extraction experiment")
    parser.add_argument("--payload", required=True, help="JSON experiment payload")
    args = parser.parse_args()

    payload = json.loads(args.payload)
    name = payload["name"]
    extractor_type = payload["extractor"]
    params = payload.get("params", {})
    split = payload.get("split", "dev")
    cv_folds = payload.get("cv_folds", 0)
    goldset_dir = payload["goldset_dir"]
    output_path = payload["output_path"]

    print(f"Experiment: {name} (extractor={extractor_type})", flush=True)
    start = time.time()

    # Track memory
    import resource
    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Build extractor
    print(f"Building {extractor_type} extractor...", flush=True)
    extractor_fn = build_extractor(extractor_type, params)

    # Load data
    data = load_goldset(goldset_dir, split)
    print(f"Loaded {len(data)} messages from {split}", flush=True)

    # Run evaluation
    if cv_folds > 0:
        print(f"Running {cv_folds}-fold cross-validation...", flush=True)
        results = cross_validate(data, extractor_fn, n_folds=cv_folds)
    else:
        print("Running evaluation...", flush=True)
        fold_result = evaluate_fold(data, extractor_fn, train_data=None)
        pl = fold_result["per_label"]
        results = build_metrics(
            defaultdict(int, {lb: c["tp"] for lb, c in pl.items()}),
            defaultdict(int, {lb: c["fp"] for lb, c in pl.items()}),
            defaultdict(int, {lb: c["fn"] for lb, c in pl.items()}),
            fold_result["total_tp"],
            fold_result["total_fp"],
            fold_result["total_fn"],
            len(data),
            fold_result.get("fp_examples", []),
            fold_result.get("fn_examples", []),
        )

    elapsed = time.time() - start
    rss_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Add metadata
    results["name"] = name
    results["extractor"] = extractor_type
    results["params"] = params
    results["split"] = split
    results["cv_folds"] = cv_folds
    results["elapsed_s"] = round(elapsed, 1)
    results["rss_mb"] = round(rss_end / (1024 * 1024), 1)  # macOS returns bytes
    results["status"] = "ok"

    # Write results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"\nResults: micro_f1={results['micro_f1']},"
        f" P={results['micro_precision']},"
        f" R={results['micro_recall']}",
        flush=True,
    )
    print(
        f"TP={results['total_tp']},"
        f" FP={results['total_fp']},"
        f" FN={results['total_fn']}",
        flush=True,
    )
    print(f"Time: {elapsed:.1f}s, RSS: {results['rss_mb']}MB", flush=True)


if __name__ == "__main__":
    main()
