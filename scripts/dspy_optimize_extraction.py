#!/usr/bin/env python3
"""DSPy prompt optimizer for fact extraction.

Connects to a local mlx_lm server (OpenAI-compatible API) and uses
BootstrapFewShot to auto-optimize extraction prompts against our goldset.

Prerequisites:
    # Start the MLX server in a separate terminal:
    uv run python -m mlx_lm.server \
        --model LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit --port 8080

Usage:
    uv run python scripts/dspy_optimize_extraction.py
    uv run python scripts/dspy_optimize_extraction.py --port 8080 --n-train 30
    uv run python scripts/dspy_optimize_extraction.py --output results/dspy_optimized_prompt.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from eval_shared import spans_match

# ─── Constants ──────────────────────────────────────────────────────────────

GOLD_PATH = Path("training_data/goldset_v6/train.json")
OUTPUT_PATH = Path("results/dspy_optimized_prompt.json")

# Same aliases as bakeoff_v2
LLM_LABEL_ALIASES: dict[str, set[str]] = {
    "location": {
        "current_location", "past_location", "future_location", "place", "hometown",
    },
    "person": {"friend_name", "partner_name", "person_name", "family_member"},
    "job": {"employer", "job_role", "job_title"},
    "school": {"school"},
    "health": {"allergy", "health_condition", "dietary"},
    "relationship": {"family_member", "partner_name"},
    "preference": {"food_like", "food_dislike", "food_item", "preference", "hobby"},
    "activity": {"activity", "hobby"},
}


# ─── DSPy Setup ─────────────────────────────────────────────────────────────


def build_training_examples(gold_path: Path, n_train: int = 20, n_val: int = 20) -> tuple:
    """Build DSPy training/validation examples from goldset.

    Returns (train_examples, val_examples) as lists of dspy.Example.
    """
    import dspy

    with open(gold_path) as f:
        records = json.load(f)

    # Split: positive examples first (more informative for few-shot)
    positive = [r for r in records if r.get("expected_candidates")]
    negative = [r for r in records if not r.get("expected_candidates")]

    # Balanced split for training
    n_pos = n_train // 2
    n_neg = n_train - n_pos
    train_records = positive[:n_pos] + negative[:n_neg]

    # Validation from remaining
    n_val_pos = n_val // 2
    n_val_neg = n_val - n_val_pos
    val_records = positive[n_pos:n_pos + n_val_pos] + negative[n_neg:n_neg + n_val_neg]

    def record_to_example(rec: dict) -> "dspy.Example":
        cands = rec.get("expected_candidates", [])
        if cands:
            facts_str = "; ".join(
                f"{c['span_label']}: {c['span_text']}" for c in cands
            )
        else:
            facts_str = "none"

        return dspy.Example(
            message=rec["message_text"],
            facts=facts_str,
        ).with_inputs("message")

    train = [record_to_example(r) for r in train_records]
    val = [record_to_example(r) for r in val_records]

    print(f"Built {len(train)} training, {len(val)} validation examples", flush=True)
    return train, val


def build_metric_fn(gold_path: Path, n_val: int = 20):
    """Build a DSPy metric function that scores extraction quality."""
    with open(gold_path) as f:
        records = json.load(f)

    # Build lookup: message_text -> expected_candidates
    gold_lookup: dict[str, list[dict]] = {}
    for r in records:
        gold_lookup[r["message_text"]] = r.get("expected_candidates", [])

    def extraction_metric(example, pred, trace=None) -> float:
        """Score a DSPy prediction against gold.

        Returns F1 score (0.0 - 1.0).
        """
        msg_text = example.message
        gold_cands = gold_lookup.get(msg_text, [])

        # Parse prediction
        pred_text = pred.facts if hasattr(pred, "facts") else str(pred)
        parsed_facts = _parse_dspy_output(pred_text)

        if not gold_cands and not parsed_facts:
            return 1.0  # Correct: no facts expected, none predicted
        if not gold_cands and parsed_facts:
            return 0.0  # FP
        if gold_cands and not parsed_facts:
            return 0.0  # FN

        # Score with spans_match
        gold_matched = [False] * len(gold_cands)
        pred_matched = [False] * len(parsed_facts)

        for gi, gc in enumerate(gold_cands):
            for pi, (pred_cat, pred_val) in enumerate(parsed_facts):
                if pred_matched[pi]:
                    continue
                if spans_match(
                    pred_val, pred_cat,
                    gc.get("span_text", ""), gc.get("span_label", ""),
                    label_aliases=LLM_LABEL_ALIASES,
                ):
                    gold_matched[gi] = True
                    pred_matched[pi] = True
                    break

        tp = sum(gold_matched)
        fp = len(parsed_facts) - sum(pred_matched)
        fn = len(gold_cands) - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return f1

    return extraction_metric


def _parse_dspy_output(text: str) -> list[tuple[str, str]]:
    """Parse DSPy output -> list of (category, value)."""
    text = text.strip()
    if text.lower() in ("none", "no facts", ""):
        return []

    facts = []
    # Try semicolon-separated first (our training format)
    parts = text.split(";")
    for part in parts:
        part = part.strip()
        if ":" in part:
            cat, _, val = part.partition(":")
            cat = cat.strip().lower()
            val = val.strip()
            if val and val.lower() != "none":
                facts.append((cat, val))

    # If semicolon parsing failed, try newline-separated
    if not facts:
        for line in text.split("\n"):
            line = line.strip().lstrip("- ")
            if ":" in line:
                cat, _, val = line.partition(":")
                cat = cat.strip().lower()
                val = val.strip()
                if val and val.lower() != "none":
                    facts.append((cat, val))

    return facts


# ─── DSPy Signature & Module ────────────────────────────────────────────────


def create_extraction_module():
    """Create the DSPy extraction module."""
    import dspy

    class ExtractFacts(dspy.Signature):
        """Extract personal facts from a text message.

        Categories: location, person, relationship, preference, job, school, health, activity.
        Only extract facts explicitly stated in the message.
        Output format: category: value (semicolon-separated if multiple). Output 'none' if no facts.
        """
        message: str = dspy.InputField(desc="The text message to extract facts from")
        facts: str = dspy.OutputField(
            desc="Extracted facts as 'category: value; category: value' or 'none'"
        )

    class FactExtractor(dspy.Module):
        def __init__(self):
            super().__init__()
            self.extract = dspy.Predict(ExtractFacts)

        def forward(self, message: str):
            return self.extract(message=message)

    return FactExtractor()


# ─── Main ────────────────────────────────────────────────────────────────────


def run_optimizer(
    port: int = 8080,
    n_train: int = 20,
    n_val: int = 20,
    max_bootstrapped: int = 4,
    gold_path: Path = GOLD_PATH,
    output_path: Path = OUTPUT_PATH,
) -> dict:
    """Run the DSPy optimizer and save the optimized prompt."""
    import dspy

    print("=" * 60, flush=True)
    print("DSPy Extraction Prompt Optimizer", flush=True)
    print("=" * 60, flush=True)

    # Connect to local mlx_lm server
    print(f"\nConnecting to local server on port {port}...", flush=True)
    lm = dspy.LM(
        model="openai/local-model",
        api_base=f"http://localhost:{port}/v1",
        api_key="not-needed",
        temperature=0.0,
        max_tokens=200,
    )
    dspy.configure(lm=lm)
    print("Connected.", flush=True)

    # Build examples
    print("\nBuilding training examples...", flush=True)
    train_examples, val_examples = build_training_examples(
        gold_path, n_train=n_train, n_val=n_val
    )

    # Build metric
    metric_fn = build_metric_fn(gold_path, n_val=n_val)

    # Create module
    extractor = create_extraction_module()

    # Quick baseline evaluation
    print("\nBaseline evaluation (no optimization)...", flush=True)
    baseline_scores = []
    for ex in val_examples[:10]:
        try:
            pred = extractor(message=ex.message)
            score = metric_fn(ex, pred)
            baseline_scores.append(score)
        except Exception as e:
            print(f"  Baseline error: {e}", flush=True)
            baseline_scores.append(0.0)
    baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
    print(f"  Baseline avg F1: {baseline_avg:.3f} (on {len(baseline_scores)} examples)", flush=True)

    # Run BootstrapFewShot optimizer
    print(f"\nRunning BootstrapFewShot (max_bootstrapped={max_bootstrapped})...", flush=True)
    t0 = time.time()

    optimizer = dspy.BootstrapFewShot(
        metric=metric_fn,
        max_bootstrapped_demos=max_bootstrapped,
        max_labeled_demos=max_bootstrapped,
    )

    optimized = optimizer.compile(
        extractor,
        trainset=train_examples,
    )

    elapsed = time.time() - t0
    print(f"Optimization complete in {elapsed:.1f}s", flush=True)

    # Evaluate optimized
    print("\nOptimized evaluation...", flush=True)
    opt_scores = []
    for ex in val_examples[:10]:
        try:
            pred = optimized(message=ex.message)
            score = metric_fn(ex, pred)
            opt_scores.append(score)
        except Exception as e:
            print(f"  Eval error: {e}", flush=True)
            opt_scores.append(0.0)
    opt_avg = sum(opt_scores) / len(opt_scores) if opt_scores else 0.0
    print(f"  Optimized avg F1: {opt_avg:.3f} (on {len(opt_scores)} examples)", flush=True)
    print(f"  Improvement: {opt_avg - baseline_avg:+.3f}", flush=True)

    # Extract the optimized prompt
    # DSPy stores the demos in the module's predictors
    demos = []
    if hasattr(optimized, 'extract') and hasattr(optimized.extract, 'demos'):
        for demo in optimized.extract.demos:
            demos.append({
                "message": getattr(demo, "message", ""),
                "facts": getattr(demo, "facts", ""),
            })

    # Build a prompt template from the optimized module
    prompt_data = {
        "system_prompt": (
            "Extract personal facts from text messages. "
            "Categories: location, person, relationship, preference, "
            "job, school, health, activity. "
            "Only extract facts explicitly stated. "
            "Output format: category: value (semicolon-separated). Output 'none' if no facts."
        ),
        "user_prompt_template": "Message: \"{text}\"\n\nFacts:",
        "parse_mode": "kv",
        "demos": demos,
        "metadata": {
            "optimizer": "BootstrapFewShot",
            "max_bootstrapped": max_bootstrapped,
            "n_train": n_train,
            "baseline_f1": round(baseline_avg, 4),
            "optimized_f1": round(opt_avg, 4),
            "improvement": round(opt_avg - baseline_avg, 4),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    # If demos exist, build a few-shot user prompt template
    if demos:
        few_shot_lines = []
        for d in demos:
            few_shot_lines.append(f'Message: "{d["message"]}"')
            few_shot_lines.append(f'Facts: {d["facts"]}')
            few_shot_lines.append("")
        few_shot_prefix = "\n".join(few_shot_lines)
        prompt_data["user_prompt_template"] = (
            few_shot_prefix + 'Message: "{text}"\nFacts:'
        )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(prompt_data, f, indent=2)
    print(f"\nSaved optimized prompt to {output_path}", flush=True)

    # Print summary
    print(f"\n{'=' * 60}", flush=True)
    print("OPTIMIZATION RESULTS", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Baseline F1:  {baseline_avg:.3f}", flush=True)
    print(f"  Optimized F1: {opt_avg:.3f}", flush=True)
    print(f"  Demos found:  {len(demos)}", flush=True)
    print(f"  Output:       {output_path}", flush=True)
    print("\nTo use in bakeoff:", flush=True)
    print(
        f"  uv run python scripts/extraction_bakeoff_v2.py"
        f" --dspy-prompt {output_path}",
        flush=True,
    )

    return prompt_data


def main():
    parser = argparse.ArgumentParser(description="DSPy extraction prompt optimizer")
    parser.add_argument("--port", type=int, default=8080, help="mlx_lm server port")
    parser.add_argument("--n-train", type=int, default=20, help="Number of training examples")
    parser.add_argument("--n-val", type=int, default=20, help="Number of validation examples")
    parser.add_argument(
        "--max-bootstrapped", type=int, default=4,
        help="Max bootstrapped demos for BootstrapFewShot",
    )
    parser.add_argument(
        "--gold", type=str, default=str(GOLD_PATH), help="Path to goldset JSON",
    )
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_PATH),
        help="Output path for optimized prompt JSON",
    )
    args = parser.parse_args()

    gold_path = Path(args.gold)
    if not gold_path.exists():
        print(f"Goldset not found: {gold_path}", flush=True)
        sys.exit(1)

    run_optimizer(
        port=args.port,
        n_train=args.n_train,
        n_val=args.n_val,
        max_bootstrapped=args.max_bootstrapped,
        gold_path=gold_path,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
