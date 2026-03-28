#!/usr/bin/env python3
"""DSPy optimization: compile better prompts + few-shot examples.

Supports two modes:
1. Global: optimize a single program across all test cases (original behavior)
2. Per-category: optimize separate programs for each category (MIPRO v2)

Uses ZAI GLM 4.7 via Cerebras as teacher to bootstrap demonstrations for the
local MLX 1.2B student model, then evaluates with the same judge.

Usage:
    uv run python evals/dspy_optimize.py                         # BootstrapFewShot (global)
    uv run python evals/dspy_optimize.py --optimizer mipro       # MIPROv2 (global)
    uv run python evals/dspy_optimize.py --per-category          # Per-category MIPROv2
    uv run python evals/dspy_optimize.py --eval-only             # Evaluate saved global program
    uv run python evals/dspy_optimize.py --eval-only --per-category  # Eval all categories
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

import dspy  # noqa: E402

from evals.dspy_client import DSPYMLXClient  # noqa: E402
from evals.dspy_reply import (  # noqa: E402
    TRAIN_EXAMPLES,
    CategoryReplyModule,
    ReplyModule,
    clean_reply,
    get_all_categories,
    get_category_examples,
    judge_metric,
)

# Save paths
SAVE_DIR = PROJECT_ROOT / "evals" / "optimized_reply.json"
CATEGORY_SAVE_DIR = PROJECT_ROOT / "evals" / "optimized_categories"


def build_teacher_lm() -> dspy.LM:
    """Cerebras ZAI GLM 4.7 as the teacher/demo generator."""
    from evals.judge_config import JUDGE_BASE_URL, JUDGE_MODEL, get_judge_api_key

    key = get_judge_api_key()
    return dspy.LM(
        model=f"openai/{JUDGE_MODEL}",
        api_base=JUDGE_BASE_URL,
        api_key=key,
        temperature=0.7,
        max_tokens=300,
    )


def build_student_lm() -> DSPYMLXClient:
    """Local MLX 1.2B as the student model."""
    return DSPYMLXClient(max_tokens=50, temperature=0.1)


def run_bootstrap(
    student: ReplyModule,
    trainset: list[dspy.Example],
    teacher_lm: dspy.LM,
) -> ReplyModule:
    """BootstrapFewShot: fast, bootstraps few-shot demos from teacher."""
    print("Optimizer: BootstrapFewShot", flush=True)
    print("  max_bootstrapped_demos=3, max_labeled_demos=4", flush=True)
    print(f"  trainset size: {len(trainset)}", flush=True)

    optimizer = dspy.BootstrapFewShot(
        metric=judge_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=4,
    )

    with dspy.context(lm=teacher_lm):
        compiled = optimizer.compile(student=student, trainset=trainset)

    return compiled


def run_mipro(
    student: dspy.Module,
    trainset: list[dspy.Example],
    teacher_lm: dspy.LM,
    num_candidates: int = 5,
    num_trials: int = 15,
    max_bootstrapped_demos: int = 3,
) -> dspy.Module:
    """MIPROv2: optimizes instruction text + few-shot demos."""
    print("Optimizer: MIPROv2", flush=True)
    print(
        f"  num_candidates={num_candidates}, num_trials={num_trials}, "
        f"max_bootstrapped_demos={max_bootstrapped_demos}",
        flush=True,
    )
    print(f"  trainset size: {len(trainset)}", flush=True)

    optimizer = dspy.MIPROv2(
        metric=judge_metric,
        prompt_model=teacher_lm,
        auto=None,
        num_candidates=num_candidates,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=4,
    )

    compiled = optimizer.compile(
        student=student,
        trainset=trainset,
        num_trials=num_trials,
        minibatch=False,
    )

    return compiled


@dataclass
class EvalCaseResult:
    """Result of evaluating a single test case."""

    query: str
    raw_reply: str
    cleaned_reply: str
    score: float  # Continuous 0.0-1.0 score from judge
    passed: bool  # score >= 0.7
    has_artifacts: bool  # True if raw != cleaned (DSPy leak detected)
    error: str | None = None


def evaluate_program(
    program: dspy.Module, trainset: list[dspy.Example]
) -> tuple[float, list[EvalCaseResult]]:
    """Run the metric on all examples, return (avg score, detailed results).

    Uses continuous 0-1 scoring for better signal. Reports both average score
    and pass rate (>= 0.7 threshold).
    """
    scores: list[float] = []
    details: list[EvalCaseResult] = []
    for ex in tqdm(trainset, desc="Evaluating"):
        try:
            pred = program(**{k: ex[k] for k in ["context", "last_message", "tone", "user_style"]})
            raw = pred.reply.strip()
            cleaned = clean_reply(pred.reply)
            has_artifacts = raw != cleaned
            score = judge_metric(ex, pred)
            scores.append(score)
            passed = score >= 0.7
            status = f"{score * 10:.0f}/10"
            artifact_tag = " [ARTIFACT]" if has_artifacts else ""
            print(f"  {status}: {ex.last_message[:40]!r} -> {raw!r}{artifact_tag}", flush=True)
            if has_artifacts:
                print(f"         cleaned: {cleaned!r}", flush=True)
            details.append(
                EvalCaseResult(
                    query=ex.last_message,
                    raw_reply=raw,
                    cleaned_reply=cleaned,
                    score=score,
                    passed=passed,
                    has_artifacts=has_artifacts,
                )
            )
        except Exception as e:
            print(f"  ERROR: {ex.last_message[:40]!r} -> {e}", flush=True)
            scores.append(0.0)
            details.append(
                EvalCaseResult(
                    query=ex.last_message,
                    raw_reply="",
                    cleaned_reply="",
                    score=0.0,
                    passed=False,
                    has_artifacts=False,
                    error=str(e),
                )
            )
    avg_score = sum(scores) / len(scores) if scores else 0
    n_passed = sum(1 for d in details if d.passed)
    artifact_fails = sum(1 for d in details if not d.passed and d.has_artifacts)
    print(f"\nAvg score: {avg_score * 10:.1f}/10", flush=True)
    print(
        f"Pass rate (>=7): {n_passed}/{len(trainset)} ({n_passed / len(trainset) * 100:.0f}%)",
        flush=True,
    )
    if artifact_fails:
        print(
            f"  ({artifact_fails} failures had DSPy artifacts - fixable with post-processing)",
            flush=True,
        )
    return avg_score, details


# ---------------------------------------------------------------------------
# Per-category optimization
# ---------------------------------------------------------------------------


def run_per_category(
    teacher_lm: dspy.LM,
    optimizer_name: str = "mipro",
) -> dict[str, float]:
    """Run optimization separately for each category.

    Returns dict of category -> pass rate.
    """
    categories = get_all_categories()
    print(f"\nRunning per-category optimization for: {', '.join(categories)}", flush=True)
    print(f"Optimizer: {optimizer_name}", flush=True)
    print(flush=True)

    CATEGORY_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, float] = {}
    all_details: dict[str, list[EvalCaseResult]] = {}

    for cat in tqdm(categories, desc="Categories"):
        cat_examples = get_category_examples(cat)
        if len(cat_examples) < 3:
            print(f"SKIP {cat}: only {len(cat_examples)} examples (need >= 3)", flush=True)
            continue

        print("=" * 70, flush=True)
        print(f"CATEGORY: {cat} ({len(cat_examples)} examples)", flush=True)
        print("=" * 70, flush=True)

        student = CategoryReplyModule(cat)
        start = time.perf_counter()

        if optimizer_name == "mipro":
            compiled = run_mipro(
                student=student,
                trainset=cat_examples,
                teacher_lm=teacher_lm,
                num_candidates=5,
                num_trials=15,
                max_bootstrapped_demos=3,
            )
        else:
            compiled = run_bootstrap(student, cat_examples, teacher_lm)

        elapsed = time.perf_counter() - start
        print(f"\n{cat} optimization took {elapsed:.1f}s", flush=True)

        # Save per-category program
        save_path = CATEGORY_SAVE_DIR / f"optimized_{cat}.json"
        compiled.save(str(save_path))
        print(f"Saved to {save_path}", flush=True)

        # Evaluate
        print(f"\nEvaluating {cat}:", flush=True)
        rate, cat_details = evaluate_program(compiled, cat_examples)
        results[cat] = rate
        all_details[cat] = cat_details
        print(flush=True)

    return results, all_details


def eval_per_category() -> tuple[dict[str, float], dict[str, list[EvalCaseResult]]]:
    """Evaluate all saved per-category programs."""
    categories = get_all_categories()
    results: dict[str, float] = {}
    all_details: dict[str, list[EvalCaseResult]] = {}

    for cat in tqdm(categories, desc="Evaluating categories"):
        save_path = CATEGORY_SAVE_DIR / f"optimized_{cat}.json"
        if not save_path.exists():
            print(f"SKIP {cat}: no saved program at {save_path}", flush=True)
            continue

        cat_examples = get_category_examples(cat)
        print(f"\n{'=' * 70}", flush=True)
        print(f"CATEGORY: {cat} ({len(cat_examples)} examples)", flush=True)
        print(f"{'=' * 70}", flush=True)

        program = CategoryReplyModule(cat)
        program.load(str(save_path))

        rate, cat_details = evaluate_program(program, cat_examples)
        results[cat] = rate
        all_details[cat] = cat_details

    return results, all_details


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="DSPy optimization for JARVIS reply generation")
    parser.add_argument(
        "--optimizer",
        choices=["bootstrap", "mipro"],
        default="bootstrap",
        help="Optimization strategy (default: bootstrap)",
    )
    parser.add_argument(
        "--per-category",
        action="store_true",
        help="Run per-category optimization instead of global",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip optimization, just evaluate saved program(s)",
    )
    args = parser.parse_args()

    # Setup logging
    log_path = PROJECT_ROOT / "results" / "dspy_optimize.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger(__name__)

    print("=" * 70, flush=True)
    print("JARVIS DSPy Optimization Pipeline", flush=True)
    print("=" * 70, flush=True)

    # Build student LM and configure as default
    student_lm = build_student_lm()
    dspy.configure(lm=student_lm)

    trainset = TRAIN_EXAMPLES
    print(f"Training examples: {len(trainset)}", flush=True)
    print(f"Categories: {', '.join(get_all_categories())}", flush=True)

    # --- Eval-only mode ---
    if args.eval_only:
        if args.per_category:
            print("\nEvaluating per-category compiled programs:", flush=True)
            results, all_details = eval_per_category()
            if results:
                print("\n" + "=" * 70, flush=True)
                print("PER-CATEGORY SUMMARY", flush=True)
                print("=" * 70, flush=True)
                for cat, rate in sorted(results.items()):
                    print(f"  {cat:20s}  {rate:.0%}", flush=True)
                avg = sum(results.values()) / len(results)
                print(f"  {'AVERAGE':20s}  {avg:.0%}", flush=True)
            return 0
        else:
            if not SAVE_DIR.exists():
                print(f"ERROR: No saved program at {SAVE_DIR}", flush=True)
                return 1
            print(f"\nLoading compiled program from {SAVE_DIR}", flush=True)
            program = ReplyModule()
            program.load(str(SAVE_DIR))
            print("\nEvaluating compiled program:", flush=True)
            evaluate_program(program, trainset)  # results printed inline
            return 0

    # --- Optimization mode ---
    teacher_lm = build_teacher_lm()
    print("Teacher: ZAI GLM 4.7 via Cerebras", flush=True)
    print("Student: MLX local 1.2B", flush=True)
    print(flush=True)

    if args.per_category:
        # Per-category optimization (default: mipro for per-category)
        optimizer_name = args.optimizer if args.optimizer != "bootstrap" else "mipro"
        results, all_details = run_per_category(teacher_lm, optimizer_name)

        print("\n" + "=" * 70, flush=True)
        print("PER-CATEGORY RESULTS", flush=True)
        print("=" * 70, flush=True)
        for cat, rate in sorted(results.items()):
            print(f"  {cat:20s}  {rate:.0%}", flush=True)
        if results:
            avg = sum(results.values()) / len(results)
            print(f"  {'AVERAGE':20s}  {avg:.0%}", flush=True)

        # Save detailed summary with per-case logs
        summary_path = CATEGORY_SAVE_DIR / "summary.json"
        detail_data: dict = {}
        for cat, cases in all_details.items():
            detail_data[cat] = [
                {
                    "query": c.query,
                    "raw_reply": c.raw_reply,
                    "cleaned_reply": c.cleaned_reply,
                    "passed": c.passed,
                    "has_artifacts": c.has_artifacts,
                    "error": c.error,
                }
                for c in cases
            ]
        summary_path.write_text(
            json.dumps(
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "optimizer": optimizer_name,
                    "temperature": 0.1,
                    "per_category_pass_rates": results,
                    "per_case_details": detail_data,
                },
                indent=2,
            )
        )
        print(f"\nSummary saved to {summary_path}", flush=True)
    else:
        # Global optimization (original behavior)
        start = time.perf_counter()
        student = ReplyModule()

        if args.optimizer == "mipro":
            compiled = run_mipro(student, trainset, teacher_lm)
        else:
            compiled = run_bootstrap(student, trainset, teacher_lm)

        elapsed = time.perf_counter() - start
        print(f"\nOptimization took {elapsed:.1f}s", flush=True)

        # Save
        SAVE_DIR.parent.mkdir(parents=True, exist_ok=True)
        compiled.save(str(SAVE_DIR))
        print(f"Saved compiled program to {SAVE_DIR}", flush=True)

        # Evaluate the compiled program
        print("\n" + "-" * 70, flush=True)
        print("Evaluating compiled program:", flush=True)
        evaluate_program(compiled, trainset)

    print("=" * 70, flush=True)
    print("Done. Run batch_eval with --optimized to compare:", flush=True)
    print("  uv run python evals/batch_eval.py --judge --optimized", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
