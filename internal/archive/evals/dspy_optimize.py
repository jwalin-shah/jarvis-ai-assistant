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
  # noqa: E402
# Load .env  # noqa: E402
_env_path = PROJECT_ROOT / ".env"  # noqa: E402
if _env_path.exists():  # noqa: E402
    for line in _env_path.read_text().splitlines():  # noqa: E402
        line = line.strip()  # noqa: E402
        if line and not line.startswith("#") and "=" in line:  # noqa: E402
            key, _, val = line.partition("=")  # noqa: E402
            os.environ.setdefault(key.strip(), val.strip())  # noqa: E402
  # noqa: E402
import dspy  # noqa: E402
from evals.dspy_client import DSPYMLXClient  # noqa: E402
from evals.dspy_reply import (  # noqa: E402
    TRAIN_EXAMPLES,  # noqa: E402
    CategoryReplyModule,  # noqa: E402
    ReplyModule,  # noqa: E402
    clean_reply,  # noqa: E402
    get_all_categories,  # noqa: E402
    get_category_examples,  # noqa: E402
    judge_metric,  # noqa: E402
)  # noqa: E402

  # noqa: E402
# Save paths  # noqa: E402
SAVE_DIR = PROJECT_ROOT / "evals" / "optimized_reply.json"  # noqa: E402
CATEGORY_SAVE_DIR = PROJECT_ROOT / "evals" / "optimized_categories"  # noqa: E402
  # noqa: E402
  # noqa: E402
def build_teacher_lm() -> dspy.LM:  # noqa: E402
    """Cerebras ZAI GLM 4.7 as the teacher/demo generator."""  # noqa: E402
    from evals.judge_config import JUDGE_BASE_URL, JUDGE_MODEL, get_judge_api_key  # noqa: E402
  # noqa: E402
    key = get_judge_api_key()  # noqa: E402
    return dspy.LM(  # noqa: E402
        model=f"openai/{JUDGE_MODEL}",  # noqa: E402
        api_base=JUDGE_BASE_URL,  # noqa: E402
        api_key=key,  # noqa: E402
        temperature=0.7,  # noqa: E402
        max_tokens=300,  # noqa: E402
    )  # noqa: E402
  # noqa: E402
  # noqa: E402
def build_student_lm() -> DSPYMLXClient:  # noqa: E402
    """Local MLX 1.2B as the student model."""  # noqa: E402
    return DSPYMLXClient(max_tokens=50, temperature=0.1)  # noqa: E402
  # noqa: E402
  # noqa: E402
def run_bootstrap(  # noqa: E402
    student: ReplyModule,  # noqa: E402
    trainset: list[dspy.Example],  # noqa: E402
    teacher_lm: dspy.LM,  # noqa: E402
) -> ReplyModule:  # noqa: E402
    """BootstrapFewShot: fast, bootstraps few-shot demos from teacher."""  # noqa: E402
    print("Optimizer: BootstrapFewShot", flush=True)  # noqa: E402
    print("  max_bootstrapped_demos=3, max_labeled_demos=4", flush=True)  # noqa: E402
    print(f"  trainset size: {len(trainset)}", flush=True)  # noqa: E402
  # noqa: E402
    optimizer = dspy.BootstrapFewShot(  # noqa: E402
        metric=judge_metric,  # noqa: E402
        max_bootstrapped_demos=3,  # noqa: E402
        max_labeled_demos=4,  # noqa: E402
    )  # noqa: E402
  # noqa: E402
    with dspy.context(lm=teacher_lm):  # noqa: E402
        compiled = optimizer.compile(student=student, trainset=trainset)  # noqa: E402
  # noqa: E402
    return compiled  # noqa: E402
  # noqa: E402
  # noqa: E402
def run_mipro(  # noqa: E402
    student: dspy.Module,  # noqa: E402
    trainset: list[dspy.Example],  # noqa: E402
    teacher_lm: dspy.LM,  # noqa: E402
    num_candidates: int = 5,  # noqa: E402
    num_trials: int = 15,  # noqa: E402
    max_bootstrapped_demos: int = 3,  # noqa: E402
) -> dspy.Module:  # noqa: E402
    """MIPROv2: optimizes instruction text + few-shot demos."""  # noqa: E402
    print("Optimizer: MIPROv2", flush=True)  # noqa: E402
    print(  # noqa: E402
        f"  num_candidates={num_candidates}, num_trials={num_trials}, "  # noqa: E402
        f"max_bootstrapped_demos={max_bootstrapped_demos}",  # noqa: E402
        flush=True,  # noqa: E402
    )  # noqa: E402
    print(f"  trainset size: {len(trainset)}", flush=True)  # noqa: E402
  # noqa: E402
    optimizer = dspy.MIPROv2(  # noqa: E402
        metric=judge_metric,  # noqa: E402
        prompt_model=teacher_lm,  # noqa: E402
        auto=None,  # noqa: E402
        num_candidates=num_candidates,  # noqa: E402
        max_bootstrapped_demos=max_bootstrapped_demos,  # noqa: E402
        max_labeled_demos=4,  # noqa: E402
    )  # noqa: E402
  # noqa: E402
    compiled = optimizer.compile(  # noqa: E402
        student=student,  # noqa: E402
        trainset=trainset,  # noqa: E402
        num_trials=num_trials,  # noqa: E402
        minibatch=False,  # noqa: E402
    )  # noqa: E402
  # noqa: E402
    return compiled  # noqa: E402
  # noqa: E402
  # noqa: E402
@dataclass  # noqa: E402
class EvalCaseResult:  # noqa: E402
    """Result of evaluating a single test case."""  # noqa: E402
  # noqa: E402
    query: str  # noqa: E402
    raw_reply: str  # noqa: E402
    cleaned_reply: str  # noqa: E402
    score: float  # Continuous 0.0-1.0 score from judge  # noqa: E402
    passed: bool  # score >= 0.7  # noqa: E402
    has_artifacts: bool  # True if raw != cleaned (DSPy leak detected)  # noqa: E402
    error: str | None = None  # noqa: E402
  # noqa: E402
  # noqa: E402
def evaluate_program(  # noqa: E402
    program: dspy.Module, trainset: list[dspy.Example]  # noqa: E402
) -> tuple[float, list[EvalCaseResult]]:  # noqa: E402
    """Run the metric on all examples, return (avg score, detailed results).  # noqa: E402
  # noqa: E402
    Uses continuous 0-1 scoring for better signal. Reports both average score  # noqa: E402
    and pass rate (>= 0.7 threshold).  # noqa: E402
    """  # noqa: E402
    scores: list[float] = []  # noqa: E402
    details: list[EvalCaseResult] = []  # noqa: E402
    for ex in tqdm(trainset, desc="Evaluating"):  # noqa: E402
        try:  # noqa: E402
            pred = program(**{k: ex[k] for k in ["context", "last_message", "tone", "user_style"]})  # noqa: E402
            raw = pred.reply.strip()  # noqa: E402
            cleaned = clean_reply(pred.reply)  # noqa: E402
            has_artifacts = raw != cleaned  # noqa: E402
            score = judge_metric(ex, pred)  # noqa: E402
            scores.append(score)  # noqa: E402
            passed = score >= 0.7  # noqa: E402
            status = f"{score * 10:.0f}/10"  # noqa: E402
            artifact_tag = " [ARTIFACT]" if has_artifacts else ""  # noqa: E402
            print(f"  {status}: {ex.last_message[:40]!r} -> {raw!r}{artifact_tag}", flush=True)  # noqa: E402
            if has_artifacts:  # noqa: E402
                print(f"         cleaned: {cleaned!r}", flush=True)  # noqa: E402
            details.append(  # noqa: E402
                EvalCaseResult(  # noqa: E402
                    query=ex.last_message,  # noqa: E402
                    raw_reply=raw,  # noqa: E402
                    cleaned_reply=cleaned,  # noqa: E402
                    score=score,  # noqa: E402
                    passed=passed,  # noqa: E402
                    has_artifacts=has_artifacts,  # noqa: E402
                )  # noqa: E402
            )  # noqa: E402
        except Exception as e:  # noqa: E402
            print(f"  ERROR: {ex.last_message[:40]!r} -> {e}", flush=True)  # noqa: E402
            scores.append(0.0)  # noqa: E402
            details.append(  # noqa: E402
                EvalCaseResult(  # noqa: E402
                    query=ex.last_message,  # noqa: E402
                    raw_reply="",  # noqa: E402
                    cleaned_reply="",  # noqa: E402
                    score=0.0,  # noqa: E402
                    passed=False,  # noqa: E402
                    has_artifacts=False,  # noqa: E402
                    error=str(e),  # noqa: E402
                )  # noqa: E402
            )  # noqa: E402
    avg_score = sum(scores) / len(scores) if scores else 0  # noqa: E402
    n_passed = sum(1 for d in details if d.passed)  # noqa: E402
    artifact_fails = sum(1 for d in details if not d.passed and d.has_artifacts)  # noqa: E402
    print(f"\nAvg score: {avg_score * 10:.1f}/10", flush=True)  # noqa: E402
    print(  # noqa: E402
        f"Pass rate (>=7): {n_passed}/{len(trainset)} ({n_passed / len(trainset) * 100:.0f}%)",  # noqa: E402
        flush=True,  # noqa: E402
    )  # noqa: E402
    if artifact_fails:  # noqa: E402
        print(  # noqa: E402
            f"  ({artifact_fails} failures had DSPy artifacts - fixable with post-processing)",  # noqa: E402
            flush=True,  # noqa: E402
        )  # noqa: E402
    return avg_score, details  # noqa: E402
  # noqa: E402
  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
# Per-category optimization  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
  # noqa: E402
  # noqa: E402
def run_per_category(  # noqa: E402
    teacher_lm: dspy.LM,  # noqa: E402
    optimizer_name: str = "mipro",  # noqa: E402
) -> dict[str, float]:  # noqa: E402
    """Run optimization separately for each category.  # noqa: E402
  # noqa: E402
    Returns dict of category -> pass rate.  # noqa: E402
    """  # noqa: E402
    categories = get_all_categories()  # noqa: E402
    print(f"\nRunning per-category optimization for: {', '.join(categories)}", flush=True)  # noqa: E402
    print(f"Optimizer: {optimizer_name}", flush=True)  # noqa: E402
    print(flush=True)  # noqa: E402
  # noqa: E402
    CATEGORY_SAVE_DIR.mkdir(parents=True, exist_ok=True)  # noqa: E402
    results: dict[str, float] = {}  # noqa: E402
    all_details: dict[str, list[EvalCaseResult]] = {}  # noqa: E402
  # noqa: E402
    for cat in tqdm(categories, desc="Categories"):  # noqa: E402
        cat_examples = get_category_examples(cat)  # noqa: E402
        if len(cat_examples) < 3:  # noqa: E402
            print(f"SKIP {cat}: only {len(cat_examples)} examples (need >= 3)", flush=True)  # noqa: E402
            continue  # noqa: E402
  # noqa: E402
        print("=" * 70, flush=True)  # noqa: E402
        print(f"CATEGORY: {cat} ({len(cat_examples)} examples)", flush=True)  # noqa: E402
        print("=" * 70, flush=True)  # noqa: E402
  # noqa: E402
        student = CategoryReplyModule(cat)  # noqa: E402
        start = time.perf_counter()  # noqa: E402
  # noqa: E402
        if optimizer_name == "mipro":  # noqa: E402
            compiled = run_mipro(  # noqa: E402
                student=student,  # noqa: E402
                trainset=cat_examples,  # noqa: E402
                teacher_lm=teacher_lm,  # noqa: E402
                num_candidates=5,  # noqa: E402
                num_trials=15,  # noqa: E402
                max_bootstrapped_demos=3,  # noqa: E402
            )  # noqa: E402
        else:  # noqa: E402
            compiled = run_bootstrap(student, cat_examples, teacher_lm)  # noqa: E402
  # noqa: E402
        elapsed = time.perf_counter() - start  # noqa: E402
        print(f"\n{cat} optimization took {elapsed:.1f}s", flush=True)  # noqa: E402
  # noqa: E402
        # Save per-category program  # noqa: E402
        save_path = CATEGORY_SAVE_DIR / f"optimized_{cat}.json"  # noqa: E402
        compiled.save(str(save_path))  # noqa: E402
        print(f"Saved to {save_path}", flush=True)  # noqa: E402
  # noqa: E402
        # Evaluate  # noqa: E402
        print(f"\nEvaluating {cat}:", flush=True)  # noqa: E402
        rate, cat_details = evaluate_program(compiled, cat_examples)  # noqa: E402
        results[cat] = rate  # noqa: E402
        all_details[cat] = cat_details  # noqa: E402
        print(flush=True)  # noqa: E402
  # noqa: E402
    return results, all_details  # noqa: E402
  # noqa: E402
  # noqa: E402
def eval_per_category() -> tuple[dict[str, float], dict[str, list[EvalCaseResult]]]:  # noqa: E402
    """Evaluate all saved per-category programs."""  # noqa: E402
    categories = get_all_categories()  # noqa: E402
    results: dict[str, float] = {}  # noqa: E402
    all_details: dict[str, list[EvalCaseResult]] = {}  # noqa: E402
  # noqa: E402
    for cat in tqdm(categories, desc="Evaluating categories"):  # noqa: E402
        save_path = CATEGORY_SAVE_DIR / f"optimized_{cat}.json"  # noqa: E402
        if not save_path.exists():  # noqa: E402
            print(f"SKIP {cat}: no saved program at {save_path}", flush=True)  # noqa: E402
            continue  # noqa: E402
  # noqa: E402
        cat_examples = get_category_examples(cat)  # noqa: E402
        print(f"\n{'=' * 70}", flush=True)  # noqa: E402
        print(f"CATEGORY: {cat} ({len(cat_examples)} examples)", flush=True)  # noqa: E402
        print(f"{'=' * 70}", flush=True)  # noqa: E402
  # noqa: E402
        program = CategoryReplyModule(cat)  # noqa: E402
        program.load(str(save_path))  # noqa: E402
  # noqa: E402
        rate, cat_details = evaluate_program(program, cat_examples)  # noqa: E402
        results[cat] = rate  # noqa: E402
        all_details[cat] = cat_details  # noqa: E402
  # noqa: E402
    return results, all_details  # noqa: E402
  # noqa: E402
  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
# Main  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
  # noqa: E402
  # noqa: E402
def main() -> int:  # noqa: E402
    parser = argparse.ArgumentParser(description="DSPy optimization for JARVIS reply generation")  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--optimizer",  # noqa: E402
        choices=["bootstrap", "mipro"],  # noqa: E402
        default="bootstrap",  # noqa: E402
        help="Optimization strategy (default: bootstrap)",  # noqa: E402
    )  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--per-category",  # noqa: E402
        action="store_true",  # noqa: E402
        help="Run per-category optimization instead of global",  # noqa: E402
    )  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--eval-only",  # noqa: E402
        action="store_true",  # noqa: E402
        help="Skip optimization, just evaluate saved program(s)",  # noqa: E402
    )  # noqa: E402
    args = parser.parse_args()  # noqa: E402
  # noqa: E402
    # Setup logging  # noqa: E402
    log_path = PROJECT_ROOT / "results" / "dspy_optimize.log"  # noqa: E402
    log_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E402
    logging.basicConfig(  # noqa: E402
        level=logging.INFO,  # noqa: E402
        format="%(asctime)s - %(levelname)s - %(message)s",  # noqa: E402
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],  # noqa: E402
    )  # noqa: E402
    logging.getLogger(__name__)  # noqa: E402
  # noqa: E402
    print("=" * 70, flush=True)  # noqa: E402
    print("JARVIS DSPy Optimization Pipeline", flush=True)  # noqa: E402
    print("=" * 70, flush=True)  # noqa: E402
  # noqa: E402
    # Build student LM and configure as default  # noqa: E402
    student_lm = build_student_lm()  # noqa: E402
    dspy.configure(lm=student_lm)  # noqa: E402
  # noqa: E402
    trainset = TRAIN_EXAMPLES  # noqa: E402
    print(f"Training examples: {len(trainset)}", flush=True)  # noqa: E402
    print(f"Categories: {', '.join(get_all_categories())}", flush=True)  # noqa: E402
  # noqa: E402
    # --- Eval-only mode ---  # noqa: E402
    if args.eval_only:  # noqa: E402
        if args.per_category:  # noqa: E402
            print("\nEvaluating per-category compiled programs:", flush=True)  # noqa: E402
            results, all_details = eval_per_category()  # noqa: E402
            if results:  # noqa: E402
                print("\n" + "=" * 70, flush=True)  # noqa: E402
                print("PER-CATEGORY SUMMARY", flush=True)  # noqa: E402
                print("=" * 70, flush=True)  # noqa: E402
                for cat, rate in sorted(results.items()):  # noqa: E402
                    print(f"  {cat:20s}  {rate:.0%}", flush=True)  # noqa: E402
                avg = sum(results.values()) / len(results)  # noqa: E402
                print(f"  {'AVERAGE':20s}  {avg:.0%}", flush=True)  # noqa: E402
            return 0  # noqa: E402
        else:  # noqa: E402
            if not SAVE_DIR.exists():  # noqa: E402
                print(f"ERROR: No saved program at {SAVE_DIR}", flush=True)  # noqa: E402
                return 1  # noqa: E402
            print(f"\nLoading compiled program from {SAVE_DIR}", flush=True)  # noqa: E402
            program = ReplyModule()  # noqa: E402
            program.load(str(SAVE_DIR))  # noqa: E402
            print("\nEvaluating compiled program:", flush=True)  # noqa: E402
            evaluate_program(program, trainset)  # results printed inline  # noqa: E402
            return 0  # noqa: E402
  # noqa: E402
    # --- Optimization mode ---  # noqa: E402
    teacher_lm = build_teacher_lm()  # noqa: E402
    print("Teacher: ZAI GLM 4.7 via Cerebras", flush=True)  # noqa: E402
    print("Student: MLX local 1.2B", flush=True)  # noqa: E402
    print(flush=True)  # noqa: E402
  # noqa: E402
    if args.per_category:  # noqa: E402
        # Per-category optimization (default: mipro for per-category)  # noqa: E402
        optimizer_name = args.optimizer if args.optimizer != "bootstrap" else "mipro"  # noqa: E402
        results, all_details = run_per_category(teacher_lm, optimizer_name)  # noqa: E402
  # noqa: E402
        print("\n" + "=" * 70, flush=True)  # noqa: E402
        print("PER-CATEGORY RESULTS", flush=True)  # noqa: E402
        print("=" * 70, flush=True)  # noqa: E402
        for cat, rate in sorted(results.items()):  # noqa: E402
            print(f"  {cat:20s}  {rate:.0%}", flush=True)  # noqa: E402
        if results:  # noqa: E402
            avg = sum(results.values()) / len(results)  # noqa: E402
            print(f"  {'AVERAGE':20s}  {avg:.0%}", flush=True)  # noqa: E402
  # noqa: E402
        # Save detailed summary with per-case logs  # noqa: E402
        summary_path = CATEGORY_SAVE_DIR / "summary.json"  # noqa: E402
        detail_data: dict = {}  # noqa: E402
        for cat, cases in all_details.items():  # noqa: E402
            detail_data[cat] = [  # noqa: E402
                {  # noqa: E402
                    "query": c.query,  # noqa: E402
                    "raw_reply": c.raw_reply,  # noqa: E402
                    "cleaned_reply": c.cleaned_reply,  # noqa: E402
                    "passed": c.passed,  # noqa: E402
                    "has_artifacts": c.has_artifacts,  # noqa: E402
                    "error": c.error,  # noqa: E402
                }  # noqa: E402
                for c in cases  # noqa: E402
            ]  # noqa: E402
        summary_path.write_text(  # noqa: E402
            json.dumps(  # noqa: E402
                {  # noqa: E402
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),  # noqa: E402
                    "optimizer": optimizer_name,  # noqa: E402
                    "temperature": 0.1,  # noqa: E402
                    "per_category_pass_rates": results,  # noqa: E402
                    "per_case_details": detail_data,  # noqa: E402
                },  # noqa: E402
                indent=2,  # noqa: E402
            )  # noqa: E402
        )  # noqa: E402
        print(f"\nSummary saved to {summary_path}", flush=True)  # noqa: E402
    else:  # noqa: E402
        # Global optimization (original behavior)  # noqa: E402
        start = time.perf_counter()  # noqa: E402
        student = ReplyModule()  # noqa: E402
  # noqa: E402
        if args.optimizer == "mipro":  # noqa: E402
            compiled = run_mipro(student, trainset, teacher_lm)  # noqa: E402
        else:  # noqa: E402
            compiled = run_bootstrap(student, trainset, teacher_lm)  # noqa: E402
  # noqa: E402
        elapsed = time.perf_counter() - start  # noqa: E402
        print(f"\nOptimization took {elapsed:.1f}s", flush=True)  # noqa: E402
  # noqa: E402
        # Save  # noqa: E402
        SAVE_DIR.parent.mkdir(parents=True, exist_ok=True)  # noqa: E402
        compiled.save(str(SAVE_DIR))  # noqa: E402
        print(f"Saved compiled program to {SAVE_DIR}", flush=True)  # noqa: E402
  # noqa: E402
        # Evaluate the compiled program  # noqa: E402
        print("\n" + "-" * 70, flush=True)  # noqa: E402
        print("Evaluating compiled program:", flush=True)  # noqa: E402
        evaluate_program(compiled, trainset)  # noqa: E402
  # noqa: E402
    print("=" * 70, flush=True)  # noqa: E402
    print("Done. Run batch_eval with --optimized to compare:", flush=True)  # noqa: E402
    print("  uv run python evals/batch_eval.py --judge --optimized", flush=True)  # noqa: E402
    return 0  # noqa: E402
  # noqa: E402
  # noqa: E402
if __name__ == "__main__":  # noqa: E402
    sys.exit(main())  # noqa: E402
