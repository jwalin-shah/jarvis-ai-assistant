#!/usr/bin/env python3  # noqa: E501
"""DSPy optimization: compile better prompts + few-shot examples.  # noqa: E501
  # noqa: E501
Supports two modes:  # noqa: E501
1. Global: optimize a single program across all test cases (original behavior)  # noqa: E501
2. Per-category: optimize separate programs for each category (MIPRO v2)  # noqa: E501
  # noqa: E501
Uses ZAI GLM 4.7 via Cerebras as teacher to bootstrap demonstrations for the  # noqa: E501
local MLX 1.2B student model, then evaluates with the same judge.  # noqa: E501
  # noqa: E501
Usage:  # noqa: E501
    uv run python evals/dspy_optimize.py                         # BootstrapFewShot (global)  # noqa: E501
    uv run python evals/dspy_optimize.py --optimizer mipro       # MIPROv2 (global)  # noqa: E501
    uv run python evals/dspy_optimize.py --per-category          # Per-category MIPROv2  # noqa: E501
    uv run python evals/dspy_optimize.py --eval-only             # Evaluate saved global program  # noqa: E501
    uv run python evals/dspy_optimize.py --eval-only --per-category  # Eval all categories  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from __future__ import annotations  # noqa: E402  # noqa: E501

# noqa: E501
import argparse  # noqa: E501
import json  # noqa: E501
import logging  # noqa: E501
import os  # noqa: E501
import sys  # noqa: E501
import time  # noqa: E501
from dataclasses import dataclass  # noqa: E402  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501

# noqa: E501
from tqdm import tqdm  # noqa: E402  # noqa: E501

  # noqa: E501
PROJECT_ROOT = Path(__file__).parent.parent  # noqa: E501
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E501
  # noqa: E501
# Load .env  # noqa: E501
_env_path = PROJECT_ROOT / ".env"  # noqa: E501
if _env_path.exists():  # noqa: E501
    for line in _env_path.read_text().splitlines():  # noqa: E501
        line = line.strip()  # noqa: E501
        if line and not line.startswith("#") and "=" in line:  # noqa: E501
            key, _, val = line.partition("=")  # noqa: E501
            os.environ.setdefault(key.strip(), val.strip())  # noqa: E501
  # noqa: E501
import dspy  # noqa: E402  # noqa: E501
from evals.dspy_client import DSPYMLXClient  # noqa: E402  # noqa: E501
from evals.dspy_reply import (  # noqa: E402  # noqa: E501
    TRAIN_EXAMPLES,  # noqa: E501
    CategoryReplyModule,  # noqa: E501
    ReplyModule,  # noqa: E501
    clean_reply,  # noqa: E501
    get_all_categories,  # noqa: E501
    get_category_examples,  # noqa: E501
    judge_metric,  # noqa: E501
)  # noqa: E501

  # noqa: E501
# Save paths  # noqa: E501
SAVE_DIR = PROJECT_ROOT / "evals" / "optimized_reply.json"  # noqa: E501
CATEGORY_SAVE_DIR = PROJECT_ROOT / "evals" / "optimized_categories"  # noqa: E501
  # noqa: E501
  # noqa: E501
def build_teacher_lm() -> dspy.LM:  # noqa: E501
    """Cerebras ZAI GLM 4.7 as the teacher/demo generator."""  # noqa: E501
    from evals.judge_config import JUDGE_BASE_URL, JUDGE_MODEL, get_judge_api_key  # noqa: E501
  # noqa: E501
    key = get_judge_api_key()  # noqa: E501
    return dspy.LM(  # noqa: E501
        model=f"openai/{JUDGE_MODEL}",  # noqa: E501
        api_base=JUDGE_BASE_URL,  # noqa: E501
        api_key=key,  # noqa: E501
        temperature=0.7,  # noqa: E501
        max_tokens=300,  # noqa: E501
    )  # noqa: E501
  # noqa: E501
  # noqa: E501
def build_student_lm() -> DSPYMLXClient:  # noqa: E501
    """Local MLX 1.2B as the student model."""  # noqa: E501
    return DSPYMLXClient(max_tokens=50, temperature=0.1)  # noqa: E501
  # noqa: E501
  # noqa: E501
def run_bootstrap(  # noqa: E501
    student: ReplyModule,  # noqa: E501
    trainset: list[dspy.Example],  # noqa: E501
    teacher_lm: dspy.LM,  # noqa: E501
) -> ReplyModule:  # noqa: E501
    """BootstrapFewShot: fast, bootstraps few-shot demos from teacher."""  # noqa: E501
    print("Optimizer: BootstrapFewShot", flush=True)  # noqa: E501
    print("  max_bootstrapped_demos=3, max_labeled_demos=4", flush=True)  # noqa: E501
    print(f"  trainset size: {len(trainset)}", flush=True)  # noqa: E501
  # noqa: E501
    optimizer = dspy.BootstrapFewShot(  # noqa: E501
        metric=judge_metric,  # noqa: E501
        max_bootstrapped_demos=3,  # noqa: E501
        max_labeled_demos=4,  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    with dspy.context(lm=teacher_lm):  # noqa: E501
        compiled = optimizer.compile(student=student, trainset=trainset)  # noqa: E501
  # noqa: E501
    return compiled  # noqa: E501
  # noqa: E501
  # noqa: E501
def run_mipro(  # noqa: E501
    student: dspy.Module,  # noqa: E501
    trainset: list[dspy.Example],  # noqa: E501
    teacher_lm: dspy.LM,  # noqa: E501
    num_candidates: int = 5,  # noqa: E501
    num_trials: int = 15,  # noqa: E501
    max_bootstrapped_demos: int = 3,  # noqa: E501
) -> dspy.Module:  # noqa: E501
    """MIPROv2: optimizes instruction text + few-shot demos."""  # noqa: E501
    print("Optimizer: MIPROv2", flush=True)  # noqa: E501
    print(  # noqa: E501
        f"  num_candidates={num_candidates}, num_trials={num_trials}, "  # noqa: E501
        f"max_bootstrapped_demos={max_bootstrapped_demos}",  # noqa: E501
        flush=True,  # noqa: E501
    )  # noqa: E501
    print(f"  trainset size: {len(trainset)}", flush=True)  # noqa: E501
  # noqa: E501
    optimizer = dspy.MIPROv2(  # noqa: E501
        metric=judge_metric,  # noqa: E501
        prompt_model=teacher_lm,  # noqa: E501
        auto=None,  # noqa: E501
        num_candidates=num_candidates,  # noqa: E501
        max_bootstrapped_demos=max_bootstrapped_demos,  # noqa: E501
        max_labeled_demos=4,  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    compiled = optimizer.compile(  # noqa: E501
        student=student,  # noqa: E501
        trainset=trainset,  # noqa: E501
        num_trials=num_trials,  # noqa: E501
        minibatch=False,  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    return compiled  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class EvalCaseResult:  # noqa: E501
    """Result of evaluating a single test case."""  # noqa: E501
  # noqa: E501
    query: str  # noqa: E501
    raw_reply: str  # noqa: E501
    cleaned_reply: str  # noqa: E501
    score: float  # Continuous 0.0-1.0 score from judge  # noqa: E501
    passed: bool  # score >= 0.7  # noqa: E501
    has_artifacts: bool  # True if raw != cleaned (DSPy leak detected)  # noqa: E501
    error: str | None = None  # noqa: E501
  # noqa: E501
  # noqa: E501
def evaluate_program(  # noqa: E501
    program: dspy.Module, trainset: list[dspy.Example]  # noqa: E501
) -> tuple[float, list[EvalCaseResult]]:  # noqa: E501
    """Run the metric on all examples, return (avg score, detailed results).  # noqa: E501
  # noqa: E501
    Uses continuous 0-1 scoring for better signal. Reports both average score  # noqa: E501
    and pass rate (>= 0.7 threshold).  # noqa: E501
    """  # noqa: E501
    scores: list[float] = []  # noqa: E501
    details: list[EvalCaseResult] = []  # noqa: E501
    for ex in tqdm(trainset, desc="Evaluating"):  # noqa: E501
        try:  # noqa: E501
            pred = program(**{k: ex[k] for k in ["context", "last_message", "tone", "user_style"]})  # noqa: E501
            raw = pred.reply.strip()  # noqa: E501
            cleaned = clean_reply(pred.reply)  # noqa: E501
            has_artifacts = raw != cleaned  # noqa: E501
            score = judge_metric(ex, pred)  # noqa: E501
            scores.append(score)  # noqa: E501
            passed = score >= 0.7  # noqa: E501
            status = f"{score * 10:.0f}/10"  # noqa: E501
            artifact_tag = " [ARTIFACT]" if has_artifacts else ""  # noqa: E501
            print(f"  {status}: {ex.last_message[:40]!r} -> {raw!r}{artifact_tag}", flush=True)  # noqa: E501
            if has_artifacts:  # noqa: E501
                print(f"         cleaned: {cleaned!r}", flush=True)  # noqa: E501
            details.append(  # noqa: E501
                EvalCaseResult(  # noqa: E501
                    query=ex.last_message,  # noqa: E501
                    raw_reply=raw,  # noqa: E501
                    cleaned_reply=cleaned,  # noqa: E501
                    score=score,  # noqa: E501
                    passed=passed,  # noqa: E501
                    has_artifacts=has_artifacts,  # noqa: E501
                )  # noqa: E501
            )  # noqa: E501
        except Exception as e:  # noqa: E501
            print(f"  ERROR: {ex.last_message[:40]!r} -> {e}", flush=True)  # noqa: E501
            scores.append(0.0)  # noqa: E501
            details.append(  # noqa: E501
                EvalCaseResult(  # noqa: E501
                    query=ex.last_message,  # noqa: E501
                    raw_reply="",  # noqa: E501
                    cleaned_reply="",  # noqa: E501
                    score=0.0,  # noqa: E501
                    passed=False,  # noqa: E501
                    has_artifacts=False,  # noqa: E501
                    error=str(e),  # noqa: E501
                )  # noqa: E501
            )  # noqa: E501
    avg_score = sum(scores) / len(scores) if scores else 0  # noqa: E501
    n_passed = sum(1 for d in details if d.passed)  # noqa: E501
    artifact_fails = sum(1 for d in details if not d.passed and d.has_artifacts)  # noqa: E501
    print(f"\nAvg score: {avg_score * 10:.1f}/10", flush=True)  # noqa: E501
    print(  # noqa: E501
        f"Pass rate (>=7): {n_passed}/{len(trainset)} ({n_passed / len(trainset) * 100:.0f}%)",  # noqa: E501
        flush=True,  # noqa: E501
    )  # noqa: E501
    if artifact_fails:  # noqa: E501
        print(  # noqa: E501
            f"  ({artifact_fails} failures had DSPy artifacts - fixable with post-processing)",  # noqa: E501
            flush=True,  # noqa: E501
        )  # noqa: E501
    return avg_score, details  # noqa: E501
  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# Per-category optimization  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
  # noqa: E501
  # noqa: E501
def run_per_category(  # noqa: E501
    teacher_lm: dspy.LM,  # noqa: E501
    optimizer_name: str = "mipro",  # noqa: E501
) -> dict[str, float]:  # noqa: E501
    """Run optimization separately for each category.  # noqa: E501
  # noqa: E501
    Returns dict of category -> pass rate.  # noqa: E501
    """  # noqa: E501
    categories = get_all_categories()  # noqa: E501
    print(f"\nRunning per-category optimization for: {', '.join(categories)}", flush=True)  # noqa: E501
    print(f"Optimizer: {optimizer_name}", flush=True)  # noqa: E501
    print(flush=True)  # noqa: E501
  # noqa: E501
    CATEGORY_SAVE_DIR.mkdir(parents=True, exist_ok=True)  # noqa: E501
    results: dict[str, float] = {}  # noqa: E501
    all_details: dict[str, list[EvalCaseResult]] = {}  # noqa: E501
  # noqa: E501
    for cat in tqdm(categories, desc="Categories"):  # noqa: E501
        cat_examples = get_category_examples(cat)  # noqa: E501
        if len(cat_examples) < 3:  # noqa: E501
            print(f"SKIP {cat}: only {len(cat_examples)} examples (need >= 3)", flush=True)  # noqa: E501
            continue  # noqa: E501
  # noqa: E501
        print("=" * 70, flush=True)  # noqa: E501
        print(f"CATEGORY: {cat} ({len(cat_examples)} examples)", flush=True)  # noqa: E501
        print("=" * 70, flush=True)  # noqa: E501
  # noqa: E501
        student = CategoryReplyModule(cat)  # noqa: E501
        start = time.perf_counter()  # noqa: E501
  # noqa: E501
        if optimizer_name == "mipro":  # noqa: E501
            compiled = run_mipro(  # noqa: E501
                student=student,  # noqa: E501
                trainset=cat_examples,  # noqa: E501
                teacher_lm=teacher_lm,  # noqa: E501
                num_candidates=5,  # noqa: E501
                num_trials=15,  # noqa: E501
                max_bootstrapped_demos=3,  # noqa: E501
            )  # noqa: E501
        else:  # noqa: E501
            compiled = run_bootstrap(student, cat_examples, teacher_lm)  # noqa: E501
  # noqa: E501
        elapsed = time.perf_counter() - start  # noqa: E501
        print(f"\n{cat} optimization took {elapsed:.1f}s", flush=True)  # noqa: E501
  # noqa: E501
        # Save per-category program  # noqa: E501
        save_path = CATEGORY_SAVE_DIR / f"optimized_{cat}.json"  # noqa: E501
        compiled.save(str(save_path))  # noqa: E501
        print(f"Saved to {save_path}", flush=True)  # noqa: E501
  # noqa: E501
        # Evaluate  # noqa: E501
        print(f"\nEvaluating {cat}:", flush=True)  # noqa: E501
        rate, cat_details = evaluate_program(compiled, cat_examples)  # noqa: E501
        results[cat] = rate  # noqa: E501
        all_details[cat] = cat_details  # noqa: E501
        print(flush=True)  # noqa: E501
  # noqa: E501
    return results, all_details  # noqa: E501
  # noqa: E501
  # noqa: E501
def eval_per_category() -> tuple[dict[str, float], dict[str, list[EvalCaseResult]]]:  # noqa: E501
    """Evaluate all saved per-category programs."""  # noqa: E501
    categories = get_all_categories()  # noqa: E501
    results: dict[str, float] = {}  # noqa: E501
    all_details: dict[str, list[EvalCaseResult]] = {}  # noqa: E501
  # noqa: E501
    for cat in tqdm(categories, desc="Evaluating categories"):  # noqa: E501
        save_path = CATEGORY_SAVE_DIR / f"optimized_{cat}.json"  # noqa: E501
        if not save_path.exists():  # noqa: E501
            print(f"SKIP {cat}: no saved program at {save_path}", flush=True)  # noqa: E501
            continue  # noqa: E501
  # noqa: E501
        cat_examples = get_category_examples(cat)  # noqa: E501
        print(f"\n{'=' * 70}", flush=True)  # noqa: E501
        print(f"CATEGORY: {cat} ({len(cat_examples)} examples)", flush=True)  # noqa: E501
        print(f"{'=' * 70}", flush=True)  # noqa: E501
  # noqa: E501
        program = CategoryReplyModule(cat)  # noqa: E501
        program.load(str(save_path))  # noqa: E501
  # noqa: E501
        rate, cat_details = evaluate_program(program, cat_examples)  # noqa: E501
        results[cat] = rate  # noqa: E501
        all_details[cat] = cat_details  # noqa: E501
  # noqa: E501
    return results, all_details  # noqa: E501
  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# Main  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
  # noqa: E501
  # noqa: E501
def main() -> int:  # noqa: E501
    parser = argparse.ArgumentParser(description="DSPy optimization for JARVIS reply generation")  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--optimizer",  # noqa: E501
        choices=["bootstrap", "mipro"],  # noqa: E501
        default="bootstrap",  # noqa: E501
        help="Optimization strategy (default: bootstrap)",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--per-category",  # noqa: E501
        action="store_true",  # noqa: E501
        help="Run per-category optimization instead of global",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--eval-only",  # noqa: E501
        action="store_true",  # noqa: E501
        help="Skip optimization, just evaluate saved program(s)",  # noqa: E501
    )  # noqa: E501
    args = parser.parse_args()  # noqa: E501
  # noqa: E501
    # Setup logging  # noqa: E501
    log_path = PROJECT_ROOT / "results" / "dspy_optimize.log"  # noqa: E501
    log_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E501
    logging.basicConfig(  # noqa: E501
        level=logging.INFO,  # noqa: E501
        format="%(asctime)s - %(levelname)s - %(message)s",  # noqa: E501
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],  # noqa: E501
    )  # noqa: E501
    logging.getLogger(__name__)  # noqa: E501
  # noqa: E501
    print("=" * 70, flush=True)  # noqa: E501
    print("JARVIS DSPy Optimization Pipeline", flush=True)  # noqa: E501
    print("=" * 70, flush=True)  # noqa: E501
  # noqa: E501
    # Build student LM and configure as default  # noqa: E501
    student_lm = build_student_lm()  # noqa: E501
    dspy.configure(lm=student_lm)  # noqa: E501
  # noqa: E501
    trainset = TRAIN_EXAMPLES  # noqa: E501
    print(f"Training examples: {len(trainset)}", flush=True)  # noqa: E501
    print(f"Categories: {', '.join(get_all_categories())}", flush=True)  # noqa: E501
  # noqa: E501
    # --- Eval-only mode ---  # noqa: E501
    if args.eval_only:  # noqa: E501
        if args.per_category:  # noqa: E501
            print("\nEvaluating per-category compiled programs:", flush=True)  # noqa: E501
            results, all_details = eval_per_category()  # noqa: E501
            if results:  # noqa: E501
                print("\n" + "=" * 70, flush=True)  # noqa: E501
                print("PER-CATEGORY SUMMARY", flush=True)  # noqa: E501
                print("=" * 70, flush=True)  # noqa: E501
                for cat, rate in sorted(results.items()):  # noqa: E501
                    print(f"  {cat:20s}  {rate:.0%}", flush=True)  # noqa: E501
                avg = sum(results.values()) / len(results)  # noqa: E501
                print(f"  {'AVERAGE':20s}  {avg:.0%}", flush=True)  # noqa: E501
            return 0  # noqa: E501
        else:  # noqa: E501
            if not SAVE_DIR.exists():  # noqa: E501
                print(f"ERROR: No saved program at {SAVE_DIR}", flush=True)  # noqa: E501
                return 1  # noqa: E501
            print(f"\nLoading compiled program from {SAVE_DIR}", flush=True)  # noqa: E501
            program = ReplyModule()  # noqa: E501
            program.load(str(SAVE_DIR))  # noqa: E501
            print("\nEvaluating compiled program:", flush=True)  # noqa: E501
            evaluate_program(program, trainset)  # results printed inline  # noqa: E501
            return 0  # noqa: E501
  # noqa: E501
    # --- Optimization mode ---  # noqa: E501
    teacher_lm = build_teacher_lm()  # noqa: E501
    print("Teacher: ZAI GLM 4.7 via Cerebras", flush=True)  # noqa: E501
    print("Student: MLX local 1.2B", flush=True)  # noqa: E501
    print(flush=True)  # noqa: E501
  # noqa: E501
    if args.per_category:  # noqa: E501
        # Per-category optimization (default: mipro for per-category)  # noqa: E501
        optimizer_name = args.optimizer if args.optimizer != "bootstrap" else "mipro"  # noqa: E501
        results, all_details = run_per_category(teacher_lm, optimizer_name)  # noqa: E501
  # noqa: E501
        print("\n" + "=" * 70, flush=True)  # noqa: E501
        print("PER-CATEGORY RESULTS", flush=True)  # noqa: E501
        print("=" * 70, flush=True)  # noqa: E501
        for cat, rate in sorted(results.items()):  # noqa: E501
            print(f"  {cat:20s}  {rate:.0%}", flush=True)  # noqa: E501
        if results:  # noqa: E501
            avg = sum(results.values()) / len(results)  # noqa: E501
            print(f"  {'AVERAGE':20s}  {avg:.0%}", flush=True)  # noqa: E501
  # noqa: E501
        # Save detailed summary with per-case logs  # noqa: E501
        summary_path = CATEGORY_SAVE_DIR / "summary.json"  # noqa: E501
        detail_data: dict = {}  # noqa: E501
        for cat, cases in all_details.items():  # noqa: E501
            detail_data[cat] = [  # noqa: E501
                {  # noqa: E501
                    "query": c.query,  # noqa: E501
                    "raw_reply": c.raw_reply,  # noqa: E501
                    "cleaned_reply": c.cleaned_reply,  # noqa: E501
                    "passed": c.passed,  # noqa: E501
                    "has_artifacts": c.has_artifacts,  # noqa: E501
                    "error": c.error,  # noqa: E501
                }  # noqa: E501
                for c in cases  # noqa: E501
            ]  # noqa: E501
        summary_path.write_text(  # noqa: E501
            json.dumps(  # noqa: E501
                {  # noqa: E501
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),  # noqa: E501
                    "optimizer": optimizer_name,  # noqa: E501
                    "temperature": 0.1,  # noqa: E501
                    "per_category_pass_rates": results,  # noqa: E501
                    "per_case_details": detail_data,  # noqa: E501
                },  # noqa: E501
                indent=2,  # noqa: E501
            )  # noqa: E501
        )  # noqa: E501
        print(f"\nSummary saved to {summary_path}", flush=True)  # noqa: E501
    else:  # noqa: E501
        # Global optimization (original behavior)  # noqa: E501
        start = time.perf_counter()  # noqa: E501
        student = ReplyModule()  # noqa: E501
  # noqa: E501
        if args.optimizer == "mipro":  # noqa: E501
            compiled = run_mipro(student, trainset, teacher_lm)  # noqa: E501
        else:  # noqa: E501
            compiled = run_bootstrap(student, trainset, teacher_lm)  # noqa: E501
  # noqa: E501
        elapsed = time.perf_counter() - start  # noqa: E501
        print(f"\nOptimization took {elapsed:.1f}s", flush=True)  # noqa: E501
  # noqa: E501
        # Save  # noqa: E501
        SAVE_DIR.parent.mkdir(parents=True, exist_ok=True)  # noqa: E501
        compiled.save(str(SAVE_DIR))  # noqa: E501
        print(f"Saved compiled program to {SAVE_DIR}", flush=True)  # noqa: E501
  # noqa: E501
        # Evaluate the compiled program  # noqa: E501
        print("\n" + "-" * 70, flush=True)  # noqa: E501
        print("Evaluating compiled program:", flush=True)  # noqa: E501
        evaluate_program(compiled, trainset)  # noqa: E501
  # noqa: E501
    print("=" * 70, flush=True)  # noqa: E501
    print("Done. Run batch_eval with --optimized to compare:", flush=True)  # noqa: E501
    print("  uv run python evals/batch_eval.py --judge --optimized", flush=True)  # noqa: E501
    return 0  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    sys.exit(main())  # noqa: E501
