#!/usr/bin/env python3
"""DSPy optimization: compile better prompts + few-shot examples.

Uses Cerebras Llama 3.3 70B as teacher to bootstrap demonstrations for the
local MLX 1.2B student model, then evaluates with the same judge.

Usage:
    uv run python evals/dspy_optimize.py                    # BootstrapFewShot (fast)
    uv run python evals/dspy_optimize.py --optimizer mipro  # MIPROv2 (slower)
    uv run python evals/dspy_optimize.py --eval-only        # just evaluate a saved program
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

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

from evals.dspy_reply import (  # noqa: E402
    TRAIN_EXAMPLES,
    ReplyModule,
    cerebras_judge_metric,
)
from jarvis.dspy_client import DSPYMLXClient  # noqa: E402

SAVE_DIR = PROJECT_ROOT / "evals" / "optimized_reply"


def _get_cerebras_key() -> str:
    key = os.environ.get("CEREBRAS_API_KEY", "")
    if not key or key == "your-key-here":
        print("ERROR: CEREBRAS_API_KEY not set in .env")
        print("       Required for teacher model and judge scoring.")
        sys.exit(1)
    return key


def build_teacher_lm() -> dspy.LM:
    """Cerebras Llama 3.3 70B as the teacher/demo generator."""
    key = _get_cerebras_key()
    return dspy.LM(
        model="openai/llama-3.3-70b",
        api_base="https://api.cerebras.ai/v1",
        api_key=key,
        temperature=0.7,
        max_tokens=100,
    )


def build_student_lm() -> DSPYMLXClient:
    """Local MLX 1.2B as the student model."""
    return DSPYMLXClient(max_tokens=50, temperature=0.7)


def run_bootstrap(
    student: ReplyModule,
    trainset: list[dspy.Example],
    teacher_lm: dspy.LM,
) -> ReplyModule:
    """BootstrapFewShot: fast, bootstraps few-shot demos from teacher."""
    print("Optimizer: BootstrapFewShot")
    print("  max_bootstrapped_demos=3, max_labeled_demos=4")
    print(f"  trainset size: {len(trainset)}")

    optimizer = dspy.BootstrapFewShot(
        metric=cerebras_judge_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=4,
    )

    with dspy.context(lm=teacher_lm):
        compiled = optimizer.compile(student=student, trainset=trainset)

    return compiled


def run_mipro(
    student: ReplyModule,
    trainset: list[dspy.Example],
    teacher_lm: dspy.LM,
    student_lm: DSPYMLXClient,
) -> ReplyModule:
    """MIPROv2: slower, also optimizes the instruction text."""
    print("Optimizer: MIPROv2")
    print("  num_candidates=5, max_bootstrapped_demos=3")
    print(f"  trainset size: {len(trainset)}")

    optimizer = dspy.MIPROv2(
        metric=cerebras_judge_metric,
        num_candidates=5,
        max_bootstrapped_demos=3,
        max_labeled_demos=4,
    )

    compiled = optimizer.compile(
        student=student,
        trainset=trainset,
        teacher=teacher_lm,
        requires_permission_to_run=False,
    )

    return compiled


def evaluate_program(program: ReplyModule, trainset: list[dspy.Example]) -> float:
    """Run the metric on all examples, return pass rate."""
    passed = 0
    for ex in trainset:
        try:
            pred = program(**{k: ex[k] for k in ["context", "last_message", "tone", "user_style"]})
            if cerebras_judge_metric(ex, pred):
                passed += 1
                print(f"  PASS: {ex.last_message[:40]!r} -> {pred.reply!r}")
            else:
                print(f"  FAIL: {ex.last_message[:40]!r} -> {pred.reply!r}")
        except Exception as e:
            print(f"  ERROR: {ex.last_message[:40]!r} -> {e}")
    rate = passed / len(trainset) if trainset else 0
    print(f"\nPass rate: {passed}/{len(trainset)} ({rate:.0%})")
    return rate


def main() -> int:
    parser = argparse.ArgumentParser(description="DSPy optimization for JARVIS reply generation")
    parser.add_argument(
        "--optimizer",
        choices=["bootstrap", "mipro"],
        default="bootstrap",
        help="Optimization strategy (default: bootstrap)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip optimization, just evaluate saved program",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("JARVIS DSPy Optimization Pipeline")
    print("=" * 70)

    # Build student LM and configure as default
    student_lm = build_student_lm()
    dspy.configure(lm=student_lm)

    trainset = TRAIN_EXAMPLES
    print(f"Training examples: {len(trainset)}")

    if args.eval_only:
        if not SAVE_DIR.exists():
            print(f"ERROR: No saved program at {SAVE_DIR}")
            return 1
        print(f"\nLoading compiled program from {SAVE_DIR}")
        program = ReplyModule()
        program.load(str(SAVE_DIR))
        print("\nEvaluating compiled program:")
        evaluate_program(program, trainset)
        return 0

    # Build teacher
    teacher_lm = build_teacher_lm()
    print("Teacher: Cerebras llama-3.3-70b")
    print("Student: MLX local 1.2B")
    print()

    # Run optimization
    start = time.perf_counter()
    student = ReplyModule()

    if args.optimizer == "mipro":
        compiled = run_mipro(student, trainset, teacher_lm, student_lm)
    else:
        compiled = run_bootstrap(student, trainset, teacher_lm)

    elapsed = time.perf_counter() - start
    print(f"\nOptimization took {elapsed:.1f}s")

    # Save
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    compiled.save(str(SAVE_DIR))
    print(f"Saved compiled program to {SAVE_DIR}")

    # Evaluate the compiled program
    print("\n" + "-" * 70)
    print("Evaluating compiled program:")
    evaluate_program(compiled, trainset)

    print("=" * 70)
    print("Done. Run batch_eval with --optimized to compare:")
    print("  uv run python evals/batch_eval.py --judge --optimized")
    return 0


if __name__ == "__main__":
    sys.exit(main())
