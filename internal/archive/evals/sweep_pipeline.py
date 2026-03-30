#!/usr/bin/env python3
"""Pipeline Sweep: Systematically test context depths and optimized prompts.

Uses GPT-OSS-120B to generate an 'optimal' instruction for each depth,
then runs evaluation on the student model.
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evals.dspy_reply import TRAIN_EXAMPLES, clean_reply, judge_metric  # noqa: E402  # noqa: E402
from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402  # noqa: E402

from models.loader import get_model  # noqa: E402  # noqa: E402


def generate_optimized_instruction(depth: int, judge_client):
    """Ask GPT-OSS-120B to write the perfect instruction for a specific context depth."""
    prompt = (
        f"You are an expert prompt engineer. We are building a text message reply assistant. "
        f"The model has access to the last {depth} messages of context. "
        f"The model is a small local 0.7B/1.2B parameter model. "
        f"Write a concise system instruction that tells the model how to reply. "
        f"It must match the user's style, be brief, casual, and sound like a human, not an AI. "
        f"Avoid all AI filler words. Respond ONLY with the instruction text itself."
    )

    resp = judge_client.chat.completions.create(
        model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.7
    )
    return resp.choices[0].message.content.strip()


def run_eval(
    depth: int, instruction: str, examples: list, student_loader, rep_penalty: float = 1.05
):
    """Evaluate a specific depth/instruction combo with ChatML."""
    scores = []

    for ex in tqdm(examples, desc=f"Eval D={depth} RP={rep_penalty}"):
        ctx = ex.context[-depth:] if isinstance(ex.context, list) else []
        ctx_str = "\n".join(ctx)

        # PROPER LIQUID AI CHATML FORMAT
        prompt = (
            f"<|im_start|>system\n{instruction}<|im_end|>\n"
            f"<|im_start|>user\nContext:\n{ctx_str}\n\nLast Message: {ex.last_message}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        result = student_loader.generate_sync(
            prompt=prompt,
            max_tokens=50,
            temperature=0.1,
            repetition_penalty=rep_penalty,
            top_p=0.9,
            top_k=40,
        )
        reply = clean_reply(result.text)

        from dspy import Prediction

        pred = Prediction(reply=reply)

        score = judge_metric(ex, pred)
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", type=str, default="3,5")
    parser.add_argument("--penalties", type=str, default="1.0,1.05,1.1")
    args = parser.parse_args()

    depths = [int(d.strip()) for d in args.depths.split(",")]
    penalties = [float(p.strip()) for p in args.penalties.split(",")]

    judge_client = get_judge_client()
    student_loader = get_model()
    if not student_loader.is_loaded():
        student_loader.load()

    print("=" * 70)
    print("PIPELINE SWEEP: ChatML + Repetition Penalty")
    print("=" * 70)

    results = []

    for depth in depths:
        instr = generate_optimized_instruction(depth, judge_client)

        for rp in penalties:
            print(f"\n>>> TESTING DEPTH: {depth} | RP: {rp}")
            avg_score = run_eval(depth, instr, TRAIN_EXAMPLES, student_loader, rp)
            print(f"Average Score: {avg_score:.3f}")

            results.append(
                {"depth": depth, "repetition_penalty": rp, "instruction": instr, "score": avg_score}
            )

    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    results.sort(key=lambda x: x["score"], reverse=True)
    for r in results:
        print(f"Depth {r['depth']}: {r['score']:.3f}")

    winner = results[0]
    print(f"\n🏆 WINNER: Depth {winner['depth']} with score {winner['score']:.3f}")

    # Save winner
    output_path = Path("evals/optimized_pipeline_config.json")
    with open(output_path, "w") as f:
        json.dump(winner, f, indent=2)
    print(f"Config saved to {output_path}")


if __name__ == "__main__":
    main()
