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
  # noqa: E402
from evals.dspy_reply import TRAIN_EXAMPLES, clean_reply, judge_metric  # noqa: E402
from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402

# noqa: E402
from models.loader import get_model  # noqa: E402


  # noqa: E402
  # noqa: E402
def generate_optimized_instruction(depth: int, judge_client):  # noqa: E402
    """Ask GPT-OSS-120B to write the perfect instruction for a specific context depth."""  # noqa: E402
    prompt = (  # noqa: E402
        f"You are an expert prompt engineer. We are building a text message reply assistant. "  # noqa: E402
        f"The model has access to the last {depth} messages of context. "  # noqa: E402
        f"The model is a small local 0.7B/1.2B parameter model. "  # noqa: E402
        f"Write a concise system instruction that tells the model how to reply. "  # noqa: E402
        f"It must match the user's style, be brief, casual, and sound like a human, not an AI. "  # noqa: E402
        f"Avoid all AI filler words. Respond ONLY with the instruction text itself."  # noqa: E402
    )  # noqa: E402
  # noqa: E402
    resp = judge_client.chat.completions.create(  # noqa: E402
        model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.7  # noqa: E402
    )  # noqa: E402
    return resp.choices[0].message.content.strip()  # noqa: E402
  # noqa: E402
  # noqa: E402
def run_eval(  # noqa: E402
    depth: int, instruction: str, examples: list, student_loader, rep_penalty: float = 1.05  # noqa: E402
):  # noqa: E402
    """Evaluate a specific depth/instruction combo with ChatML."""  # noqa: E402
    scores = []  # noqa: E402
  # noqa: E402
    for ex in tqdm(examples, desc=f"Eval D={depth} RP={rep_penalty}"):  # noqa: E402
        ctx = ex.context[-depth:] if isinstance(ex.context, list) else []  # noqa: E402
        ctx_str = "\n".join(ctx)  # noqa: E402
  # noqa: E402
        # PROPER LIQUID AI CHATML FORMAT  # noqa: E402
        prompt = (  # noqa: E402
            f"<|im_start|>system\n{instruction}<|im_end|>\n"  # noqa: E402
            f"<|im_start|>user\nContext:\n{ctx_str}\n\nLast Message: {ex.last_message}<|im_end|>\n"  # noqa: E402
            f"<|im_start|>assistant\n"  # noqa: E402
        )  # noqa: E402
  # noqa: E402
        result = student_loader.generate_sync(  # noqa: E402
            prompt=prompt,  # noqa: E402
            max_tokens=50,  # noqa: E402
            temperature=0.1,  # noqa: E402
            repetition_penalty=rep_penalty,  # noqa: E402
            top_p=0.9,  # noqa: E402
            top_k=40,  # noqa: E402
        )  # noqa: E402
        reply = clean_reply(result.text)  # noqa: E402
  # noqa: E402
        from dspy import Prediction  # noqa: E402
  # noqa: E402
        pred = Prediction(reply=reply)  # noqa: E402
  # noqa: E402
        score = judge_metric(ex, pred)  # noqa: E402
        scores.append(score)  # noqa: E402
  # noqa: E402
    return sum(scores) / len(scores) if scores else 0  # noqa: E402
  # noqa: E402
  # noqa: E402
def main():  # noqa: E402
    parser = argparse.ArgumentParser()  # noqa: E402
    parser.add_argument("--depths", type=str, default="3,5")  # noqa: E402
    parser.add_argument("--penalties", type=str, default="1.0,1.05,1.1")  # noqa: E402
    args = parser.parse_args()  # noqa: E402
  # noqa: E402
    depths = [int(d.strip()) for d in args.depths.split(",")]  # noqa: E402
    penalties = [float(p.strip()) for p in args.penalties.split(",")]  # noqa: E402
  # noqa: E402
    judge_client = get_judge_client()  # noqa: E402
    student_loader = get_model()  # noqa: E402
    if not student_loader.is_loaded():  # noqa: E402
        student_loader.load()  # noqa: E402
  # noqa: E402
    print("=" * 70)  # noqa: E402
    print("PIPELINE SWEEP: ChatML + Repetition Penalty")  # noqa: E402
    print("=" * 70)  # noqa: E402
  # noqa: E402
    results = []  # noqa: E402
  # noqa: E402
    for depth in depths:  # noqa: E402
        instr = generate_optimized_instruction(depth, judge_client)  # noqa: E402
  # noqa: E402
        for rp in penalties:  # noqa: E402
            print(f"\n>>> TESTING DEPTH: {depth} | RP: {rp}")  # noqa: E402
            avg_score = run_eval(depth, instr, TRAIN_EXAMPLES, student_loader, rp)  # noqa: E402
            print(f"Average Score: {avg_score:.3f}")  # noqa: E402
  # noqa: E402
            results.append(  # noqa: E402
                {"depth": depth, "repetition_penalty": rp, "instruction": instr, "score": avg_score}  # noqa: E402
            )  # noqa: E402
  # noqa: E402
    # Summary  # noqa: E402
    print("\n" + "=" * 70)  # noqa: E402
    print("FINAL RESULTS")  # noqa: E402
    print("=" * 70)  # noqa: E402
    results.sort(key=lambda x: x["score"], reverse=True)  # noqa: E402
    for r in results:  # noqa: E402
        print(f"Depth {r['depth']}: {r['score']:.3f}")  # noqa: E402
  # noqa: E402
    winner = results[0]  # noqa: E402
    print(f"\n🏆 WINNER: Depth {winner['depth']} with score {winner['score']:.3f}")  # noqa: E402
  # noqa: E402
    # Save winner  # noqa: E402
    output_path = Path("evals/optimized_pipeline_config.json")  # noqa: E402
    with open(output_path, "w") as f:  # noqa: E402
        json.dump(winner, f, indent=2)  # noqa: E402
    print(f"Config saved to {output_path}")  # noqa: E402
  # noqa: E402
  # noqa: E402
if __name__ == "__main__":  # noqa: E402
    main()  # noqa: E402
