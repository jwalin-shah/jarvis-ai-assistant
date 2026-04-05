#!/usr/bin/env python3
"""Pipeline optimization: Sweep through context depths and optimize prompts via MIPROv2.

This script find the best 'structural' variables (like context_depth)
AND the best 'textual' variables (instructions/demos) at the same time.

Usage:
    uv run python evals/optimize_universal_prompt.py --trials 10 --depths 3,5,10
"""

import argparse
import sys
from pathlib import Path

import dspy

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
  # noqa: E402
from evals.dspy_client import DSPYMLXClient  # noqa: E402
from evals.dspy_reply import (  # noqa: E402
    TRAIN_EXAMPLES,  # noqa: E402
    judge_metric,  # noqa: E402
)  # noqa: E402


  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
# Student Module  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
  # noqa: E402
  # noqa: E402
class SimpleTextAdapter(dspy.Adapter):  # noqa: E402
    """Bypasses JSON/formatting and just takes the raw LM output as the 'reply'."""  # noqa: E402
  # noqa: E402
    def __call__(self, lm, lm_kwargs, signature, demos, inputs):  # noqa: E402
        # Build a simple text prompt  # noqa: E402
        prompt = ""  # noqa: E402
        # Handle instruction if it exists in signature (MIPRO v2 puts it there)  # noqa: E402
        if hasattr(signature, "instructions"):  # noqa: E402
            prompt += f"{signature.instructions}\n\n"  # noqa: E402
  # noqa: E402
        for field_name, value in inputs.items():  # noqa: E402
            # Skip signature metadata  # noqa: E402
            if field_name in ["instructions"]:  # noqa: E402
                continue  # noqa: E402
            prompt += f"{field_name}: {value}\n"  # noqa: E402
        prompt += "reply:"  # noqa: E402
  # noqa: E402
        # Generate  # noqa: E402
        response = lm(prompt, **lm_kwargs)  # noqa: E402
        if isinstance(response, str):  # noqa: E402
            text = response  # noqa: E402
        elif hasattr(response, "choices"):  # noqa: E402
            text = (  # noqa: E402
                response.choices[0].message.content  # noqa: E402
                if hasattr(response.choices[0], "message")  # noqa: E402
                else response.choices[0].text  # noqa: E402
            )  # noqa: E402
        else:  # noqa: E402
            text = str(response)  # noqa: E402
  # noqa: E402
        # Strip potential label if model repeats it  # noqa: E402
        if text.lower().startswith("reply:"):  # noqa: E402
            text = text[6:].strip()  # noqa: E402
  # noqa: E402
        # Create prediction object matching signature  # noqa: E402
        return dspy.Prediction(reply=text)  # noqa: E402
  # noqa: E402
  # noqa: E402
class UniversalReplySignature(dspy.Signature):  # noqa: E402
    """Generate a natural, human-like text message reply."""  # noqa: E402
  # noqa: E402
    context: str = dspy.InputField(desc="Conversation history with timestamps")  # noqa: E402
    last_message: str = dspy.InputField(desc="The message to reply to")  # noqa: E402
    tone: str = dspy.InputField(desc="Tone: casual or professional")  # noqa: E402
    user_style: str = dspy.InputField(desc="User's texting style description")  # noqa: E402
    reply: str = dspy.OutputField(  # noqa: E402
        desc="Brief, natural reply matching the user's style. No AI filler."  # noqa: E402
    )  # noqa: E402
  # noqa: E402
  # noqa: E402
class UniversalReplyModule(dspy.Module):  # noqa: E402
    def __init__(self):  # noqa: E402
        super().__init__()  # noqa: E402
        self.generate = dspy.Predict(UniversalReplySignature)  # noqa: E402
  # noqa: E402
    def forward(self, **kwargs):  # noqa: E402
        return self.generate(**kwargs)  # noqa: E402
  # noqa: E402
  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
# Optimization Logic  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
  # noqa: E402
  # noqa: E402
def get_context_at_depth(full_context: list[str], depth: int) -> str:  # noqa: E402
    """Helper to format context for DSPy examples."""  # noqa: E402
    selected = full_context[-depth:] if len(full_context) >= depth else full_context  # noqa: E402
    return "\n".join(selected)  # noqa: E402
  # noqa: E402
  # noqa: E402
def prepare_trainset(depth: int):  # noqa: E402
    """Re-prepare the training set with the specified context depth."""  # noqa: E402
    new_trainset = []  # noqa: E402
    for ex in TRAIN_EXAMPLES:  # noqa: E402
        # TRAIN_EXAMPLES context is usually a list of strings  # noqa: E402
        if isinstance(ex.context, list):  # noqa: E402
            ctx_str = get_context_at_depth(ex.context, depth)  # noqa: E402
        else:  # noqa: E402
            ctx_str = ex.context  # noqa: E402
  # noqa: E402
        # Create new example with modified context  # noqa: E402
        new_ex = dspy.Example(  # noqa: E402
            context=ctx_str, last_message=ex.last_message, tone=ex.tone, user_style=ex.user_style  # noqa: E402
        ).with_inputs("context", "last_message", "tone", "user_style")  # noqa: E402
  # noqa: E402
        # Transfer metadata (needed for judge_metric)  # noqa: E402
        for attr in ["_rubric", "_max_words", "_max_chars", "_banned", "_category"]:  # noqa: E402
            if hasattr(ex, attr):  # noqa: E402
                setattr(new_ex, attr, getattr(ex, attr))  # noqa: E402
  # noqa: E402
        new_trainset.append(new_ex)  # noqa: E402
    return new_trainset  # noqa: E402
  # noqa: E402
  # noqa: E402
def build_teacher_lm():  # noqa: E402
    from evals.judge_config import JUDGE_BASE_URL, JUDGE_MODEL, get_judge_api_key  # noqa: E402
  # noqa: E402
    key = get_judge_api_key()  # noqa: E402
    # Teacher models (Cerebras) can handle JSON/Chat via ChatAdapter  # noqa: E402
    return dspy.LM(  # noqa: E402
        model=f"openai/{JUDGE_MODEL}",  # noqa: E402
        api_base=JUDGE_BASE_URL,  # noqa: E402
        api_key=key,  # noqa: E402
        temperature=0.7,  # noqa: E402
        max_tokens=500,  # noqa: E402
    )  # noqa: E402
  # noqa: E402
  # noqa: E402
def main():  # noqa: E402
    parser = argparse.ArgumentParser()  # noqa: E402
    parser.add_argument("--trials", type=int, default=15, help="MIPROv2 trials per depth")  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--depths", type=str, default="3,5,10", help="Comma-separated context depths to test"  # noqa: E402
    )  # noqa: E402
    parser.add_argument("--candidates", type=int, default=5, help="MIPROv2 candidates")  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--demos", type=int, default=3, help="Max bootstrapped demos (set to 0 to test no few-shot)"  # noqa: E402
    )  # noqa: E402
    args = parser.parse_args()  # noqa: E402
  # noqa: E402
    depths = [int(d.strip()) for d in args.depths.split(",")]  # noqa: E402
  # noqa: E402
    print("=" * 70)  # noqa: E402
    print("JARVIS FULL PIPELINE OPTIMIZATION (MIPROv2 + Context Sweep)")  # noqa: E402
    print(f"Teacher/Judge: {args.demos} bootstrapped demos")  # noqa: E402
    print("=" * 70)  # noqa: E402
  # noqa: E402
    teacher_lm = build_teacher_lm()  # noqa: E402
    # Configure teacher to use ChatAdapter for instruction generation/proposing  # noqa: E402
    dspy.configure(lm=teacher_lm, adapter=dspy.ChatAdapter())  # noqa: E402
  # noqa: E402
    student_lm = DSPYMLXClient(max_tokens=50, temperature=0.1)  # noqa: E402
    student_adapter = SimpleTextAdapter()  # noqa: E402
  # noqa: E402
    # We DON'T set a global adapter here, so student uses default "Field: Value" text  # noqa: E402
    # When we want to use the student, we'll use a local dspy.context  # noqa: E402
  # noqa: E402
    best_overall_score = -1.0  # noqa: E402
    best_config = None  # noqa: E402
    results = []  # noqa: E402
  # noqa: E402
    for depth in depths:  # noqa: E402
        print(f"\n>>> OPTIMIZING FOR CONTEXT_DEPTH: {depth}")  # noqa: E402
        trainset = prepare_trainset(depth)  # noqa: E402
  # noqa: E402
        student = UniversalReplyModule()  # noqa: E402
  # noqa: E402
        optimizer = dspy.MIPROv2(  # noqa: E402
            metric=judge_metric,  # noqa: E402
            prompt_model=teacher_lm,  # noqa: E402
            auto=None,  # noqa: E402
            num_candidates=args.candidates,  # noqa: E402
            max_bootstrapped_demos=args.demos,  # noqa: E402
            max_labeled_demos=4,  # noqa: E402
        )  # noqa: E402
  # noqa: E402
        with dspy.context(lm=student_lm, adapter=student_adapter):  # noqa: E402
            # Student evaluation runs here using SimpleTextAdapter  # noqa: E402
            compiled = optimizer.compile(  # noqa: E402
                student=student,  # noqa: E402
                trainset=trainset,  # noqa: E402
                num_trials=args.trials,  # noqa: E402
                minibatch=False,  # noqa: E402
            )  # noqa: E402
  # noqa: E402
        # Evaluate the compiled program for this depth  # noqa: E402
        scores = []  # noqa: E402
        with dspy.context(lm=student_lm, adapter=student_adapter):  # noqa: E402
            for ex in trainset:  # noqa: E402
                inputs = {  # noqa: E402
                    "context": ex.context,  # noqa: E402
                    "last_message": ex.last_message,  # noqa: E402
                    "tone": ex.tone,  # noqa: E402
                    "user_style": ex.user_style,  # noqa: E402
                }  # noqa: E402
                pred = compiled(**inputs)  # noqa: E402
                score = judge_metric(ex, pred)  # noqa: E402
                scores.append(score)  # noqa: E402
  # noqa: E402
        avg_score = sum(scores) / len(scores) if scores else 0  # noqa: E402
        print(f"Final Optimized Score for Depth {depth}: {avg_score:.3f}")  # noqa: E402
  # noqa: E402
        # Save results  # noqa: E402
        save_path = PROJECT_ROOT / "evals" / f"optimized_universal_d{depth}.json"  # noqa: E402
        compiled.save(str(save_path))  # noqa: E402
  # noqa: E402
        res = {"depth": depth, "avg_score": avg_score, "save_path": str(save_path)}  # noqa: E402
        results.append(res)  # noqa: E402
  # noqa: E402
        if avg_score > best_overall_score:  # noqa: E402
            best_overall_score = avg_score  # noqa: E402
            best_config = res  # noqa: E402
  # noqa: E402
    print("\n" + "=" * 70)  # noqa: E402
    print("OPTIMIZATION COMPLETE")  # noqa: E402
    print("=" * 70)  # noqa: E402
    for r in results:  # noqa: E402
        print(f"Depth {r['depth']}: {r['avg_score']:.3f}")  # noqa: E402
  # noqa: E402
    print(f"\n🏆 WINNER: Depth {best_config['depth']} with score {best_config['avg_score']:.3f}")  # noqa: E402
    print(f"Saved at: {best_config['save_path']}")  # noqa: E402
  # noqa: E402
  # noqa: E402
if __name__ == "__main__":  # noqa: E402
    main()  # noqa: E402
