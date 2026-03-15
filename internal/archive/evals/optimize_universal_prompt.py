#!/usr/bin/env python3  # noqa: E501
"""Pipeline optimization: Sweep through context depths and optimize prompts via MIPROv2.  # noqa: E501
  # noqa: E501
This script find the best 'structural' variables (like context_depth)  # noqa: E501
AND the best 'textual' variables (instructions/demos) at the same time.  # noqa: E501
  # noqa: E501
Usage:  # noqa: E501
    uv run python evals/optimize_universal_prompt.py --trials 10 --depths 3,5,10  # noqa: E501
"""  # noqa: E501
  # noqa: E501
import argparse  # noqa: E501
import sys  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501

# noqa: E501
import dspy  # noqa: E501

  # noqa: E501
PROJECT_ROOT = Path(__file__).parent.parent  # noqa: E501
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E501
  # noqa: E501
from evals.dspy_client import DSPYMLXClient  # noqa: E402  # noqa: E501
from evals.dspy_reply import (  # noqa: E402  # noqa: E501
    TRAIN_EXAMPLES,  # noqa: E501
    judge_metric,  # noqa: E501
)  # noqa: E501


  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# Student Module  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
  # noqa: E501
  # noqa: E501
class SimpleTextAdapter(dspy.Adapter):  # noqa: E501
    """Bypasses JSON/formatting and just takes the raw LM output as the 'reply'."""  # noqa: E501
  # noqa: E501
    def __call__(self, lm, lm_kwargs, signature, demos, inputs):  # noqa: E501
        # Build a simple text prompt  # noqa: E501
        prompt = ""  # noqa: E501
        # Handle instruction if it exists in signature (MIPRO v2 puts it there)  # noqa: E501
        if hasattr(signature, "instructions"):  # noqa: E501
            prompt += f"{signature.instructions}\n\n"  # noqa: E501
  # noqa: E501
        for field_name, value in inputs.items():  # noqa: E501
            # Skip signature metadata  # noqa: E501
            if field_name in ["instructions"]:  # noqa: E501
                continue  # noqa: E501
            prompt += f"{field_name}: {value}\n"  # noqa: E501
        prompt += "reply:"  # noqa: E501
  # noqa: E501
        # Generate  # noqa: E501
        response = lm(prompt, **lm_kwargs)  # noqa: E501
        if isinstance(response, str):  # noqa: E501
            text = response  # noqa: E501
        elif hasattr(response, "choices"):  # noqa: E501
            text = (  # noqa: E501
                response.choices[0].message.content  # noqa: E501
                if hasattr(response.choices[0], "message")  # noqa: E501
                else response.choices[0].text  # noqa: E501
            )  # noqa: E501
        else:  # noqa: E501
            text = str(response)  # noqa: E501
  # noqa: E501
        # Strip potential label if model repeats it  # noqa: E501
        if text.lower().startswith("reply:"):  # noqa: E501
            text = text[6:].strip()  # noqa: E501
  # noqa: E501
        # Create prediction object matching signature  # noqa: E501
        return dspy.Prediction(reply=text)  # noqa: E501
  # noqa: E501
  # noqa: E501
class UniversalReplySignature(dspy.Signature):  # noqa: E501
    """Generate a natural, human-like text message reply."""  # noqa: E501
  # noqa: E501
    context: str = dspy.InputField(desc="Conversation history with timestamps")  # noqa: E501
    last_message: str = dspy.InputField(desc="The message to reply to")  # noqa: E501
    tone: str = dspy.InputField(desc="Tone: casual or professional")  # noqa: E501
    user_style: str = dspy.InputField(desc="User's texting style description")  # noqa: E501
    reply: str = dspy.OutputField(  # noqa: E501
        desc="Brief, natural reply matching the user's style. No AI filler."  # noqa: E501
    )  # noqa: E501
  # noqa: E501
  # noqa: E501
class UniversalReplyModule(dspy.Module):  # noqa: E501
    def __init__(self):  # noqa: E501
        super().__init__()  # noqa: E501
        self.generate = dspy.Predict(UniversalReplySignature)  # noqa: E501
  # noqa: E501
    def forward(self, **kwargs):  # noqa: E501
        return self.generate(**kwargs)  # noqa: E501
  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# Optimization Logic  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
  # noqa: E501
  # noqa: E501
def get_context_at_depth(full_context: list[str], depth: int) -> str:  # noqa: E501
    """Helper to format context for DSPy examples."""  # noqa: E501
    selected = full_context[-depth:] if len(full_context) >= depth else full_context  # noqa: E501
    return "\n".join(selected)  # noqa: E501
  # noqa: E501
  # noqa: E501
def prepare_trainset(depth: int):  # noqa: E501
    """Re-prepare the training set with the specified context depth."""  # noqa: E501
    new_trainset = []  # noqa: E501
    for ex in TRAIN_EXAMPLES:  # noqa: E501
        # TRAIN_EXAMPLES context is usually a list of strings  # noqa: E501
        if isinstance(ex.context, list):  # noqa: E501
            ctx_str = get_context_at_depth(ex.context, depth)  # noqa: E501
        else:  # noqa: E501
            ctx_str = ex.context  # noqa: E501
  # noqa: E501
        # Create new example with modified context  # noqa: E501
        new_ex = dspy.Example(  # noqa: E501
            context=ctx_str, last_message=ex.last_message, tone=ex.tone, user_style=ex.user_style  # noqa: E501
        ).with_inputs("context", "last_message", "tone", "user_style")  # noqa: E501
  # noqa: E501
        # Transfer metadata (needed for judge_metric)  # noqa: E501
        for attr in ["_rubric", "_max_words", "_max_chars", "_banned", "_category"]:  # noqa: E501
            if hasattr(ex, attr):  # noqa: E501
                setattr(new_ex, attr, getattr(ex, attr))  # noqa: E501
  # noqa: E501
        new_trainset.append(new_ex)  # noqa: E501
    return new_trainset  # noqa: E501
  # noqa: E501
  # noqa: E501
def build_teacher_lm():  # noqa: E501
    from evals.judge_config import JUDGE_BASE_URL, JUDGE_MODEL, get_judge_api_key  # noqa: E501
  # noqa: E501
    key = get_judge_api_key()  # noqa: E501
    # Teacher models (Cerebras) can handle JSON/Chat via ChatAdapter  # noqa: E501
    return dspy.LM(  # noqa: E501
        model=f"openai/{JUDGE_MODEL}",  # noqa: E501
        api_base=JUDGE_BASE_URL,  # noqa: E501
        api_key=key,  # noqa: E501
        temperature=0.7,  # noqa: E501
        max_tokens=500,  # noqa: E501
    )  # noqa: E501
  # noqa: E501
  # noqa: E501
def main():  # noqa: E501
    parser = argparse.ArgumentParser()  # noqa: E501
    parser.add_argument("--trials", type=int, default=15, help="MIPROv2 trials per depth")  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--depths", type=str, default="3,5,10", help="Comma-separated context depths to test"  # noqa: E501
    )  # noqa: E501
    parser.add_argument("--candidates", type=int, default=5, help="MIPROv2 candidates")  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--demos", type=int, default=3, help="Max bootstrapped demos (set to 0 to test no few-shot)"  # noqa: E501
    )  # noqa: E501
    args = parser.parse_args()  # noqa: E501
  # noqa: E501
    depths = [int(d.strip()) for d in args.depths.split(",")]  # noqa: E501
  # noqa: E501
    print("=" * 70)  # noqa: E501
    print("JARVIS FULL PIPELINE OPTIMIZATION (MIPROv2 + Context Sweep)")  # noqa: E501
    print(f"Teacher/Judge: {args.demos} bootstrapped demos")  # noqa: E501
    print("=" * 70)  # noqa: E501
  # noqa: E501
    teacher_lm = build_teacher_lm()  # noqa: E501
    # Configure teacher to use ChatAdapter for instruction generation/proposing  # noqa: E501
    dspy.configure(lm=teacher_lm, adapter=dspy.ChatAdapter())  # noqa: E501
  # noqa: E501
    student_lm = DSPYMLXClient(max_tokens=50, temperature=0.1)  # noqa: E501
    student_adapter = SimpleTextAdapter()  # noqa: E501
  # noqa: E501
    # We DON'T set a global adapter here, so student uses default "Field: Value" text  # noqa: E501
    # When we want to use the student, we'll use a local dspy.context  # noqa: E501
  # noqa: E501
    best_overall_score = -1.0  # noqa: E501
    best_config = None  # noqa: E501
    results = []  # noqa: E501
  # noqa: E501
    for depth in depths:  # noqa: E501
        print(f"\n>>> OPTIMIZING FOR CONTEXT_DEPTH: {depth}")  # noqa: E501
        trainset = prepare_trainset(depth)  # noqa: E501
  # noqa: E501
        student = UniversalReplyModule()  # noqa: E501
  # noqa: E501
        optimizer = dspy.MIPROv2(  # noqa: E501
            metric=judge_metric,  # noqa: E501
            prompt_model=teacher_lm,  # noqa: E501
            auto=None,  # noqa: E501
            num_candidates=args.candidates,  # noqa: E501
            max_bootstrapped_demos=args.demos,  # noqa: E501
            max_labeled_demos=4,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
        with dspy.context(lm=student_lm, adapter=student_adapter):  # noqa: E501
            # Student evaluation runs here using SimpleTextAdapter  # noqa: E501
            compiled = optimizer.compile(  # noqa: E501
                student=student,  # noqa: E501
                trainset=trainset,  # noqa: E501
                num_trials=args.trials,  # noqa: E501
                minibatch=False,  # noqa: E501
            )  # noqa: E501
  # noqa: E501
        # Evaluate the compiled program for this depth  # noqa: E501
        scores = []  # noqa: E501
        with dspy.context(lm=student_lm, adapter=student_adapter):  # noqa: E501
            for ex in trainset:  # noqa: E501
                inputs = {  # noqa: E501
                    "context": ex.context,  # noqa: E501
                    "last_message": ex.last_message,  # noqa: E501
                    "tone": ex.tone,  # noqa: E501
                    "user_style": ex.user_style,  # noqa: E501
                }  # noqa: E501
                pred = compiled(**inputs)  # noqa: E501
                score = judge_metric(ex, pred)  # noqa: E501
                scores.append(score)  # noqa: E501
  # noqa: E501
        avg_score = sum(scores) / len(scores) if scores else 0  # noqa: E501
        print(f"Final Optimized Score for Depth {depth}: {avg_score:.3f}")  # noqa: E501
  # noqa: E501
        # Save results  # noqa: E501
        save_path = PROJECT_ROOT / "evals" / f"optimized_universal_d{depth}.json"  # noqa: E501
        compiled.save(str(save_path))  # noqa: E501
  # noqa: E501
        res = {"depth": depth, "avg_score": avg_score, "save_path": str(save_path)}  # noqa: E501
        results.append(res)  # noqa: E501
  # noqa: E501
        if avg_score > best_overall_score:  # noqa: E501
            best_overall_score = avg_score  # noqa: E501
            best_config = res  # noqa: E501
  # noqa: E501
    print("\n" + "=" * 70)  # noqa: E501
    print("OPTIMIZATION COMPLETE")  # noqa: E501
    print("=" * 70)  # noqa: E501
    for r in results:  # noqa: E501
        print(f"Depth {r['depth']}: {r['avg_score']:.3f}")  # noqa: E501
  # noqa: E501
    print(f"\n🏆 WINNER: Depth {best_config['depth']} with score {best_config['avg_score']:.3f}")  # noqa: E501
    print(f"Saved at: {best_config['save_path']}")  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    main()  # noqa: E501
