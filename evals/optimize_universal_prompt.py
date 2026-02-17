#!/usr/bin/env python3
"""Pipeline optimization: Sweep through context depths and optimize prompts via MIPROv2.

This script find the best 'structural' variables (like context_depth) 
AND the best 'textual' variables (instructions/demos) at the same time.

Usage:
    uv run python evals/optimize_universal_prompt.py --trials 10 --depths 3,5,10
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import dspy

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evals.dspy_client import DSPYMLXClient
from evals.dspy_reply import (
    TRAIN_EXAMPLES,
    clean_reply,
    judge_metric,
)

# ---------------------------------------------------------------------------
# Student Module
# ---------------------------------------------------------------------------

class SimpleTextAdapter(dspy.Adapter):
    """Bypasses JSON/formatting and just takes the raw LM output as the 'reply'."""
    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        # Build a simple text prompt
        prompt = ""
        # Handle instruction if it exists in signature (MIPRO v2 puts it there)
        if hasattr(signature, "instructions"):
            prompt += f"{signature.instructions}\n\n"
            
        for field_name, value in inputs.items():
            # Skip signature metadata
            if field_name in ["instructions"]: continue
            prompt += f"{field_name}: {value}\n"
        prompt += "reply:"
        
        # Generate
        response = lm(prompt, **lm_kwargs)
        if isinstance(response, str):
            text = response
        elif hasattr(response, "choices"):
            text = response.choices[0].message.content if hasattr(response.choices[0], "message") else response.choices[0].text
        else:
            text = str(response)
            
        # Strip potential label if model repeats it
        if text.lower().startswith("reply:"):
            text = text[6:].strip()
            
        # Create prediction object matching signature
        return dspy.Prediction(reply=text)

class UniversalReplySignature(dspy.Signature):
    """Generate a natural, human-like text message reply."""
    
    context: str = dspy.InputField(desc="Conversation history with timestamps")
    last_message: str = dspy.InputField(desc="The message to reply to")
    tone: str = dspy.InputField(desc="Tone: casual or professional")
    user_style: str = dspy.InputField(desc="User's texting style description")
    reply: str = dspy.OutputField(desc="Brief, natural reply matching the user's style. No AI filler.")

class UniversalReplyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(UniversalReplySignature)

    def forward(self, **kwargs):
        return self.generate(**kwargs)

# ---------------------------------------------------------------------------
# Optimization Logic
# ---------------------------------------------------------------------------

def get_context_at_depth(full_context: list[str], depth: int) -> str:
    """Helper to format context for DSPy examples."""
    selected = full_context[-depth:] if len(full_context) >= depth else full_context
    return "\n".join(selected)

def prepare_trainset(depth: int):
    """Re-prepare the training set with the specified context depth."""
    new_trainset = []
    for ex in TRAIN_EXAMPLES:
        # TRAIN_EXAMPLES context is usually a list of strings
        if isinstance(ex.context, list):
            ctx_str = get_context_at_depth(ex.context, depth)
        else:
            ctx_str = ex.context
            
        # Create new example with modified context
        new_ex = dspy.Example(
            context=ctx_str,
            last_message=ex.last_message,
            tone=ex.tone,
            user_style=ex.user_style
        ).with_inputs("context", "last_message", "tone", "user_style")
        
        # Transfer metadata (needed for judge_metric)
        for attr in ["_rubric", "_max_words", "_max_chars", "_banned", "_category"]:
            if hasattr(ex, attr):
                setattr(new_ex, attr, getattr(ex, attr))
                
        new_trainset.append(new_ex)
    return new_trainset

def build_teacher_lm():
    from evals.judge_config import JUDGE_BASE_URL, JUDGE_MODEL, get_judge_api_key
    key = get_judge_api_key()
    # Teacher models (Cerebras) can handle JSON/Chat via ChatAdapter
    return dspy.LM(
        model=f"openai/{JUDGE_MODEL}",
        api_base=JUDGE_BASE_URL,
        api_key=key,
        temperature=0.7,
        max_tokens=500,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=15, help="MIPROv2 trials per depth")
    parser.add_argument("--depths", type=str, default="3,5,10", help="Comma-separated context depths to test")
    parser.add_argument("--candidates", type=int, default=5, help="MIPROv2 candidates")
    parser.add_argument("--demos", type=int, default=3, help="Max bootstrapped demos (set to 0 to test no few-shot)")
    args = parser.parse_args()

    depths = [int(d.strip()) for d in args.depths.split(",")]
    
    print("=" * 70)
    print("JARVIS FULL PIPELINE OPTIMIZATION (MIPROv2 + Context Sweep)")
    print(f"Teacher/Judge: {args.demos} bootstrapped demos")
    print("=" * 70)
    
    teacher_lm = build_teacher_lm()
    # Configure teacher to use ChatAdapter for instruction generation/proposing
    dspy.configure(lm=teacher_lm, adapter=dspy.ChatAdapter())
    
    student_lm = DSPYMLXClient(max_tokens=50, temperature=0.1)
    student_adapter = SimpleTextAdapter()
    
    # We DON'T set a global adapter here, so student uses default "Field: Value" text
    # When we want to use the student, we'll use a local dspy.context
    
    best_overall_score = -1.0
    best_config = None
    results = []

    for depth in depths:
        print(f"\n>>> OPTIMIZING FOR CONTEXT_DEPTH: {depth}")
        trainset = prepare_trainset(depth)
        
        student = UniversalReplyModule()
        
        optimizer = dspy.MIPROv2(
            metric=judge_metric,
            prompt_model=teacher_lm,
            auto=None,
            num_candidates=args.candidates,
            max_bootstrapped_demos=args.demos,
            max_labeled_demos=4,
        )

        with dspy.context(lm=student_lm, adapter=student_adapter):
            # Student evaluation runs here using SimpleTextAdapter
            compiled = optimizer.compile(
                student=student,
                trainset=trainset,
                num_trials=args.trials,
                minibatch=False,
            )
        
        # Evaluate the compiled program for this depth
        scores = []
        with dspy.context(lm=student_lm, adapter=student_adapter):
            for ex in trainset:
                inputs = {
                    "context": ex.context,
                    "last_message": ex.last_message,
                    "tone": ex.tone,
                    "user_style": ex.user_style
                }
                pred = compiled(**inputs)
                score = judge_metric(ex, pred)
                scores.append(score)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"Final Optimized Score for Depth {depth}: {avg_score:.3f}")
        
        # Save results
        save_path = PROJECT_ROOT / "evals" / f"optimized_universal_d{depth}.json"
        compiled.save(str(save_path))
        
        res = {
            "depth": depth,
            "avg_score": avg_score,
            "save_path": str(save_path)
        }
        results.append(res)
        
        if avg_score > best_overall_score:
            best_overall_score = avg_score
            best_config = res

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    for r in results:
        print(f"Depth {r['depth']}: {r['avg_score']:.3f}")
    
    print(f"\n🏆 WINNER: Depth {best_config['depth']} with score {best_config['avg_score']:.3f}")
    print(f"Saved at: {best_config['save_path']}")

if __name__ == "__main__":
    main()
