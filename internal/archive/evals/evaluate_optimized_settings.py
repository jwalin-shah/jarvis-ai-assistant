#!/usr/bin/env python3
"""Evaluate optimized generation settings with Cerebras judge."""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
  # noqa: E402
from evals.eval_pipeline import EVAL_DATASET_PATH, EvalExample, load_eval_dataset  # noqa: E402
from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402
from tqdm import tqdm  # noqa: E402

# noqa: E402
from models.loader import get_model  # noqa: E402

  # noqa: E402
# Configuration  # noqa: E402
NUM_EXAMPLES = 20  # Small batch for quick eval  # noqa: E402
  # noqa: E402
# Optimized settings (refined)  # noqa: E402
OPTIMIZED_CONFIG = {  # noqa: E402
    "temperature": 0.15,  # noqa: E402
    "repetition_penalty": 1.15,  # Higher = no echoing  # noqa: E402
    "max_tokens": 20,  # Shorter = more natural  # noqa: E402
    "top_p": 0.9,  # noqa: E402
}  # noqa: E402
  # noqa: E402
# Baseline settings (old)  # noqa: E402
BASELINE_CONFIG = {  # noqa: E402
    "temperature": 0.1,  # noqa: E402
    "repetition_penalty": 1.05,  # noqa: E402
    "max_tokens": 50,  # noqa: E402
    "top_p": 0.9,  # noqa: E402
}  # noqa: E402
  # noqa: E402
SYSTEM_PROMPT = """You are texting from your phone. Reply naturally, matching their style.  # noqa: E402
Be brief (1-2 sentences), casual, and sound like a real person."""  # noqa: E402
  # noqa: E402
  # noqa: E402
def build_prompt(context: str, last_message: str) -> str:  # noqa: E402
    """Build chat prompt."""  # noqa: E402
    return (  # noqa: E402
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"  # noqa: E402
        f"<|im_start|>user\n"  # noqa: E402
        f"Context: {context}\n"  # noqa: E402
        f"Reply to: {last_message}<|im_end|>\n"  # noqa: E402
        f"<|im_start|>assistant\n"  # noqa: E402
    )  # noqa: E402
  # noqa: E402
  # noqa: E402
def generate_reply(loader, prompt: str, config: dict) -> tuple[str, float]:  # noqa: E402
    """Generate reply with given config."""  # noqa: E402
    start = time.perf_counter()  # noqa: E402
    try:  # noqa: E402
        result = loader.generate_sync(  # noqa: E402
            prompt=prompt,  # noqa: E402
            temperature=config["temperature"],  # noqa: E402
            max_tokens=config["max_tokens"],  # noqa: E402
            repetition_penalty=config["repetition_penalty"],  # noqa: E402
            top_p=config["top_p"],  # noqa: E402
        )  # noqa: E402
        latency = (time.perf_counter() - start) * 1000  # noqa: E402
        return result.text.strip(), latency  # noqa: E402
    except Exception as e:  # noqa: E402
        latency = (time.perf_counter() - start) * 1000  # noqa: E402
        return f"[ERROR: {e}]", latency  # noqa: E402
  # noqa: E402
  # noqa: E402
def judge_example(  # noqa: E402
    client, context: str, last_message: str, ideal: str, generated: str  # noqa: E402
) -> tuple[float, str]:  # noqa: E402
    """Judge a single example."""  # noqa: E402
    prompt = f"""You are an expert evaluator of text message replies.  # noqa: E402
  # noqa: E402
Rate the generated reply on a scale of 1-10 based on how natural and appropriate it is.  # noqa: E402
  # noqa: E402
Context: {context}  # noqa: E402
Message to reply to: {last_message}  # noqa: E402
Ideal reply: {ideal}  # noqa: E402
Generated reply: {generated}  # noqa: E402
  # noqa: E402
Respond with ONLY a JSON object in this exact format:  # noqa: E402
{{"score": <number 1-10>, "reasoning": "<brief explanation>"}}  # noqa: E402
  # noqa: E402
Rating criteria:  # noqa: E402
- 9-10: Perfect natural reply, matches style  # noqa: E402
- 7-8: Good reply, minor issues  # noqa: E402
- 5-6: Acceptable but awkward or slightly off  # noqa: E402
- 3-4: Poor, unnatural or inappropriate  # noqa: E402
- 1-2: Very bad, completely wrong  # noqa: E402
"""  # noqa: E402
  # noqa: E402
    try:  # noqa: E402
        resp = client.chat.completions.create(  # noqa: E402
            model=JUDGE_MODEL,  # noqa: E402
            messages=[{"role": "user", "content": prompt}],  # noqa: E402
            temperature=0.0,  # noqa: E402
            max_tokens=200,  # noqa: E402
        )  # noqa: E402
        content = resp.choices[0].message.content  # noqa: E402
        # Extract JSON  # noqa: E402
        if "```json" in content:  # noqa: E402
            content = content.split("```json")[1].split("```")[0]  # noqa: E402
        elif "```" in content:  # noqa: E402
            content = content.split("```")[1].split("```")[0]  # noqa: E402
        result = json.loads(content.strip())  # noqa: E402
        return result.get("score", 0), result.get("reasoning", "")  # noqa: E402
    except Exception as e:  # noqa: E402
        return 0, f"Judge error: {e}"  # noqa: E402
  # noqa: E402
  # noqa: E402
def evaluate_config(loader, client, examples: list, config: dict, config_name: str) -> dict:  # noqa: E402
    """Evaluate a configuration on examples."""  # noqa: E402
    print(f"\n{'=' * 70}")  # noqa: E402
    print(f"Evaluating: {config_name}")  # noqa: E402
    print(f"Config: {config}")  # noqa: E402
    print(f"{'=' * 70}")  # noqa: E402
  # noqa: E402
    results = []  # noqa: E402
    scores = []  # noqa: E402
  # noqa: E402
    for i, ex in enumerate(tqdm(examples, desc=config_name)):  # noqa: E402
        # Handle both dict and EvalExample  # noqa: E402
        if isinstance(ex, EvalExample):  # noqa: E402
            context = "\n".join(ex.context)  # noqa: E402
            last_message = ex.last_message  # noqa: E402
            ideal = ex.ideal_response  # noqa: E402
        else:  # noqa: E402
            context = ex.get("context", "")  # noqa: E402
            last_message = ex.get("last_message", "")  # noqa: E402
            ideal = ex.get("ideal_reply", "")  # noqa: E402
  # noqa: E402
        prompt = build_prompt(context, last_message)  # noqa: E402
        reply, latency = generate_reply(loader, prompt, config)  # noqa: E402
  # noqa: E402
        # Judge  # noqa: E402
        score, reasoning = judge_example(client, context, last_message, ideal, reply)  # noqa: E402
        scores.append(score)  # noqa: E402
  # noqa: E402
        results.append(  # noqa: E402
            {  # noqa: E402
                "example_id": i,  # noqa: E402
                "last_message": last_message,  # noqa: E402
                "ideal": ideal,  # noqa: E402
                "generated": reply,  # noqa: E402
                "length": len(reply),  # noqa: E402
                "latency_ms": latency,  # noqa: E402
                "score": score,  # noqa: E402
                "reasoning": reasoning,  # noqa: E402
            }  # noqa: E402
        )  # noqa: E402
  # noqa: E402
        time.sleep(2.1)  # Rate limit: 30 req/min  # noqa: E402
  # noqa: E402
    avg_score = sum(scores) / len(scores) if scores else 0  # noqa: E402
    pass_rate = sum(1 for s in scores if s >= 6) / len(scores) if scores else 0  # noqa: E402
    avg_length = sum(r["length"] for r in results) / len(results)  # noqa: E402
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)  # noqa: E402
  # noqa: E402
    return {  # noqa: E402
        "config_name": config_name,  # noqa: E402
        "config": config,  # noqa: E402
        "avg_score": avg_score,  # noqa: E402
        "pass_rate": pass_rate,  # noqa: E402
        "avg_length": avg_length,  # noqa: E402
        "avg_latency_ms": avg_latency,  # noqa: E402
        "results": results,  # noqa: E402
    }  # noqa: E402
  # noqa: E402
  # noqa: E402
def main():  # noqa: E402
    print("=" * 70)  # noqa: E402
    print("EVALUATING OPTIMIZED GENERATION SETTINGS")  # noqa: E402
    print("=" * 70)  # noqa: E402
  # noqa: E402
    # Load model  # noqa: E402
    loader = get_model()  # noqa: E402
    if not loader.is_loaded():  # noqa: E402
        print("Loading model...")  # noqa: E402
        loader.load()  # noqa: E402
  # noqa: E402
    # Load judge client  # noqa: E402
    client = get_judge_client()  # noqa: E402
    print(f"Judge: {JUDGE_MODEL}")  # noqa: E402
  # noqa: E402
    # Load examples  # noqa: E402
    examples = load_eval_dataset(EVAL_DATASET_PATH)[:NUM_EXAMPLES]  # noqa: E402
    print(f"Loaded {len(examples)} examples")  # noqa: E402
  # noqa: E402
    # Evaluate baseline  # noqa: E402
    baseline_results = evaluate_config(loader, client, examples, BASELINE_CONFIG, "Baseline (old)")  # noqa: E402
  # noqa: E402
    # Evaluate optimized  # noqa: E402
    optimized_results = evaluate_config(  # noqa: E402
        loader, client, examples, OPTIMIZED_CONFIG, "Optimized (new)"  # noqa: E402
    )  # noqa: E402
  # noqa: E402
    # Summary  # noqa: E402
    print(f"\n{'=' * 70}")  # noqa: E402
    print("FINAL COMPARISON")  # noqa: E402
    print(f"{'=' * 70}")  # noqa: E402
    print("\n📊 Baseline (rep=1.05, max_tokens=50):")  # noqa: E402
    print(f"   Average Score: {baseline_results['avg_score']:.2f}/10")  # noqa: E402
    print(f"   Pass Rate: {baseline_results['pass_rate'] * 100:.1f}%")  # noqa: E402
    print(f"   Avg Length: {baseline_results['avg_length']:.0f} chars")  # noqa: E402
    print(f"   Avg Latency: {baseline_results['avg_latency_ms']:.0f}ms")  # noqa: E402
  # noqa: E402
    print("\n🚀 Optimized (rep=1.12, max_tokens=25):")  # noqa: E402
    print(f"   Average Score: {optimized_results['avg_score']:.2f}/10")  # noqa: E402
    print(f"   Pass Rate: {optimized_results['pass_rate'] * 100:.1f}%")  # noqa: E402
    print(f"   Avg Length: {optimized_results['avg_length']:.0f} chars")  # noqa: E402
    print(f"   Avg Latency: {optimized_results['avg_latency_ms']:.0f}ms")  # noqa: E402
  # noqa: E402
    score_improvement = optimized_results["avg_score"] - baseline_results["avg_score"]  # noqa: E402
    length_reduction = baseline_results["avg_length"] - optimized_results["avg_length"]  # noqa: E402
  # noqa: E402
    print("\n📈 Improvements:")  # noqa: E402
    print(f"   Score: {score_improvement:+.2f} points")  # noqa: E402
    print(  # noqa: E402
        f"   Length: -{length_reduction:.0f} chars ({length_reduction / baseline_results['avg_length'] * 100:.0f}% shorter)"  # noqa: E402
    )  # noqa: E402
  # noqa: E402
    # Save results  # noqa: E402
    output = {  # noqa: E402
        "baseline": baseline_results,  # noqa: E402
        "optimized": optimized_results,  # noqa: E402
        "improvements": {  # noqa: E402
            "score": score_improvement,  # noqa: E402
            "length_reduction": length_reduction,  # noqa: E402
            "length_reduction_pct": length_reduction / baseline_results["avg_length"] * 100,  # noqa: E402
        },  # noqa: E402
    }  # noqa: E402
  # noqa: E402
    output_file = PROJECT_ROOT / "results" / "optimized_settings_eval.json"  # noqa: E402
    with open(output_file, "w") as f:  # noqa: E402
        json.dump(output, f, indent=2)  # noqa: E402
    print(f"\n💾 Results saved to: {output_file}")  # noqa: E402
  # noqa: E402
    return output  # noqa: E402
  # noqa: E402
  # noqa: E402
if __name__ == "__main__":  # noqa: E402
    main()  # noqa: E402
