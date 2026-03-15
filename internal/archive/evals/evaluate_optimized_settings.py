#!/usr/bin/env python3  # noqa: E501
"""Evaluate optimized generation settings with Cerebras judge."""  # noqa: E501
  # noqa: E501
import json  # noqa: E501
import sys  # noqa: E501
import time  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501

  # noqa: E501
PROJECT_ROOT = Path(__file__).parent.parent  # noqa: E501
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E501
  # noqa: E501
from evals.eval_pipeline import (  # noqa: E402  # noqa: E501
    EVAL_DATASET_PATH,
    EvalExample,
    load_eval_dataset,
)
from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E402  # noqa: E501
from tqdm import tqdm  # noqa: E402  # noqa: E501

# noqa: E501
from models.loader import get_model  # noqa: E402  # noqa: E501

  # noqa: E501
# Configuration  # noqa: E501
NUM_EXAMPLES = 20  # Small batch for quick eval  # noqa: E501
  # noqa: E501
# Optimized settings (refined)  # noqa: E501
OPTIMIZED_CONFIG = {  # noqa: E501
    "temperature": 0.15,  # noqa: E501
    "repetition_penalty": 1.15,  # Higher = no echoing  # noqa: E501
    "max_tokens": 20,  # Shorter = more natural  # noqa: E501
    "top_p": 0.9,  # noqa: E501
}  # noqa: E501
  # noqa: E501
# Baseline settings (old)  # noqa: E501
BASELINE_CONFIG = {  # noqa: E501
    "temperature": 0.1,  # noqa: E501
    "repetition_penalty": 1.05,  # noqa: E501
    "max_tokens": 50,  # noqa: E501
    "top_p": 0.9,  # noqa: E501
}  # noqa: E501
  # noqa: E501
SYSTEM_PROMPT = """You are texting from your phone. Reply naturally, matching their style.  # noqa: E501
Be brief (1-2 sentences), casual, and sound like a real person."""  # noqa: E501
  # noqa: E501
  # noqa: E501
def build_prompt(context: str, last_message: str) -> str:  # noqa: E501
    """Build chat prompt."""  # noqa: E501
    return (  # noqa: E501
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"  # noqa: E501
        f"<|im_start|>user\n"  # noqa: E501
        f"Context: {context}\n"  # noqa: E501
        f"Reply to: {last_message}<|im_end|>\n"  # noqa: E501
        f"<|im_start|>assistant\n"  # noqa: E501
    )  # noqa: E501
  # noqa: E501
  # noqa: E501
def generate_reply(loader, prompt: str, config: dict) -> tuple[str, float]:  # noqa: E501
    """Generate reply with given config."""  # noqa: E501
    start = time.perf_counter()  # noqa: E501
    try:  # noqa: E501
        result = loader.generate_sync(  # noqa: E501
            prompt=prompt,  # noqa: E501
            temperature=config["temperature"],  # noqa: E501
            max_tokens=config["max_tokens"],  # noqa: E501
            repetition_penalty=config["repetition_penalty"],  # noqa: E501
            top_p=config["top_p"],  # noqa: E501
        )  # noqa: E501
        latency = (time.perf_counter() - start) * 1000  # noqa: E501
        return result.text.strip(), latency  # noqa: E501
    except Exception as e:  # noqa: E501
        latency = (time.perf_counter() - start) * 1000  # noqa: E501
        return f"[ERROR: {e}]", latency  # noqa: E501
  # noqa: E501
  # noqa: E501
def judge_example(  # noqa: E501
    client, context: str, last_message: str, ideal: str, generated: str  # noqa: E501
) -> tuple[float, str]:  # noqa: E501
    """Judge a single example."""  # noqa: E501
    prompt = f"""You are an expert evaluator of text message replies.  # noqa: E501
  # noqa: E501
Rate the generated reply on a scale of 1-10 based on how natural and appropriate it is.  # noqa: E501
  # noqa: E501
Context: {context}  # noqa: E501
Message to reply to: {last_message}  # noqa: E501
Ideal reply: {ideal}  # noqa: E501
Generated reply: {generated}  # noqa: E501
  # noqa: E501
Respond with ONLY a JSON object in this exact format:  # noqa: E501
{{"score": <number 1-10>, "reasoning": "<brief explanation>"}}  # noqa: E501
  # noqa: E501
Rating criteria:  # noqa: E501
- 9-10: Perfect natural reply, matches style  # noqa: E501
- 7-8: Good reply, minor issues  # noqa: E501
- 5-6: Acceptable but awkward or slightly off  # noqa: E501
- 3-4: Poor, unnatural or inappropriate  # noqa: E501
- 1-2: Very bad, completely wrong  # noqa: E501
"""  # noqa: E501
  # noqa: E501
    try:  # noqa: E501
        resp = client.chat.completions.create(  # noqa: E501
            model=JUDGE_MODEL,  # noqa: E501
            messages=[{"role": "user", "content": prompt}],  # noqa: E501
            temperature=0.0,  # noqa: E501
            max_tokens=200,  # noqa: E501
        )  # noqa: E501
        content = resp.choices[0].message.content  # noqa: E501
        # Extract JSON  # noqa: E501
        if "```json" in content:  # noqa: E501
            content = content.split("```json")[1].split("```")[0]  # noqa: E501
        elif "```" in content:  # noqa: E501
            content = content.split("```")[1].split("```")[0]  # noqa: E501
        result = json.loads(content.strip())  # noqa: E501
        return result.get("score", 0), result.get("reasoning", "")  # noqa: E501
    except Exception as e:  # noqa: E501
        return 0, f"Judge error: {e}"  # noqa: E501
  # noqa: E501
  # noqa: E501
def evaluate_config(loader, client, examples: list, config: dict, config_name: str) -> dict:  # noqa: E501
    """Evaluate a configuration on examples."""  # noqa: E501
    print(f"\n{'=' * 70}")  # noqa: E501
    print(f"Evaluating: {config_name}")  # noqa: E501
    print(f"Config: {config}")  # noqa: E501
    print(f"{'=' * 70}")  # noqa: E501
  # noqa: E501
    results = []  # noqa: E501
    scores = []  # noqa: E501
  # noqa: E501
    for i, ex in enumerate(tqdm(examples, desc=config_name)):  # noqa: E501
        # Handle both dict and EvalExample  # noqa: E501
        if isinstance(ex, EvalExample):  # noqa: E501
            context = "\n".join(ex.context)  # noqa: E501
            last_message = ex.last_message  # noqa: E501
            ideal = ex.ideal_response  # noqa: E501
        else:  # noqa: E501
            context = ex.get("context", "")  # noqa: E501
            last_message = ex.get("last_message", "")  # noqa: E501
            ideal = ex.get("ideal_reply", "")  # noqa: E501
  # noqa: E501
        prompt = build_prompt(context, last_message)  # noqa: E501
        reply, latency = generate_reply(loader, prompt, config)  # noqa: E501
  # noqa: E501
        # Judge  # noqa: E501
        score, reasoning = judge_example(client, context, last_message, ideal, reply)  # noqa: E501
        scores.append(score)  # noqa: E501
  # noqa: E501
        results.append(  # noqa: E501
            {  # noqa: E501
                "example_id": i,  # noqa: E501
                "last_message": last_message,  # noqa: E501
                "ideal": ideal,  # noqa: E501
                "generated": reply,  # noqa: E501
                "length": len(reply),  # noqa: E501
                "latency_ms": latency,  # noqa: E501
                "score": score,  # noqa: E501
                "reasoning": reasoning,  # noqa: E501
            }  # noqa: E501
        )  # noqa: E501
  # noqa: E501
        time.sleep(2.1)  # Rate limit: 30 req/min  # noqa: E501
  # noqa: E501
    avg_score = sum(scores) / len(scores) if scores else 0  # noqa: E501
    pass_rate = sum(1 for s in scores if s >= 6) / len(scores) if scores else 0  # noqa: E501
    avg_length = sum(r["length"] for r in results) / len(results)  # noqa: E501
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)  # noqa: E501
  # noqa: E501
    return {  # noqa: E501
        "config_name": config_name,  # noqa: E501
        "config": config,  # noqa: E501
        "avg_score": avg_score,  # noqa: E501
        "pass_rate": pass_rate,  # noqa: E501
        "avg_length": avg_length,  # noqa: E501
        "avg_latency_ms": avg_latency,  # noqa: E501
        "results": results,  # noqa: E501
    }  # noqa: E501
  # noqa: E501
  # noqa: E501
def main():  # noqa: E501
    print("=" * 70)  # noqa: E501
    print("EVALUATING OPTIMIZED GENERATION SETTINGS")  # noqa: E501
    print("=" * 70)  # noqa: E501
  # noqa: E501
    # Load model  # noqa: E501
    loader = get_model()  # noqa: E501
    if not loader.is_loaded():  # noqa: E501
        print("Loading model...")  # noqa: E501
        loader.load()  # noqa: E501
  # noqa: E501
    # Load judge client  # noqa: E501
    client = get_judge_client()  # noqa: E501
    print(f"Judge: {JUDGE_MODEL}")  # noqa: E501
  # noqa: E501
    # Load examples  # noqa: E501
    examples = load_eval_dataset(EVAL_DATASET_PATH)[:NUM_EXAMPLES]  # noqa: E501
    print(f"Loaded {len(examples)} examples")  # noqa: E501
  # noqa: E501
    # Evaluate baseline  # noqa: E501
    baseline_results = evaluate_config(loader, client, examples, BASELINE_CONFIG, "Baseline (old)")  # noqa: E501
  # noqa: E501
    # Evaluate optimized  # noqa: E501
    optimized_results = evaluate_config(  # noqa: E501
        loader, client, examples, OPTIMIZED_CONFIG, "Optimized (new)"  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    # Summary  # noqa: E501
    print(f"\n{'=' * 70}")  # noqa: E501
    print("FINAL COMPARISON")  # noqa: E501
    print(f"{'=' * 70}")  # noqa: E501
    print("\n📊 Baseline (rep=1.05, max_tokens=50):")  # noqa: E501
    print(f"   Average Score: {baseline_results['avg_score']:.2f}/10")  # noqa: E501
    print(f"   Pass Rate: {baseline_results['pass_rate'] * 100:.1f}%")  # noqa: E501
    print(f"   Avg Length: {baseline_results['avg_length']:.0f} chars")  # noqa: E501
    print(f"   Avg Latency: {baseline_results['avg_latency_ms']:.0f}ms")  # noqa: E501
  # noqa: E501
    print("\n🚀 Optimized (rep=1.12, max_tokens=25):")  # noqa: E501
    print(f"   Average Score: {optimized_results['avg_score']:.2f}/10")  # noqa: E501
    print(f"   Pass Rate: {optimized_results['pass_rate'] * 100:.1f}%")  # noqa: E501
    print(f"   Avg Length: {optimized_results['avg_length']:.0f} chars")  # noqa: E501
    print(f"   Avg Latency: {optimized_results['avg_latency_ms']:.0f}ms")  # noqa: E501
  # noqa: E501
    score_improvement = optimized_results["avg_score"] - baseline_results["avg_score"]  # noqa: E501
    length_reduction = baseline_results["avg_length"] - optimized_results["avg_length"]  # noqa: E501
  # noqa: E501
    print("\n📈 Improvements:")  # noqa: E501
    print(f"   Score: {score_improvement:+.2f} points")  # noqa: E501
    print(  # noqa: E501
        f"   Length: -{length_reduction:.0f} chars ({length_reduction / baseline_results['avg_length'] * 100:.0f}% shorter)"  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    # Save results  # noqa: E501
    output = {  # noqa: E501
        "baseline": baseline_results,  # noqa: E501
        "optimized": optimized_results,  # noqa: E501
        "improvements": {  # noqa: E501
            "score": score_improvement,  # noqa: E501
            "length_reduction": length_reduction,  # noqa: E501
            "length_reduction_pct": length_reduction / baseline_results["avg_length"] * 100,  # noqa: E501
        },  # noqa: E501
    }  # noqa: E501
  # noqa: E501
    output_file = PROJECT_ROOT / "results" / "optimized_settings_eval.json"  # noqa: E501
    with open(output_file, "w") as f:  # noqa: E501
        json.dump(output, f, indent=2)  # noqa: E501
    print(f"\n💾 Results saved to: {output_file}")  # noqa: E501
  # noqa: E501
    return output  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    main()  # noqa: E501
