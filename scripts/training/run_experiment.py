#!/usr/bin/env python3
"""Run controlled experiment: 0.3B vs 0.7B vs 1.2B fine-tuning.

Trains all three models with identical hyperparameters and compares results.

Usage:
    # Run full experiment (trains all three sequentially)
    uv run python scripts/training/run_experiment.py
    
    # Train just one model
    uv run python scripts/training/run_experiment.py --model 0.7b
    
    # Evaluate existing adapters
    uv run python scripts/training/run_experiment.py --evaluate-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the training function
from scripts.training.train_with_early_stopping import train_with_early_stopping, compare_adapters


EXPERIMENT_CONFIGS = {
    "0.3b": PROJECT_ROOT / "ft_configs/experiment_0.3b.yaml",
    "0.7b": PROJECT_ROOT / "ft_configs/experiment_0.7b.yaml",
    "1.2b": PROJECT_ROOT / "ft_configs/experiment_1.2b.yaml",
}


def evaluate_baseline(model_name: str, test_prompts: list[dict]) -> dict:
    """Evaluate base model without adapter (baseline)."""
    from mlx_lm import load, generate
    import mlx.core as mx
    
    model_map = {
        "0.3b": "mlx-community/LFM2-350M-4bit",
        "0.7b": "mlx-community/LFM2-700M-4bit",
        "1.2b": "LiquidAI/LFM2.5-1.2B",
    }
    
    print(f"\n{'='*70}")
    print(f"BASELINE: {model_name} (no adapter)")
    print(f"{'='*70}")
    
    model, tokenizer = load(model_map[model_name])
    
    system_prompt = (
        "You are Jwalin. Reply to text messages in your natural texting style.\n"
        "Rules:\n"
        "- Match your typical reply length (9 words avg)\n"
        "- Use your abbreviations naturally: wanna, bc, gonna, kinda, btw\n"
        "- No emoji usage\n"
        "- Never sound like an AI assistant\n"
        "- No formal greetings or sign-offs\n"
        "- Just text back like you normally would"
    )
    
    results = []
    for test in test_prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test["prompt"]},
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        output = generate(model, tokenizer, prompt=formatted, max_tokens=30, verbose=False)
        results.append({
            "prompt": test["prompt"][:50],
            "expected": test["expected"],
            "generated": output,
        })
        print(f'Prompt: "{test["prompt"][:40]}..."')
        print(f'  Expected: "{test["expected"]}"')
        print(f'  Baseline: "{output}"')
        print()
        mx.clear_cache()
    
    return {"model": model_name, "results": results}


def run_experiment(models: list[str], patience: int = 3) -> list[dict]:
    """Run training experiment for specified models."""
    results = []
    
    for model_name in models:
        config_path = EXPERIMENT_CONFIGS.get(model_name)
        if not config_path or not config_path.exists():
            print(f"❌ Config not found for {model_name}")
            continue
        
        print(f"\n{'='*70}")
        print(f"TRAINING {model_name.upper()}")
        print(f"{'='*70}")
        
        try:
            result = train_with_early_stopping(
                config_path=config_path,
                patience=patience,
                cleanup_every=25,
            )
            results.append(result)
        except Exception as e:
            print(f"❌ Failed to train {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def print_experiment_summary(results: list[dict]) -> None:
    """Print formatted experiment summary."""
    print(f"\n{'='*70}")
    print("EXPERIMENT RESULTS: Model Size vs Fine-tuning Performance")
    print(f"{'='*70}")
    
    # Sort by best validation loss
    sorted_results = sorted(results, key=lambda x: x.get("best_val_loss", float("inf")))
    
    print(f"\n{'Model':<10} {'Best Iter':<12} {'Val Loss':<12} {'Status':<15} {'Time (min)':<12}")
    print("-" * 70)
    
    for r in sorted_results:
        model = Path(r["config"]).stem.replace("experiment_", "")
        best_iter = r.get("best_iteration", "N/A")
        val_loss = r.get("best_val_loss", "N/A")
        status = "Early Stopped" if r.get("stopped_early") else "Completed"
        if r.get("interrupted"):
            status = "Interrupted"
        time_min = r.get("elapsed_seconds", 0) / 60
        
        loss_str = f"{val_loss:.4f}" if isinstance(val_loss, float) else str(val_loss)
        print(f"{model:<10} {best_iter:<12} {loss_str:<12} {status:<15} {time_min:<12.1f}")
    
    # Key findings
    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}")
    
    if len(sorted_results) >= 2:
        best = sorted_results[0]
        worst = sorted_results[-1]
        improvement = ((worst["best_val_loss"] - best["best_val_loss"]) / worst["best_val_loss"] * 100)
        
        print(f"\n🏆 Best performing model: {Path(best['config']).stem}")
        print(f"   Validation loss: {best['best_val_loss']:.4f}")
        print(f"   Converged at iteration: {best['best_iteration']}")
        print(f"\n📊 Performance gap: {improvement:.1f}% improvement from worst to best")
        
        # Memory comparison hint
        print(f"\n💡 Memory usage (approximate):")
        print(f"   0.3B: ~2GB peak | 0.7B: ~3GB peak | 1.2B: ~5GB peak")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run model size experiment")
    parser.add_argument("--model", choices=["0.3b", "0.7b", "1.2b", "all"],
                       default="all", help="Which model to train")
    parser.add_argument("--patience", type=int, default=3,
                       help="Early stopping patience (default: 3)")
    parser.add_argument("--evaluate-only", action="store_true",
                       help="Skip training, just evaluate existing adapters")
    parser.add_argument("--baseline", action="store_true",
                       help="Also evaluate base models (no adapter)")
    args = parser.parse_args()
    
    # Determine which models to run
    if args.model == "all":
        models = ["0.3b", "0.7b", "1.2b"]
    else:
        models = [args.model]
    
    print(f"{'='*70}")
    print("CONTROLLED EXPERIMENT: Fine-tuning by Model Size")
    print(f"{'='*70}")
    print(f"\nModels: {', '.join(models)}")
    print(f"Early stopping patience: {args.patience}")
    print(f"\nConstants:")
    print(f"  - Data: data/personal/raw_style_variable")
    print(f"  - LoRA: layers=8, rank=8, alpha=16")
    print(f"  - Training: lr=1e-4, max_iters=500, eval_every=25")
    print(f"{'='*70}")
    
    # Train models
    if not args.evaluate_only:
        results = run_experiment(models, patience=args.patience)
        print_experiment_summary(results)
    
    # Evaluate baselines if requested
    if args.baseline:
        print(f"\n{'='*70}")
        print("EVALUATING BASELINES (no fine-tuning)")
        print(f"{'='*70}")
        
        test_prompts = [
            {"prompt": "+14089089638: oh nice! how far is austin", "expected": "she has a car too..."},
            {"prompt": "+14089089638: where are you now", "expected": "at the place across the street"},
            {"prompt": "Friend: wanna grab food tonight?", "expected": "yeah im down where u thinkin"},
        ]
        
        for model_name in models:
            evaluate_baseline(model_name, test_prompts)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    print(f"  1. Compare adapters: uv run python scripts/training/compare_best_adapters.py --all-0.7b")
    print(f"  2. Full evaluation: uv run python scripts/training/eval_adapters.py --adapter <name> --judge")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
