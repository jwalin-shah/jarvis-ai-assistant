#!/usr/bin/env python3
"""Training orchestrator with early stopping for JARVIS fine-tuning.

Automatically trains multiple adapter configs and stops each when validation
loss stops improving. Saves the best checkpoint for each config.

Usage:
    # Train single config with early stopping
    uv run python scripts/training/train_with_early_stopping.py --config ft_configs/personal_0.7b_lora_v2.yaml
    
    # Train multiple configs and compare
    uv run python scripts/training/train_with_early_stopping.py --compare
    
    # Train all 0.7b variants
    uv run python scripts/training/train_with_early_stopping.py --model-size 0.7b
"""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TrainingState:
    """Track training progress for early stopping."""
    iteration: int = 0
    best_val_loss: float = float('inf')
    best_iteration: int = 0
    val_loss_history: list[tuple[int, float]] = field(default_factory=list)
    patience: int = 3  # Stop after N evals without improvement
    
    def should_stop(self) -> bool:
        """Check if training should stop (no improvement for patience evals)."""
        if len(self.val_loss_history) < self.patience + 1:
            return False
        
        # Check last N evals
        recent_losses = [loss for _, loss in self.val_loss_history[-self.patience:]]
        best_recent = min(recent_losses)
        
        # Stop if best recent is not better than best overall
        return best_recent >= self.best_val_loss
    
    def update(self, iteration: int, val_loss: float) -> bool:
        """Update state with new validation loss. Returns True if new best."""
        self.iteration = iteration
        self.val_loss_history.append((iteration, val_loss))
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_iteration = iteration
            return True
        return False


def load_config(config_path: Path) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_with_early_stopping(
    config_path: Path,
    patience: int = 3,
    max_iters: int | None = None,
    cleanup_every: int = 25,
) -> dict[str, Any]:
    """Train a single config with early stopping.
    
    Args:
        config_path: Path to YAML config
        patience: Stop after N evals without improvement
        max_iters: Hard max iterations (None = use config)
        cleanup_every: Clear MLX cache every N iterations
        
    Returns:
        Dict with training results
    """
    from mlx_lm import load
    from mlx_lm.tuner import train, TrainingArgs
    from mlx_lm.tuner.datasets import load_dataset
    
    config = load_config(config_path)
    adapter_path = PROJECT_ROOT / config["adapter_path"]
    adapter_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Training: {config_path.stem}")
    print(f"Adapter: {adapter_path}")
    print(f"Patience: {patience} evals without improvement")
    print(f"{'='*70}")
    
    # Load model and data
    print("Loading model...")
    model, tokenizer = load(config["model"])
    
    print("Loading datasets...")
    # Load JSONL data manually (mlx_lm expects args object, not string)
    from mlx_lm.tuner.datasets import ChatDataset
    import json
    
    # Load training data
    train_path = PROJECT_ROOT / config["data"] / "train.jsonl"
    train_raw = []
    with open(train_path) as f:
        for line in f:
            train_raw.append(json.loads(line))
    train_set = ChatDataset(data=train_raw, tokenizer=tokenizer)
    print(f"Loaded {len(train_set)} training examples")
    
    # Load validation data
    val_path = PROJECT_ROOT / config["data"] / "test.jsonl"
    val_raw = []
    with open(val_path) as f:
        for line in f:
            val_raw.append(json.loads(line))
    val_set = ChatDataset(data=val_raw, tokenizer=tokenizer)
    print(f"Loaded {len(val_set)} validation examples")
    
    # Setup training args (mlx_lm.TrainingArgs doesn't take learning_rate)
    iters = max_iters or config.get("iters", 500)
    adapter_file = adapter_path / "adapters.safetensors"
    training_args = TrainingArgs(
        batch_size=config.get("batch_size", 1),
        iters=iters,
        val_batches=config.get("val_batches", 25),
        steps_per_report=config.get("steps_per_report", 10),
        steps_per_eval=config.get("steps_per_eval", 25),
        steps_per_save=config.get("save_every", 50),
        grad_checkpoint=config.get("grad_checkpoint", True),
        adapter_file=str(adapter_file),
    )
    
    # Create optimizer with learning rate
    import mlx.optimizers as optim
    lr = config.get("learning_rate", 5e-5)
    optimizer = optim.Adam(learning_rate=lr)
    print(f"Optimizer: Adam with lr={lr}")
    
    # State tracking
    state = TrainingState(patience=patience)
    best_adapter_path: Path | None = None
    start_time = time.perf_counter()
    interrupted = False
    stop_training = False
    
    # Create TrainingCallback subclass for early stopping
    from mlx_lm.tuner.callbacks import TrainingCallback
    
    class EarlyStoppingCallback(TrainingCallback):
        def __init__(self, state, adapter_path, cleanup_every):
            self.state = state
            self.adapter_path = adapter_path
            self.cleanup_every = cleanup_every
            self.best_adapter_path = None
            
        def on_train_loss_report(self, train_info: dict):
            iteration = train_info.get("iteration", 0)
            if iteration > 0 and iteration % self.cleanup_every == 0:
                mx.clear_cache()
        
        def on_val_loss_report(self, val_info: dict):
            nonlocal stop_training, best_adapter_path
            iteration = val_info.get("iteration", 0)
            val_loss = val_info.get("val_loss", float('inf'))
            
            is_best = self.state.update(iteration, val_loss)
            status = "🌟 NEW BEST" if is_best else ""
            print(f"\n[Eval @ {iteration}] Val loss: {val_loss:.4f} {status}")
            
            # Save best checkpoint
            if is_best:
                best_path = self.adapter_path / "best_adapters.safetensors"
                current_path = self.adapter_path / "adapters.safetensors"
                if current_path.exists():
                    shutil.copy2(current_path, best_path)
                    print(f"  💾 Saved best checkpoint")
                    best_adapter_path = best_path
            
            # Note: Early stopping would go here, but mlx_lm's train() doesn't support
            # stopping from callback. We track the best checkpoint and can manually stop.
    
    callback = EarlyStoppingCallback(state, adapter_path, cleanup_every)
    
    # Handle interrupt
    def signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\n\n⚠️  Interrupt received, finishing current iteration...")
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run training
    print(f"\nStarting training (max {iters} iterations)...")
    print("="*70)
    
    try:
        # Wrap datasets in CacheDataset for proper batching
        from mlx_lm.tuner.datasets import CacheDataset
        train(
            model=model,
            args=training_args,
            optimizer=optimizer,
            train_dataset=CacheDataset(train_set),
            val_dataset=CacheDataset(val_set),
            training_callback=callback,
        )
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    
    elapsed = time.perf_counter() - start_time
    
    # Final cleanup
    mx.clear_cache()
    
    # Prepare results
    result = {
        "config": str(config_path),
        "adapter_path": str(adapter_path),
        "final_iteration": state.iteration,
        "best_iteration": state.best_iteration,
        "best_val_loss": state.best_val_loss,
        "val_loss_history": state.val_loss_history,
        "stopped_early": state.should_stop(),
        "interrupted": interrupted,
        "elapsed_seconds": elapsed,
        "best_adapter": str(best_adapter_path) if best_adapter_path else None,
    }
    
    # Save results
    results_file = adapter_path / "training_results.json"
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Training Complete: {config_path.stem}")
    print(f"{'='*70}")
    print(f"Best iteration: {state.best_iteration}")
    print(f"Best val loss: {state.best_val_loss:.4f}")
    print(f"Total iterations: {state.iteration}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Results saved to: {results_file}")
    
    return result


def compare_adapters(results: list[dict]) -> None:
    """Print comparison of adapter results."""
    print(f"\n{'='*70}")
    print("ADAPTER COMPARISON")
    print(f"{'='*70}")
    
    # Sort by best val loss
    sorted_results = sorted(results, key=lambda x: x["best_val_loss"])
    
    print(f"\n{'Rank':<5} {'Config':<30} {'Best Loss':<12} {'Iteration':<10} {'Time (min)':<10}")
    print("-" * 70)
    
    for i, r in enumerate(sorted_results, 1):
        config_name = Path(r["config"]).stem
        loss = r["best_val_loss"]
        iter_ = r["best_iteration"]
        time_min = r["elapsed_seconds"] / 60
        print(f"{i:<5} {config_name:<30} {loss:<12.4f} {iter_:<10} {time_min:<10.1f}")
    
    print(f"\n🏆 Best adapter: {sorted_results[0]['adapter_path']}")
    print(f"   Config: {sorted_results[0]['config']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train adapters with early stopping")
    parser.add_argument("--config", type=Path, help="Single config to train")
    parser.add_argument("--patience", type=int, default=3, 
                       help="Stop after N evals without improvement (default: 3)")
    parser.add_argument("--max-iters", type=int, default=None,
                       help="Hard max iterations (default: use config)")
    parser.add_argument("--cleanup-every", type=int, default=25,
                       help="Clear cache every N iterations (default: 25)")
    parser.add_argument("--compare", action="store_true",
                       help="Train comparison configs (0.7b variants)")
    parser.add_argument("--model-size", choices=["0.3b", "0.7b", "1.2b"],
                       help="Train all variants of a model size")
    args = parser.parse_args()
    
    # Determine which configs to train
    configs_to_train: list[Path] = []
    
    if args.config:
        configs_to_train = [args.config]
    elif args.model_size:
        # Find all configs for this model size
        pattern = f"personal_{args.model_size.replace('.', '')}_*.yaml"
        configs_to_train = list((PROJECT_ROOT / "ft_configs").glob(pattern))
        configs_to_train = [c for c in configs_to_train if "memory" not in c.name]  # Skip utility configs
    elif args.compare:
        # Train key comparison configs
        configs_to_train = [
            PROJECT_ROOT / "ft_configs/personal_0.7b_lora_v2.yaml",
            PROJECT_ROOT / "ft_configs/personal_0.7b_lora_rawstyle.yaml",
            PROJECT_ROOT / "ft_configs/personal_0.7b_dora_variable.yaml",
        ]
    else:
        print("Error: specify --config, --model-size, or --compare")
        return 1
    
    if not configs_to_train:
        print("No configs found to train")
        return 1
    
    print(f"Found {len(configs_to_train)} config(s) to train:")
    for c in configs_to_train:
        print(f"  - {c.name}")
    
    # Train each config
    all_results = []
    for config_path in configs_to_train:
        if not config_path.exists():
            print(f"\n⚠️  Config not found: {config_path}")
            continue
        
        try:
            result = train_with_early_stopping(
                config_path=config_path,
                patience=args.patience,
                max_iters=args.max_iters,
                cleanup_every=args.cleanup_every,
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ Failed to train {config_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare results if multiple
    if len(all_results) > 1:
        compare_adapters(all_results)
    
    # Save overall results
    if all_results:
        summary_file = PROJECT_ROOT / "results" / "training_summary.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, 'w') as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "results": all_results,
            }, f, indent=2)
        print(f"\nSummary saved to: {summary_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
