#!/usr/bin/env python3
"""Memory-efficient LoRA training with periodic cache cleanup for 8GB Macs.

This wrapper around mlx_lm adds aggressive memory management to reduce
the "allocated" memory footprint shown in Activity Monitor.
"""

import argparse
import gc
import signal
import sys
import time
from pathlib import Path

import mlx.core as mx
import yaml
from mlx_lm import load, generate
from mlx_lm.tuner import train, TrainingArgs


def clear_memory():
    """Aggressively clear memory caches."""
    # Clear MLX cache
    try:
        mx.clear_cache()
        active = mx.get_active_memory() / (1024**3)
        print(f"  [Memory] Cleared MLX cache. Active: {active:.2f} GB")
    except Exception as e:
        print(f"  [Memory] MLX cache clear failed: {e}")
    
    # Hint Python GC (limited effectiveness but harmless)
    gc.collect()


class MemoryEfficientTrainer:
    """Wrapper that adds memory cleanup to training loop."""
    
    def __init__(self, cleanup_every_n_iters: int = 25):
        self.cleanup_every = cleanup_every_n_iters
        self.iteration = 0
        
    def training_callback(self, iteration: int, train_loss: float, val_loss: float = None):
        """Called after each training iteration."""
        self.iteration = iteration
        
        # Cleanup every N iterations
        if iteration > 0 and iteration % self.cleanup_every == 0:
            print(f"\n[Memory] Cleanup at iteration {iteration}")
            clear_memory()
        
        # Print memory stats
        if iteration % 10 == 0:
            active = mx.get_active_memory() / (1024**3)
            peak = mx.get_peak_memory() / (1024**3)
            print(f"  [Memory] Active: {active:.2f} GB, Peak: {peak:.2f} GB")


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Memory-efficient LoRA training")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--cleanup-every", type=int, default=25,
                       help="Clear cache every N iterations (default: 25)")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    
    # Load model
    print("Loading model...")
    model, tokenizer = load(config["model"])
    
    # Load datasets
    print("Loading datasets...")
    from mlx_lm.tuner.datasets import load_dataset
    train_set = load_dataset(config["data"], tokenizer)
    valid_set = None  # Optional: add validation set
    
    # Set up training args
    training_args = TrainingArgs(
        batch_size=config.get("batch_size", 4),
        iters=config.get("iters", 300),
        val_batches=config.get("val_batches", 25),
        learning_rate=config.get("learning_rate", 1e-4),
        steps_per_report=config.get("steps_per_report", 10),
        steps_per_eval=config.get("steps_per_eval", 50),
        grad_checkpoint=config.get("grad_checkpoint", True),
    )
    
    # Create trainer with memory management
    trainer = MemoryEfficientTrainer(cleanup_every_n_iters=args.cleanup_every)
    
    # Handle interrupt gracefully
    def signal_handler(sig, frame):
        print("\n\nInterrupted! Cleaning up memory...")
        clear_memory()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run training
    print(f"\nStarting training with cleanup every {args.cleanup_every} iterations...")
    print("=" * 60)
    
    # Clear memory before starting
    clear_memory()
    
    train(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_set,
        val_dataset=valid_set,
        training_callback=trainer.training_callback,
    )
    
    # Final cleanup
    print("\n" + "=" * 60)
    print("Training complete! Final memory cleanup...")
    clear_memory()


if __name__ == "__main__":
    main()
