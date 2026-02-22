#!/usr/bin/env python3
"""Simple training script for JARVIS LoRA fine-tuning.

This is a thin wrapper around mlx_lm.lora with sensible defaults.
All configuration is in ft_configs/lora_template.yaml.

Usage:
    # Train with default config
    uv run python scripts/training/train.py
    
    # Train with custom config
    uv run python scripts/training/train.py --config ft_configs/my_config.yaml
    
    # Quick test run (10 iterations)
    uv run python scripts/training/train.py --test
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Train JARVIS LoRA adapter")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "ft_configs" / "lora_template.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test run (10 iterations)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        help="Override number of iterations",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: Config not found: {args.config}", file=sys.stderr)
        return 1

    # Build command
    cmd = ["mlx_lm.lora", "--config", str(args.config)]

    if args.test:
        cmd.extend(["--iters", "10", "--steps-per-eval", "5"])
        print("Running test training (10 iterations)...")
    elif args.iters:
        cmd.extend(["--iters", str(args.iters)])

    print(f"Training with config: {args.config}")
    print(f"Command: {' '.join(cmd)}")
    print()

    # Run training
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
