"""Wrapper for mlx_lm.lora that sets memory limits before training.

On 8GB systems, MLX will allocate 4-5GB upfront without limits,
causing swap thrashing. This wrapper caps allocation at 1GB with
512MB cache, letting it expand gradually as needed.

Usage:
    uv run python scripts/train_personal.py --config ft_configs/convergence_0.3b.yaml
    uv run python scripts/train_personal.py --config ft_configs/personal_1.2b_lora_cataware.yaml
"""

from __future__ import annotations

import sys

import mlx.core as mx


def main() -> None:
    # Set memory limits BEFORE any model loading
    # 1GB hard limit - MLX will expand gradually, not grab 4-5GB upfront
    mx.set_memory_limit(1 * 1024 * 1024 * 1024)
    # 512MB cache limit - prevents stale tensors from hogging GPU memory
    mx.set_cache_limit(512 * 1024 * 1024)

    print(
        f"MLX memory limit: 1GB, cache limit: 512MB (device: {mx.default_device()})",
        flush=True,
    )

    # Now import and run mlx_lm.lora's main
    from mlx_lm.lora import main as lora_main

    # Pass through all CLI args (strip our script name)
    sys.argv = ["mlx_lm.lora"] + sys.argv[1:]
    lora_main()


if __name__ == "__main__":
    main()
