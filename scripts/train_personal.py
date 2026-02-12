"""Wrapper for mlx_lm.lora that sets memory limits before training.

On 8GB systems, MLX will allocate 4-5GB upfront without limits,
causing swap thrashing. This wrapper caps allocation at 1GB with
512MB cache, letting it expand gradually as needed.

Usage:
    uv run python scripts/train_personal.py --config ft_configs/convergence_0.3b.yaml
    uv run python scripts/train_personal.py --config ft_configs/personal_1.2b_lora_cataware.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from jarvis.utils.logging import setup_script_logging

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse wrapper arguments and preserve unknown args for mlx_lm.lora."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Any unrecognized arguments are forwarded to `mlx_lm.lora`. "
            "Example: --config ft_configs/personal_1.2b_lora_cataware.yaml"
        ),
    )
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=1.0,
        help="MLX hard memory limit in GiB before launching mlx_lm.lora (default: %(default)s).",
    )
    parser.add_argument(
        "--cache-limit-mb",
        type=int,
        default=512,
        help="MLX cache limit in MiB before launching mlx_lm.lora (default: %(default)s).",
    )
    return parser.parse_known_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args, passthrough = parse_args(argv)
    setup_script_logging("train_personal")
    import mlx.core as mx

    # Set memory limits BEFORE any model loading (CLI args override centralized defaults)
    memory_limit_bytes = int(args.memory_limit_gb * 1024 * 1024 * 1024)
    cache_limit_bytes = int(args.cache_limit_mb * 1024 * 1024)
    mx.set_memory_limit(memory_limit_bytes)
    mx.set_cache_limit(cache_limit_bytes)

    print(
        (
            f"MLX memory limit: {args.memory_limit_gb:g}GB, "
            f"cache limit: {args.cache_limit_mb}MB (device: {mx.default_device()})"
        ),
        flush=True,
    )

    # Now import and run mlx_lm.lora's main
    logger.info("Launching mlx_lm.lora with args: %s", passthrough)
    from mlx_lm.lora import main as lora_main

    # Pass through all CLI args (strip our script name)
    sys.argv = ["mlx_lm.lora"] + passthrough
    lora_main()


if __name__ == "__main__":
    main()
