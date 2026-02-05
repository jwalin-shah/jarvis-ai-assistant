#!/usr/bin/env python3
"""Benchmark all MLX embedding models."""

import subprocess
import sys

# Get model registry from the server script
sys.path.insert(0, "scripts")
from minimal_mlx_embed_server import MODEL_REGISTRY


def benchmark(name: str) -> str | None:
    """Run benchmark in subprocess to get accurate RAM measurement."""
    code = f"""
import resource
import time
import sys
sys.path.insert(0, "scripts")
from minimal_mlx_embed_server import MinimalEmbedder

e = MinimalEmbedder()
e.load_model("{name}")
_ = e.encode(["warmup"])

# Single text latency (5 runs)
single_times = []
for _ in range(5):
    start = time.perf_counter()
    e.encode(["test"])
    single_times.append(time.perf_counter() - start)

# Batch throughput (100 texts, 3 runs)
batch = [f"text {{i}}" for i in range(100)]
batch_times = []
for _ in range(3):
    start = time.perf_counter()
    e.encode(batch)
    batch_times.append(time.perf_counter() - start)

ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
min_single = min(single_times) * 1000
min_batch = min(batch_times) * 1000
throughput = 100 / min(batch_times)

print(f"{{e.config['num_hidden_layers']}},{{e.config['hidden_size']}},{{ram:.0f}},{{min_single:.1f}},{{min_batch:.0f}},{{throughput:.0f}}")
"""
    try:
        r = subprocess.run(
            ["uv", "run", "python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except subprocess.TimeoutExpired:
        return None


def main():
    print(
        f"{'Model':<12} {'Layers':>6} {'Dim':>5} {'RAM':>7} "
        f"{'1 text':>8} {'100 texts':>10} {'Throughput':>12}"
    )
    print("-" * 70)

    for name in MODEL_REGISTRY:
        result = benchmark(name)
        if result:
            layers, dim, ram, single, batch, tput = result.split(",")
            print(
                f"{name:<12} {layers:>6} {dim:>5} {ram:>5}MB "
                f"{single:>6}ms {batch:>8}ms {tput:>10}/s"
            )
        else:
            print(f"{name:<12} FAILED")


if __name__ == "__main__":
    main()
