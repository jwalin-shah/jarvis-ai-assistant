#!/usr/bin/env python3
"""Test if MLX accumulates memory across batches."""
import sys
from pathlib import Path
import psutil
import gc
import subprocess

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def get_system_memory():
    """Get system memory stats."""
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        'total_gb': vm.total / 1024**3,
        'available_gb': vm.available / 1024**3,
        'used_gb': vm.used / 1024**3,
        'percent': vm.percent,
        'swap_used_gb': swap.used / 1024**3,
    }

def print_memory_pressure():
    """Print system memory pressure."""
    mem = get_system_memory()
    print(f"  SYSTEM: {mem['used_gb']:.1f}/{mem['total_gb']:.1f} GB used ({mem['percent']:.1f}%), swap={mem['swap_used_gb']:.2f} GB")

proc = psutil.Process()

print("\n" + "="*70)
print("SYSTEM MEMORY AT START:")
print_memory_pressure()
print("="*70 + "\n")

from jarvis.embedding_adapter import get_embedder
embedder = get_embedder()

# Warmup
_ = embedder.encode(["warmup"])

mem_start = proc.memory_info()
print(f"\nPROCESS START: RSS={mem_start.rss/1024/1024:.1f} MB")
print("\nEncoding 15 batches of 5000 texts (simulating 75k total)...\n")

batch_size = 5000
num_batches = 15

for batch_num in range(num_batches):
    texts = [f"Message {i} in batch {batch_num}" for i in range(batch_size)]

    mem_before = proc.memory_info()
    embeddings = embedder.encode(texts, normalize=True)
    mem_after = proc.memory_info()

    delta_mb = (mem_after.rss - mem_before.rss) / 1024 / 1024
    total_mb = mem_after.rss / 1024 / 1024

    # Print every batch
    print(f"Batch {batch_num+1:2d}/15: Process RSS={total_mb:6.1f} MB  (+{delta_mb:5.1f} MB)")

    # Print system memory every 5 batches
    if (batch_num + 1) % 5 == 0:
        print_memory_pressure()

    del texts, embeddings
    gc.collect()

mem_end = proc.memory_info()
total_increase = (mem_end.rss - mem_start.rss) / 1024 / 1024

print(f"\n{'='*70}")
print(f"FINAL PROCESS: RSS={mem_end.rss/1024/1024:.1f} MB  (increased by +{total_increase:.1f} MB)")
print(f"\nSYSTEM MEMORY AT END:")
print_memory_pressure()
print(f"{'='*70}")
