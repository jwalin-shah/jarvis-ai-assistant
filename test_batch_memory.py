#!/usr/bin/env python3
"""Test memory usage during batch embedding."""
import sys
from pathlib import Path
import psutil
import time

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

proc = psutil.Process()

print("=" * 80)
print("BATCH EMBEDDING MEMORY TEST")
print("=" * 80)

mem = proc.memory_info()
print(f"\n[MAIN] START: RSS={mem.rss/1024/1024:.1f} MB")

from jarvis.embedding_adapter import get_embedder
embedder = get_embedder()

mem = proc.memory_info()
print(f"[MAIN] After embedder load: RSS={mem.rss/1024/1024:.1f} MB")

# Encode 1 text (warmup)
_ = embedder.encode(["warmup"])
mem = proc.memory_info()
print(f"[MAIN] After warmup: RSS={mem.rss/1024/1024:.1f} MB")

# Test different batch sizes
for batch_size in [100, 1000, 5000]:
    texts = [f"This is test message number {i}" for i in range(batch_size)]

    mem_before = proc.memory_info()
    print(f"\n[BATCH {batch_size}] BEFORE encode: RSS={mem_before.rss/1024/1024:.1f} MB")

    embeddings = embedder.encode(texts, normalize=True)

    mem_after = proc.memory_info()
    print(f"[BATCH {batch_size}] AFTER encode: RSS={mem_after.rss/1024/1024:.1f} MB (+{(mem_after.rss-mem_before.rss)/1024/1024:.1f})")
    print(f"[BATCH {batch_size}] Embedding shape: {embeddings.shape}")

    del embeddings, texts

mem_final = proc.memory_info()
print(f"\n[MAIN] FINAL: RSS={mem_final.rss/1024/1024:.1f} MB")

print("\n" + "=" * 80)
print("CHECK ACTIVITY MONITOR - what's total memory now?")
print(f"psutil RSS: {mem_final.rss/1024/1024:.1f} MB")
print("=" * 80)

print("\nWaiting 30 seconds for you to check Activity Monitor...")
try:
    time.sleep(30)
except KeyboardInterrupt:
    print("\nInterrupted")
