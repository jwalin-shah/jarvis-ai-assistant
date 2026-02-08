#!/usr/bin/env python3
"""Test MLX embedder memory usage with small dataset.

Tests that the memory fixes work before processing full 76k dataset.
"""

import sys
from pathlib import Path

import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def log_memory(label: str) -> float:
    """Log and return current RSS memory in MB."""
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"[MEMORY] {label}: {mem_mb:.1f} MB")
    return mem_mb


def test_embedder_memory(n_texts: int = 1000, batch_size: int = 64):
    """Test embedder with N texts and track memory."""
    print(f"Testing MLX embedder with {n_texts} texts, batch_size={batch_size}")
    print("=" * 70)

    mem_start = log_memory("START")

    # Import and load embedder
    print("\n1. Loading embedder...")
    from jarvis.embedding_adapter import get_embedder

    embedder = get_embedder()
    mem_after_load = log_memory("After get_embedder()")

    # Create test texts
    print(f"\n2. Creating {n_texts} test texts...")
    test_texts = [f"This is test message number {i} with some content." for i in range(n_texts)]
    mem_after_texts = log_memory("After creating texts")

    # Encode texts
    print(f"\n3. Encoding {n_texts} texts...")
    embeddings = embedder.encode(test_texts, normalize=True)
    mem_after_encode = log_memory("After encoding")

    print(f"\n4. Results:")
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Expected: ({n_texts}, 384)")

    # Memory deltas
    print(f"\n5. Memory deltas:")
    print(f"   Embedder load: +{mem_after_load - mem_start:.1f} MB")
    print(f"   Text creation: +{mem_after_texts - mem_after_load:.1f} MB")
    print(f"   Encoding: +{mem_after_encode - mem_after_texts:.1f} MB")
    print(f"   Total increase: +{mem_after_encode - mem_start:.1f} MB")

    # Expected memory usage
    embeddings_size = n_texts * 384 * 4 / 1024 / 1024  # MB
    print(f"\n6. Expected embeddings size: {embeddings_size:.1f} MB")

    # Verdict
    print(f"\n7. Verdict:")
    total_increase = mem_after_encode - mem_start
    if total_increase < 800:
        print(f"   ✓ PASS: Memory increase ({total_increase:.1f} MB) is reasonable")
        return True
    else:
        print(f"   ✗ FAIL: Memory increase ({total_increase:.1f} MB) is too high!")
        print(f"          Expected: <800 MB, Got: {total_increase:.1f} MB")
        return False


def main():
    print("MLX Embedder Memory Test")
    print("=" * 70)

    # Test 1: Small batch
    print("\n\nTEST 1: Small batch (100 texts)")
    print("-" * 70)
    success1 = test_embedder_memory(n_texts=100, batch_size=64)

    # Test 2: Medium batch
    print("\n\nTEST 2: Medium batch (1000 texts)")
    print("-" * 70)
    success2 = test_embedder_memory(n_texts=1000, batch_size=64)

    # Test 3: Larger batch
    print("\n\nTEST 3: Larger batch (5000 texts)")
    print("-" * 70)
    success3 = test_embedder_memory(n_texts=5000, batch_size=64)

    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Test 1 (100 texts):   {'PASS' if success1 else 'FAIL'}")
    print(f"Test 2 (1000 texts):  {'PASS' if success2 else 'FAIL'}")
    print(f"Test 3 (5000 texts):  {'PASS' if success3 else 'FAIL'}")

    if success1 and success2 and success3:
        print("\n✓ All tests passed! MLX memory fixes are working.")
        print("  Ready to process full 76k dataset.")
        return 0
    else:
        print("\n✗ Some tests failed. Memory leak still present.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
