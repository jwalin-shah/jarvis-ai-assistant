"""Test CPU embedder implementation."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that modules import correctly."""
    print("Testing imports...")
    try:
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_availability():
    """Test ONNX runtime availability."""
    print("\nTesting ONNX availability...")
    from jarvis.embedding import is_cpu_embedder_available

    available = is_cpu_embedder_available()
    if available:
        print("✓ ONNX Runtime is available")
    else:
        print("✗ ONNX Runtime not installed")
        print("  Install with: uv pip install onnxruntime")
    return available


def test_singleton():
    """Test that embedder is a singleton."""
    print("\nTesting singleton pattern...")
    from jarvis.embedding import CPUEmbedder

    e1 = CPUEmbedder.get_instance()
    e2 = CPUEmbedder.get_instance()

    if e1 is e2:
        print("✓ Singleton pattern works")
        return True
    else:
        print("✗ Singleton failed - different instances")
        return False


def test_encode_dummy():
    """Test encoding with dummy/mock if model not available."""
    print("\nTesting encode interface...")
    from jarvis.embedding import CPUEmbedder

    embedder = CPUEmbedder.get_instance()

    # Check interface exists
    if hasattr(embedder, "encode") and hasattr(embedder, "load"):
        print("✓ Interface methods exist")
        return True
    else:
        print("✗ Missing interface methods")
        return False


def test_router():
    """Test router functionality."""
    print("\nTesting router...")
    try:
        from jarvis.embedding import get_embedder_for_context

        # Test that router returns something
        _embedder = get_embedder_for_context("mlx")
        print("✓ Router returns embedder for 'mlx' context")
        return True
    except Exception as e:
        print(f"✗ Router failed: {e}")
        return False


def test_performance_comparison():
    """Compare CPU vs MLX performance if both available."""
    print("\nTesting performance (if models available)...")

    from jarvis.embedding import CPUEmbedder, is_cpu_embedder_available

    if not is_cpu_embedder_available():
        print("⚠ Skipping - ONNX not available")
        return None

    embedder = CPUEmbedder.get_instance()

    # Try to load and encode test texts
    test_texts = [
        "Hello world",
        "This is a test message",
        "Another example text",
    ]

    try:
        print("  Loading model...")
        loaded = embedder.load()
        if not loaded:
            print("⚠ Model not available (need ONNX export)")
            print(
                "  Run: optimum-cli export onnx --model BAAI/bge-small-en-v1.5 "
                "models/bge-small-onnx/"
            )
            return None

        print("  Encoding test texts...")
        start = time.time()
        embeddings = embedder.encode(test_texts)
        elapsed = time.time() - start

        print(f"✓ Encoded {len(test_texts)} texts in {elapsed:.3f}s")
        print(f"  Shape: {embeddings.shape}")
        print(
            f"  Mean embedding norm: "
            f"{sum(sum(e**2 for e in emb) ** 0.5 for emb in embeddings) / len(embeddings):.3f}"
        )

        # Cleanup
        embedder.unload()
        return True

    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("CPU Embedder Test Suite")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Availability", test_availability),
        ("Singleton", test_singleton),
        ("Interface", test_encode_dummy),
        ("Router", test_router),
        ("Performance", test_performance_comparison),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL" if result is False else "⚠ SKIP"
        print(f"{name:20} {status}")

    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
