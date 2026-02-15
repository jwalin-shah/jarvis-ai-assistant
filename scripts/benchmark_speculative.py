import logging
import time

from models.loader import MLXModelLoader, ModelConfig

logging.basicConfig(level=logging.INFO)

TARGET_MODEL = "models/lfm2-1.2b-extract-mlx-4bit"
DRAFT_MODEL = "lfm-350m"


def benchmark_speculative():
    # 1. Setup Loader
    loader = MLXModelLoader(ModelConfig(model_path=TARGET_MODEL))
    loader.load()

    prompt = "Radhika is an Air Import Agent at Expeditors. She is moving to Dallas. She"

    # --- RUN 1: Standard Generation (1.2B only) ---
    print("\n[Run 1] Standard Generation (1.2B)")
    start = time.perf_counter()
    res1 = loader.generate_sync(prompt=prompt, max_tokens=100, temperature=0.0)
    _ = time.perf_counter() - start  # timing for debugging
    print(f"  Speed: {res1.tokens_per_second} tokens/sec")
    print(f"  Result: {res1.text[:50]}...")

    # --- RUN 2: Speculative Generation (1.2B + 350M) ---
    print("\n[Run 2] Speculative Generation (1.2B + 350M Draft)")
    loader.load_draft_model(DRAFT_MODEL)
    start = time.perf_counter()
    # MLXModelLoader detects _draft_model and uses it automatically
    res2 = loader.generate_sync(prompt=prompt, max_tokens=100, temperature=0.0)
    _ = time.perf_counter() - start  # timing for debugging
    print(f"  Speed: {res2.tokens_per_second} tokens/sec")
    print(f"  Acceptance Rate: {res2.acceptance_rate:.2%}")
    print(f"  Result: {res2.text[:50]}...")

    speedup = (res2.tokens_per_second / res1.tokens_per_second) if res1.tokens_per_second > 0 else 0
    print(f"\n--- Total Speedup: {speedup:.2f}x ---")


if __name__ == "__main__":
    benchmark_speculative()
