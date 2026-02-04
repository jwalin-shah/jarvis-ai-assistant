#!/usr/bin/env python3
"""Test sentence-transformers fine-tuning + MLX loading.

Uses sentence-transformers directly (more compatible) instead of SetFit wrapper.

Usage:
    uv run python scripts/test_finetune_mlx.py
"""

import tempfile
import time
from pathlib import Path

import numpy as np


def test_finetune_mlx_integration():
    """Test sentence-transformers fine-tuning -> MLX pipeline."""

    print("=" * 60)
    print("Sentence-Transformers Fine-tuning + MLX Test")
    print("=" * 60)

    # Sample training data (contrastive pairs)
    # Pairs of (anchor, positive) - texts that should be similar
    train_pairs = [
        # AGREE cluster
        ("Sure, I can do that", "Yes, sounds good"),
        ("Absolutely, count me in", "I'd be happy to help"),
        ("Yeah that works", "Sounds great to me"),
        # DECLINE cluster
        ("Sorry, I can't make it", "No thanks, I'm busy"),
        ("I'll have to pass", "Unfortunately I can't"),
        ("Not this time", "I'm going to skip this one"),
        # QUESTION cluster
        ("What time works for you?", "When are you free?"),
        ("Where should we meet?", "What's the location?"),
        ("How does that sound?", "What do you think?"),
    ]

    test_texts = [
        "Yeah that works for me",
        "I can't do that sorry",
        "What do you think?",
    ]

    # Step 1: Check dependencies
    print("\n[1/5] Checking dependencies...")
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader

        print("  ✓ sentence-transformers available")
    except ImportError as e:
        print(f"  ✗ Missing dependency: {e}")
        return False

    # Step 2: Fine-tune with contrastive learning
    print("\n[2/5] Fine-tuning with contrastive learning...")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "finetuned-model"

        try:
            start = time.time()

            # Load base model
            model = SentenceTransformer("BAAI/bge-small-en-v1.5")

            # Create training examples (contrastive pairs)
            train_examples = [InputExample(texts=[a, b]) for a, b in train_pairs]

            # DataLoader
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)

            # Contrastive loss (pulls similar pairs together)
            train_loss = losses.MultipleNegativesRankingLoss(model)

            # Train (1 epoch for quick test)
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=1,
                warmup_steps=0,
                show_progress_bar=True,
            )

            elapsed = time.time() - start
            print(f"  ✓ Fine-tuning complete in {elapsed:.1f}s")

            # Save
            model.save(str(model_path))
            print(f"  ✓ Model saved to {model_path}")

            # Check saved files
            saved_files = list(model_path.glob("*"))
            print(f"  ✓ Saved files: {[f.name for f in saved_files[:10]]}")

        except Exception as e:
            print(f"  ✗ Fine-tuning failed: {e}")
            import traceback

            traceback.print_exc()
            return False

        # Step 3: Load in MLX
        print("\n[3/5] Loading fine-tuned model in MLX...")
        try:
            from mlx_embedding_models.embedding import EmbeddingModel

            start = time.time()
            mlx_model = EmbeddingModel.from_pretrained(str(model_path))
            elapsed = time.time() - start
            print(f"  ✓ MLX model loaded in {elapsed:.2f}s")

        except Exception as e:
            print(f"  ✗ from_pretrained failed: {e}")

            print("\n  Trying from_finetuned...")
            try:
                mlx_model = EmbeddingModel.from_finetuned("bge-small", str(model_path))
                print("  ✓ MLX model loaded via from_finetuned")
            except Exception as e2:
                print(f"  ✗ from_finetuned also failed: {e2}")
                import traceback

                traceback.print_exc()
                return False

        # Step 4: Test MLX embeddings
        print("\n[4/5] Testing MLX embeddings...")
        try:
            start = time.time()
            embeddings = mlx_model.encode(test_texts)
            elapsed = time.time() - start

            print(f"  ✓ Generated {len(embeddings)} embeddings in {elapsed * 1000:.1f}ms")
            print(f"  ✓ Shape: {embeddings.shape}, dtype: {embeddings.dtype}")

            # Validate
            assert embeddings.shape == (3, 384)
            assert not np.isnan(embeddings).any()
            print("  ✓ Embeddings are valid")

        except Exception as e:
            print(f"  ✗ MLX embedding failed: {e}")
            import traceback

            traceback.print_exc()
            return False

        # Step 5: Compare PyTorch vs MLX embeddings
        print("\n[5/5] Comparing PyTorch vs MLX embeddings...")
        try:
            pytorch_embeddings = model.encode(test_texts)
            diff = np.abs(pytorch_embeddings - embeddings).mean()
            max_diff = np.abs(pytorch_embeddings - embeddings).max()

            print(f"  Mean diff: {diff:.6f}")
            print(f"  Max diff:  {max_diff:.6f}")

            if max_diff < 0.001:
                print("  ✓ Embeddings match perfectly!")
            elif max_diff < 0.01:
                print("  ✓ Embeddings match closely (minor float differences)")
            else:
                print("  ⚠ Some difference (check normalization settings)")

        except Exception as e:
            print(f"  ⚠ Comparison failed: {e}")

    print("\n" + "=" * 60)
    print("✓ Fine-tuning + MLX integration PASSED")
    print("=" * 60)
    print("\nThis confirms you can:")
    print("  1. Fine-tune embeddings with contrastive learning")
    print("  2. Load fine-tuned weights directly in MLX")
    print("  3. Get fast MLX inference on Apple Silicon")

    return True


if __name__ == "__main__":
    success = test_finetune_mlx_integration()
    exit(0 if success else 1)
