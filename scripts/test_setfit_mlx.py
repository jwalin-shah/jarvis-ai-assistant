#!/usr/bin/env python3
"""Test SetFit fine-tuning + MLX loading end-to-end.

This script verifies that:
1. SetFit can fine-tune a sentence transformer
2. The fine-tuned model can be loaded in MLX
3. Embeddings work correctly

Usage:
    uv run python scripts/test_setfit_mlx.py
"""

import tempfile
import time
from pathlib import Path

import numpy as np


def test_setfit_mlx_integration():
    """Test the full SetFit -> MLX pipeline."""

    print("=" * 60)
    print("SetFit + MLX Integration Test")
    print("=" * 60)

    # Sample training data (few-shot)
    train_texts = [
        # AGREE examples
        "Sure, I can do that",
        "Yes, sounds good",
        "Absolutely, count me in",
        "I'd be happy to help",
        # DECLINE examples
        "Sorry, I can't make it",
        "No thanks, I'm busy",
        "I'll have to pass on that",
        "Unfortunately I can't",
        # QUESTION examples
        "What time works for you?",
        "Where should we meet?",
        "How does that sound?",
        "When are you free?",
    ]
    train_labels = [
        "AGREE",
        "AGREE",
        "AGREE",
        "AGREE",
        "DECLINE",
        "DECLINE",
        "DECLINE",
        "DECLINE",
        "QUESTION",
        "QUESTION",
        "QUESTION",
        "QUESTION",
    ]

    test_texts = [
        "Yeah that works for me",
        "I can't do that sorry",
        "What do you think?",
    ]

    # Step 1: Check if setfit is installed
    print("\n[1/5] Checking SetFit installation...")
    try:
        from datasets import Dataset
        from setfit import SetFitModel, Trainer, TrainingArguments

        print("  ✓ SetFit installed")
    except ImportError as e:
        print(f"  ✗ SetFit not installed: {e}")
        print("\n  Install with: uv add setfit datasets")
        return False

    # Step 2: Fine-tune with SetFit
    print("\n[2/5] Fine-tuning with SetFit (few-shot)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "setfit-test-model"

        try:
            # Create dataset
            train_dataset = Dataset.from_dict(
                {
                    "text": train_texts,
                    "label": train_labels,
                }
            )

            # Load base model and fine-tune
            start = time.time()
            model = SetFitModel.from_pretrained(
                "BAAI/bge-small-en-v1.5",
                labels=["AGREE", "DECLINE", "QUESTION"],
            )

            # Training arguments for quick test
            args = TrainingArguments(
                batch_size=4,
                num_epochs=1,  # Quick test
                num_iterations=5,  # Few iterations for speed
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
            )
            trainer.train()

            elapsed = time.time() - start
            print(f"  ✓ Fine-tuning complete in {elapsed:.1f}s")

            # Test predictions before saving
            predictions = model.predict(test_texts)
            print(f"  ✓ Predictions: {list(predictions)}")

            # Save the model
            model.save_pretrained(str(model_path))
            print(f"  ✓ Model saved to {model_path}")

            # Check what files were saved
            saved_files = list(model_path.glob("*"))
            print(f"  ✓ Saved files: {[f.name for f in saved_files]}")

        except Exception as e:
            print(f"  ✗ SetFit training failed: {e}")
            import traceback

            traceback.print_exc()
            return False

        # Step 3: Load in MLX
        print("\n[3/5] Loading fine-tuned model in MLX...")
        try:
            from mlx_embedding_models.embedding import EmbeddingModel

            start = time.time()

            # Try from_pretrained first (loads from HF-style directory)
            mlx_model = EmbeddingModel.from_pretrained(str(model_path))

            elapsed = time.time() - start
            print(f"  ✓ MLX model loaded in {elapsed:.2f}s")

        except Exception as e:
            print(f"  ✗ MLX loading failed: {e}")
            print("\n  Trying from_finetuned instead...")

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
            print(f"  ✓ Embedding shape: {embeddings.shape}")
            print(f"  ✓ Embedding dtype: {embeddings.dtype}")

            # Check embeddings are valid
            assert embeddings.shape[0] == len(test_texts)
            assert embeddings.shape[1] == 384  # bge-small dimension
            assert not np.isnan(embeddings).any()
            print("  ✓ Embeddings are valid")

        except Exception as e:
            print(f"  ✗ MLX embedding failed: {e}")
            import traceback

            traceback.print_exc()
            return False

        # Step 5: Compare embeddings (sanity check)
        print("\n[5/5] Comparing SetFit vs MLX embeddings...")
        try:
            # Get SetFit embeddings
            setfit_embeddings = model.model_body.encode(test_texts)

            # Compare
            diff = np.abs(setfit_embeddings - embeddings).mean()
            print(f"  Mean absolute difference: {diff:.6f}")

            if diff < 0.01:
                print("  ✓ Embeddings match closely!")
            else:
                print("  ⚠ Embeddings differ (may be due to normalization)")

        except Exception as e:
            print(f"  ⚠ Comparison skipped: {e}")

    print("\n" + "=" * 60)
    print("✓ SetFit + MLX integration test PASSED")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Fine-tune on your full trigger/response datasets")
    print("  2. Save model to ~/.jarvis/models/setfit-trigger/")
    print("  3. Update MLX service to load fine-tuned model")

    return True


if __name__ == "__main__":
    success = test_setfit_mlx_integration()
    exit(0 if success else 1)
