#!/usr/bin/env python3
"""Train SetFit models for trigger and response classification.

Fine-tunes sentence transformers using contrastive learning (SetFit).
Trains multiple base models and compares results.

IMPORTANT: This script requires the .venv-setfit environment (transformers<5.0).
           Do NOT run with `uv run` - use the setfit venv directly.

Usage:
    # Activate the setfit training environment
    source .venv-setfit/bin/activate

    # Train both classifiers on all models with baseline comparison (all models saved)
    python scripts/train_setfit.py --baseline

    # Train specific classifiers/models
    python scripts/train_setfit.py --classifier trigger --models bge-small bge-micro

    # Quick test with fewer epochs
    python scripts/train_setfit.py --quick

    # Also test MLX loading after saving
    python scripts/train_setfit.py --baseline --test-mlx

    # Deactivate when done
    deactivate
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

# =============================================================================
# Text normalization (standalone - no jarvis imports needed)
# =============================================================================

# Reaction patterns - iMessage tapbacks
REACTION_PATTERNS = [
    r'^Liked\s+".*"$',
    r'^Loved\s+".*"$',
    r'^Disliked\s+".*"$',
    r'^Laughed at\s+".*"$',
    r'^Emphasized\s+".*"$',
    r'^Questioned\s+".*"$',
    r'^Removed a like from\s+".*"$',
    r'^Removed a heart from\s+".*"$',
    r'^Removed a dislike from\s+".*"$',
    r'^Removed a laugh from\s+".*"$',
    r'^Removed an exclamation from\s+".*"$',
    r'^Removed a question mark from\s+".*"$',
]
REACTION_REGEX = re.compile("|".join(REACTION_PATTERNS), re.IGNORECASE | re.DOTALL)

# Auto-signatures to strip
AUTO_SIGNATURE_PATTERNS = [
    r"\n--\s*\n.*$",
    r"\nSent from my iPhone.*$",
    r"\nSent from my iPad.*$",
    r"\nGet Outlook for iOS.*$",
    r"\nSent via.*$",
]
AUTO_SIGNATURE_REGEX = re.compile("|".join(AUTO_SIGNATURE_PATTERNS), re.IGNORECASE | re.DOTALL)

# Repeated emoji pattern
REPEATED_EMOJI_PATTERN = re.compile(
    r"([\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0001F600-\U0001F64F"
    r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF])\1{2,}"
)
_WHITESPACE_PATTERN = re.compile(r"[ \t]+")


def normalize_text(text: str) -> str:
    """Normalize text for training. Returns empty string for reactions."""
    if not text:
        return ""

    # Filter out iMessage reactions
    if REACTION_REGEX.match(text.strip()):
        return ""

    cleaned = text
    cleaned = AUTO_SIGNATURE_REGEX.sub("", cleaned)

    # Normalize whitespace
    lines = cleaned.split("\n")
    normalized_lines = []
    for line in lines:
        line = _WHITESPACE_PATTERN.sub(" ", line.strip())
        if line:
            normalized_lines.append(line)
    cleaned = "\n".join(normalized_lines)

    # Collapse repeated emojis
    cleaned = REPEATED_EMOJI_PATTERN.sub(r"\1\1", cleaned)

    return cleaned.strip()


# Base models to fine-tune (HuggingFace model IDs)
BASE_MODELS = {
    "bge-small": "BAAI/bge-small-en-v1.5",  # 33M params, best frozen baseline
    "gte-tiny": "TaylorAI/gte-tiny",  # 15M params
    "minilm-l6": "sentence-transformers/all-MiniLM-L6-v2",  # 22M params
    "bge-micro": "TaylorAI/bge-micro-v2",  # 8M params
}


@dataclass
class SetFitResult:
    """Result from SetFit training."""

    base_model: str
    classifier_type: str
    train_size: int
    test_size: int
    test_f1: float
    test_accuracy: float
    per_class: dict[str, dict[str, float]]
    training_time: float
    num_epochs: int
    num_iterations: int


def _save_incremental_results(results: list[SetFitResult], output_dir: Path) -> None:
    """Save results incrementally to avoid data loss."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "setfit_results_incremental.json"

    results_data = [
        {
            "base_model": r.base_model,
            "classifier_type": r.classifier_type,
            "train_size": r.train_size,
            "test_size": r.test_size,
            "test_f1": r.test_f1,
            "test_accuracy": r.test_accuracy,
            "per_class": r.per_class,
            "training_time": r.training_time,
            "num_epochs": r.num_epochs,
            "num_iterations": r.num_iterations,
        }
        for r in results
    ]
    results_file.write_text(json.dumps(results_data, indent=2))


def get_memory_usage() -> str:
    """Get current memory usage string."""
    import os

    import psutil

    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024

    # Try to get MPS memory if available
    try:
        import torch

        if torch.backends.mps.is_available():
            # MPS doesn't have direct memory query, estimate from allocated tensors
            mps_allocated = torch.mps.driver_allocated_memory() / 1024 / 1024
            return f"RAM: {mem_mb:.0f}MB, MPS: {mps_allocated:.0f}MB"
    except Exception:
        pass

    return f"RAM: {mem_mb:.0f}MB"


def get_frozen_baseline(
    model_id: str,
    train_texts: list[str],
    train_labels: list[str],
    test_texts: list[str],
    test_labels: list[str],
) -> float:
    """Get baseline F1 using frozen embeddings + logistic regression.

    This shows what the base model can do WITHOUT fine-tuning,
    so you can measure the improvement from SetFit.
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    print("    Computing frozen baseline...")

    # Get embeddings
    model = SentenceTransformer(model_id)
    train_embeddings = model.encode(train_texts, show_progress_bar=False)
    test_embeddings = model.encode(test_texts, show_progress_bar=False)

    # Clean up embedding model immediately
    del model
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

    # Train simple classifier
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(train_embeddings, train_labels)

    # Evaluate
    predictions = clf.predict(test_embeddings)
    baseline_f1 = f1_score(test_labels, predictions, average="macro", zero_division=0)

    return baseline_f1


def select_best_examples(
    texts: list[str],
    labels: list[str],
    model_id: str,
    samples_per_class: int = 16,
) -> tuple[list[str], list[str]]:
    """Select the most representative examples per class using centroid selection.

    Uses embeddings to find examples closest to each class centroid,
    which gives the clearest/most typical examples of each class.

    Args:
        texts: All text samples
        labels: Corresponding labels
        model_id: Sentence transformer model to use for embeddings
        samples_per_class: Number of examples to select per class

    Returns:
        Tuple of (selected_texts, selected_labels)
    """
    from sentence_transformers import SentenceTransformer

    print(f"    Selecting {samples_per_class} best examples per class...")

    # Get embeddings for all texts
    model = SentenceTransformer(model_id)
    embeddings = model.encode(texts, show_progress_bar=True)

    # Clean up model
    del model
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

    # Convert to numpy for easier manipulation
    texts_arr = np.array(texts)
    labels_arr = np.array(labels)

    selected_texts = []
    selected_labels = []

    unique_labels = sorted(set(labels))
    for label in unique_labels:
        # Get indices for this class
        mask = labels_arr == label
        class_indices = np.where(mask)[0]
        class_embeddings = embeddings[mask]

        # Compute centroid
        centroid = class_embeddings.mean(axis=0)

        # Compute distances to centroid
        distances = np.linalg.norm(class_embeddings - centroid, axis=1)

        # Select closest to centroid (most representative)
        n_select = min(samples_per_class, len(class_indices))
        closest_indices = np.argsort(distances)[:n_select]

        # Get the original indices
        original_indices = class_indices[closest_indices]

        selected_texts.extend(texts_arr[original_indices].tolist())
        selected_labels.extend([label] * n_select)

        print(f"      {label}: selected {n_select}/{len(class_indices)} examples")

    print(f"    Total selected: {len(selected_texts)} examples")
    return selected_texts, selected_labels


def load_trigger_data(
    files: list[str],
    apply_normalization: bool = True,
) -> tuple[list[str], list[str]]:
    """Load labeled trigger data from JSONL files."""
    texts = []
    labels = []
    seen: set[str] = set()

    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            print(f"  Warning: {file_path} not found, skipping")
            continue

        count = 0
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text", "").strip()
                label = row.get("label") or row.get("auto_label")

                if text and label:
                    if apply_normalization:
                        text = normalize_text(text)
                        if not text:
                            continue

                    if text in seen:
                        continue
                    seen.add(text)

                    texts.append(text)
                    labels.append(label.lower())
                    count += 1

        print(f"  {path.name}: {count} samples")

    return texts, labels


def load_response_data(
    files: list[str],
    apply_normalization: bool = True,
) -> tuple[list[str], list[str]]:
    """Load labeled response data from JSONL files."""
    texts = []
    labels = []
    seen: set[str] = set()

    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            print(f"  Warning: {file_path} not found, skipping")
            continue

        count = 0
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("response", "").strip() or row.get("text", "").strip()
                label = row.get("label")

                if text and label:
                    if apply_normalization:
                        text = normalize_text(text)
                        if not text:
                            continue

                    if text in seen:
                        continue
                    seen.add(text)

                    texts.append(text)
                    labels.append(label.upper())
                    count += 1

        print(f"  {path.name}: {count} samples")

    return texts, labels


def train_setfit_model(
    base_model_id: str,
    train_texts: list[str],
    train_labels: list[str],
    test_texts: list[str],
    test_labels: list[str],
    classifier_type: str,
    num_epochs: int = 1,
    num_iterations: int = 20,
    batch_size: int = 16,
) -> tuple[SetFitResult, any]:
    """Train a SetFit model and evaluate it.

    Args:
        base_model_id: HuggingFace model ID (e.g., "BAAI/bge-small-en-v1.5")
        train_texts: Training texts
        train_labels: Training labels
        test_texts: Test texts
        test_labels: Test labels
        classifier_type: "trigger" or "response"
        num_epochs: Number of training epochs (contrastive + classifier head)
        num_iterations: Number of text pairs to generate per class
        batch_size: Training batch size

    Returns:
        Tuple of (SetFitResult, trained model)
    """
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    # Create datasets
    train_dataset = Dataset.from_dict(
        {
            "text": train_texts,
            "label": train_labels,
        }
    )

    # Get unique labels
    unique_labels = sorted(set(train_labels))
    print(f"    Labels: {unique_labels}")

    # Load base model
    print(f"    Loading {base_model_id}...")
    model = SetFitModel.from_pretrained(
        base_model_id,
        labels=unique_labels,
    )

    # Training arguments
    # SetFit uses contrastive learning on pairs, then trains a classification head
    args = TrainingArguments(
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_iterations=num_iterations,  # Pairs per class for contrastive learning
        # With 4800 samples, we have plenty of data - can use fewer iterations
        # Default is 20, which generates 20 * num_classes * 2 pairs per epoch
    )

    # Train
    print(f"    Training (epochs={num_epochs}, iterations={num_iterations})...")
    start = time.time()

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )
    trainer.train()

    training_time = time.time() - start
    print(f"    Training complete in {training_time:.1f}s")

    # Evaluate on test set
    print("    Evaluating...")
    predictions = model.predict(test_texts)
    predictions = [str(p) for p in predictions]  # Ensure string labels

    # Calculate metrics
    test_f1 = f1_score(test_labels, predictions, average="macro", zero_division=0)
    test_accuracy = sum(p == t for p, t in zip(predictions, test_labels)) / len(test_labels)

    # Per-class metrics
    report = classification_report(test_labels, predictions, output_dict=True, zero_division=0)
    per_class = {
        k: v for k, v in report.items() if k not in ["accuracy", "macro avg", "weighted avg"]
    }

    result = SetFitResult(
        base_model=base_model_id,
        classifier_type=classifier_type,
        train_size=len(train_texts),
        test_size=len(test_texts),
        test_f1=test_f1,
        test_accuracy=test_accuracy,
        per_class=per_class,
        training_time=training_time,
        num_epochs=num_epochs,
        num_iterations=num_iterations,
    )

    return result, model


def save_setfit_model(
    model: any,
    classifier_type: str,
    base_model_name: str,
    output_dir: Path | None = None,
) -> Path:
    """Save a trained SetFit model.

    Args:
        model: Trained SetFit model
        classifier_type: "trigger" or "response"
        base_model_name: Short name like "bge-small"
        output_dir: Output directory (default: ~/.jarvis/models/)

    Returns:
        Path where model was saved
    """
    if output_dir is None:
        output_dir = Path.home() / ".jarvis" / "models"

    model_path = output_dir / f"setfit-{classifier_type}-{base_model_name}"
    model_path.mkdir(parents=True, exist_ok=True)

    print(f"  Saving to {model_path}...")
    model.save_pretrained(str(model_path))

    # List saved files
    saved_files = list(model_path.glob("*"))
    print(f"  Saved files: {[f.name for f in saved_files]}")

    return model_path


def test_mlx_loading(model_path: Path) -> bool:
    """Test that a saved SetFit model can be loaded in MLX.

    Returns True if loading succeeds.
    """
    print(f"  Testing MLX loading from {model_path}...")
    try:
        from mlx_embedding_models.embedding import EmbeddingModel

        mlx_model = EmbeddingModel.from_pretrained(str(model_path))

        # Quick test
        test_texts = ["hello world", "test message"]
        embeddings = mlx_model.encode(test_texts)

        assert embeddings.shape[0] == 2
        assert not np.isnan(embeddings).any()

        print(f"  ✓ MLX loading successful (embeddings shape: {embeddings.shape})")
        return True

    except Exception as e:
        print(f"  ✗ MLX loading failed: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SetFit models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(BASE_MODELS.keys()),
        choices=list(BASE_MODELS.keys()),
        help="Base models to fine-tune",
    )
    parser.add_argument(
        "--classifier",
        nargs="+",
        default=["trigger", "response"],
        choices=["trigger", "response"],
        help="Classifier types to train",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Pairs per class for contrastive learning (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (1 epoch, 5 iterations)",
    )
    parser.add_argument(
        "--test-mlx",
        action="store_true",
        help="Test MLX loading after saving each model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / ".jarvis" / "models",
        help="Output directory for saved models",
    )
    parser.add_argument(
        "--trigger-data",
        type=Path,
        default=Path("data/trigger_labeling.jsonl"),
        help="Path to trigger labeling data",
    )
    parser.add_argument(
        "--response-data",
        type=Path,
        default=Path("data/response_labeling.jsonl"),
        help="Path to response labeling data",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Compute frozen baseline for comparison (shows improvement from fine-tuning)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only run preflight check (verify all models load) without training",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run hyperparameter sweep over epochs/iterations combinations",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=0,
        help="Select N best examples per class using centroid selection (0 = use all data)",
    )
    parser.add_argument(
        "--use-curated",
        action="store_true",
        help="Use hand-curated examples from results/setfit_training/selected_*.jsonl",
    )
    parser.add_argument(
        "--select-only",
        action="store_true",
        help="Only select and show best examples without training (use with --samples-per-class)",
    )
    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.epochs = 1
        args.iterations = 5
        print("Quick mode: epochs=1, iterations=5")

    print("=" * 70)
    print("SETFIT TRAINING")
    print("=" * 70)
    print(f"Base models: {args.models}")
    print(f"Classifiers: {args.classifier}")
    print(f"Epochs: {args.epochs}")
    print(f"Iterations per class: {args.iterations}")
    print(f"Batch size: {args.batch_size}")
    print(f"Test split: {args.test_split}")

    # Check SetFit installation
    try:
        from setfit import SetFitModel  # noqa: F401

        print("✓ SetFit installed")
    except ImportError:
        print("✗ SetFit not installed. Run: uv add setfit datasets")
        return

    # Preflight check: verify all models can load
    print("\n" + "=" * 70)
    print("PREFLIGHT CHECK")
    print("=" * 70)
    from sentence_transformers import SentenceTransformer

    all_ok = True
    for model_name in args.models:
        model_id = BASE_MODELS[model_name]
        try:
            print(f"  Loading {model_name}...", end=" ", flush=True)
            model = SentenceTransformer(model_id)
            # Quick test
            emb = model.encode(["test message"], show_progress_bar=False)
            assert emb.shape[0] == 1
            print(f"✓ OK (dim={emb.shape[1]})")
            # Clean up
            del model, emb
            gc.collect()
            try:
                import torch

                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass
        except Exception as e:
            print(f"✗ FAILED: {e}")
            all_ok = False

    if not all_ok:
        print("\n✗ Some models failed preflight check. Aborting.")
        return

    print("✓ All models passed preflight check")

    if args.check_only:
        print("\n--check-only specified, exiting without training.")
        return

    # Pre-load all data upfront (before loading any models)
    all_results: list[SetFitResult] = []
    best_models: dict[str, SetFitResult] = {}  # classifier_type -> best result
    baselines: dict[str, dict[str, float]] = {}  # classifier_type -> {model_name: baseline_f1}
    datasets: dict[str, dict] = {}  # classifier_type -> {train_texts, test_texts, ...}

    for classifier_type in args.classifier:
        print(f"\nLoading {classifier_type} data...")
        if classifier_type == "trigger":
            texts, labels = load_trigger_data([str(args.trigger_data)])
        else:
            texts, labels = load_response_data([str(args.response_data)])

        print(f"  Total: {len(texts)} samples")
        print(f"  Distribution: {dict(Counter(labels))}")

        # Split train/test
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts,
            labels,
            test_size=args.test_split,
            random_state=42,
            stratify=labels,
        )
        print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}")

        datasets[classifier_type] = {
            "train_texts": train_texts,
            "train_texts_full": train_texts,  # Keep full set for baseline
            "test_texts": test_texts,
            "train_labels": train_labels,
            "train_labels_full": train_labels,  # Keep full set for baseline
            "test_labels": test_labels,
        }
        baselines[classifier_type] = {}

    # Select best examples if requested (do this once per classifier, before model loop)
    if args.samples_per_class > 0:
        print(f"\n{'=' * 70}")
        print(f"SELECTING BEST EXAMPLES ({args.samples_per_class} per class)")
        print(f"{'=' * 70}")

        # Use the first model for selection (they should give similar results)
        selection_model_id = BASE_MODELS[args.models[0]]

        for classifier_type in args.classifier:
            data = datasets[classifier_type]
            print(f"\n{classifier_type}:")

            selected_texts, selected_labels = select_best_examples(
                texts=data["train_texts_full"],
                labels=data["train_labels_full"],
                model_id=selection_model_id,
                samples_per_class=args.samples_per_class,
            )

            # Update training data with selected subset
            datasets[classifier_type]["train_texts"] = selected_texts
            datasets[classifier_type]["train_labels"] = selected_labels

            # Save selected examples for review
            selected_file = Path(f"results/setfit_training/selected_{classifier_type}.jsonl")
            selected_file.parent.mkdir(parents=True, exist_ok=True)
            with open(selected_file, "w") as f:
                for text, label in zip(selected_texts, selected_labels):
                    f.write(json.dumps({"text": text, "label": label}) + "\n")
            print(f"    Saved selected examples to: {selected_file}")

        # Show sample of selected examples
        print(f"\n{'=' * 70}")
        print("SAMPLE OF SELECTED EXAMPLES (review for quality)")
        print(f"{'=' * 70}")
        for classifier_type in args.classifier:
            data = datasets[classifier_type]
            unique_labels = sorted(set(data["train_labels"]))
            print(f"\n{classifier_type.upper()}:")
            for label in unique_labels:
                print(f"\n  [{label}]")
                # Show first 3 examples of each class
                examples = [
                    t for t, l in zip(data["train_texts"], data["train_labels"]) if l == label
                ][:3]
                for i, ex in enumerate(examples, 1):
                    # Truncate long examples
                    display = ex[:80] + "..." if len(ex) > 80 else ex
                    print(f"    {i}. {display!r}")

        if args.select_only:
            print("\n--select-only specified. Review the examples above and in:")
            for classifier_type in args.classifier:
                print(f"  results/setfit_training/selected_{classifier_type}.jsonl")
            print("\nTo train, run again without --select-only")
            return

    # Load hand-curated examples if requested
    if args.use_curated:
        print(f"\n{'=' * 70}")
        print("LOADING CURATED EXAMPLES")
        print(f"{'=' * 70}")

        for classifier_type in args.classifier:
            curated_file = Path(f"results/setfit_training/selected_{classifier_type}.jsonl")
            if not curated_file.exists():
                print(f"  ✗ {curated_file} not found!")
                print("    Run with --samples-per-class first, then manually curate the file.")
                return

            # Load curated examples
            curated_texts = []
            curated_labels = []
            with open(curated_file) as f:
                for line in f:
                    item = json.loads(line)
                    curated_texts.append(item["text"])
                    curated_labels.append(item["label"])

            datasets[classifier_type]["train_texts"] = curated_texts
            datasets[classifier_type]["train_labels"] = curated_labels

            # Show distribution
            dist = Counter(curated_labels)
            print(f"  {classifier_type}: {len(curated_texts)} examples")
            print(f"    Distribution: {dict(dist)}")

    # Helper to clean up GPU memory
    def cleanup_model(model, msg=""):
        if model is not None:
            del model
        gc.collect()
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        if msg:
            print(f"    ({msg})")

    # Train each model on all classifier types before moving to next model
    # This keeps only one model in memory at a time
    for model_name in args.models:
        model_id = BASE_MODELS[model_name]
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_name} ({model_id})")
        print(f"{'=' * 70}")

        for classifier_type in args.classifier:
            data = datasets[classifier_type]
            print(f"\n--- {classifier_type.upper()} ---")

            # Compute frozen baseline if requested
            # Uses same training data as SetFit for fair comparison
            if args.baseline:
                baseline_f1 = get_frozen_baseline(
                    model_id,
                    data["train_texts"],
                    data["train_labels"],
                    data["test_texts"],
                    data["test_labels"],
                )
                baselines[classifier_type][model_name] = baseline_f1
                n_samples = len(data["train_texts"])
                print(f"    Frozen baseline F1: {baseline_f1:.3f} (LogReg on {n_samples} samples)")

            train_size = len(data["train_texts"])
            print(f"    Training SetFit on {train_size} samples...")

            result, model = train_setfit_model(
                base_model_id=model_id,
                train_texts=data["train_texts"],
                train_labels=data["train_labels"],
                test_texts=data["test_texts"],
                test_labels=data["test_labels"],
                classifier_type=classifier_type,
                num_epochs=args.epochs,
                num_iterations=args.iterations,
                batch_size=args.batch_size,
            )

            all_results.append(result)
            print(f"    Test F1: {result.test_f1:.3f}")
            print(f"    Test Accuracy: {result.test_accuracy:.3f}")

            # Save incrementally (don't lose progress if crashes)
            _save_incremental_results(all_results, Path("results/setfit_training"))

            # Track best per classifier type (just the result, not the model)
            if (
                classifier_type not in best_models
                or result.test_f1 > best_models[classifier_type].test_f1
            ):
                best_models[classifier_type] = result

            # Always save the model, then clean up immediately
            model_path = save_setfit_model(model, classifier_type, model_name, args.output)
            if args.test_mlx:
                test_mlx_loading(model_path)

            # Clean up immediately after saving
            cleanup_model(model, f"cleaned up, {get_memory_usage()}")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    if args.baseline:
        print(
            f"\n{'Model':<15} {'Type':<10} {'Baseline':>10} {'SetFit F1':>10} {'Δ':>8} {'Time':>10}"
        )
        print("-" * 75)

        for r in sorted(all_results, key=lambda x: (x.classifier_type, -x.test_f1)):
            model_name = next(k for k, v in BASE_MODELS.items() if v == r.base_model)
            model_short = r.base_model.split("/")[-1][:15]
            baseline = baselines.get(r.classifier_type, {}).get(model_name, 0)
            delta = r.test_f1 - baseline if baseline else 0
            delta_str = f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}"
            print(
                f"{model_short:<15} {r.classifier_type:<10} {baseline:>10.3f} "
                f"{r.test_f1:>10.3f} {delta_str:>8} {r.training_time:>8.1f}s"
            )
    else:
        print(f"\n{'Model':<15} {'Type':<10} {'Test F1':>10} {'Accuracy':>10} {'Time':>10}")
        print("-" * 60)

        for r in sorted(all_results, key=lambda x: (x.classifier_type, -x.test_f1)):
            model_short = r.base_model.split("/")[-1][:15]
            print(
                f"{model_short:<15} {r.classifier_type:<10} {r.test_f1:>10.3f} "
                f"{r.test_accuracy:>10.3f} {r.training_time:>8.1f}s"
            )

    # Show best models per classifier type
    print("\n" + "=" * 70)
    print("BEST MODELS")
    print("=" * 70)
    for classifier_type, result in best_models.items():
        model_name = next(k for k, v in BASE_MODELS.items() if v == result.base_model)
        print(f"  {classifier_type}: {model_name} (F1={result.test_f1:.3f})")
        print(f"    Saved at: {args.output}/setfit-{classifier_type}-{model_name}")

    # Save comprehensive results to JSON
    results_dir = Path("results/setfit_training")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "setfit_results.json"

    results_data = []
    for r in all_results:
        model_name = next(k for k, v in BASE_MODELS.items() if v == r.base_model)
        baseline = baselines.get(r.classifier_type, {}).get(model_name)
        results_data.append(
            {
                "base_model": r.base_model,
                "model_name": model_name,
                "classifier_type": r.classifier_type,
                "train_size": r.train_size,
                "test_size": r.test_size,
                "baseline_f1": baseline,
                "test_f1": r.test_f1,
                "improvement": r.test_f1 - baseline if baseline else None,
                "test_accuracy": r.test_accuracy,
                "per_class": r.per_class,
                "training_time": r.training_time,
                "num_epochs": r.num_epochs,
                "num_iterations": r.num_iterations,
            }
        )
    results_file.write_text(json.dumps(results_data, indent=2))
    print(f"\nResults saved to {results_file}")

    # Save detailed per-class comparison
    print("\n" + "=" * 70)
    print("PER-CLASS F1 SCORES")
    print("=" * 70)

    for classifier_type in args.classifier:
        type_results = [r for r in all_results if r.classifier_type == classifier_type]
        if not type_results:
            continue

        print(f"\n{classifier_type.upper()}:")

        # Get all class labels
        all_classes = set()
        for r in type_results:
            all_classes.update(r.per_class.keys())
        all_classes = sorted(all_classes)

        # Print header
        header = f"{'Model':<20}"
        for cls in all_classes:
            header += f" {cls[:8]:>8}"
        header += f" {'MACRO':>8}"
        print(header)
        print("-" * len(header))

        # Print each model's per-class F1
        for r in sorted(type_results, key=lambda x: -x.test_f1):
            model_short = r.base_model.split("/")[-1][:20]
            row = f"{model_short:<20}"
            for cls in all_classes:
                if cls in r.per_class:
                    row += f" {r.per_class[cls]['f1-score']:>8.3f}"
                else:
                    row += f" {'N/A':>8}"
            row += f" {r.test_f1:>8.3f}"
            print(row)

    # Save as CSV for easy comparison
    csv_file = results_dir / "setfit_comparison.csv"
    with open(csv_file, "w") as f:
        # Header
        f.write("classifier_type,model_name,base_model,baseline_f1,test_f1,improvement,")
        f.write("test_accuracy,train_size,training_time,")
        # Add per-class columns dynamically
        all_classes_combined = set()
        for r in all_results:
            all_classes_combined.update(r.per_class.keys())
        all_classes_combined = sorted(all_classes_combined)
        f.write(",".join(f"f1_{cls}" for cls in all_classes_combined))
        f.write("\n")

        # Data rows
        for r in all_results:
            model_name = next(k for k, v in BASE_MODELS.items() if v == r.base_model)
            model_short = r.base_model.split("/")[-1]
            baseline = baselines.get(r.classifier_type, {}).get(model_name)
            improvement = r.test_f1 - baseline if baseline else None

            f.write(f"{r.classifier_type},{model_name},{model_short},")
            f.write(f"{baseline:.4f}," if baseline else ",")
            f.write(f"{r.test_f1:.4f},")
            f.write(f"{improvement:.4f}," if improvement else ",")
            f.write(f"{r.test_accuracy:.4f},{r.train_size},{r.training_time:.1f},")

            per_class_f1s = []
            for cls in all_classes_combined:
                if cls in r.per_class:
                    per_class_f1s.append(f"{r.per_class[cls]['f1-score']:.4f}")
                else:
                    per_class_f1s.append("")
            f.write(",".join(per_class_f1s))
            f.write("\n")

    print(f"\nCSV saved to {csv_file}")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Review results - did fine-tuning improve F1 scores?
   All models have been saved to: ~/.jarvis/models/setfit-{classifier}-{model}/

2. Update jarvis config to use the best model:
   - MLX can load it with: EmbeddingModel.from_pretrained(path)

3. For production, you may want to:
   - Increase epochs (try 2-3) for better convergence
   - Increase iterations (try 30-50) for more contrastive pairs
   - Add more training data if available
""")


if __name__ == "__main__":
    main()
