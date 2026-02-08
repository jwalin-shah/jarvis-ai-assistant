#!/usr/bin/env python3
"""Quick script to train and deploy LinearSVC category classifier."""
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

from jarvis.embeddings.bert_embedder import BERTEmbedder
from jarvis.classifiers.feature_extractor import FeatureExtractor


def load_llm_labels(path: str = "llm_category_labels.jsonl"):
    """Load LLM-generated category labels."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def main():
    print("="*70, flush=True)
    print("DEPLOYING LINEARSVC CATEGORY CLASSIFIER", flush=True)
    print("="*70, flush=True)

    # Load data
    print("\nLoading LLM labels...", flush=True)
    examples = load_llm_labels()
    print(f"Loaded {len(examples)} labeled examples", flush=True)

    # Split train/test
    print("\nSplitting train/test (80/20, stratified)...", flush=True)
    labels = [ex["category"] for ex in examples]
    train_idx, test_idx = train_test_split(
        range(len(examples)),
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )
    train_examples = [examples[i] for i in train_idx]
    test_examples = [examples[i] for i in test_idx]
    print(f"Train: {len(train_examples)}, Test: {len(test_examples)}", flush=True)

    # Extract features
    print("\nLoading BERT embedder...", flush=True)
    embedder = BERTEmbedder()
    feature_extractor = FeatureExtractor(embedder)

    print(f"\nExtracting features for {len(train_examples)} training examples...", flush=True)
    X_train, y_train = feature_extractor.extract_batch(train_examples)

    print(f"\nExtracting features for {len(test_examples)} test examples...", flush=True)
    X_test, y_test = feature_extractor.extract_batch(test_examples)

    print(f"\nFeature matrix: {X_train.shape[0]} samples, {X_train.shape[1]} features", flush=True)

    # Train LinearSVC with grid search
    print("\n" + "="*70, flush=True)
    print("TRAINING LINEARSVC", flush=True)
    print("="*70, flush=True)

    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "max_iter": [5000],
    }

    svm = LinearSVC(
        class_weight="balanced",
        random_state=42,
    )

    grid = GridSearchCV(
        svm,
        param_grid,
        cv=5,
        scoring="f1_macro",
        verbose=2,
        n_jobs=1,  # CRITICAL: n_jobs=1 to avoid swap on 8GB RAM
    )

    start = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"\nTraining completed in {elapsed:.1f}s", flush=True)
    print(f"Best params: {grid.best_params_}", flush=True)
    print(f"Best CV F1 (macro): {grid.best_score_:.4f}", flush=True)

    # Evaluate on test set
    print("\nTEST SET PERFORMANCE:", flush=True)
    y_pred = grid.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)
    print(report, flush=True)

    # Save model
    output_path = Path("models/category_svm_v2.joblib")
    output_path.parent.mkdir(exist_ok=True)

    print(f"\nSaving model to {output_path}...", flush=True)
    joblib.dump(grid.best_estimator_, output_path)

    print(f"\n✅ DEPLOYMENT COMPLETE!", flush=True)
    print(f"   Model: {output_path}", flush=True)
    print(f"   Test F1 (macro): {grid.best_score_:.4f}", flush=True)


if __name__ == "__main__":
    main()
