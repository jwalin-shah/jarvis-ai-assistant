#!/usr/bin/env python3
"""
Train 6-category classifier with BERT embeddings + hand-crafted + spaCy features.

Training data: llm_category_labels.jsonl (Groq Llama 3.3 70B labels)
Categories: closing, acknowledge, question, request, emotion, statement

Feature pipeline:
- 384-dim BERT embeddings (via get_embedder().encode(), normalized)
- 26 hand-crafted features (message structure, context, style, reactions, emotions)
- ~69 spaCy features (14 original + 55 new targeted features)
- 8 new hand-crafted features (from error analysis)
Total: ~487 features

Model: Pipeline with ColumnTransformer (scaling) + LinearSVC
GridSearchCV with n_jobs=1 for 8GB RAM constraint

CRITICAL: Training uses normalize=True for BERT to match serving (FIX train/serve skew)
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis.embedding_adapter import get_embedder
from jarvis.features import CategoryFeatureExtractor, FeatureConfig


def load_training_data(path: Path) -> tuple[list[str], list[str], list[list[str]]]:
    """Load LLM-labeled examples from JSONL."""
    texts = []
    labels = []
    contexts = []

    print(f"Loading training data from {path}...", flush=True)
    with open(path) as f:
        for line in f:
            example = json.loads(line)
            texts.append(example["text"])
            labels.append(example["label"])
            contexts.append(example.get("context", []))

    print(f"Loaded {len(texts)} examples", flush=True)

    # Print class distribution
    from collections import Counter
    dist = Counter(labels)
    total = len(labels)
    print("\nClass distribution:", flush=True)
    for label, count in sorted(dist.items()):
        print(f"  {label}: {count} ({count/total*100:.1f}%)", flush=True)

    return texts, labels, contexts


def extract_bert_embeddings(
    texts: list[str],
    embedder,
    batch_size: int = 100,
    cache_path: Path | None = None,
) -> np.ndarray:
    """Extract BERT embeddings with optional caching.

    CRITICAL: Uses normalize=True to match serving path (FIX train/serve skew).
    """
    # Check cache first
    if cache_path and cache_path.exists():
        print(f"Loading cached BERT embeddings from {cache_path}...", flush=True)
        return np.load(cache_path)

    print(f"\nEncoding BERT embeddings (batches of {batch_size})...", flush=True)
    bert_embeds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT encoding"):
        batch = texts[i:i+batch_size]
        # CRITICAL: normalize=True to match serving
        embeds = embedder.encode(batch, normalize=True)
        bert_embeds.append(embeds)
    bert_embeds = np.vstack(bert_embeds)
    print(f"BERT embeddings shape: {bert_embeds.shape}", flush=True)

    # Save to cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, bert_embeds)
        print(f"Cached BERT embeddings to {cache_path}", flush=True)

    return bert_embeds


def extract_non_bert_features(
    texts: list[str],
    contexts: list[list[str]],
    extractor: CategoryFeatureExtractor,
) -> np.ndarray:
    """Extract all non-BERT features (~103 dims)."""
    print("Extracting non-BERT features (~103 dims)...", flush=True)
    features = []
    for text, context in tqdm(zip(texts, contexts), total=len(texts), desc="Non-BERT features"):
        # extract_all returns 26 + ~69 + 8 = ~103 features
        features.append(extractor.extract_all(text, context))
    features = np.vstack(features)
    print(f"Non-BERT features shape: {features.shape}", flush=True)
    return features


def build_pipeline() -> Pipeline:
    """Build Pipeline with ColumnTransformer scaling + LinearSVC.

    Feature groups:
    - [0:384] = BERT (already normalized, passthrough)
    - [384:391] = Mobilization one-hots (binary, passthrough)
    - [391:~487] = Other features (scale with StandardScaler)
    """
    bert_indices, binary_indices, scale_indices = FeatureConfig.get_scaling_indices()

    preprocessor = ColumnTransformer(
        transformers=[
            ('bert_pass', 'passthrough', bert_indices),
            ('binary_pass', 'passthrough', binary_indices),
            ('count_scale', StandardScaler(), scale_indices),
        ],
        remainder='drop',  # Should never hit this
    )

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('svm', LinearSVC(max_iter=5000, class_weight="balanced", random_state=42)),
    ])

    return pipeline


def train_with_gridsearch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[Pipeline, float]:
    """Train Pipeline with GridSearchCV."""
    print("\n" + "="*70, flush=True)
    print("TRAINING PIPELINE (ColumnTransformer + LinearSVC)", flush=True)
    print("="*70, flush=True)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}", flush=True)
    print(f"Feature dims: {X_train.shape[1]} (~487 expected)", flush=True)

    pipeline = build_pipeline()

    param_grid = {
        "svm__C": [0.01, 0.1, 1.0, 10.0],
    }

    grid = GridSearchCV(
        pipeline,
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

    # Confusion matrix
    print("\nCONFUSION MATRIX:", flush=True)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
    print("Labels:", sorted(set(y_test)), flush=True)
    print(cm, flush=True)

    return grid.best_estimator_, grid.best_score_


def ablation_study(
    bert_train: np.ndarray,
    bert_test: np.ndarray,
    non_bert_train: np.ndarray,
    non_bert_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
):
    """Run ablation experiments to measure incremental feature value.

    A: BERT only (384)
    B: BERT + hand-crafted (26)
    C: B + old spaCy (14)
    D: C + new spaCy (55)
    E: D + new hand-crafted (8)
    F: E with ColumnTransformer scaling
    """
    print("\n" + "="*70, flush=True)
    print("ABLATION STUDY", flush=True)
    print("="*70, flush=True)

    results = []

    # A: BERT only
    print("\nA: BERT only (384 dims)", flush=True)
    X_train_a = bert_train
    X_test_a = bert_test
    svm_a = LinearSVC(C=1.0, max_iter=5000, class_weight="balanced", random_state=42)
    svm_a.fit(X_train_a, y_train)
    y_pred_a = svm_a.predict(X_test_a)
    report_a = classification_report(y_test, y_pred_a, output_dict=True)
    results.append(("A: BERT only", report_a["accuracy"], report_a["macro avg"]["f1-score"]))
    print(f"Accuracy: {report_a['accuracy']:.4f}, F1 (macro): {report_a['macro avg']['f1-score']:.4f}", flush=True)

    # B: BERT + hand-crafted (26)
    print("\nB: BERT + hand-crafted (410 dims)", flush=True)
    X_train_b = np.hstack([bert_train, non_bert_train[:, :26]])
    X_test_b = np.hstack([bert_test, non_bert_test[:, :26]])
    svm_b = LinearSVC(C=1.0, max_iter=5000, class_weight="balanced", random_state=42)
    svm_b.fit(X_train_b, y_train)
    y_pred_b = svm_b.predict(X_test_b)
    report_b = classification_report(y_test, y_pred_b, output_dict=True)
    results.append(("B: + hand-crafted (26)", report_b["accuracy"], report_b["macro avg"]["f1-score"]))
    print(f"Accuracy: {report_b['accuracy']:.4f}, F1 (macro): {report_b['macro avg']['f1-score']:.4f}", flush=True)

    # C: B + old spaCy (14)
    print("\nC: B + old spaCy (424 dims)", flush=True)
    X_train_c = np.hstack([bert_train, non_bert_train[:, :40]])  # 26 + 14
    X_test_c = np.hstack([bert_test, non_bert_test[:, :40]])
    svm_c = LinearSVC(C=1.0, max_iter=5000, class_weight="balanced", random_state=42)
    svm_c.fit(X_train_c, y_train)
    y_pred_c = svm_c.predict(X_test_c)
    report_c = classification_report(y_test, y_pred_c, output_dict=True)
    results.append(("C: + old spaCy (14)", report_c["accuracy"], report_c["macro avg"]["f1-score"]))
    print(f"Accuracy: {report_c['accuracy']:.4f}, F1 (macro): {report_c['macro avg']['f1-score']:.4f}", flush=True)

    # D: C + new spaCy (55)
    print("\nD: C + new spaCy (479 dims)", flush=True)
    X_train_d = np.hstack([bert_train, non_bert_train[:, :95]])  # 26 + 69
    X_test_d = np.hstack([bert_test, non_bert_test[:, :95]])
    svm_d = LinearSVC(C=1.0, max_iter=5000, class_weight="balanced", random_state=42)
    svm_d.fit(X_train_d, y_train)
    y_pred_d = svm_d.predict(X_test_d)
    report_d = classification_report(y_test, y_pred_d, output_dict=True)
    results.append(("D: + new spaCy (55)", report_d["accuracy"], report_d["macro avg"]["f1-score"]))
    print(f"Accuracy: {report_d['accuracy']:.4f}, F1 (macro): {report_d['macro avg']['f1-score']:.4f}", flush=True)

    # E: D + new hand-crafted (8)
    print("\nE: D + new hand-crafted (487 dims)", flush=True)
    X_train_e = np.hstack([bert_train, non_bert_train])  # All features
    X_test_e = np.hstack([bert_test, non_bert_test])
    svm_e = LinearSVC(C=1.0, max_iter=5000, class_weight="balanced", random_state=42)
    svm_e.fit(X_train_e, y_train)
    y_pred_e = svm_e.predict(X_test_e)
    report_e = classification_report(y_test, y_pred_e, output_dict=True)
    results.append(("E: + new hand-crafted (8)", report_e["accuracy"], report_e["macro avg"]["f1-score"]))
    print(f"Accuracy: {report_e['accuracy']:.4f}, F1 (macro): {report_e['macro avg']['f1-score']:.4f}", flush=True)

    # F: E with ColumnTransformer scaling (FINAL)
    print("\nF: E with ColumnTransformer scaling (487 dims)", flush=True)
    pipeline_f = build_pipeline()
    pipeline_f.fit(X_train_e, y_train)
    y_pred_f = pipeline_f.predict(X_test_e)
    report_f = classification_report(y_test, y_pred_f, output_dict=True)
    results.append(("F: + scaling (ColumnTransformer)", report_f["accuracy"], report_f["macro avg"]["f1-score"]))
    print(f"Accuracy: {report_f['accuracy']:.4f}, F1 (macro): {report_f['macro avg']['f1-score']:.4f}", flush=True)

    # Summary table
    print("\n" + "="*70, flush=True)
    print("ABLATION SUMMARY", flush=True)
    print("="*70, flush=True)
    print(f"{'Experiment':<40} {'Accuracy':<10} {'F1 (macro)':<10}", flush=True)
    print("-"*70, flush=True)
    for exp, acc, f1 in results:
        print(f"{exp:<40} {acc:<10.4f} {f1:<10.4f}", flush=True)

    return pipeline_f


def main():
    project_root = Path(__file__).parent.parent
    data_path = project_root / "llm_category_labels.jsonl"
    model_path = project_root / "models" / "category_svm_v2.joblib"
    metadata_path = project_root / "models" / "category_svm_v2_metadata.json"
    cache_path = project_root / "models" / "cache" / "bert_embeddings.npy"

    # Ensure directories exist
    model_path.parent.mkdir(exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    texts, labels, contexts = load_training_data(data_path)
    y = np.array(labels)

    # Stratified train/test split (80/20)
    print("\nSplitting train/test (80/20, stratified)...", flush=True)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(texts, y))

    texts_train = [texts[i] for i in train_idx]
    texts_test = [texts[i] for i in test_idx]
    contexts_train = [contexts[i] for i in train_idx]
    contexts_test = [contexts[i] for i in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    print(f"Train: {len(texts_train)}, Test: {len(texts_test)}", flush=True)

    # Load embedder and feature extractor
    print("\nLoading BERT embedder...", flush=True)
    embedder = get_embedder()

    print("Loading feature extractor...", flush=True)
    extractor = CategoryFeatureExtractor()

    # Extract BERT embeddings (cached)
    bert_train = extract_bert_embeddings(texts_train, embedder, cache_path=None)  # No cache for train set
    bert_test = extract_bert_embeddings(texts_test, embedder, cache_path=None)

    # Extract non-BERT features
    non_bert_train = extract_non_bert_features(texts_train, contexts_train, extractor)
    non_bert_test = extract_non_bert_features(texts_test, contexts_test, extractor)

    # Full feature matrix
    X_train = np.hstack([bert_train, non_bert_train])
    X_test = np.hstack([bert_test, non_bert_test])
    print(f"\nFull feature matrix: {X_train.shape} (expected ~487 dims)", flush=True)

    # Ablation study
    best_pipeline = ablation_study(
        bert_train, bert_test,
        non_bert_train, non_bert_test,
        y_train, y_test
    )

    # Train final model with GridSearchCV (on full feature set)
    print("\n" + "="*70, flush=True)
    print("FINAL MODEL TRAINING (GridSearchCV)", flush=True)
    print("="*70, flush=True)
    final_pipeline, final_score = train_with_gridsearch(X_train, y_train, X_test, y_test)

    # Save pipeline
    print(f"\nSaving pipeline to {model_path}...", flush=True)
    joblib.dump(final_pipeline, model_path)

    # Save metadata
    metadata = {
        "model_type": "Pipeline (ColumnTransformer + LinearSVC)",
        "n_features": X_train.shape[1],
        "feature_breakdown": {
            "bert_embeddings": FeatureConfig.BERT_DIM,
            "hand_crafted": FeatureConfig.HAND_CRAFTED_DIM,
            "spacy": FeatureConfig.SPACY_DIM,
            "new_hand_crafted": FeatureConfig.NEW_HAND_CRAFTED_DIM,
            "total_non_bert": FeatureConfig.TOTAL_NON_BERT,
        },
        "categories": sorted(set(labels)),
        "n_train": len(texts_train),
        "n_test": len(texts_test),
        "cv_f1_macro": float(final_score),
        "best_params": final_pipeline.get_params(),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {metadata_path}", flush=True)
    print(f"\n✓ Training complete! Pipeline saved to {model_path}", flush=True)


if __name__ == "__main__":
    main()
