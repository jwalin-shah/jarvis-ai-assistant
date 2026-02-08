#!/usr/bin/env python3
"""
Train 6-category classifier with BERT embeddings + hand-crafted + spaCy features.

Training data: llm_category_labels.jsonl (Groq Llama 3.3 70B labels)
Categories: closing, acknowledge, question, request, emotion, statement

Feature pipeline:
- 384-dim BERT embeddings (via get_embedder().encode())
- 26 hand-crafted features (message structure, context, style, reactions, emotions)
- 14 spaCy features (imperatives, modals, agreement, etc.)
Total: 424 features

Model: LinearSVC with GridSearchCV (balanced classes, n_jobs=1 for 8GB RAM)
"""

import json
import re
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import spacy
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import LinearSVC
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis.embedding_adapter import get_embedder

# Load spaCy model
print("Loading spaCy model...", flush=True)
nlp = spacy.load("en_core_web_sm")

# Regex patterns for hand-crafted features (synced with production)
EMOJI_RE = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
    r"\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]"
)

PROFESSIONAL_KEYWORDS_RE = re.compile(
    r"\b(meeting|deadline|project|report|schedule|conference|presentation|"
    r"budget|client|invoice|proposal)\b",
    re.IGNORECASE,
)

ABBREVIATION_RE = re.compile(
    r"\b(lol|lmao|omg|wtf|brb|btw|smh|tbh|imo|idk|ngl|fr|rn|ong|nvm|wya|hmu|"
    r"fyi|asap|dm|irl|fomo|goat|sus|bet|cap|no cap)\b",
    re.IGNORECASE,
)


def extract_spacy_features(text: str) -> np.ndarray:
    """Extract 14 SpaCy linguistic features."""
    doc = nlp(text)
    features = []

    # 1. has_imperative: Check for imperative verbs (VB at start)
    has_imperative = 0.0
    if len(doc) > 0 and doc[0].pos_ == "VERB" and doc[0].tag_ == "VB":
        has_imperative = 1.0
    features.append(has_imperative)

    # 2. you_modal: "can you", "could you", "would you", "will you"
    text_lower = text.lower()
    you_modal = 1.0 if any(p in text_lower for p in ["can you", "could you", "would you", "will you", "should you"]) else 0.0
    features.append(you_modal)

    # 3. request_verb: Common request verbs
    request_verbs = {"send", "give", "help", "tell", "show", "let", "call", "get", "make", "take"}
    has_request = 1.0 if any(token.lemma_ in request_verbs for token in doc) else 0.0
    features.append(has_request)

    # 4. starts_modal: Starts with modal verb
    starts_modal = 0.0
    if len(doc) > 0 and doc[0].tag_ in ("MD", "VB"):
        starts_modal = 1.0
    features.append(starts_modal)

    # 5. directive_question: Questions that are really directives
    directive_q = 1.0 if you_modal and "?" in text else 0.0
    features.append(directive_q)

    # 6. i_will: "I'll", "I will", "I'm gonna"
    i_will = 1.0 if any(p in text_lower for p in ["i'll", "i will", "i'm gonna", "ima", "imma"]) else 0.0
    features.append(i_will)

    # 7. promise_verb: Promise/commitment verbs
    promise_verbs = {"promise", "guarantee", "commit", "swear"}
    has_promise = 1.0 if any(token.lemma_ in promise_verbs for token in doc) else 0.0
    features.append(has_promise)

    # 8. first_person_count
    first_person = sum(1 for token in doc if token.text.lower() in ("i", "me", "my", "mine", "myself"))
    features.append(float(first_person))

    # 9. agreement: Agreement words
    agreement_words = {"sure", "okay", "ok", "yes", "yeah", "yep", "yup", "sounds good", "bet", "fs"}
    has_agreement = 1.0 if any(word in text_lower for word in agreement_words) else 0.0
    features.append(has_agreement)

    # 10. modal_count
    modal_count = sum(1 for token in doc if token.tag_ == "MD")
    features.append(float(modal_count))

    # 11. verb_count
    verb_count = sum(1 for token in doc if token.pos_ == "VERB")
    features.append(float(verb_count))

    # 12. second_person_count
    second_person = sum(1 for token in doc if token.text.lower() in ("you", "your", "yours", "yourself"))
    features.append(float(second_person))

    # 13. has_negation
    has_neg = 1.0 if any(token.dep_ == "neg" for token in doc) else 0.0
    features.append(has_neg)

    # 14. is_interrogative: Question indicators
    is_question = 1.0 if "?" in text or any(token.tag_ in ("WDT", "WP", "WP$", "WRB") for token in doc) else 0.0
    features.append(is_question)

    return np.array(features, dtype=np.float32)


def extract_hand_crafted_features(text: str, context: list[str]) -> np.ndarray:
    """Extract 26 hand-crafted features (enhanced with reaction/emotion detection)."""
    features: list[float] = []
    text_lower = text.lower()
    words = text.split()
    total_words = len(words)

    # Message structure (5)
    features.append(float(len(text)))
    features.append(float(total_words))
    features.append(float(text.count("?")))
    features.append(float(text.count("!")))
    features.append(float(len(EMOJI_RE.findall(text))))

    # Mobilization one-hots (7) - default to "none" and "answer" for training
    # These will be provided by mobilization classifier at inference
    for level in ("high", "medium", "low", "none"):
        features.append(1.0 if level == "none" else 0.0)
    for rtype in ("commitment", "answer", "emotional"):
        features.append(1.0 if rtype == "answer" else 0.0)

    # Tone flags (2)
    features.append(1.0 if PROFESSIONAL_KEYWORDS_RE.search(text) else 0.0)
    features.append(1.0 if ABBREVIATION_RE.search(text) else 0.0)

    # Context features (3)
    features.append(float(len(context)))
    avg_ctx_len = float(np.mean([len(m) for m in context])) if context else 0.0
    features.append(avg_ctx_len)
    features.append(1.0 if len(context) == 0 else 0.0)

    # Style features (2)
    abbr_count = len(ABBREVIATION_RE.findall(text))
    features.append(abbr_count / max(total_words, 1))
    capitalized = sum(1 for w in words[1:] if w[0].isupper()) if len(words) > 1 else 0
    features.append(capitalized / max(len(words) - 1, 1))

    # NEW: Reaction/emotion features (7)
    # 1. Is this an iMessage reaction/tapback?
    reaction_patterns = ["Laughed at", "Loved", "Liked", "Disliked", "Emphasized", "Questioned"]
    is_reaction = 1.0 if any(text.startswith(p) for p in reaction_patterns) else 0.0
    features.append(is_reaction)

    # 2. Emotional marker count (lmao, lol, xd, haha, bruh, rip, omg)
    emotional_markers = ["lmao", "lol", "xd", "haha", "omg", "bruh", "rip", "lmfao", "rofl"]
    emotional_count = sum(text_lower.count(marker) for marker in emotional_markers)
    features.append(float(emotional_count))

    # 3. Does message END with emotional marker?
    last_word = words[-1].lower() if words else ""
    ends_with_emotion = 1.0 if last_word in emotional_markers else 0.0
    features.append(ends_with_emotion)

    # 4. Question word at start (what, why, how, when, where, who, did, do, does)
    question_starters = {"what", "why", "how", "when", "where", "who", "did", "do", "does", "can", "could", "would", "will", "should"}
    first_word = words[0].lower() if words else ""
    question_first = 1.0 if first_word in question_starters else 0.0
    features.append(question_first)

    # 5. Imperative verb at start (make, send, get, tell, show, give, come, take)
    imperative_verbs = {"make", "send", "get", "tell", "show", "give", "come", "take", "call", "help", "let"}
    imperative_first = 1.0 if first_word in imperative_verbs else 0.0
    features.append(imperative_first)

    # 6. Brief agreement phrase (ok, yeah, sure, cool, bet)
    brief_agreements = {"ok", "okay", "k", "yeah", "yep", "yup", "sure", "cool", "bet", "fs", "aight"}
    is_brief_agreement = 1.0 if total_words <= 3 and any(w in brief_agreements for w in words) else 0.0
    features.append(is_brief_agreement)

    # 7. Exclamatory ending (!, multiple !!, or all caps)
    exclamatory = 1.0 if (text.endswith("!") or text.isupper() and total_words <= 5) else 0.0
    features.append(exclamatory)

    return np.array(features, dtype=np.float32)


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


def extract_features(
    texts: list[str],
    contexts: list[list[str]],
    embedder,
    batch_size: int = 100,
) -> np.ndarray:
    """Extract all 424 features: 384 BERT + 26 hand-crafted + 14 spaCy."""
    print(f"\nExtracting features for {len(texts)} examples...", flush=True)

    # 1. BERT embeddings (384-dim) in batches
    print("Encoding BERT embeddings (batches of 100)...", flush=True)
    bert_embeds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT encoding"):
        batch = texts[i:i+batch_size]
        embeds = embedder.encode(batch)
        bert_embeds.append(embeds)
    bert_embeds = np.vstack(bert_embeds)
    print(f"BERT embeddings shape: {bert_embeds.shape}", flush=True)

    # 2. Hand-crafted features (26-dim)
    print("Extracting hand-crafted features...", flush=True)
    hand_crafted = []
    for text, context in tqdm(zip(texts, contexts), total=len(texts), desc="Hand-crafted"):
        hand_crafted.append(extract_hand_crafted_features(text, context))
    hand_crafted = np.vstack(hand_crafted)
    print(f"Hand-crafted features shape: {hand_crafted.shape}", flush=True)

    # 3. SpaCy features (14-dim)
    print("Extracting spaCy features...", flush=True)
    spacy_feats = []
    for text in tqdm(texts, desc="SpaCy"):
        spacy_feats.append(extract_spacy_features(text))
    spacy_feats = np.vstack(spacy_feats)
    print(f"SpaCy features shape: {spacy_feats.shape}", flush=True)

    # 4. Concatenate all features
    X = np.hstack([bert_embeds, hand_crafted, spacy_feats])
    print(f"Final feature matrix: {X.shape} (384 BERT + 26 hand + 14 spaCy = {X.shape[1]})", flush=True)

    return X


def train_linearsvc(X_train, y_train, X_test, y_test):
    """Train LinearSVC with GridSearchCV."""
    print("\n" + "="*70, flush=True)
    print("TRAINING LINEARSVC", flush=True)
    print("="*70, flush=True)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}", flush=True)

    # GridSearchCV with n_jobs=1 for 8GB RAM constraint
    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
    }

    svm = LinearSVC(
        max_iter=5000,
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

    return grid.best_estimator_, grid.best_score_


def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM with GridSearchCV."""
    print("\n" + "="*70, flush=True)
    print("TRAINING LIGHTGBM", flush=True)
    print("="*70, flush=True)

    # Encode string labels to integers
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [5, 10],
        "num_leaves": [31, 50],
    }

    lgbm = lgb.LGBMClassifier(
        random_state=42,
        class_weight="balanced",
        verbose=-1,
    )

    grid = GridSearchCV(
        lgbm,
        param_grid,
        cv=5,
        scoring="f1_macro",
        verbose=2,
        n_jobs=1,
    )

    start = time.time()
    grid.fit(X_train, y_train_encoded)
    elapsed = time.time() - start

    print(f"\nTraining completed in {elapsed:.1f}s", flush=True)
    print(f"Best params: {grid.best_params_}", flush=True)
    print(f"Best CV F1 (macro): {grid.best_score_:.4f}", flush=True)

    # Evaluate on test set
    print("\nTEST SET PERFORMANCE:", flush=True)
    y_pred_encoded = grid.predict(X_test)
    y_pred = le.inverse_transform(y_pred_encoded)
    report = classification_report(y_test, y_pred, digits=4)
    print(report, flush=True)

    # Store label encoder in model for later use
    grid.best_estimator_.label_encoder_ = le

    return grid.best_estimator_, grid.best_score_


def main():
    project_root = Path(__file__).parent.parent
    data_path = project_root / "llm_category_labels.jsonl"
    model_path = project_root / "models" / "category_svm_v2.joblib"
    metadata_path = project_root / "models" / "category_svm_v2_metadata.json"

    # Ensure models directory exists
    model_path.parent.mkdir(exist_ok=True)

    # Load data
    texts, labels, contexts = load_training_data(data_path)

    # Convert to numpy arrays
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

    # Load embedder
    print("\nLoading BERT embedder...", flush=True)
    embedder = get_embedder()

    # Extract features
    X_train = extract_features(texts_train, contexts_train, embedder)
    X_test = extract_features(texts_test, contexts_test, embedder)

    # Train LinearSVC only (skip LightGBM for deployment)
    svm_model, svm_score = train_linearsvc(X_train, y_train, X_test, y_test)

    # Select LinearSVC
    print("\n" + "="*70, flush=True)
    print("DEPLOYING LINEARSVC", flush=True)
    print("="*70, flush=True)
    print(f"LinearSVC CV F1:  {svm_score:.4f}", flush=True)
    best_model = svm_model
    model_type = "LinearSVC"

    # Save best model and metadata
    print(f"\nSaving {model_type} to {model_path}...", flush=True)
    joblib.dump(best_model, model_path)

    metadata = {
        "model_type": model_type,
        "n_features": X_train.shape[1],
        "feature_breakdown": {
            "bert_embeddings": 384,
            "hand_crafted": 26,
            "spacy": 14,
        },
        "categories": sorted(set(labels)),
        "n_train": len(texts_train),
        "n_test": len(texts_test),
        "cv_f1_macro": float(svm_score),
        "best_params": best_model.get_params(),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {metadata_path}", flush=True)
    print(f"\n✓ Training complete! Best model: {model_type}", flush=True)


if __name__ == "__main__":
    main()
