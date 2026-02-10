#!/usr/bin/env python3
"""
Train a lightweight classifier to filter GLiNER fact candidates.

This is the second stage of the two-stage pipeline:
1. GLiNER extracts candidates (high recall)
2. This classifier filters false positives (high precision)

The classifier is intentionally small (LogisticRegression or LinearSVC)
to run fast on CPU with minimal memory (~MB, not GB).

Usage:
    # Train on synthetic/heuristic labeled data
    python scripts/train_fact_filter.py --input training_data/fact_candidates.jsonl

    # Evaluate on test set
    python scripts/train_fact_filter.py --input train.jsonl --test test.jsonl --evaluate
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure logging with both file and stream handlers."""
    log_file = Path(__file__).resolve().parent.parent / "logs" / "train_fact_filter.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(),
        ],
    )
    logger.info("Logging to %s", log_file)


class FactFilterFeatures:
    """Feature extractor for fact candidate classification."""

    # Vague words that suggest false positive
    VAGUE_WORDS = {
        "it", "that", "this", "them", "there", "these", "those",
        "what", "something", "thing", "stuff", "anything",
    }

    # Preference verbs
    PREF_POSITIVE = {"love", "like", "enjoy", "prefer", "obsessed", "addicted", "favorite"}
    PREF_NEGATIVE = {"hate", "dislike", "can't stand", "despise"}

    # Location markers
    LOCATION_MARKERS = {
        "live", "lived", "living", "moving", "moved", "from", "based",
        "grew up", "born", "located", "staying", "visiting",
    }

    # Relationship markers
    RELATIONSHIP_MARKERS = {
        "my", "our", "mom", "dad", "mother", "father", "sister", "brother",
        "wife", "husband", "girlfriend", "boyfriend", "partner", "friend",
    }

    # Bot indicators
    BOT_INDICATORS = [
        "CVS Pharmacy", "Rx Ready", "Check out this job",
        "prescription is ready", "unsubscribe",
    ]

    # Canonical stage-1 labels currently emitted by CandidateExtractor.
    # Kept alongside legacy labels for backward compatibility.
    ENTITY_TYPES = [
        "person_name",
        "family_member",
        "place",
        "org",
        "date_ref",
        "food_item",
        "job_role",
        "health_condition",
        "activity",
        # Legacy/optional aliases
        "food_preference",
        "disliked_food",
        "allergy",
        "current_location",
        "past_location",
        "future_location",
        "employer",
        "friend_name",
        "partner_name",
        "thing_preference",
    ]

    ENTITY_ALIASES = {
        "food_preference": "food_item",
        "disliked_food": "food_item",
        "allergy": "health_condition",
        "employer": "org",
    }

    FACT_PREFIXES = [
        "relationship",
        "preference",
        "location",
        "work",
        "health",
        "personal",
        "other",
    ]

    def extract(
        self,
        text: str,
        candidate: str,
        entity_type: str,
        *,
        fact_type: str = "",
        gliner_score: float | None = None,
    ) -> dict[str, float]:
        """Extract features from a candidate fact.

        Returns a dictionary of feature names to values.
        """
        text_lower = text.lower()
        candidate_lower = candidate.lower()
        words = candidate_lower.split()
        normalized_entity = self.ENTITY_ALIASES.get(entity_type, entity_type)

        features: dict[str, float] = {}

        # 1. Candidate structure features
        features["candidate_word_count"] = len(words)
        features["candidate_char_count"] = len(candidate)
        features["candidate_has_caps"] = 1.0 if any(c.isupper() for c in candidate) else 0.0
        features["candidate_all_lower"] = 1.0 if candidate.islower() else 0.0

        # 2. Vagueness features
        features["is_vague_pronoun"] = 1.0 if candidate_lower in self.VAGUE_WORDS else 0.0
        vague_word_count = sum(1 for w in words if w in self.VAGUE_WORDS)
        features["vague_word_ratio"] = vague_word_count / len(words) if words else 0.0

        # 3. Category-specific features
        # Preference indicators
        has_pos_pref = any(w in text_lower for w in self.PREF_POSITIVE)
        has_neg_pref = any(w in text_lower for w in self.PREF_NEGATIVE)
        features["has_preference_verb"] = 1.0 if (has_pos_pref or has_neg_pref) else 0.0
        features["is_negative_pref"] = 1.0 if has_neg_pref else 0.0

        # Location indicators
        features["has_location_marker"] = 1.0 if any(
            m in text_lower for m in self.LOCATION_MARKERS
        ) else 0.0

        # Relationship indicators
        features["has_relationship_marker"] = 1.0 if any(
            m in text_lower for m in self.RELATIONSHIP_MARKERS
        ) else 0.0

        # 4. Text quality features
        features["text_length"] = len(text)
        features["text_word_count"] = len(text.split())

        # First person indicator (higher confidence if "I" statement)
        features["is_first_person"] = 1.0 if text_lower.startswith(("i ", "my ")) else 0.0

        # 5. Bot/spam detection
        features["is_likely_bot"] = 1.0 if any(
            b.lower() in text_lower for b in self.BOT_INDICATORS
        ) else 0.0

        # 6. Entity type features
        entity_type_onehot = {k: 0.0 for k in self.ENTITY_TYPES}
        if normalized_entity in entity_type_onehot:
            entity_type_onehot[normalized_entity] = 1.0
        elif entity_type in entity_type_onehot:
            entity_type_onehot[entity_type] = 1.0
        features.update({f"etype_{k}": v for k, v in entity_type_onehot.items()})

        # 7. Candidate score feature from GLiNER (when available).
        score = float(gliner_score) if gliner_score is not None else 0.0
        features["gliner_score"] = score
        features["score_ge_05"] = 1.0 if score >= 0.5 else 0.0
        features["score_ge_07"] = 1.0 if score >= 0.7 else 0.0

        # 8. Coarse fact-type features (optional predicted fact_type).
        prefix = fact_type.split(".", 1)[0] if fact_type else ""
        for p in self.FACT_PREFIXES:
            features[f"factprefix_{p}"] = 1.0 if prefix == p else 0.0

        return features

    def get_feature_names(self) -> list[str]:
        """Get list of feature names (for consistency)."""
        dummy = self.extract("test", "test", "food_preference")
        return list(dummy.keys())


class FactFilterClassifier:
    """Lightweight classifier for filtering GLiNER candidates."""

    def __init__(self, model_type: str = "logistic") -> None:
        self.model_type = model_type
        self.feature_extractor = FactFilterFeatures()
        self.model: Any = None
        self.scaler: Any = None
        self.is_fitted = False

    def fit(
        self,
        texts: list[str],
        candidates: list[str],
        entity_types: list[str],
        labels: list[int],
        fact_types: list[str] | None = None,
        gliner_scores: list[float] | None = None,
    ) -> None:
        """Train the classifier."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC

        if fact_types is None:
            fact_types = [""] * len(texts)
        if gliner_scores is None:
            gliner_scores = [0.0] * len(texts)

        # Extract features
        X = []
        total = len(texts)
        for i, (text, cand, etype, ftype, score) in enumerate(zip(
            texts, candidates, entity_types, fact_types, gliner_scores
        )):
            if (i + 1) % 500 == 0 or i + 1 == total:
                print(f"Extracting features {i + 1}/{total}...", flush=True)
            features = self.feature_extractor.extract(
                text,
                cand,
                etype,
                fact_type=ftype,
                gliner_score=score,
            )
            X.append(list(features.values()))

        X = np.array(X)
        y = np.array(labels)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        if self.model_type == "logistic":
            self.model = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                C=1.0,
            )
        else:  # svm
            self.model = LinearSVC(
                class_weight="balanced",
                max_iter=2000,
                C=1.0,
            )

        self.model.fit(X_scaled, y)
        self.is_fitted = True

        # Log training stats
        n_positive = sum(labels)
        n_negative = len(labels) - n_positive
        logger.info(f"Trained on {len(labels)} examples")
        logger.info(f"  Positive: {n_positive}, Negative: {n_negative}")

    def predict(
        self,
        text: str,
        candidate: str,
        entity_type: str,
        *,
        fact_type: str = "",
        gliner_score: float | None = None,
    ) -> tuple[int, float]:
        """Predict if candidate is valid.

        Returns:
            (prediction, confidence_score)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not trained yet")

        features = self.feature_extractor.extract(
            text,
            candidate,
            entity_type,
            fact_type=fact_type,
            gliner_score=gliner_score,
        )
        X = np.array([list(features.values())])
        X_scaled = self.scaler.transform(X)

        prediction = self.model.predict(X_scaled)[0]

        # Get confidence
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_scaled)[0]
            confidence = proba[1] if prediction == 1 else proba[0]
        else:
            # For SVM, use decision function
            decision = self.model.decision_function(X_scaled)[0]
            confidence = 1 / (1 + np.exp(-decision))  # Sigmoid

        return int(prediction), float(confidence)

    def save(self, path: Path) -> None:
        """Save classifier to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "is_fitted": self.is_fitted,
                "model_type": self.model_type,
            }, f)

        logger.info(f"Saved classifier to {path}")

    @classmethod
    def load(cls, path: Path) -> FactFilterClassifier:
        """Load classifier from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(model_type=data.get("model_type", "logistic"))
        instance.model = data["model"]
        instance.scaler = data["scaler"]
        instance.is_fitted = data["is_fitted"]

        return instance


def load_training_data(path: Path) -> dict[str, list]:
    """Load training data from JSONL file."""
    texts, candidates, entity_types, labels = [], [], [], []
    fact_types, gliner_scores = [], []

    with open(path) as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["text"])
            candidates.append(data["candidate"])
            entity_types.append(data.get("entity_type", ""))
            labels.append(data["label"])
            fact_types.append(data.get("fact_type", ""))
            try:
                gliner_scores.append(float(data.get("gliner_score", 0.0)))
            except (TypeError, ValueError):
                gliner_scores.append(0.0)

    return {
        "texts": texts,
        "candidates": candidates,
        "entity_types": entity_types,
        "labels": labels,
        "fact_types": fact_types,
        "gliner_scores": gliner_scores,
    }


def evaluate_classifier(classifier: FactFilterClassifier, test_data: dict) -> dict[str, float]:
    """Evaluate classifier on test data."""
    predictions = []
    confidences = []

    total = len(test_data["texts"])
    for i, (text, cand, etype, ftype, score) in enumerate(zip(
        test_data["texts"],
        test_data["candidates"],
        test_data["entity_types"],
        test_data.get("fact_types", [""] * len(test_data["texts"])),
        test_data.get("gliner_scores", [0.0] * len(test_data["texts"])),
    )):
        if (i + 1) % 500 == 0 or i + 1 == total:
            print(f"Evaluating {i + 1}/{total}...", flush=True)
        pred, conf = classifier.predict(
            text,
            cand,
            etype,
            fact_type=ftype,
            gliner_score=score,
        )
        predictions.append(pred)
        confidences.append(conf)

    y_true = np.array(test_data["labels"])
    y_pred = np.array(predictions)

    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train fact filter classifier"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Training data JSONL file",
    )
    parser.add_argument(
        "--test",
        type=Path,
        help="Test data JSONL file (optional)",
    )
    parser.add_argument(
        "--model-type",
        choices=["logistic", "svm"],
        default="logistic",
        help="Classifier type",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/fact_filter.pkl"),
        help="Output model path",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation",
    )

    args = parser.parse_args()
    _setup_logging()

    # Load training data
    logger.info(f"Loading training data from {args.input}")
    train_data = load_training_data(args.input)

    # Train classifier
    logger.info(f"Training {args.model_type} classifier...")
    classifier = FactFilterClassifier(model_type=args.model_type)
    classifier.fit(
        train_data["texts"],
        train_data["candidates"],
        train_data["entity_types"],
        train_data["labels"],
        fact_types=train_data.get("fact_types"),
        gliner_scores=train_data.get("gliner_scores"),
    )

    # Evaluate on training data (overfit check)
    train_metrics = evaluate_classifier(classifier, train_data)
    logger.info("Training metrics:")
    for k, v in train_metrics.items():
        logger.info(f"  {k}: {v:.3f}")

    # Evaluate on test data
    if args.test:
        logger.info(f"Loading test data from {args.test}")
        test_data = load_training_data(args.test)
        test_metrics = evaluate_classifier(classifier, test_data)
        logger.info("Test metrics:")
        for k, v in test_metrics.items():
            logger.info(f"  {k}: {v:.3f}")

    # Save model
    classifier.save(args.output)

    # Demo predictions
    print("\n" + "=" * 70, flush=True)
    print("Demo predictions:", flush=True)
    print("=" * 70, flush=True)

    test_cases = [
        ("I love Thai food", "Thai food", "food_preference"),
        ("yeah same I love it", "it", "thing_preference"),
        ("My sister Sarah is coming", "Sarah", "family_member"),
        ("that thing is great", "that thing", "thing_preference"),
        ("I work at Google", "Google", "employer"),
    ]

    for text, cand, etype in test_cases:
        pred, conf = classifier.predict(text, cand, etype)
        status = "VALID" if pred == 1 else "INVALID"
        print(f"\nText: {text}", flush=True)
        print(f"Candidate: '{cand}' ({etype})", flush=True)
        print(f"Prediction: {status} (confidence: {conf:.2f})", flush=True)


if __name__ == "__main__":
    main()
