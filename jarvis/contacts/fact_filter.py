"""Message-level and candidate-level fact filtering.

Provides lightweight gates to skip messages that don't contain personal facts,
and to filter noisy candidates after extraction.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Fallback feature extraction logic if scripts/train_message_gate.py is not available
# This matches the logic in scripts/train_message_gate.py exactly.


class MessageGateFeatures:
    """Extract lightweight numeric features from messages for the gate model."""

    PREF_WORDS = {"love", "like", "hate", "prefer", "obsessed", "favorite", "enjoy", "allergic"}
    LOCATION_WORDS = {
        "live",
        "living",
        "moving",
        "moved",
        "from",
        "to",
        "based",
        "relocating",
        "sf",
        "dallas",
        "nyc",
        "austin",
    }
    REL_WORDS = {
        "my",
        "mom",
        "dad",
        "sister",
        "brother",
        "wife",
        "husband",
        "girlfriend",
        "boyfriend",
        "partner",
        "friend",
        "neighbor",
    }
    HEALTH_WORDS = {
        "pain",
        "hospital",
        "injury",
        "allergic",
        "anxious",
        "depressed",
        "headache",
        "surgery",
        "therapy",
        "dental",
    }
    WORK_WORDS = {
        "work",
        "job",
        "hired",
        "fired",
        "interview",
        "company",
        "office",
        "career",
        "salary",
        "raise",
    }
    PERSONAL_WORDS = {
        "jacket",
        "car",
        "tesla",
        "dog",
        "zodiac",
        "gemini",
        "bday",
        "birthday",
        "gift",
        "bought",
    }
    BOT_PATTERNS = {
        "cvs pharmacy",
        "prescription is ready",
        "unsubscribe",
        "check out this job",
        "apply now",
    }

    def __init__(self) -> None:
        self._feature_names = [
            "char_len",
            "word_len",
            "upper_ratio",
            "digit_ratio",
            "has_question",
            "has_exclaim",
            "first_person",
            "pref_marker",
            "location_marker",
            "relationship_marker",
            "health_marker",
            "work_marker",
            "personal_marker",
            "likely_bot",
            "is_short_msg",
            "is_from_me",
            "bucket_random",
            "bucket_likely",
            "bucket_negative",
            "bucket_other",
        ]

    def transform_single(
        self, text: str, is_from_me: bool = False, bucket: str = "other"
    ) -> np.ndarray:
        """Convert a single message to numeric feature vector."""
        t = (text or "").strip()
        lower = t.lower()
        words = lower.split()

        char_len = len(t)
        word_len = len(words)
        upper_count = sum(1 for c in t if c.isupper())
        digit_count = sum(1 for c in t if c.isdigit())

        row = [
            float(char_len),
            float(word_len),
            (upper_count / char_len) if char_len else 0.0,
            (digit_count / char_len) if char_len else 0.0,
            1.0 if "?" in t else 0.0,
            1.0 if "!" in t else 0.0,
            1.0 if any(w in {"i", "i'm", "my", "me", "we"} for w in words[:5]) else 0.0,
            1.0 if any(w in lower for w in self.PREF_WORDS) else 0.0,
            1.0 if any(w in lower for w in self.LOCATION_WORDS) else 0.0,
            1.0 if any(w in lower for w in self.REL_WORDS) else 0.0,
            1.0 if any(w in lower for w in self.HEALTH_WORDS) else 0.0,
            1.0 if any(w in lower for w in self.WORK_WORDS) else 0.0,
            1.0 if any(w in lower for w in self.PERSONAL_WORDS) else 0.0,
            1.0 if any(p in lower for p in self.BOT_PATTERNS) else 0.0,
            1.0 if word_len <= 3 else 0.0,
            1.0 if is_from_me else 0.0,
            1.0 if bucket == "random" else 0.0,
            1.0 if bucket == "likely" else 0.0,
            1.0 if bucket == "negative" else 0.0,
            1.0 if bucket not in {"random", "likely", "negative"} else 0.0,
        ]
        return np.asarray([row], dtype=np.float32)


class MessageGate:
    """Message-level keep/discard classifier."""

    def __init__(self, model_path: str | Path = "models/message_gate.pkl") -> None:
        self.model_path = Path(model_path)
        self.model: Any = None
        self.vectorizer: Any = None
        self.scaler: Any = None
        self.threshold: float = 0.35  # Lowered from 0.5
        self.num_features = MessageGateFeatures()
        self._loaded = False

    def load(self) -> bool:
        """Lazy-load the model."""
        if self._loaded:
            return True

        if not self.model_path.exists():
            logger.debug("Message gate model not found at %s", self.model_path)
            return False

        try:
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)

            self.model = data["model"]
            self.vectorizer = data["vectorizer"]
            self.scaler = data["scaler"]
            self.threshold = data.get("threshold", 0.5)
            self._loaded = True
            return True
        except Exception as e:
            logger.error("Failed to load message gate: %s", e)
            return False

    def predict_score(self, text: str, is_from_me: bool = False) -> float:
        """Get the probability that this message contains a fact."""
        if not self.load():
            return 1.0  # Default to keep if model missing

        from scipy.sparse import csr_matrix, hstack

        try:
            x_text = self.vectorizer.transform([text])
            x_num_arr = self.num_features.transform_single(text, is_from_me=is_from_me)
            x_num = csr_matrix(self.scaler.transform(x_num_arr))
            x = hstack([x_text, x_num], format="csr")

            if hasattr(self.model, "predict_proba"):
                return float(self.model.predict_proba(x)[0, 1])

            decision = self.model.decision_function(x)[0]
            return float(1.0 / (1.0 + np.exp(-decision)))
        except Exception as e:
            logger.debug("Gate prediction failed: %s", e)
            return 1.0

    def is_fact_likely(
        self, text: str, is_from_me: bool = False, threshold: float | None = None
    ) -> bool:
        """True if the message should be processed for fact extraction."""
        score = self.predict_score(text, is_from_me)
        thr = threshold if threshold is not None else self.threshold
        return score >= thr


# Global instances for easy access
_message_gate: MessageGate | None = None


def get_message_gate() -> MessageGate:
    """Get or create the global message gate instance."""
    global _message_gate
    if _message_gate is None:
        _message_gate = MessageGate()
    return _message_gate


def is_fact_likely(text: str, is_from_me: bool = False, threshold: float | None = None) -> bool:
    """Check if a message likely contains a fact using the message gate model."""
    return get_message_gate().is_fact_likely(text, is_from_me, threshold)
