"""Tests for jarvis.contacts.fact_filter — MessageGate and MessageGateFeatures.

Covers feature extraction, model loading/fallback, prediction flow,
threshold filtering, bucket encoding, and the module-level singleton.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jarvis.contacts.fact_filter import (
    MessageGate,
    MessageGateFeatures,
    get_message_gate,
    is_fact_likely,
)

# ---------------------------------------------------------------------------
# MessageGateFeatures tests
# ---------------------------------------------------------------------------


class TestFeatureVectorShape:
    """transform_single returns correct shape and dtype."""

    def test_shape_is_1x20(self):
        feats = MessageGateFeatures()
        vec = feats.transform_single("hello world")
        assert vec.shape == (1, 20)
        assert vec.dtype == np.float32

    def test_empty_text_produces_valid_vector(self):
        feats = MessageGateFeatures()
        vec = feats.transform_single("")
        assert vec.shape == (1, 20)
        # char_len and word_len should be 0
        assert vec[0, 0] == 0.0  # char_len
        assert vec[0, 1] == 0.0  # word_len

    def test_none_text_treated_as_empty(self):
        feats = MessageGateFeatures()
        vec = feats.transform_single(None)
        assert vec.shape == (1, 20)
        assert vec[0, 0] == 0.0


class TestFeatureExtraction:
    """Individual feature values are correct."""

    def setup_method(self):
        self.feats = MessageGateFeatures()

    def test_char_and_word_length(self):
        vec = self.feats.transform_single("hello world")
        assert vec[0, 0] == 11.0  # char_len
        assert vec[0, 1] == 2.0  # word_len

    def test_upper_ratio(self):
        vec = self.feats.transform_single("HELLO world")  # 5 upper / 11 chars
        ratio = vec[0, 2]
        assert abs(ratio - 5 / 11) < 1e-5

    def test_digit_ratio(self):
        vec = self.feats.transform_single("abc123")  # 3 digits / 6 chars
        ratio = vec[0, 3]
        assert abs(ratio - 0.5) < 1e-5

    def test_question_mark_detection(self):
        vec_q = self.feats.transform_single("how are you?")
        vec_no = self.feats.transform_single("how are you")
        assert vec_q[0, 4] == 1.0
        assert vec_no[0, 4] == 0.0

    def test_exclamation_mark_detection(self):
        vec_e = self.feats.transform_single("wow!")
        vec_no = self.feats.transform_single("wow")
        assert vec_e[0, 5] == 1.0
        assert vec_no[0, 5] == 0.0

    def test_first_person_markers(self):
        for text in ["I like sushi", "I'm hungry", "my dog is cute", "help me"]:
            vec = self.feats.transform_single(text)
            assert vec[0, 6] == 1.0, f"Expected first_person=1 for '{text}'"

    def test_no_first_person(self):
        vec = self.feats.transform_single("the weather is nice today")
        assert vec[0, 6] == 0.0

    def test_first_person_only_checks_first_5_words(self):
        # "i" appears at word index 6 (0-based), beyond [:5]
        vec = self.feats.transform_single("one two three four five six i said")
        assert vec[0, 6] == 0.0

    @pytest.mark.parametrize(
        "text,index",
        [
            ("I love pizza", 7),  # pref_marker
            ("I live in Austin", 8),  # location_marker
            ("my brother is tall", 9),  # relationship_marker
            ("I have a headache", 10),  # health_marker
        ],
    )
    def test_category_word_markers(self, text, index):
        vec = self.feats.transform_single(text)
        assert vec[0, index] == 1.0

    def test_bot_pattern_detection(self):
        vec = self.feats.transform_single("Your CVS Pharmacy prescription is ready")
        assert vec[0, 13] == 1.0  # likely_bot

    def test_normal_message_not_bot(self):
        vec = self.feats.transform_single("Hey want to grab dinner tonight?")
        assert vec[0, 13] == 0.0

    def test_short_message_flag(self):
        vec_short = self.feats.transform_single("ok sure")  # 2 words
        vec_long = self.feats.transform_single("this is a longer message")  # 5 words
        assert vec_short[0, 14] == 1.0
        assert vec_long[0, 14] == 0.0

    def test_is_from_me_flag(self):
        vec_me = self.feats.transform_single("hello", is_from_me=True)
        vec_other = self.feats.transform_single("hello", is_from_me=False)
        assert vec_me[0, 15] == 1.0
        assert vec_other[0, 15] == 0.0


class TestBucketEncoding:
    """One-hot bucket encoding for training labels."""

    def setup_method(self):
        self.feats = MessageGateFeatures()

    def test_random_bucket(self):
        vec = self.feats.transform_single("text", bucket="random")
        assert vec[0, 16] == 1.0  # bucket_random
        assert vec[0, 17] == 0.0
        assert vec[0, 18] == 0.0
        assert vec[0, 19] == 0.0

    def test_likely_bucket(self):
        vec = self.feats.transform_single("text", bucket="likely")
        assert vec[0, 16] == 0.0
        assert vec[0, 17] == 1.0  # bucket_likely
        assert vec[0, 18] == 0.0
        assert vec[0, 19] == 0.0

    def test_negative_bucket(self):
        vec = self.feats.transform_single("text", bucket="negative")
        assert vec[0, 16] == 0.0
        assert vec[0, 17] == 0.0
        assert vec[0, 18] == 1.0  # bucket_negative
        assert vec[0, 19] == 0.0

    def test_other_bucket_default(self):
        vec = self.feats.transform_single("text", bucket="other")
        assert vec[0, 16] == 0.0
        assert vec[0, 17] == 0.0
        assert vec[0, 18] == 0.0
        assert vec[0, 19] == 1.0  # bucket_other

    def test_unknown_bucket_maps_to_other(self):
        vec = self.feats.transform_single("text", bucket="xyz_unknown")
        assert vec[0, 19] == 1.0  # bucket_other


# ---------------------------------------------------------------------------
# MessageGate tests
# ---------------------------------------------------------------------------


class TestMessageGateLoadMissing:
    """Graceful fallback when model file is missing."""

    def test_load_returns_false_when_missing(self):
        gate = MessageGate(model_path="/nonexistent/path/model.pkl")
        assert gate.load() is False
        assert gate._loaded is False

    def test_predict_score_returns_1_when_missing(self):
        gate = MessageGate(model_path="/nonexistent/path/model.pkl")
        score = gate.predict_score("I love sushi")
        assert score == 1.0

    def test_is_fact_likely_returns_true_when_missing(self):
        gate = MessageGate(model_path="/nonexistent/path/model.pkl")
        assert gate.is_fact_likely("anything") is True


class TestMessageGateLoadValid:
    """Model loading from a valid pickle file."""

    def _make_model_file(self, tmp_path: Path) -> Path:
        """Create a minimal valid gate model pickle with real sklearn objects."""
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler

        vectorizer = CountVectorizer(max_features=5)
        vectorizer.fit(["hello world", "I love sushi"])

        scaler = StandardScaler()
        scaler.fit(np.zeros((2, 20)))

        model = SGDClassifier(loss="log_loss")
        # Fit with dummy data matching vectorizer + scaler output dims
        from scipy.sparse import csr_matrix, hstack

        x_text = vectorizer.transform(["hello", "sushi"])
        x_num = csr_matrix(scaler.transform(np.zeros((2, 20))))
        x = hstack([x_text, x_num], format="csr")
        model.fit(x, [0, 1])

        model_path = tmp_path / "gate.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "vectorizer": vectorizer,
                    "scaler": scaler,
                    "threshold": 0.4,
                },
                f,
            )
        return model_path

    def test_load_returns_true(self, tmp_path):
        path = self._make_model_file(tmp_path)
        gate = MessageGate(model_path=path)
        assert gate.load() is True
        assert gate._loaded is True
        assert gate.threshold == 0.4

    def test_load_is_idempotent(self, tmp_path):
        path = self._make_model_file(tmp_path)
        gate = MessageGate(model_path=path)
        gate.load()
        gate.load()  # second call should be a no-op
        assert gate._loaded is True


class TestMessageGatePrediction:
    """Prediction flow with mocked internals."""

    def _make_gate_with_mock_model(self, proba: float) -> MessageGate:
        """Create a gate with an already-loaded mock model."""
        gate = MessageGate(model_path="/fake")
        gate._loaded = True

        gate.model = MagicMock()
        gate.model.predict_proba.return_value = np.array([[1 - proba, proba]])

        from scipy.sparse import csr_matrix

        gate.vectorizer = MagicMock()
        gate.vectorizer.transform.return_value = csr_matrix(np.zeros((1, 5)))

        gate.scaler = MagicMock()
        gate.scaler.transform.return_value = np.zeros((1, 20))

        return gate

    def test_predict_score_returns_proba(self):
        gate = self._make_gate_with_mock_model(0.85)
        score = gate.predict_score("I live in Austin")
        assert abs(score - 0.85) < 1e-5

    def test_predict_score_range(self):
        for p in [0.0, 0.5, 1.0]:
            gate = self._make_gate_with_mock_model(p)
            score = gate.predict_score("test")
            assert 0.0 <= score <= 1.0

    def test_decision_function_fallback(self):
        """When model lacks predict_proba, falls back to sigmoid(decision_function)."""
        gate = MessageGate(model_path="/fake")
        gate._loaded = True

        model = MagicMock(spec=[])  # no predict_proba attribute
        model.decision_function = MagicMock(return_value=np.array([0.0]))
        gate.model = model

        from scipy.sparse import csr_matrix

        gate.vectorizer = MagicMock()
        gate.vectorizer.transform.return_value = csr_matrix(np.zeros((1, 5)))
        gate.scaler = MagicMock()
        gate.scaler.transform.return_value = np.zeros((1, 20))

        score = gate.predict_score("test")
        # sigmoid(0) = 0.5
        assert abs(score - 0.5) < 1e-5

    def test_predict_score_returns_1_on_exception(self):
        gate = MessageGate(model_path="/fake")
        gate._loaded = True
        gate.model = MagicMock()
        gate.vectorizer = MagicMock(side_effect=RuntimeError("vectorizer broken"))

        score = gate.predict_score("test")
        assert score == 1.0

    def test_is_fact_likely_uses_threshold(self):
        gate = self._make_gate_with_mock_model(0.6)
        gate.threshold = 0.5
        assert gate.is_fact_likely("test") is True

        gate2 = self._make_gate_with_mock_model(0.4)
        gate2.threshold = 0.5
        assert gate2.is_fact_likely("test") is False

    def test_is_fact_likely_custom_threshold(self):
        gate = self._make_gate_with_mock_model(0.6)
        gate.threshold = 0.5
        # Override with higher threshold
        assert gate.is_fact_likely("test", threshold=0.9) is False
        # Override with lower threshold
        assert gate.is_fact_likely("test", threshold=0.1) is True


class TestModuleLevelSingleton:
    """Module-level get_message_gate() and is_fact_likely() wrappers."""

    def test_get_message_gate_returns_same_instance(self):
        # Reset global
        import jarvis.contacts.fact_filter as ff

        ff._message_gate = None
        g1 = get_message_gate()
        g2 = get_message_gate()
        assert g1 is g2

    def test_is_fact_likely_delegates(self):
        """Module-level is_fact_likely delegates to singleton."""
        with patch.object(MessageGate, "is_fact_likely", return_value=True) as mock:
            result = is_fact_likely("test text", is_from_me=True, threshold=0.3)
            assert result is True
            mock.assert_called_once_with("test text", True, 0.3)
