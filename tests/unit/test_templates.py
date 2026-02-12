"""Tests for models/templates.py - Template matching and custom templates."""

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from models.templates import (
    CustomTemplate,
    CustomTemplateStore,
    EmbeddingCache,
    ResponseTemplate,
    TemplateMatch,
    TemplateMatcher,
)


# =============================================================================
# EmbeddingCache Tests
# =============================================================================


class TestEmbeddingCache:
    """Test the LRU embedding cache."""

    def test_set_and_get(self):
        cache = EmbeddingCache(maxsize=10)
        vec = np.array([1.0, 2.0, 3.0])
        cache.set("key1", vec)
        result = cache.get("key1", track_analytics=False)
        np.testing.assert_array_equal(result, vec)

    def test_get_missing_returns_none(self):
        cache = EmbeddingCache(maxsize=10)
        assert cache.get("missing", track_analytics=False) is None

    def test_lru_eviction(self):
        cache = EmbeddingCache(maxsize=2)
        cache.set("a", np.array([1.0]))
        cache.set("b", np.array([2.0]))
        cache.set("c", np.array([3.0]))  # Evicts "a"
        assert cache.get("a", track_analytics=False) is None
        assert cache.get("c", track_analytics=False) is not None

    def test_access_refreshes_lru(self):
        cache = EmbeddingCache(maxsize=2)
        cache.set("a", np.array([1.0]))
        cache.set("b", np.array([2.0]))
        cache.get("a", track_analytics=False)  # Refresh "a"
        cache.set("c", np.array([3.0]))  # Evicts "b" (oldest)
        assert cache.get("a", track_analytics=False) is not None
        assert cache.get("b", track_analytics=False) is None

    def test_clear(self):
        cache = EmbeddingCache(maxsize=10)
        cache.set("a", np.array([1.0]))
        cache.clear()
        assert cache.get("a", track_analytics=False) is None
        stats = cache.stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_hit_rate(self):
        cache = EmbeddingCache(maxsize=10)
        cache.set("a", np.array([1.0]))
        cache.get("a", track_analytics=False)  # hit
        cache.get("b", track_analytics=False)  # miss
        assert cache.hit_rate == pytest.approx(0.5)

    def test_stats(self):
        cache = EmbeddingCache(maxsize=100)
        cache.set("a", np.array([1.0]))
        cache.get("a", track_analytics=False)
        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["maxsize"] == 100
        assert stats["hits"] == 1

    def test_thread_safety(self):
        cache = EmbeddingCache(maxsize=50)
        errors = []

        def worker(prefix):
            try:
                for i in range(30):
                    cache.set(f"{prefix}_{i}", np.array([float(i)]))
                    cache.get(f"{prefix}_{i}", track_analytics=False)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
        assert not errors


# =============================================================================
# ResponseTemplate / TemplateMatch Tests
# =============================================================================


class TestResponseTemplate:
    """Test ResponseTemplate dataclass."""

    def test_basic_creation(self):
        t = ResponseTemplate(
            name="greeting",
            patterns=["hello", "hi there"],
            response="Hey! What's up?",
        )
        assert t.name == "greeting"
        assert len(t.patterns) == 2
        assert t.is_group_template is False
        assert t.min_group_size is None

    def test_group_template(self):
        t = ResponseTemplate(
            name="group_greeting",
            patterns=["hello everyone"],
            response="Hey team!",
            is_group_template=True,
            min_group_size=3,
            max_group_size=10,
        )
        assert t.is_group_template is True
        assert t.min_group_size == 3
        assert t.max_group_size == 10


class TestTemplateMatch:
    """Test TemplateMatch dataclass."""

    def test_creation(self):
        template = ResponseTemplate(name="test", patterns=["hi"], response="hello")
        match = TemplateMatch(template=template, similarity=0.95, matched_pattern="hi")
        assert match.similarity == 0.95
        assert match.matched_pattern == "hi"
        assert match.template.name == "test"


# =============================================================================
# CustomTemplate Tests
# =============================================================================


class TestCustomTemplate:
    """Test CustomTemplate serialization and conversion."""

    def test_to_dict_roundtrip(self):
        ct = CustomTemplate(
            name="work_reply",
            template_text="I'll get back to you on that.",
            trigger_phrases=["can you do this", "please handle"],
            category="work",
            tags=["professional"],
        )
        d = ct.to_dict()
        restored = CustomTemplate.from_dict(d)
        assert restored.name == ct.name
        assert restored.template_text == ct.template_text
        assert restored.trigger_phrases == ct.trigger_phrases
        assert restored.category == ct.category
        assert restored.tags == ct.tags

    def test_from_dict_defaults(self):
        ct = CustomTemplate.from_dict({"name": "minimal"})
        assert ct.template_text == ""
        assert ct.category == "general"
        assert ct.enabled is True
        assert ct.usage_count == 0

    def test_to_response_template(self):
        ct = CustomTemplate(
            id="abc123",
            name="test",
            template_text="response text",
            trigger_phrases=["trigger1", "trigger2"],
        )
        rt = ct.to_response_template()
        assert rt.name == "custom_abc123"
        assert rt.patterns == ["trigger1", "trigger2"]
        assert rt.response == "response text"


class TestCustomTemplateStore:
    """Test custom template storage."""

    @pytest.fixture()
    def store(self, tmp_path):
        """Create a store with a temp storage path."""
        return CustomTemplateStore(storage_path=tmp_path / "templates.json")

    def test_add_and_get(self, store):
        ct = CustomTemplate(name="test", template_text="hello")
        store.add(ct)
        retrieved = store.get(ct.id)
        assert retrieved is not None
        assert retrieved.name == "test"

    def test_list_all(self, store):
        store.add(CustomTemplate(name="a", template_text="hello"))
        store.add(CustomTemplate(name="b", template_text="world"))
        templates = store.list_all()
        assert len(templates) == 2

    def test_remove(self, store):
        ct = CustomTemplate(name="test", template_text="hello")
        store.add(ct)
        store.remove(ct.id)
        assert store.get(ct.id) is None

    def test_update(self, store):
        ct = CustomTemplate(name="old_name", template_text="old")
        store.add(ct)
        ct.name = "new_name"
        ct.template_text = "new"
        store.update(ct)
        retrieved = store.get(ct.id)
        assert retrieved.name == "new_name"

    def test_persistence(self, tmp_path):
        path = tmp_path / "templates.json"
        store1 = CustomTemplateStore(storage_path=path)
        store1.add(CustomTemplate(id="persist1", name="persistent", template_text="hi"))

        # Create new store from same file
        store2 = CustomTemplateStore(storage_path=path)
        assert store2.get("persist1") is not None
        assert store2.get("persist1").name == "persistent"

    def test_corrupted_file_handled(self, tmp_path):
        path = tmp_path / "templates.json"
        path.write_text("not valid json {{{")
        store = CustomTemplateStore(storage_path=path)
        assert len(store.list_all()) == 0  # Gracefully handles corruption


# =============================================================================
# TemplateMatcher Tests
# =============================================================================


class TestTemplateMatcher:
    """Test semantic template matching with mocked embedder."""

    @pytest.fixture()
    def mock_embedder_fn(self):
        """Mock _get_sentence_model to return a fake embedder."""
        mock = MagicMock()
        # Default: return random normalized embeddings
        def fake_encode(texts, normalize=True):
            if isinstance(texts, str):
                texts = [texts]
            return np.random.randn(len(texts), 384).astype(np.float32)

        mock.encode.side_effect = fake_encode
        mock.is_available.return_value = True
        return mock

    def test_match_returns_none_below_threshold(self, mock_embedder_fn):
        """Low similarity should return None."""
        templates = [
            ResponseTemplate(name="greeting", patterns=["hello"], response="hi"),
        ]

        with patch("models.templates._get_sentence_model", return_value=mock_embedder_fn):
            matcher = TemplateMatcher(templates=templates)
            # Force low similarity: pattern embedding orthogonal to query
            matcher._pattern_embeddings = np.array([[1, 0, 0] + [0] * 381], dtype=np.float32)
            matcher._pattern_norms = np.array([1.0])
            matcher._pattern_to_template = [("hello", templates[0])]

            # Query embedding perpendicular to pattern
            with patch.object(matcher, "_get_query_embedding", return_value=np.array([0, 1, 0] + [0] * 381, dtype=np.float32)):
                result = matcher.match("something unrelated", track_analytics=False)
                assert result is None

    def test_match_returns_match_above_threshold(self, mock_embedder_fn):
        """High similarity should return a TemplateMatch."""
        templates = [
            ResponseTemplate(name="greeting", patterns=["hello"], response="hi there"),
        ]

        with patch("models.templates._get_sentence_model", return_value=mock_embedder_fn):
            matcher = TemplateMatcher(templates=templates)
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            matcher._pattern_embeddings = vec.reshape(1, -1)
            matcher._pattern_norms = np.array([1.0])
            matcher._pattern_to_template = [("hello", templates[0])]

            # Query embedding = same as pattern (similarity = 1.0)
            with patch.object(matcher, "_get_query_embedding", return_value=vec):
                result = matcher.match("hello", track_analytics=False)
                assert result is not None
                assert isinstance(result, TemplateMatch)
                assert result.template.name == "greeting"
                assert result.similarity >= 0.7

    def test_match_selects_best_template(self, mock_embedder_fn):
        """Should return the template with highest similarity."""
        templates = [
            ResponseTemplate(name="greeting", patterns=["hello"], response="hi"),
            ResponseTemplate(name="farewell", patterns=["goodbye"], response="bye"),
        ]

        with patch("models.templates._get_sentence_model", return_value=mock_embedder_fn):
            matcher = TemplateMatcher(templates=templates)

            # Pattern 1: [1, 0, ...], Pattern 2: [0, 1, ...]
            p1 = np.zeros(384, dtype=np.float32); p1[0] = 1.0
            p2 = np.zeros(384, dtype=np.float32); p2[1] = 1.0
            matcher._pattern_embeddings = np.vstack([p1, p2])
            matcher._pattern_norms = np.array([1.0, 1.0])
            matcher._pattern_to_template = [("hello", templates[0]), ("goodbye", templates[1])]

            # Query closer to pattern 2
            q = np.zeros(384, dtype=np.float32); q[1] = 0.9; q[0] = 0.1
            q = q / np.linalg.norm(q)

            with patch.object(matcher, "_get_query_embedding", return_value=q):
                result = matcher.match("bye bye", track_analytics=False)
                assert result is not None
                assert result.template.name == "farewell"

    def test_match_with_invalid_embedder_raises(self, mock_embedder_fn):
        """Passing non-Embedder should raise TypeError."""
        templates = [
            ResponseTemplate(name="test", patterns=["hi"], response="hello"),
        ]

        with patch("models.templates._get_sentence_model", return_value=mock_embedder_fn):
            matcher = TemplateMatcher(templates=templates)
            matcher._pattern_embeddings = np.ones((1, 384), dtype=np.float32)
            matcher._pattern_norms = np.array([1.0])
            matcher._pattern_to_template = [("hi", templates[0])]

            with pytest.raises(TypeError, match="Embedder protocol"):
                matcher.match("hi", embedder="not_an_embedder", track_analytics=False)

    def test_empty_templates(self, mock_embedder_fn):
        """Matcher with no templates should return None."""
        with patch("models.templates._get_sentence_model", return_value=mock_embedder_fn):
            mock_embedder_fn.encode.return_value = np.zeros((0, 384), dtype=np.float32)
            matcher = TemplateMatcher(templates=[])
            # No patterns means no embeddings to match against
            result = matcher.match("anything", track_analytics=False)
            assert result is None
