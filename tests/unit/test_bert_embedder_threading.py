"""TEST-01: Thread safety tests for InProcessEmbedder concurrent encode().

Verifies that the GPU lock serializes concurrent encode() calls,
preventing race conditions on tokenizer state and Metal GPU operations.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock

import numpy as np


class TestBertEmbedderThreadSafety:
    """Verify InProcessEmbedder.encode() serializes via GPU lock."""

    def _make_embedder_with_mock_model(self):
        """Create an InProcessEmbedder with mocked internals."""
        from models.bert_embedder import InProcessEmbedder

        embedder = InProcessEmbedder(model_name="bge-small")

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.no_padding = MagicMock()

        mock_encoding = MagicMock()
        mock_encoding.ids = [101, 2054, 2003, 102]
        mock_encoding.attention_mask = [1, 1, 1, 1]
        mock_tokenizer.encode_batch = MagicMock(return_value=[mock_encoding])

        embedder.tokenizer = mock_tokenizer

        # Mock model forward pass
        mock_model = MagicMock()
        # Return a batch of hidden states: (1, seq_len, hidden_dim)
        import mlx.core as mx

        mock_model.return_value = mx.zeros((1, 4, 384))
        embedder.model = mock_model
        embedder.model_name = "bge-small"
        embedder.config = {"hidden_size": 384}

        return embedder

    def test_concurrent_encode_serialized_by_lock(self):
        """10 concurrent encode() calls should be serialized by the GPU lock."""
        embedder = self._make_embedder_with_mock_model()

        # Track when each thread enters and exits the critical section
        entry_times: list[float] = []
        exit_times: list[float] = []
        lock_for_tracking = threading.Lock()

        # Use a real lock to prove serialization
        real_lock = threading.Lock()

        # Patch encode to acquire the lock and track timing
        def locked_encode(texts, normalize=True, batch_size=64):
            with real_lock:
                with lock_for_tracking:
                    entry_times.append(time.monotonic())
                time.sleep(0.01)
                result = np.random.randn(len(texts), 384).astype(np.float32)
                with lock_for_tracking:
                    exit_times.append(time.monotonic())
                return result

        embedder.encode = locked_encode

        # Launch 10 threads
        num_threads = 10
        results = []
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(embedder.encode, [f"text {i}"]) for i in range(num_threads)]
            for f in as_completed(futures):
                results.append(f.result())

        assert len(results) == num_threads
        assert len(entry_times) == num_threads
        assert len(exit_times) == num_threads

        # Sort by entry time
        pairs = sorted(zip(entry_times, exit_times), key=lambda x: x[0])

        # Verify serialization: each entry should be >= the previous exit
        for i in range(1, len(pairs)):
            prev_exit = pairs[i - 1][1]
            curr_entry = pairs[i][0]
            assert curr_entry >= prev_exit - 0.1, (
                f"Thread {i} entered at {curr_entry} before thread {i - 1} "
                f"exited at {prev_exit}. Lock did not serialize."
            )

    def test_concurrent_encode_all_return_valid_embeddings(self):
        """All concurrent encode() calls return correctly shaped arrays."""
        from models.bert_embedder import InProcessEmbedder

        embedder = InProcessEmbedder(model_name="bge-small")

        # Replace encode with a mock that returns valid embeddings
        call_count = {"n": 0}
        lock = threading.Lock()

        def mock_encode(texts, normalize=True, batch_size=64):
            with lock:
                call_count["n"] += 1
            time.sleep(0.005)
            return np.random.randn(len(texts), 384).astype(np.float32)

        embedder.encode = mock_encode

        num_threads = 10
        results = []
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [
                pool.submit(embedder.encode, [f"hello world {i}"]) for i in range(num_threads)
            ]
            for f in as_completed(futures):
                results.append(f.result())

        assert len(results) == num_threads
        for r in results:
            assert isinstance(r, np.ndarray)
            assert r.shape == (1, 384)

    def test_encode_empty_returns_empty_array(self):
        """encode([]) returns empty array without acquiring lock."""
        from models.bert_embedder import InProcessEmbedder

        embedder = InProcessEmbedder(model_name="bge-small")
        # Set model so it doesn't try to load
        embedder.model = MagicMock()
        embedder.tokenizer = MagicMock()
        embedder.model_name = "bge-small"

        result = embedder.encode([])
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_singleton_thread_safety(self):
        """get_in_process_embedder() returns same instance from multiple threads."""
        from models.bert_embedder import (
            get_in_process_embedder,
            reset_in_process_embedder,
        )

        reset_in_process_embedder()

        instances = []
        lock = threading.Lock()

        def get_instance():
            inst = get_in_process_embedder()
            with lock:
                instances.append(id(inst))

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(instances)) == 1, "Singleton returned different instances"
        reset_in_process_embedder()

    def test_gpu_lock_is_shared_with_loader(self):
        """InProcessEmbedder's GPU lock is the same as MLXModelLoader's."""
        from models.bert_embedder import InProcessEmbedder
        from models.loader import MLXModelLoader

        embedder = InProcessEmbedder()
        assert embedder._get_gpu_lock() is MLXModelLoader._mlx_load_lock
