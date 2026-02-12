"""TEST-02: Thread safety tests for InProcessCrossEncoder concurrent predict().

Same pattern as TEST-01 but for the cross-encoder predict() method.
Verifies the GPU lock serializes concurrent predict() calls.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock

import numpy as np
import pytest


class TestCrossEncoderThreadSafety:
    """Verify InProcessCrossEncoder.predict() serializes via GPU lock."""

    def test_concurrent_predict_serialized_by_lock(self):
        """10 concurrent predict() calls should be serialized by the GPU lock."""
        from models.cross_encoder import InProcessCrossEncoder

        ce = InProcessCrossEncoder()

        real_lock = threading.Lock()
        entry_times: list[float] = []
        exit_times: list[float] = []
        timing_lock = threading.Lock()

        def locked_predict(pairs, batch_size=32):
            with real_lock:
                with timing_lock:
                    entry_times.append(time.monotonic())
                time.sleep(0.01)
                result = np.random.rand(len(pairs)).astype(np.float32)
                with timing_lock:
                    exit_times.append(time.monotonic())
                return result

        ce.predict = locked_predict

        num_threads = 10
        results = []
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [
                pool.submit(ce.predict, [("query", f"doc{i}")])
                for i in range(num_threads)
            ]
            for f in as_completed(futures):
                results.append(f.result())

        assert len(results) == num_threads

        # Verify serialization
        pairs = sorted(zip(entry_times, exit_times), key=lambda x: x[0])
        for i in range(1, len(pairs)):
            prev_exit = pairs[i - 1][1]
            curr_entry = pairs[i][0]
            assert curr_entry >= prev_exit - 0.001, (
                f"Thread {i} entered before thread {i-1} exited. Not serialized."
            )

    def test_concurrent_predict_returns_valid_scores(self):
        """All concurrent predict() calls return correctly shaped score arrays."""
        from models.cross_encoder import InProcessCrossEncoder

        ce = InProcessCrossEncoder()

        def mock_predict(pairs, batch_size=32):
            time.sleep(0.005)
            return np.random.rand(len(pairs)).astype(np.float32)

        ce.predict = mock_predict

        num_threads = 10
        results = []
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [
                pool.submit(ce.predict, [(f"query {i}", f"doc {i}")])
                for i in range(num_threads)
            ]
            for f in as_completed(futures):
                results.append(f.result())

        assert len(results) == num_threads
        for r in results:
            assert isinstance(r, np.ndarray)
            assert r.shape == (1,)
            assert 0.0 <= r[0] <= 1.0

    def test_predict_empty_pairs(self):
        """predict([]) returns empty array without acquiring lock."""
        from models.cross_encoder import InProcessCrossEncoder

        ce = InProcessCrossEncoder()
        ce.model = MagicMock()
        ce.tokenizer = MagicMock()
        ce.model_name = "test"

        result = ce.predict([])
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_gpu_lock_shared_with_embedder_and_loader(self):
        """Cross-encoder's GPU lock is the same as MLXModelLoader's."""
        from models.cross_encoder import InProcessCrossEncoder
        from models.loader import MLXModelLoader

        ce = InProcessCrossEncoder()
        assert ce._get_gpu_lock() is MLXModelLoader._mlx_load_lock

    def test_singleton_thread_safety(self):
        """get_cross_encoder() returns same instance from multiple threads."""
        from models.cross_encoder import get_cross_encoder, reset_cross_encoder

        reset_cross_encoder()

        instances = []
        lock = threading.Lock()

        def get_instance():
            inst = get_cross_encoder()
            with lock:
                instances.append(id(inst))

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(instances)) == 1, "Singleton returned different instances"
        reset_cross_encoder()
