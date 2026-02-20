import sys
from unittest.mock import MagicMock

# Mock dependencies that might be missing or platform-specific
sys.modules["mlx"] = MagicMock()
sys.modules["mlx.core"] = MagicMock()
sys.modules["mlx.nn"] = MagicMock()

# Mock jarvis.embedding_adapter to avoid complex imports
mock_adapter = MagicMock()
sys.modules["jarvis.embedding_adapter"] = mock_adapter

# Mock models.bert_embedder to avoid import errors
sys.modules["models.bert_embedder"] = MagicMock()

import time
import numpy as np
from unittest.mock import patch
# We need to make sure we can import jarvis.search.vec_search
# It imports orjson, numpy. We assume these are available or we need to install them.
# It imports jarvis.db.
try:
    from jarvis.search.vec_search import VecSearcher
except ImportError:
    # If imports fail due to other dependencies, mock them too
    sys.modules["jarvis.db"] = MagicMock()
    from jarvis.search.vec_search import VecSearcher

from contracts.imessage import Message
from datetime import datetime

def benchmark():
    # Mock DB
    mock_db = MagicMock()
    mock_conn = MagicMock()
    mock_db.connection.return_value.__enter__.return_value = mock_conn

    # Mock cursor to avoid failures in existing_ids check
    mock_conn.execute.return_value.fetchall.return_value = []

    # Mock Embedder
    mock_embedder = MagicMock()
    # Ensure get_embedder returns our mock
    mock_adapter.get_embedder.return_value = mock_embedder

    # Create dummy messages
    count = 100000
    messages = []
    now = datetime.now()
    for i in range(count):
        messages.append(Message(
            id=i,
            text=f"message {i}",
            date=now,
            is_from_me=True,
            chat_id="chat1",
            sender="me",
            sender_name="Me"
        ))

    # Mock encode to return random embeddings
    embedding_dim = 384
    # Pre-generate random embeddings
    embeddings = np.random.rand(count, embedding_dim).astype(np.float32)
    mock_embedder.encode.return_value = embeddings

    # Instantiate VecSearcher
    # We mocked jarvis.embedding_adapter.get_embedder, which VecSearcher uses in __init__
    searcher = VecSearcher(mock_db)
    # Manually set embedder just in case
    searcher._embedder = mock_embedder

    print(f"Benchmarking index_messages with {count} messages...")

    # Warmup (optional)

    # Benchmark
    start = time.time()
    searcher.index_messages(messages)
    end = time.time()

    print(f"Time taken for {count} messages: {end - start:.4f}s")

if __name__ == "__main__":
    benchmark()
