import sys
from unittest.mock import MagicMock

sys.modules["mlx"] = MagicMock()
sys.modules["mlx.core"] = MagicMock()
sys.modules["mlx.nn"] = MagicMock()

mock_adapter = MagicMock()
sys.modules["jarvis.embedding_adapter"] = mock_adapter

sys.modules["models.bert_embedder"] = MagicMock()

import time  # noqa: E402

import numpy as np  # noqa: E402

try:
    from jarvis.search.vec_search import VecSearcher  # noqa: E402
except ImportError:
    sys.modules["jarvis.db"] = MagicMock()
    from jarvis.search.vec_search import VecSearcher  # noqa: E402

from datetime import datetime  # noqa: E402

from contracts.imessage import Message  # noqa: E402


def benchmark():
    mock_db = MagicMock()
    mock_conn = MagicMock()
    mock_db.connection.return_value.__enter__.return_value = mock_conn

    mock_conn.execute.return_value.fetchall.return_value = []

    mock_embedder = MagicMock()
    mock_adapter.get_embedder.return_value = mock_embedder

    count = 100000
    messages = []
    now = datetime.now()
    for i in range(count):
        messages.append(
            Message(
                id=i,
                text=f"message {i}",
                date=now,
                is_from_me=True,
                chat_id="chat1",
                sender="me",
                sender_name="Me",
            )
        )

    embedding_dim = 384
    embeddings = np.random.rand(count, embedding_dim).astype(np.float32)
    mock_embedder.encode.return_value = embeddings

    searcher = VecSearcher(mock_db)
    searcher._embedder = mock_embedder

    print(f"Benchmarking index_messages with {count} messages...")

    start = time.time()
    searcher.index_messages(messages)
    end = time.time()

    print(f"Time taken for {count} messages: {end - start:.4f}s")


if __name__ == "__main__":
    benchmark()
