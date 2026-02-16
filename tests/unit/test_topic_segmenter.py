import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

# Mock dependencies BEFORE importing module under test
mock_adapter = MagicMock()
sys.modules["jarvis.embedding_adapter"] = mock_adapter

# We need to ensure we import from the source, assuming it's in python path
try:
    from jarvis.contracts.imessage import Message

    from jarvis.topics.topic_segmenter import segment_conversation
except ImportError:
    # If not in path (e.g. running as script), assume we are in root
    import os

    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())
    from jarvis.topics.topic_segmenter import segment_conversation

    # Mock Message protocol if not importable
    try:
        from jarvis.contracts.imessage import Message
    except ImportError:
        Message = Any


@dataclass
class MockMessage:
    id: int
    text: str
    date: datetime
    chat_id: str = "chat1"
    is_from_me: bool = True
    sender: str = "me"

    # Add attributes expected by Message protocol
    guid: str = "guid"
    handle_id: int = 0
    account: str = ""
    account_guid: str = ""
    service: str = ""
    group_id: str = None
    is_sent: bool = True
    is_read: bool = True
    error: int = 0
    date_read: datetime = None
    date_delivered: datetime = None
    cache_roomnames: str = None
    is_audio_message: bool = False
    date_played: datetime = None
    item_type: int = 0
    other_handle: int = 0
    share_status: int = 0
    share_direction: int = 0
    is_expirable: bool = False
    expire_state: int = 0
    message_action_type: int = 0
    message_source: int = 0


def test_segment_conversation_drifts():
    # Setup messages
    n = 5
    messages = []
    start_date = datetime.now()
    for i in range(n):
        messages.append(
            MockMessage(id=i, text=f"Message {i}", date=start_date + timedelta(minutes=i))
        )

    # Setup embeddings
    # v0, v1, v2 are aligned (high sim)
    # v3 is orthogonal (low sim)
    # v4 is None

    v0 = np.array([1.0, 0.0], dtype=np.float32)
    v1 = np.array([0.9, 0.1], dtype=np.float32)  # Sim ~0.99
    v2 = np.array([1.0, 0.0], dtype=np.float32)  # Sim ~0.99
    v3 = np.array([0.0, 1.0], dtype=np.float32)  # Sim 0.0 vs v2
    v4 = None

    embeddings = [v0, v1, v2, v3, v4]

    # Expected drifts:
    # i=1 (v0, v1): Sim ~0.99. Drift ~0.01
    # i=2 (v1, v2): Sim ~0.99. Drift ~0.01
    # i=3 (v2, v3): Sim 0.0. Drift 1.0
    # i=4 (v3, v4): v4 is None. Skip.

    with (
        patch("jarvis.topics.utils.prepare_messages_for_segmentation") as mock_prep,
        patch("jarvis.topics.utils.get_embeddings_for_segmentation") as mock_get_embs,
        patch("jarvis.topics.entity_anchor.get_tracker") as mock_get_tracker,
        patch("jarvis.topics.topic_segmenter._compute_segment_metadata"),
        patch("jarvis.topics.topic_segmenter._get_segmentation_config") as mock_config,
    ):
        mock_prep.return_value = (messages, ["text"] * n)
        mock_get_embs.return_value = embeddings

        mock_tracker = MagicMock()
        mock_tracker.get_anchors.return_value = set()
        mock_get_tracker.return_value = mock_tracker

        mock_config_obj = MagicMock()
        mock_config_obj.topic_shift_weight = 0.0
        # High threshold to force segments on high drift
        mock_config_obj.boundary_threshold = 0.5
        mock_config_obj.use_topic_shift_markers = False
        mock_config.return_value = mock_config_obj

        segments = segment_conversation(messages)

        # We expect a split at i=3 (drift=1.0 > 0.5)
        # So segments: [0, 1, 2], [3], [4] (maybe?)

        # Verify message counts
        # Segment 1: [0, 1, 2] -> 3 messages
        # Segment 2: [3] -> 1 message
        # Message 4 is missing.

        total_msgs = sum(len(s.messages) for s in segments)
        assert total_msgs == 4
        assert len(segments) == 2
        assert len(segments[0].messages) == 3
        assert len(segments[1].messages) == 1

        print("Verification passed!")


if __name__ == "__main__":
    test_segment_conversation_drifts()
