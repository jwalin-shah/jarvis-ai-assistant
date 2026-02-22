"""Shared fixtures for integration tests.

Provides common mocks and helpers for testing the RAG flow:
- Mock iMessage reader
- Mock generator
- Mock context fetcher
- Sample message data
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from jarvis.contracts.imessage import Message
from jarvis.contracts.models import GenerationResponse
from jarvis.core.health import reset_degradation_controller
from jarvis.core.memory import reset_memory_controller
from tests.helpers import create_mock_message

# --- Context Managers for Mocking ---


@contextmanager
def patch_imessage_reader():
    """Context manager to patch the iMessage reader."""
    mock_reader = MagicMock()
    mock_reader.check_access.return_value = True
    mock_reader.get_messages.return_value = []
    mock_reader.get_conversations.return_value = []
    mock_reader.search.return_value = []

    with patch("integrations.imessage.ChatDBReader") as mock_class:
        mock_class.return_value.__enter__ = MagicMock(return_value=mock_reader)
        mock_class.return_value.__exit__ = MagicMock(return_value=False)
        yield mock_reader


@contextmanager
def patch_generator():
    """Context manager to patch the model generator."""
    mock_gen = MagicMock()
    mock_gen.is_loaded.return_value = True
    mock_gen.generate.return_value = GenerationResponse(
        text="Default mock response",
        tokens_used=10,
        generation_time_ms=100.0,
        model_name="mock-model",
        used_template=False,
        template_name=None,
        finish_reason="stop",
    )

    with patch("models.get_generator", return_value=mock_gen):
        yield mock_gen


@contextmanager
def patch_services():
    """Context manager to patch all RAG services for testing."""
    with patch_imessage_reader() as mock_reader:
        with patch_generator() as mock_gen:
            with patch("core.health.get_degradation_controller") as mock_deg:
                # Configure degradation controller to pass through
                mock_controller = MagicMock()
                mock_controller.execute.side_effect = lambda feature, func, *args: func(*args)
                mock_deg.return_value = mock_controller
                yield {
                    "reader": mock_reader,
                    "generator": mock_gen,
                    "degradation": mock_controller,
                }


# --- Pytest Fixtures ---


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons before and after each test."""
    reset_memory_controller()
    reset_degradation_controller()
    yield
    reset_memory_controller()
    reset_degradation_controller()


@pytest.fixture
def mock_messages() -> list[Message]:
    """Sample conversation messages for testing."""
    return [
        create_mock_message(
            "Hey, dinner tomorrow?",
            is_from_me=False,
            msg_id=1,
            date=datetime(2024, 1, 15, 18, 0),
        ),
        create_mock_message(
            "Sure, what time?",
            is_from_me=True,
            msg_id=2,
            date=datetime(2024, 1, 15, 18, 5),
        ),
        create_mock_message(
            "7pm at my place?",
            is_from_me=False,
            msg_id=3,
            date=datetime(2024, 1, 15, 18, 10),
        ),
    ]


@pytest.fixture
def mock_work_messages() -> list[Message]:
    """Sample work-related conversation for testing."""
    return [
        create_mock_message(
            "Did you finish the report?",
            is_from_me=False,
            sender_name="Boss",
            msg_id=1,
            date=datetime(2024, 1, 15, 9, 0),
        ),
        create_mock_message(
            "Almost done, just reviewing the numbers",
            is_from_me=True,
            msg_id=2,
            date=datetime(2024, 1, 15, 9, 30),
        ),
        create_mock_message(
            "Great, can you send it by EOD?",
            is_from_me=False,
            sender_name="Boss",
            msg_id=3,
            date=datetime(2024, 1, 15, 9, 35),
        ),
    ]
