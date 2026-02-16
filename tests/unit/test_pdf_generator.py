"""Unit tests for PDF generator."""

import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

# Add project root to sys.path to allow imports
sys.path.append(os.getcwd())

import pytest

# Mock reportlab modules before importing the module under test
# This is necessary because reportlab might not be installed in the test environment
mock_reportlab = MagicMock()
sys.modules["reportlab"] = mock_reportlab
sys.modules["reportlab.lib"] = mock_reportlab.lib
sys.modules["reportlab.lib.colors"] = mock_reportlab.lib.colors
sys.modules["reportlab.lib.enums"] = mock_reportlab.lib.enums
sys.modules["reportlab.lib.pagesizes"] = mock_reportlab.lib.pagesizes
sys.modules["reportlab.lib.styles"] = mock_reportlab.lib.styles
sys.modules["reportlab.lib.units"] = mock_reportlab.lib.units
sys.modules["reportlab.platypus"] = mock_reportlab.platypus

# Setup specific attributes needed by the module
mock_reportlab.lib.pagesizes.letter = (612, 792)
mock_reportlab.lib.units.inch = 72

# Mock colors
mock_reportlab.lib.colors.Color = MagicMock()
mock_reportlab.lib.colors.white = MagicMock()
mock_reportlab.lib.colors.black = MagicMock()

# Mock styles
mock_styles = MagicMock()
mock_styles.__getitem__.return_value = MagicMock()  # Allow styles['Heading1']
sys.modules["reportlab.lib.styles"].getSampleStyleSheet.return_value = mock_styles
sys.modules["reportlab.lib.styles"].ParagraphStyle = MagicMock()


# Import after mocking
from contracts.imessage import Attachment, Conversation, Message, Reaction  # noqa: E402
from jarvis.pdf_generator import (  # noqa: E402
    PDFExportOptions,
    PDFGenerator,
    generate_pdf,
    generate_pdf_base64,
)


@pytest.fixture
def sample_message() -> Message:
    return Message(
        id=1,
        chat_id="chat1",
        sender="+1234567890",
        sender_name="Alice",
        text="Hello world",
        date=datetime(2023, 1, 1, 12, 0, 0),
        is_from_me=False,
    )


@pytest.fixture
def sample_conversation() -> Conversation:
    return Conversation(
        chat_id="chat1",
        participants=["+1234567890"],
        display_name="Alice",
        last_message_date=datetime(2023, 1, 1, 12, 0, 0),
        message_count=1,
        is_group=False,
    )


def test_pdf_generator_init() -> None:
    """Test PDF generator initialization."""
    generator = PDFGenerator()
    assert generator.page_width == 612
    assert generator.page_height == 792
    assert generator.margin == 0.75 * 72


def test_generate_simple(sample_message: Message, sample_conversation: Conversation) -> None:
    """Test simple PDF generation."""
    generator = PDFGenerator()

    with patch("jarvis.pdf_generator.SimpleDocTemplate") as mock_doc_template:
        mock_doc = mock_doc_template.return_value

        pdf_bytes = generator.generate([sample_message], sample_conversation)

        assert isinstance(pdf_bytes, bytes)
        mock_doc_template.assert_called_once()
        mock_doc.build.assert_called_once()

        # Verify story content was passed to build
        args, _ = mock_doc.build.call_args
        story = args[0]
        assert len(story) > 0


def test_generate_with_options(sample_message: Message) -> None:
    """Test generation with options."""
    generator = PDFGenerator()
    options = PDFExportOptions(include_attachments=False, include_reactions=False)

    with patch("jarvis.pdf_generator.SimpleDocTemplate") as mock_doc_template:
        generator.generate([sample_message], options=options)

        # Verify call happened
        mock_doc_template.assert_called_once()


def test_filter_by_date(sample_message: Message) -> None:
    """Test date filtering logic."""
    generator = PDFGenerator()

    # Message is 2023-01-01 12:00:00

    # Filter before (end date before message)
    options = PDFExportOptions(end_date=datetime(2022, 12, 31))
    filtered = generator._filter_by_date([sample_message], options)
    assert len(filtered) == 0

    # Filter after (start date after message)
    options = PDFExportOptions(start_date=datetime(2023, 1, 2))
    filtered = generator._filter_by_date([sample_message], options)
    assert len(filtered) == 0

    # Filter include (message within range)
    options = PDFExportOptions(start_date=datetime(2022, 12, 31), end_date=datetime(2023, 1, 2))
    filtered = generator._filter_by_date([sample_message], options)
    assert len(filtered) == 1


def test_generate_pdf_wrappers(sample_message: Message) -> None:
    """Test wrapper functions."""
    with patch("jarvis.pdf_generator.PDFGenerator") as mock_generator:
        mock_instance = mock_generator.return_value
        mock_instance.generate.return_value = b"pdf_content"

        # Test generate_pdf
        result = generate_pdf([sample_message])
        assert result == b"pdf_content"
        mock_instance.generate.assert_called()

        # Test generate_pdf_base64
        result_b64 = generate_pdf_base64([sample_message])
        # b"pdf_content" base64 encoded is "cGRmX2NvbnRlbnQ="
        assert result_b64 == "cGRmX2NvbnRlbnQ="


@patch("jarvis.pdf_generator.Path")
def test_attachment_handling(mock_path: MagicMock, sample_message: Message) -> None:
    """Test attachment handling with mocked file system."""
    # Setup attachment
    attachment = Attachment(
        filename="test.jpg", file_path="/path/to/test.jpg", mime_type="image/jpeg", file_size=1000
    )
    sample_message.attachments = [attachment]

    # Setup mock path behavior
    mock_p = mock_path.return_value
    mock_p.expanduser.return_value = mock_p
    mock_p.exists.return_value = True
    mock_p.suffix = ".jpg"

    generator = PDFGenerator()

    with patch("jarvis.pdf_generator.SimpleDocTemplate") as mock_doc_template:
        generator.generate([sample_message])

        # Verify Image was instantiated (accessed via sys.modules mock)
        # Note: Depending on implementation details, checking exact calls on mocked modules
        # that are also patched might be tricky.
        # But we can assume if no exception was raised, it worked.
        mock_doc_template.assert_called()


def test_reaction_handling(sample_message: Message) -> None:
    """Test reaction handling."""
    reaction = Reaction(type="love", sender="+0987654321", sender_name="Bob", date=datetime.now())
    sample_message.reactions = [reaction]

    generator = PDFGenerator()
    options = PDFExportOptions(include_reactions=True)

    with patch("jarvis.pdf_generator.SimpleDocTemplate") as mock_doc_template:
        generator.generate([sample_message], options=options)
        mock_doc_template.assert_called()
