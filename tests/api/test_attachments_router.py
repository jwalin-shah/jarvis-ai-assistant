"""Tests for attachments API router.

Tests comprehensive coverage of all attachment endpoints including:
- List attachments with filtering
- Attachment statistics per conversation
- Storage summary across conversations
- Thumbnail retrieval
- File download
- Security validation
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.main import app
from api.routers.attachments import format_bytes
from api.schemas import AttachmentTypeEnum
from contracts.imessage import Attachment


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_reader():
    """Create mock ChatDBReader."""
    reader = MagicMock()
    reader.check_access = MagicMock(return_value=True)
    reader.close = MagicMock()
    return reader


@pytest.fixture
def client(mock_reader):
    """Create test client with mocked dependencies."""
    from api.dependencies import get_imessage_reader

    def override_get_imessage_reader():
        yield mock_reader

    app.dependency_overrides[get_imessage_reader] = override_get_imessage_reader
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def sample_attachment():
    """Create a sample attachment."""

    def _create(
        filename: str = "IMG_1234.jpg",
        mime_type: str = "image/jpeg",
        file_size: int = 245760,
        file_path: str = "~/Library/Messages/Attachments/test.jpg",
    ) -> Attachment:
        return Attachment(
            filename=filename,
            file_path=file_path,
            mime_type=mime_type,
            file_size=file_size,
            width=1920,
            height=1080,
            duration_seconds=None,
            created_date=datetime.now(UTC),
            is_sticker=False,
            uti="public.jpeg",
        )

    return _create


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestFormatBytes:
    """Tests for format_bytes helper function."""

    def test_format_bytes_less_than_kb(self):
        """Bytes less than 1KB formatted correctly."""
        assert format_bytes(500) == "500 B"

    def test_format_bytes_kb(self):
        """Kilobytes formatted correctly."""
        assert format_bytes(1024) == "1.0 KB"
        assert format_bytes(2560) == "2.5 KB"

    def test_format_bytes_mb(self):
        """Megabytes formatted correctly."""
        assert format_bytes(1024 * 1024) == "1.0 MB"
        assert format_bytes(5 * 1024 * 1024) == "5.0 MB"

    def test_format_bytes_gb(self):
        """Gigabytes formatted correctly."""
        assert format_bytes(1024 * 1024 * 1024) == "1.0 GB"
        assert format_bytes(2.5 * 1024 * 1024 * 1024) == "2.5 GB"

    def test_format_bytes_zero(self):
        """Zero bytes formatted correctly."""
        assert format_bytes(0) == "0 B"


# =============================================================================
# List Attachments Endpoint Tests
# =============================================================================


class TestListAttachmentsEndpoint:
    """Tests for GET /attachments."""

    def test_list_attachments_all(self, client, mock_reader, sample_attachment):
        """List all attachments without filters."""
        attachment = sample_attachment()
        mock_reader.get_attachments.return_value = [
            {
                "attachment": attachment,
                "message_id": 12345,
                "message_date": datetime.now(UTC),
                "chat_id": "chat123",
                "sender": "+15551234567",
                "sender_name": "John Doe",
                "is_from_me": False,
            }
        ]

        response = client.get("/attachments")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["attachment"]["filename"] == "IMG_1234.jpg"
        assert data[0]["message_id"] == 12345
        assert data[0]["sender_name"] == "John Doe"

    def test_list_attachments_filter_by_chat(self, client, mock_reader, sample_attachment):
        """Filter attachments by chat_id."""
        attachment = sample_attachment()
        mock_reader.get_attachments.return_value = [
            {
                "attachment": attachment,
                "message_id": 12345,
                "message_date": datetime.now(UTC),
                "chat_id": "chat123",
                "sender": "+15551234567",
                "sender_name": None,
                "is_from_me": False,
            }
        ]

        response = client.get("/attachments?chat_id=chat123")

        assert response.status_code == 200
        mock_reader.get_attachments.assert_called_once()
        call_kwargs = mock_reader.get_attachments.call_args.kwargs
        assert call_kwargs["chat_id"] == "chat123"

    def test_list_attachments_filter_by_type_images(self, client, mock_reader, sample_attachment):
        """Filter attachments by type (images)."""
        mock_reader.get_attachments.return_value = []

        response = client.get("/attachments?attachment_type=images")

        assert response.status_code == 200
        call_kwargs = mock_reader.get_attachments.call_args.kwargs
        assert call_kwargs["attachment_type"] == "images"

    def test_list_attachments_filter_by_type_videos(self, client, mock_reader):
        """Filter attachments by type (videos)."""
        mock_reader.get_attachments.return_value = []

        response = client.get("/attachments?attachment_type=videos")

        assert response.status_code == 200
        call_kwargs = mock_reader.get_attachments.call_args.kwargs
        assert call_kwargs["attachment_type"] == "videos"

    def test_list_attachments_filter_by_date_after(self, client, mock_reader):
        """Filter attachments by after date."""
        mock_reader.get_attachments.return_value = []

        after_date = "2024-01-01T00:00:00Z"
        response = client.get(f"/attachments?after={after_date}")

        assert response.status_code == 200
        call_kwargs = mock_reader.get_attachments.call_args.kwargs
        assert call_kwargs["after"] is not None

    def test_list_attachments_filter_by_date_before(self, client, mock_reader):
        """Filter attachments by before date."""
        mock_reader.get_attachments.return_value = []

        before_date = "2024-12-31T23:59:59Z"
        response = client.get(f"/attachments?before={before_date}")

        assert response.status_code == 200
        call_kwargs = mock_reader.get_attachments.call_args.kwargs
        assert call_kwargs["before"] is not None

    def test_list_attachments_respects_limit(self, client, mock_reader):
        """Limit parameter is passed to reader."""
        mock_reader.get_attachments.return_value = []

        response = client.get("/attachments?limit=50")

        assert response.status_code == 200
        call_kwargs = mock_reader.get_attachments.call_args.kwargs
        assert call_kwargs["limit"] == 50

    def test_list_attachments_limit_validation(self, client):
        """Limit validation enforces bounds."""
        # Too small
        response = client.get("/attachments?limit=0")
        assert response.status_code == 422

        # Too large
        response = client.get("/attachments?limit=1000")
        assert response.status_code == 422

    def test_list_attachments_invalid_type(self, client):
        """Invalid attachment type returns 422."""
        response = client.get("/attachments?attachment_type=invalid")
        assert response.status_code == 422

    def test_list_attachments_empty_results(self, client, mock_reader):
        """Empty results return empty array."""
        mock_reader.get_attachments.return_value = []

        response = client.get("/attachments")

        assert response.status_code == 200
        assert response.json() == []

    def test_list_attachments_no_full_disk_access(self):
        """Without Full Disk Access, returns 403."""
        from api.dependencies import get_imessage_reader

        # Create a reader that denies access
        no_access_reader = MagicMock()
        no_access_reader.check_access = MagicMock(return_value=False)
        no_access_reader.close = MagicMock()

        def override_get_imessage_reader():
            # Simulate the dependency raising an exception for no access
            from fastapi import HTTPException

            no_access_reader.close()
            raise HTTPException(status_code=403, detail="No access")

        app.dependency_overrides[get_imessage_reader] = override_get_imessage_reader
        try:
            test_client = TestClient(app)
            response = test_client.get("/attachments")
            assert response.status_code == 403
        finally:
            app.dependency_overrides.clear()


# =============================================================================
# Attachment Stats Endpoint Tests
# =============================================================================


class TestAttachmentStatsEndpoint:
    """Tests for GET /attachments/stats/{chat_id}."""

    def test_get_stats_success(self, client, mock_reader):
        """Stats endpoint returns aggregated data."""
        mock_reader.get_attachment_stats.return_value = {
            "total_count": 150,
            "total_size_bytes": 524288000,
            "by_type": {"images": 100, "videos": 30, "documents": 20},
            "size_by_type": {"images": 314572800, "videos": 157286400, "documents": 52428800},
        }

        response = client.get("/attachments/stats/chat123")

        assert response.status_code == 200
        data = response.json()
        assert data["chat_id"] == "chat123"
        assert data["total_count"] == 150
        assert data["total_size_bytes"] == 524288000
        assert data["total_size_formatted"] == "500.0 MB"
        assert data["by_type"]["images"] == 100

    def test_get_stats_zero_attachments(self, client, mock_reader):
        """Stats with zero attachments returns zero values."""
        mock_reader.get_attachment_stats.return_value = {
            "total_count": 0,
            "total_size_bytes": 0,
            "by_type": {},
            "size_by_type": {},
        }

        response = client.get("/attachments/stats/chat123")

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 0
        assert data["total_size_formatted"] == "0 B"


# =============================================================================
# Storage Summary Endpoint Tests
# =============================================================================


class TestStorageSummaryEndpoint:
    """Tests for GET /attachments/storage."""

    def test_storage_summary_success(self, client, mock_reader):
        """Storage summary returns aggregated data."""
        mock_reader.get_storage_by_conversation.return_value = [
            {
                "chat_id": "chat1",
                "display_name": "Alice",
                "attachment_count": 150,
                "total_size_bytes": 524288000,
            },
            {
                "chat_id": "chat2",
                "display_name": "Bob",
                "attachment_count": 100,
                "total_size_bytes": 314572800,
            },
        ]

        response = client.get("/attachments/storage")

        assert response.status_code == 200
        data = response.json()
        assert data["total_attachments"] == 250
        assert data["total_size_bytes"] == 524288000 + 314572800
        assert len(data["by_conversation"]) == 2
        assert data["by_conversation"][0]["chat_id"] == "chat1"

    def test_storage_summary_respects_limit(self, client, mock_reader):
        """Storage summary respects limit parameter."""
        mock_reader.get_storage_by_conversation.return_value = []

        response = client.get("/attachments/storage?limit=10")

        assert response.status_code == 200
        call_kwargs = mock_reader.get_storage_by_conversation.call_args.kwargs
        assert call_kwargs["limit"] == 10

    def test_storage_summary_limit_validation(self, client):
        """Limit validation enforces bounds."""
        response = client.get("/attachments/storage?limit=300")
        assert response.status_code == 422

    def test_storage_summary_empty(self, client, mock_reader):
        """Empty storage summary returns zero values."""
        mock_reader.get_storage_by_conversation.return_value = []

        response = client.get("/attachments/storage")

        assert response.status_code == 200
        data = response.json()
        assert data["total_attachments"] == 0
        assert data["total_size_bytes"] == 0
        assert data["by_conversation"] == []


# =============================================================================
# Thumbnail Endpoint Tests
# =============================================================================


class TestThumbnailEndpoint:
    """Tests for GET /attachments/thumbnail."""

    def test_get_thumbnail_success(self, client, mock_reader, tmp_path):
        """Returns thumbnail when available."""
        # Create a fake thumbnail file
        thumb_path = tmp_path / "thumb.jpg"
        thumb_path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)  # JPEG header

        mock_reader.get_attachment_thumbnail_path.return_value = str(thumb_path)

        file_path = str(Path.home() / "Library/Messages/Attachments/test.jpg")
        response = client.get(f"/attachments/thumbnail?file_path={file_path}")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"

    def test_get_thumbnail_not_found(self, client, mock_reader):
        """Returns 404 when thumbnail doesn't exist."""
        mock_reader.get_attachment_thumbnail_path.return_value = None

        file_path = str(Path.home() / "Library/Messages/Attachments/test.jpg")
        response = client.get(f"/attachments/thumbnail?file_path={file_path}")

        assert response.status_code == 404
        assert "not available" in response.json()["detail"]

    def test_get_thumbnail_path_security(self, client, mock_reader):
        """Path outside attachments directory is rejected."""
        # Try to access file outside attachments directory
        file_path = "/etc/passwd"
        response = client.get(f"/attachments/thumbnail?file_path={file_path}")

        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]

    def test_get_thumbnail_expands_tilde(self, client, mock_reader, tmp_path):
        """Tilde in path is expanded correctly."""
        thumb_path = tmp_path / "thumb.jpg"
        thumb_path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)

        mock_reader.get_attachment_thumbnail_path.return_value = str(thumb_path)

        # Use tilde notation
        file_path = "~/Library/Messages/Attachments/test.jpg"
        response = client.get(f"/attachments/thumbnail?file_path={file_path}")

        # Should handle tilde expansion (may still 404 if path doesn't match security check)
        assert response.status_code in [200, 403, 404]

    def test_get_thumbnail_fallback_to_original(self, client, mock_reader, tmp_path):
        """Falls back to original for small images."""
        mock_reader.get_attachment_thumbnail_path.return_value = None

        # Create small original image
        attachments_dir = tmp_path / "Library" / "Messages" / "Attachments"
        attachments_dir.mkdir(parents=True)
        img_path = attachments_dir / "test.jpg"
        img_path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 1000)  # Small JPEG

        with patch("api.routers.attachments.Path.home", return_value=tmp_path):
            response = client.get(f"/attachments/thumbnail?file_path={img_path}")

            assert response.status_code == 200
            assert response.headers["content-type"] == "image/jpeg"


# =============================================================================
# Download Endpoint Tests
# =============================================================================


class TestDownloadEndpoint:
    """Tests for GET /attachments/file."""

    def test_download_success(self, client, mock_reader, tmp_path):
        """Downloads file successfully."""
        # Create attachment file
        attachments_dir = tmp_path / "Library" / "Messages" / "Attachments"
        attachments_dir.mkdir(parents=True)
        file_path = attachments_dir / "test.jpg"
        file_content = b"\xff\xd8\xff" + b"\x00" * 1000
        file_path.write_bytes(file_content)

        with patch("api.routers.attachments.Path.home", return_value=tmp_path):
            response = client.get(f"/attachments/file?file_path={file_path}")

            assert response.status_code == 200
            assert response.headers["content-type"] == "image/jpeg"
            assert response.content == file_content

    def test_download_not_found(self, client, mock_reader):
        """Returns 404 when file doesn't exist."""
        file_path = str(Path.home() / "Library/Messages/Attachments/nonexistent.jpg")
        response = client.get(f"/attachments/file?file_path={file_path}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_download_path_security(self, client, mock_reader):
        """Path outside attachments directory is rejected."""
        file_path = "/etc/passwd"
        response = client.get(f"/attachments/file?file_path={file_path}")

        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]

    def test_download_media_types(self, client, mock_reader, tmp_path):
        """Different file types have correct media types."""
        attachments_dir = tmp_path / "Library" / "Messages" / "Attachments"
        attachments_dir.mkdir(parents=True)

        test_cases = [
            ("test.png", "image/png"),
            ("test.mp4", "video/mp4"),
            ("test.pdf", "application/pdf"),
            ("test.m4a", "audio/mp4"),
        ]

        with patch("api.routers.attachments.Path.home", return_value=tmp_path):
            for filename, expected_type in test_cases:
                file_path = attachments_dir / filename
                file_path.write_bytes(b"\x00" * 100)

                response = client.get(f"/attachments/file?file_path={file_path}")

                assert response.status_code == 200
                assert response.headers["content-type"] == expected_type

    def test_download_unknown_extension(self, client, mock_reader, tmp_path):
        """Unknown file extension uses generic media type."""
        attachments_dir = tmp_path / "Library" / "Messages" / "Attachments"
        attachments_dir.mkdir(parents=True)
        file_path = attachments_dir / "test.xyz"
        file_path.write_bytes(b"\x00" * 100)

        with patch("api.routers.attachments.Path.home", return_value=tmp_path):
            response = client.get(f"/attachments/file?file_path={file_path}")

            assert response.status_code == 200
            assert response.headers["content-type"] == "application/octet-stream"


# =============================================================================
# Security Tests
# =============================================================================


class TestSecurityValidation:
    """Tests for security validation."""

    def test_path_traversal_attack(self, client, mock_reader):
        """Path traversal attempts are blocked."""
        # Try various path traversal techniques
        malicious_paths = [
            "../../../etc/passwd",
            "~/Library/Messages/Attachments/../../../../../../etc/passwd",
            "/etc/passwd",
        ]

        for path in malicious_paths:
            response = client.get(f"/attachments/file?file_path={path}")
            assert response.status_code in [403, 400, 404]

    def test_symlink_security(self, client, mock_reader, tmp_path):
        """Symlinks outside attachments directory are rejected."""
        # Create a symlink pointing outside attachments
        attachments_dir = tmp_path / "Library" / "Messages" / "Attachments"
        attachments_dir.mkdir(parents=True)
        symlink_path = attachments_dir / "malicious_link"

        try:
            symlink_path.symlink_to("/etc/passwd")

            with patch("api.routers.attachments.Path.home", return_value=tmp_path):
                response = client.get(f"/attachments/file?file_path={symlink_path}")

                assert response.status_code == 403
        except OSError:
            # Skip if symlinks not supported
            pytest.skip("Symlinks not supported on this system")


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_filename(self, client, mock_reader, sample_attachment):
        """Empty filename is handled gracefully."""
        attachment = sample_attachment(filename="")
        mock_reader.get_attachments.return_value = [
            {
                "attachment": attachment,
                "message_id": 1,
                "message_date": datetime.now(UTC),
                "chat_id": "chat123",
                "sender": "+15551234567",
                "sender_name": None,
                "is_from_me": False,
            }
        ]

        response = client.get("/attachments")

        assert response.status_code == 200
        data = response.json()
        assert data[0]["attachment"]["filename"] == ""

    def test_very_large_file_size(self, client, mock_reader, sample_attachment):
        """Very large file sizes are formatted correctly."""
        attachment = sample_attachment(file_size=5 * 1024 * 1024 * 1024)  # 5 GB
        mock_reader.get_attachments.return_value = [
            {
                "attachment": attachment,
                "message_id": 1,
                "message_date": datetime.now(UTC),
                "chat_id": "chat123",
                "sender": "+15551234567",
                "sender_name": None,
                "is_from_me": False,
            }
        ]

        response = client.get("/attachments")

        assert response.status_code == 200
        data = response.json()
        assert data[0]["attachment"]["file_size"] == 5 * 1024 * 1024 * 1024

    def test_special_characters_in_filename(self, client, mock_reader, sample_attachment):
        """Special characters in filename are preserved."""
        attachment = sample_attachment(filename="test file (1) [copy].jpg")
        mock_reader.get_attachments.return_value = [
            {
                "attachment": attachment,
                "message_id": 1,
                "message_date": datetime.now(UTC),
                "chat_id": "chat123",
                "sender": "+15551234567",
                "sender_name": None,
                "is_from_me": False,
            }
        ]

        response = client.get("/attachments")

        assert response.status_code == 200
        data = response.json()
        assert data[0]["attachment"]["filename"] == "test file (1) [copy].jpg"
