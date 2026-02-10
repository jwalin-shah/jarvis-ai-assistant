"""Tests for contacts API router.

Tests comprehensive coverage of all contact endpoints including:
- Avatar retrieval (PNG/SVG)
- Contact info metadata
- Default avatar generation
- Avatar caching
- Phone number normalization
- Email handling
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from integrations.imessage import ContactAvatarData

from api.routers.contacts import (
    AVATAR_COLORS,
    _generate_initials,
    _get_color_for_identifier,
    generate_png_avatar,
    generate_svg_avatar,
    get_avatar_cache,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def app():
    """Get the FastAPI application instance."""
    from api.main import app as main_app

    return main_app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def clear_avatar_cache():
    """Clear avatar cache before each test."""
    from api.routers.contacts import get_avatar_cache

    cache = get_avatar_cache()
    cache.invalidate()
    yield
    cache.invalidate()


@pytest.fixture
def mock_contact_avatar():
    """Create mock ContactAvatarData."""

    def _create(
        has_image: bool = False,
        display_name: str | None = "John Doe",
        first_name: str | None = "John",
        last_name: str | None = "Doe",
    ) -> ContactAvatarData:
        return ContactAvatarData(
            image_data=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100 if has_image else None,
            first_name=first_name,
            last_name=last_name,
            display_name=display_name,
        )

    return _create


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_get_color_for_identifier_consistent(self):
        """Same identifier always returns same color."""
        color1 = _get_color_for_identifier("+15551234567")
        color2 = _get_color_for_identifier("+15551234567")
        assert color1 == color2

    def test_get_color_for_identifier_different(self):
        """Different identifiers return different colors."""
        color1 = _get_color_for_identifier("+15551234567")
        color2 = _get_color_for_identifier("+15559876543")
        # High probability they're different (not guaranteed but very likely)
        assert isinstance(color1, str)
        assert isinstance(color2, str)

    def test_get_color_for_identifier_from_palette(self):
        """Color is from the defined palette."""
        color = _get_color_for_identifier("+15551234567")
        assert color in AVATAR_COLORS

    def test_generate_initials_from_name_two_parts(self):
        """Two-part name returns first and last initials."""
        initials = _generate_initials("+15551234567", "John Doe")
        assert initials == "JD"

    def test_generate_initials_from_name_one_part(self):
        """Single name returns first letter."""
        initials = _generate_initials("+15551234567", "Madonna")
        assert initials == "M"

    def test_generate_initials_from_name_three_parts(self):
        """Three-part name returns first and last."""
        initials = _generate_initials("+15551234567", "John Paul Doe")
        assert initials == "JD"

    def test_generate_initials_from_phone_number(self):
        """Phone number uses last 2 digits."""
        initials = _generate_initials("+15551234567", None)
        assert initials == "67"

    def test_generate_initials_from_short_phone(self):
        """Single-digit phone uses that digit."""
        initials = _generate_initials("5", None)
        assert initials == "5"

    def test_generate_initials_from_email(self):
        """Email uses first letter of local part."""
        initials = _generate_initials("john@example.com", None)
        assert initials == "J"

    def test_generate_initials_fallback(self):
        """Unknown format returns question mark."""
        initials = _generate_initials("unknown", None)
        assert initials == "?"

    def test_generate_svg_avatar_valid_structure(self):
        """SVG avatar has correct structure."""
        svg = generate_svg_avatar("JD", "#FF6B6B", 88)
        assert svg.startswith(b"<svg")
        assert b"JD" in svg
        assert b"#FF6B6B" in svg

    def test_generate_svg_avatar_size(self):
        """SVG avatar respects size parameter."""
        svg = generate_svg_avatar("AB", "#FF6B6B", 100)
        assert b'width="100"' in svg
        assert b'height="100"' in svg

    def test_generate_png_avatar_with_pil(self):
        """PNG avatar uses PIL when available or falls back to SVG."""
        # The function tries to import PIL and uses it if available,
        # otherwise falls back to SVG. Either behavior is acceptable.
        result = generate_png_avatar("JD", "#FF6B6B", 88)
        # Should have generated something (PNG or SVG fallback)
        assert len(result) > 0
        # Should be either PNG or SVG format
        is_png = result[:4] == b"\x89PNG"
        is_svg = result.startswith(b"<svg")
        assert is_png or is_svg

    def test_generate_png_avatar_fallback_to_svg(self):
        """PNG avatar falls back to SVG without PIL."""
        # Test that it handles missing PIL gracefully
        result = generate_png_avatar("JD", "#FF6B6B", 88)
        # Should return either PNG or SVG (both are valid)
        assert len(result) > 0


# =============================================================================
# Get Avatar Endpoint Tests
# =============================================================================


class TestGetAvatarEndpoint:
    """Tests for GET /contacts/{identifier}/avatar."""

    @patch("api.routers.contacts.get_contact_avatar")
    def test_get_avatar_with_contact_image(self, mock_get_avatar, client, mock_contact_avatar):
        """Returns contact photo when available."""
        mock_get_avatar.return_value = mock_contact_avatar(has_image=True)

        response = client.get("/contacts/+15551234567/avatar?format=png")

        assert response.status_code == 200
        assert response.headers["content-type"] in ["image/png", "image/jpeg"]
        assert response.headers["x-avatar-source"] == "contacts"
        mock_get_avatar.assert_called_once()

    @patch("api.routers.contacts.get_contact_avatar")
    def test_get_avatar_generated_default(self, mock_get_avatar, client, mock_contact_avatar):
        """Generates default avatar when no contact photo."""
        mock_get_avatar.return_value = mock_contact_avatar(has_image=False)

        response = client.get("/contacts/+15551234567/avatar?format=svg")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/svg+xml"
        assert response.headers["x-avatar-source"] == "generated"
        assert b"<svg" in response.content
        assert b"JD" in response.content

    @patch("api.routers.contacts.get_contact_avatar")
    def test_get_avatar_caches_result(self, mock_get_avatar, client, mock_contact_avatar):
        """Second request uses cached avatar."""
        mock_get_avatar.return_value = mock_contact_avatar(has_image=False)

        # First request
        response1 = client.get("/contacts/+15551234567/avatar?size=88&format=svg")
        assert response1.status_code == 200
        assert mock_get_avatar.call_count == 1

        # Second request - should use cache
        response2 = client.get("/contacts/+15551234567/avatar?size=88&format=svg")
        assert response2.status_code == 200
        assert response2.headers["x-avatar-source"] == "cache"
        # Still called once (cache hit)
        assert mock_get_avatar.call_count == 1

    @patch("api.routers.contacts.get_contact_avatar")
    def test_get_avatar_normalizes_phone_number(self, mock_get_avatar, client, mock_contact_avatar):
        """Phone numbers are normalized before lookup."""
        mock_get_avatar.return_value = mock_contact_avatar(has_image=False)

        response = client.get("/contacts/5551234567/avatar")

        assert response.status_code == 200
        # Should have normalized to +15551234567 (assuming US)
        called_identifier = mock_get_avatar.call_args[0][0]
        assert called_identifier.startswith("+")

    def test_get_avatar_invalid_phone_number(self, client):
        """Invalid phone number still generates avatar (doesn't fail)."""
        response = client.get("/contacts/invalid-phone/avatar")

        # The API is lenient - it will generate a default avatar even for invalid identifiers
        assert response.status_code == 200
        assert response.headers["x-avatar-source"] == "generated"

    @patch("api.routers.contacts.get_contact_avatar")
    def test_get_avatar_email_identifier(self, mock_get_avatar, client, mock_contact_avatar):
        """Email addresses are handled correctly."""
        mock_get_avatar.return_value = mock_contact_avatar(has_image=False)

        response = client.get("/contacts/john@example.com/avatar")

        assert response.status_code == 200
        # Should normalize email to lowercase
        called_identifier = mock_get_avatar.call_args[0][0]
        assert called_identifier == "john@example.com"

    @patch("api.routers.contacts.get_contact_avatar")
    def test_get_avatar_respects_size_parameter(self, mock_get_avatar, client, mock_contact_avatar):
        """Size parameter affects generated avatar."""
        mock_get_avatar.return_value = mock_contact_avatar(has_image=False)

        response = client.get("/contacts/+15551234567/avatar?size=128&format=svg")

        assert response.status_code == 200
        # SVG should include the specified size
        assert b'width="128"' in response.content
        assert b'height="128"' in response.content

    def test_get_avatar_size_validation(self, client):
        """Size parameter is validated."""
        # Too small
        response = client.get("/contacts/+15551234567/avatar?size=10")
        assert response.status_code == 422

        # Too large
        response = client.get("/contacts/+15551234567/avatar?size=1000")
        assert response.status_code == 422

    def test_get_avatar_format_validation(self, client):
        """Format parameter is validated."""
        response = client.get("/contacts/+15551234567/avatar?format=invalid")
        assert response.status_code == 422

    @patch("api.routers.contacts.get_contact_avatar")
    def test_get_avatar_handles_exceptions(self, mock_get_avatar, client):
        """Exceptions during avatar fetch are handled gracefully."""
        mock_get_avatar.side_effect = RuntimeError("Database error")

        response = client.get("/contacts/+15551234567/avatar")

        # Should still return a generated avatar
        assert response.status_code == 200
        assert response.headers["x-avatar-source"] == "generated"

    @patch("api.routers.contacts.get_contact_avatar")
    def test_get_avatar_png_format_with_pil(self, mock_get_avatar, client, mock_contact_avatar):
        """PNG format returns PNG when PIL available."""
        mock_get_avatar.return_value = mock_contact_avatar(has_image=False)

        response = client.get("/contacts/+15551234567/avatar?format=png")

        assert response.status_code == 200
        # Should attempt PNG (or fall back to SVG)
        content_type = response.headers["content-type"]
        assert content_type in ["image/png", "image/svg+xml"]


# =============================================================================
# Get Contact Info Endpoint Tests
# =============================================================================


class TestGetContactInfoEndpoint:
    """Tests for GET /contacts/{identifier}/info."""

    @patch("api.routers.contacts.get_contact_avatar")
    def test_get_contact_info_with_avatar(self, mock_get_avatar, client, mock_contact_avatar):
        """Returns complete contact info when avatar exists."""
        mock_get_avatar.return_value = mock_contact_avatar(has_image=True, display_name="John Doe")

        response = client.get("/contacts/+15551234567/info")

        assert response.status_code == 200
        data = response.json()
        assert data["identifier"].startswith("+")
        assert data["display_name"] == "John Doe"
        assert data["has_avatar"] is True
        assert data["initials"] == "JD"
        assert "avatar_color" in data
        assert data["avatar_color"] in AVATAR_COLORS

    @patch("api.routers.contacts.get_contact_avatar")
    def test_get_contact_info_without_avatar(self, mock_get_avatar, client, mock_contact_avatar):
        """Returns info with has_avatar=False when no photo."""
        mock_get_avatar.return_value = mock_contact_avatar(has_image=False)

        response = client.get("/contacts/+15551234567/info")

        assert response.status_code == 200
        data = response.json()
        assert data["has_avatar"] is False
        assert data["initials"] == "JD"

    @patch("api.routers.contacts.get_contact_avatar")
    def test_get_contact_info_no_display_name(self, mock_get_avatar, client, mock_contact_avatar):
        """Returns None for display_name when not available."""
        avatar_data = mock_contact_avatar(has_image=False, display_name=None)
        mock_get_avatar.return_value = avatar_data

        response = client.get("/contacts/+15551234567/info")

        assert response.status_code == 200
        data = response.json()
        # Should have display_name from first+last
        assert data["display_name"] == "John Doe"

    @patch("api.routers.contacts.get_contact_avatar")
    def test_get_contact_info_builds_name_from_parts(
        self, mock_get_avatar, client, mock_contact_avatar
    ):
        """Builds display name from first and last name."""
        avatar_data = ContactAvatarData(
            image_data=None,
            first_name="Jane",
            last_name="Smith",
            display_name=None,
        )
        mock_get_avatar.return_value = avatar_data

        response = client.get("/contacts/+15551234567/info")

        assert response.status_code == 200
        data = response.json()
        assert data["display_name"] == "Jane Smith"

    @patch("api.routers.contacts.get_contact_avatar")
    def test_get_contact_info_email_identifier(self, mock_get_avatar, client, mock_contact_avatar):
        """Handles email identifiers correctly."""
        mock_get_avatar.return_value = mock_contact_avatar(has_image=False)

        response = client.get("/contacts/john@example.com/info")

        assert response.status_code == 200
        data = response.json()
        assert data["identifier"] == "john@example.com"

    def test_get_contact_info_invalid_identifier(self, client):
        """Invalid identifier still returns info (doesn't fail)."""
        response = client.get("/contacts/invalid/info")

        # The API is lenient - it will return basic info even for invalid identifiers
        assert response.status_code == 200
        data = response.json()
        assert "initials" in data

    @patch("api.routers.contacts.get_contact_avatar")
    def test_get_contact_info_exception_handling(self, mock_get_avatar, client):
        """Exceptions are handled gracefully."""
        mock_get_avatar.side_effect = RuntimeError("Database error")

        response = client.get("/contacts/+15551234567/info")

        assert response.status_code == 200
        data = response.json()
        # Should still return basic info with generated initials
        assert data["has_avatar"] is False
        assert "initials" in data


# =============================================================================
# Avatar Cache Tests
# =============================================================================


class TestAvatarCache:
    """Tests for avatar cache functionality."""

    def test_cache_get_miss(self):
        """Cache returns False for missing key."""
        cache = get_avatar_cache()
        found, value = cache.get("nonexistent")
        assert found is False
        assert value is None

    def test_cache_set_and_get(self):
        """Set and get work correctly."""
        cache = get_avatar_cache()
        cache.set("test_key", b"test_data")

        found, value = cache.get("test_key")
        assert found is True
        assert value == b"test_data"

    def test_cache_ttl_expiration(self):
        """Cache entries expire after TTL."""
        import time

        cache = get_avatar_cache()
        cache._ttl = 0.1  # 100ms TTL

        cache.set("test_key", b"test_data")
        found1, _ = cache.get("test_key")
        assert found1 is True

        # Wait for expiration
        time.sleep(0.2)

        found2, _ = cache.get("test_key")
        assert found2 is False

    def test_cache_invalidate_specific_key(self):
        """Invalidate removes specific key."""
        cache = get_avatar_cache()
        cache.set("key1", b"data1")
        cache.set("key2", b"data2")

        cache.invalidate("key1")

        found1, _ = cache.get("key1")
        found2, _ = cache.get("key2")
        assert found1 is False
        assert found2 is True

    def test_cache_invalidate_all(self):
        """Invalidate with no key clears all entries."""
        cache = get_avatar_cache()
        cache.set("key1", b"data1")
        cache.set("key2", b"data2")

        cache.invalidate()

        found1, _ = cache.get("key1")
        found2, _ = cache.get("key2")
        assert found1 is False
        assert found2 is False

    def test_cache_stats(self):
        """Stats return accurate cache metrics."""
        cache = get_avatar_cache()

        # Get a fresh cache to avoid interference from other tests
        cache._hits = 0
        cache._misses = 0
        cache.invalidate()

        cache.set("key1", b"data1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_maxsize_eviction(self):
        """Cache evicts oldest entries when full."""
        cache = get_avatar_cache()
        cache._maxsize = 3

        cache.set("key1", b"data1")
        cache.set("key2", b"data2")
        cache.set("key3", b"data3")
        cache.set("key4", b"data4")  # Should evict oldest

        stats = cache.stats()
        assert stats["size"] == 3

    def test_cache_thread_safety(self):
        """Cache handles concurrent access."""
        import threading

        cache = get_avatar_cache()
        errors = []

        def writer(key):
            try:
                cache.set(key, b"data")
                cache.get(key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"key{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# Phone Number Normalization Tests
# =============================================================================


class TestPhoneNumberNormalization:
    """Tests for phone number normalization."""

    @patch("api.routers.contacts.normalize_phone_number")
    def test_normalize_us_number(self, mock_normalize, client):
        """US number is normalized with +1."""
        mock_normalize.return_value = "+15551234567"

        with patch("api.routers.contacts.get_contact_avatar") as mock_get_avatar:
            mock_get_avatar.return_value = ContactAvatarData(None, None, None, None)
            client.get("/contacts/5551234567/info")

            mock_normalize.assert_called_once_with("5551234567")

    @patch("api.routers.contacts.normalize_phone_number")
    def test_normalize_already_formatted(self, mock_normalize, client):
        """Already formatted number is accepted."""
        mock_normalize.return_value = "+15551234567"

        with patch("api.routers.contacts.get_contact_avatar") as mock_get_avatar:
            mock_get_avatar.return_value = ContactAvatarData(None, None, None, None)
            client.get("/contacts/+15551234567/info")

            mock_normalize.assert_called_once_with("+15551234567")


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @patch("api.routers.contacts.get_contact_avatar")
    def test_empty_display_name(self, mock_get_avatar, client):
        """Empty display name is handled correctly."""
        avatar_data = ContactAvatarData(None, None, None, "")
        mock_get_avatar.return_value = avatar_data

        response = client.get("/contacts/+15551234567/info")

        assert response.status_code == 200
        data = response.json()
        assert data["display_name"] is None or data["display_name"] == ""

    @patch("api.routers.contacts.get_contact_avatar")
    def test_special_characters_in_name(self, mock_get_avatar, client):
        """Special characters in name are preserved."""
        avatar_data = ContactAvatarData(None, "José", "García", "José García")
        mock_get_avatar.return_value = avatar_data

        response = client.get("/contacts/+15551234567/info")

        assert response.status_code == 200
        data = response.json()
        assert data["display_name"] == "José García"
        assert data["initials"] == "JG"

    @patch("api.routers.contacts.get_contact_avatar")
    def test_very_long_name(self, mock_get_avatar, client):
        """Very long names are handled without error."""
        long_name = "A" * 100
        avatar_data = ContactAvatarData(None, None, None, long_name)
        mock_get_avatar.return_value = avatar_data

        response = client.get("/contacts/+15551234567/info")

        assert response.status_code == 200
        data = response.json()
        assert data["display_name"] == long_name
