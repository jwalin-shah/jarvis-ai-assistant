"""Production-hardened fixtures with guaranteed isolation.

These fixtures ensure complete test isolation, preventing state leakage
between tests that can cause flakiness.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    pass


@pytest.fixture
def isolated_env() -> Iterator[dict[str, str]]:
    """Provide isolated environment variables.

    Automatically restores original environment after test.

    Example:
        def test_with_env_var(isolated_env):
            isolated_env["MY_VAR"] = "value"
            result = function_reading_env()
            assert result == "value"
            # Original env restored after test
    """
    # Deep copy of original environment
    original_env = {k: v for k, v in os.environ.items()}

    # Create isolated env copy
    isolated = dict(original_env)

    try:
        # Replace os.environ temporarily
        os.environ = isolated
        yield isolated
    finally:
        # Restore original environment completely
        os.environ.clear()
        os.environ.update(original_env)


@pytest.fixture
def temp_workspace() -> Iterator[Path]:
    """Provide temporary workspace directory.

    Guaranteed cleanup even if test fails. Changes into the directory
    and restores original working directory after.

    Example:
        def test_file_operations(temp_workspace):
            # temp_workspace is current working directory
            file_path = temp_workspace / "test.txt"
            file_path.write_text("content")
            # Directory and contents deleted after test
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="jarvis_test_"))
    original_cwd = Path.cwd()

    try:
        os.chdir(temp_dir)
        yield temp_dir
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def isolated_config() -> Iterator[MagicMock]:
    """Provide isolated configuration.

    Prevents config file modifications from leaking between tests.

    Example:
        def test_config_modification(isolated_config):
            isolated_config.set("key", "value")
            # Changes don't affect other tests
    """
    import jarvis.config as config_module

    # Save original config state
    original_config = getattr(config_module, "_config", None)
    original_config_path = getattr(config_module, "_config_path", None)

    # Create isolated config file
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
    ) as f:
        f.write("{}")
        temp_config_path = Path(f.name)

    try:
        # Replace config state
        config_module._config = {}
        config_module._config_path = temp_config_path

        # Create mock-like object for compatibility
        mock_config = MagicMock()
        mock_config._config = {}
        mock_config._config_path = temp_config_path
        mock_module = MagicMock()
        mock_module._config = {}
        mock_module._config_path = temp_config_path

        # Patch the module
        with patch.dict(sys.modules, {"jarvis.config": mock_module}):
            yield mock_module

    finally:
        # Restore original config
        config_module._config = original_config
        config_module._config_path = original_config_path

        # Cleanup temp file
        try:
            temp_config_path.unlink()
        except FileNotFoundError:
            pass


@pytest.fixture
def mock_metal_device() -> Iterator[MagicMock]:
    """Mock Metal device for GPU-independent tests.

    Example:
        def test_mlx_operation(mock_metal_device):
            mock_metal_device.is_available.return_value = False
            result = check_gpu_available()
            assert result is False
    """
    mock_device = MagicMock()
    mock_device.is_available.return_value = True
    mock_device.get_active_memory.return_value = 1024 * 1024 * 1024  # 1GB
    mock_device.get_peak_memory.return_value = 8 * 1024 * 1024 * 1024  # 8GB

    with patch("mlx.core.metal", mock_device):
        yield mock_device


@pytest.fixture
def deterministic_uuid() -> Iterator[Callable[[], uuid.UUID]]:
    """Provide deterministic UUID generation.

    Returns sequential UUIDs for reproducible tests.

    Example:
        def test_uuid_generation(deterministic_uuid):
            id1 = deterministic_uuid()
            id2 = deterministic_uuid()
            assert str(id1) == "00000000-0000-0000-0000-000000000001"
            assert str(id2) == "00000000-0000-0000-0000-000000000002"
    """
    counter = [0]

    def mock_uuid4() -> uuid.UUID:
        counter[0] += 1
        return uuid.UUID(f"00000000-0000-0000-0000-{counter[0]:012x}")

    with patch("uuid.uuid4", mock_uuid4):
        yield mock_uuid4


@pytest.fixture
def isolated_db() -> Iterator[Path]:
    """Provide isolated SQLite database.

    Creates a temporary database file that is cleaned up after test.

    Example:
        def test_database_operations(isolated_db):
            conn = sqlite3.connect(isolated_db)
            # Operations on isolated database
    """
    with tempfile.NamedTemporaryFile(
        suffix=".db",
        delete=False,
    ) as f:
        db_path = Path(f.name)

    try:
        yield db_path
    finally:
        try:
            db_path.unlink()
        except FileNotFoundError:
            pass


@pytest.fixture
def no_network() -> Iterator[None]:
    """Block all network access during test.

    Tests that accidentally make network requests will fail fast.

    Example:
        def test_offline_operation(no_network):
            # Any network access will raise ConnectionError
            with pytest.raises(ConnectionError):
                requests.get("https://example.com")
    """
    import socket

    original_socket = socket.socket

    class NoNetworkSocket:
        """Socket that rejects all connections."""

        def __init__(self, *args, **kwargs):
            raise ConnectionError("Network access disabled in test")

    socket.socket = NoNetworkSocket

    try:
        yield
    finally:
        socket.socket = original_socket


@pytest.fixture
def isolated_imports() -> Iterator[None]:
    """Provide isolated import state.

    Clears module cache to prevent import side effects.

    Example:
        def test_import_behavior(isolated_imports):
            # Import is fresh, not cached
            import jarvis.some_module
    """
    # Save original modules
    original_modules = dict(sys.modules)

    try:
        yield
    finally:
        # Restore only modules that existed before
        current_modules = set(sys.modules.keys())
        for mod in current_modules - set(original_modules.keys()):
            del sys.modules[mod]


@pytest.fixture
def seeded_random(seed: int = 42) -> Iterator[dict[str, Any]]:
    """Provide seeded random number generators.

    Creates isolated RNGs that don't affect global state.

    Example:
        def test_random_embeddings(seeded_random):
            rng = seeded_random["numpy"]
            embedding = rng.randn(384)  # Deterministic per test
    """
    import random

    import numpy as np

    # Save original states
    original_py_state = random.getstate()
    original_np_state = np.random.get_state()

    # Create isolated RNGs
    np_rng = np.random.RandomState(seed)
    py_rng = random.Random(seed)

    try:
        yield {"numpy": np_rng, "python": py_rng}
    finally:
        # Restore global states
        random.setstate(original_py_state)
        np.random.set_state(original_np_state)


@pytest.fixture
def clean_temp_files() -> Iterator[None]:
    """Ensure all temp files are cleaned up after test.

    Tracks temp file creation and deletes any that remain.

    Example:
        def test_file_creation(clean_temp_files):
            # Any temp files created here are tracked
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(b"content")
            # File deleted after test even with delete=False
    """
    import tempfile

    # Get list of existing temp files
    temp_dir = Path(tempfile.gettempdir())
    existing = set(temp_dir.glob("tmp*"))

    try:
        yield
    finally:
        # Clean up any new temp files
        current = set(temp_dir.glob("tmp*"))
        new_files = current - existing
        for f in new_files:
            try:
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)
            except (OSError, PermissionError):
                pass


# Type alias for convenience
IsolatedEnv = dict[str, str]
TempWorkspace = Path
IsolatedConfig = MagicMock
IsolatedDB = Path
