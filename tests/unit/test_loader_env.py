import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Mock modules
# We explicitly do NOT mock mlx_lm initially to verify it's not imported
# But we need mlx.core and psutil and numpy
sys.modules["mlx"] = MagicMock()
sys.modules["mlx.core"] = MagicMock()
sys.modules["mlx.nn"] = MagicMock()
sys.modules["psutil"] = MagicMock()
sys.modules["numpy"] = MagicMock()

# We need to mock models.bert_embedder because models/__init__ imports it
sys.modules["models.bert_embedder"] = MagicMock()

class TestLoaderEnv(unittest.TestCase):
    def setUp(self):
        # Clean up environment variables
        if "HF_HUB_OFFLINE" in os.environ:
            del os.environ["HF_HUB_OFFLINE"]
        if "TRANSFORMERS_OFFLINE" in os.environ:
            del os.environ["TRANSFORMERS_OFFLINE"]

        # Unload models.loader if it was already loaded
        if "models.loader" in sys.modules:
            del sys.modules["models.loader"]

        # Ensure mlx_lm is not in sys.modules (if it was from previous tests)
        keys_to_remove = [k for k in sys.modules if k.startswith("mlx_lm")]
        for k in keys_to_remove:
            del sys.modules[k]

    def tearDown(self):
        if "HF_HUB_OFFLINE" in os.environ:
            del os.environ["HF_HUB_OFFLINE"]
        if "TRANSFORMERS_OFFLINE" in os.environ:
            del os.environ["TRANSFORMERS_OFFLINE"]

    def test_lazy_imports_and_env_config(self):
        # 1. Import models.loader
        try:
            import models.loader
            from models.loader import MLXModelLoader
        except ImportError as e:
            self.fail(f"Failed to import models.loader: {e}")

        # Assert environment variables are NOT set just by importing
        self.assertIsNone(os.environ.get("HF_HUB_OFFLINE"))
        self.assertIsNone(os.environ.get("TRANSFORMERS_OFFLINE"))

        # Assert mlx_lm is NOT imported yet
        self.assertNotIn("mlx_lm", sys.modules)

        # 2. Instantiate MLXModelLoader
        loader = MLXModelLoader()

        # Assert environment variables ARE set after instantiation
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
        self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")

        # Assert mlx_lm is STILL not imported (instantiation doesn't trigger it, only usage)
        self.assertNotIn("mlx_lm", sys.modules)

        # 3. Call a method that triggers import (we need to mock mlx_lm now)
        # We inject a mock into sys.modules so the lazy import succeeds
        mock_mlx_lm = MagicMock()
        # Mock load returning (model, tokenizer)
        mock_mlx_lm.load.return_value = (MagicMock(), MagicMock())
        sys.modules["mlx_lm"] = mock_mlx_lm
        sys.modules["mlx_lm.sample_utils"] = MagicMock()

        # Call load()
        # We need to mock psutil virtual_memory to allow load check to pass
        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.available = 100 * 1024 * 1024 * 1024 # 100GB
            loader.config.model_path = "dummy"

            # We also need to mock os.path.isdir to return True so it thinks model is local
            with patch("os.path.isdir", return_value=True):
                 loader.load()

        # Now mlx_lm should be used (load called on mock)
        mock_mlx_lm.load.assert_called()

if __name__ == "__main__":
    unittest.main()
