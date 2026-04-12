import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


# Save paths
SAVE_DIR = PROJECT_ROOT / "evals" / "optimized_reply.json"
CATEGORY_SAVE_DIR = PROJECT_ROOT / "evals" / "optimized_categories"

# Ensure category save dir exists
CATEGORY_SAVE_DIR.mkdir(parents=True, exist_ok=True)


def train_general():
    print(f"\n{'=' * 60}\nOptimizing GENERAL Universal Prompt\n{'=' * 60}\n")
    pass
