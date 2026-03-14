import os
import sys

# CI runs this script via `python evals/quality_gate.py ...`
# The actual evaluation scripts have been moved to internal/archive/evals
# We need to act as a proxy

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import and run the actual script's main
from internal.archive.evals.quality_gate import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
