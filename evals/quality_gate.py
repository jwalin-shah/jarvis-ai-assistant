"""Proxy script to maintain CI compatibility after evals/ move.

This script imports and runs the main function from the new location
at internal/archive/evals/quality_gate.py to ensure existing CI
workflows calling `python evals/quality_gate.py` continue to function.
"""

import sys
from pathlib import Path

# Add project root to sys.path to allow absolute imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the actual quality gate script
from internal.archive.evals.quality_gate import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
