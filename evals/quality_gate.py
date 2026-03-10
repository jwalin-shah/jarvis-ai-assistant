"""Proxy script for CI workflows.

CI workflows expect this file to exist and support specific flags.
This forwards execution to the relocated script in internal.archive.
"""

import sys
from pathlib import Path

# Add project root to sys.path to allow imports from internal.archive
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import and run the actual script
from internal.archive.evals.quality_gate import main

if __name__ == "__main__":
    sys.exit(main())
