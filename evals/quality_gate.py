"""Proxy script for CI to call the relocated quality_gate.py."""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the actual module and run its main function
from internal.archive.evals import quality_gate

if __name__ == "__main__":
    sys.exit(quality_gate.main())
