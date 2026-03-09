#!/usr/bin/env python3
import os
import sys

# Add the project root to sys.path so we can import from internal.archive
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import and run the actual script
from internal.archive.evals import quality_gate

if __name__ == "__main__":
    sys.exit(quality_gate.main())
