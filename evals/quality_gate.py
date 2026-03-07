import sys
import argparse

# The actual implementation of quality_gate was moved to internal/archive/evals/quality_gate.py
# This serves as a proxy for the CI pipeline which still expects it here.

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "internal" / "archive"))

try:
    from evals.quality_gate import main
except ImportError:
    def main():
        print("ERROR: Could not find internal/archive/evals/quality_gate.py", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
