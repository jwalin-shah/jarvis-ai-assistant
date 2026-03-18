import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the actual script
from internal.archive.evals import quality_gate

if __name__ == "__main__":
    quality_gate.main()
