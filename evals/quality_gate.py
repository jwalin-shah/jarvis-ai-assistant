import sys
import os

# Manipulate path to allow import from the correct location
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from internal.archive.evals import quality_gate

if __name__ == "__main__":
    # Remove the script itself from argv before passing it on
    sys.exit(quality_gate.main())
