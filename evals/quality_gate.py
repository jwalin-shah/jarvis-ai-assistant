import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from internal.archive.evals.quality_gate import main

if __name__ == "__main__":
    sys.exit(main())
