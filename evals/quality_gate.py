import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import actual script and execute it
from internal.archive.evals import quality_gate

if __name__ == "__main__":
    sys.exit(quality_gate.main())
