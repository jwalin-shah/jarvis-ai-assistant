import sys
from pathlib import Path

# Add project root to sys.path so we can import from internal
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the actual quality gate script from the archive
from internal.archive.evals import quality_gate

if __name__ == "__main__":
    quality_gate.main()
