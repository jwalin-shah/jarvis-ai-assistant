import sys
from pathlib import Path

# Add project root to python path to allow imports from internal
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

# Redirect execution to the new location
from internal.archive.evals import quality_gate

if __name__ == "__main__":
    quality_gate.main()
