import sys
from pathlib import Path

# Add project root to sys.path so we can import from internal
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Forward execution to the relocated script
if __name__ == "__main__":
    from internal.archive.evals import quality_gate
    sys.exit(quality_gate.main())
