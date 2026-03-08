import sys
from pathlib import Path

# Add the project root to sys.path so we can import from internal.archive.evals
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Forward the execution to the actual script in its new location
if __name__ == "__main__":
    from internal.archive.evals.quality_gate import main
    sys.exit(main())
