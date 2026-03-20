import sys
import os
from pathlib import Path

# Add project root to sys.path so we can import internal.archive.evals
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the actual script
try:
    from internal.archive.evals import quality_gate
    # Delegate to the actual script's main function if it has one,
    # or just let it run if it executes on import.
    if hasattr(quality_gate, 'main'):
        sys.exit(quality_gate.main())
except ImportError as e:
    print(f"Error importing internal.archive.evals.quality_gate: {e}")
    sys.exit(1)
