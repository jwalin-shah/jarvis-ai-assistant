import sys
from pathlib import Path

# Redirect execution to the new location of quality_gate.py
new_path = (
    Path(__file__).resolve().parent.parent / "internal" / "archive" / "evals" / "quality_gate.py"
)

if new_path.exists():
    import runpy

    sys.argv[0] = str(new_path)
    runpy.run_path(str(new_path), run_name="__main__")
else:
    print(f"Error: Could not find original quality_gate script at {new_path}")
    sys.exit(1)
