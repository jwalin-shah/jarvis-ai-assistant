import sys
from pathlib import Path

# Add project root to sys.path so internal can be resolved
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import and run the actual script
from internal.archive.evals import quality_gate


def main():
    # If the user didn't specify baseline/candidate, quality_gate will use defaults
    # Let's override sys.argv to supply the internal/archive/evals/ paths
    # if they aren't explicitly provided, so that defaults work.

    has_baseline = any(arg == "--baseline" for arg in sys.argv)
    has_candidate = any(arg == "--candidate" for arg in sys.argv)

    if not has_baseline:
        sys.argv.extend(["--baseline", "internal/archive/evals/baselines/baseline_20260221.json"])

    if not has_candidate:
        sys.argv.extend(
            ["--candidate", "internal/archive/evals/results/eval_pipeline_baseline.json"]
        )

    # Intercept missing file errors gracefully when --allow-missing-candidate is set
    # Actually quality_gate.main() handles this internally, but it requires the
    # candidate file arg to be resolved correctly.

    # We should let the main method run
    sys.exit(quality_gate.main())


if __name__ == "__main__":
    main()
