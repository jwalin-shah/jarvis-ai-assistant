"""Training entry point script.

Wraps the actual training logic to provide a clean CLI interface.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train a model using JARVIS infrastructure")
    parser.add_argument("--config", type=str, help="Path to training config", default="ft_configs/config.yaml")
    parser.add_argument("--test", action="store_true", help="Run a quick test training loop")

    args = parser.parse_args()

    # Ensure we are in the project root
    project_root = Path(__file__).parent.parent.parent

    # Construct the command to run the actual training script
    # We use uv run to ensure dependencies are met
    cmd = ["uv", "run", "python", "-m", "scripts.training.run_train"]

    if args.config:
        cmd.extend(["--config", args.config])

    if args.test:
        cmd.append("--test")

    print(f"Running training command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, cwd=project_root, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
