"""Entry point for jarvis-tools CLI.

Usage:
    python -m jarvis.tools --help
    python -m jarvis.tools train category --data-dir data/
"""

import sys

from jarvis.tools.cli import app

if __name__ == "__main__":
    sys.exit(app())
