"""Entry point for python -m jarvis.setup.

This allows running the setup wizard as:
    python -m jarvis.setup
"""

import sys

from jarvis.setup import main

if __name__ == "__main__":
    sys.exit(main())
