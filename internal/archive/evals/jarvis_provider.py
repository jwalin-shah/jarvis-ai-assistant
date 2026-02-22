#!/usr/bin/env python3
"""Example provider script for JARVIS.

This script demonstrates how to provide external context or tools to the JARVIS
system during evaluation or runtime. It sets up the environment and imports
necessary configurations.
"""

import logging
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from jarvis.prompts.generation_config import DEFAULT_REPETITION_PENALTY  # noqa: E402

logging.basicConfig(
    level=logging.WARNING, format="%(levelname)s: %(message)s", stream=sys.stderr
)

print(f"Loaded config with penalty: {DEFAULT_REPETITION_PENALTY}")
