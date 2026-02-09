"""Centralized LLM judge configuration for evals and scripts.

All eval/script files should import from here instead of hardcoding
model names and API endpoints. Change the judge model in ONE place.

Usage:
    from evals.judge_config import JUDGE_MODEL, get_judge_client

    client = get_judge_client()
    resp = client.chat.completions.create(model=JUDGE_MODEL, ...)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Load .env from project root
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# Judge configuration (change here to switch providers)
# ---------------------------------------------------------------------------

# Groq Llama 3.3 70B (FREE, ultra-fast, 30 req/min, 6k req/day)
JUDGE_MODEL = "llama-3.3-70b-versatile"
JUDGE_BASE_URL = "https://api.groq.com/openai/v1"
JUDGE_API_KEY_ENV = "GROQ_API_KEY"

# Alternative: Cerebras Llama 3.3 70B (FREE, 30 req/min, 14.4k req/day)
# JUDGE_MODEL = "llama-3.3-70b"
# JUDGE_BASE_URL = "https://api.cerebras.ai/v1"
# JUDGE_API_KEY_ENV = "CEREBRAS_API_KEY"


def get_judge_api_key() -> str | None:
    """Get the judge API key from environment.

    Returns:
        The API key string, or None if not configured.
    """
    key = os.environ.get(JUDGE_API_KEY_ENV, "")
    if not key or key == "your-key-here":
        return None
    return key


def get_judge_client():
    """Create OpenAI-compatible client for the judge model.

    Returns None if the API key is not set (non-fatal for optional judge usage).
    """
    key = os.environ.get(JUDGE_API_KEY_ENV, "")
    if not key or key == "your-key-here":
        return None
    from openai import OpenAI

    return OpenAI(base_url=JUDGE_BASE_URL, api_key=key)
