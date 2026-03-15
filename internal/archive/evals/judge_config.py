"""Centralized LLM judge configuration for evals and scripts.  # noqa: E501
  # noqa: E501
All eval/script files should import from here instead of hardcoding  # noqa: E501
model names and API endpoints. Change the judge model in ONE place.  # noqa: E501
  # noqa: E501
Usage:  # noqa: E501
    from evals.judge_config import JUDGE_MODEL, get_judge_client  # noqa: E501
  # noqa: E501
    client = get_judge_client()  # noqa: E501
    resp = client.chat.completions.create(model=JUDGE_MODEL, ...)  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from __future__ import annotations  # noqa: E402  # noqa: E501

# noqa: E501
import os  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501

  # noqa: E501
# Load .env from project root  # noqa: E501
_env_path = Path(__file__).parent.parent / ".env"  # noqa: E501
if _env_path.exists():  # noqa: E501
    for _line in _env_path.read_text().splitlines():  # noqa: E501
        _line = _line.strip()  # noqa: E501
        if _line and not _line.startswith("#") and "=" in _line:  # noqa: E501
            _k, _, _v = _line.partition("=")  # noqa: E501
            os.environ.setdefault(_k.strip(), _v.strip())  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# Judge configuration (change here to switch providers)  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
  # noqa: E501
# Cerebras Model Options (FREE tier: 30 req/min, 14.4k req/day)  # noqa: E501
# Change JUDGE_MODEL to switch between them  # noqa: E501
  # noqa: E501
# Option 1: GPT-OSS-120B - Ultra fast, high intelligence  # noqa: E501
JUDGE_MODEL = "gpt-oss-120b"  # noqa: E501
  # noqa: E501
# Option 2: Llama 3.3 70B - Reliable, well-tested (RECOMMENDED)  # noqa: E501
# JUDGE_MODEL = "llama-3.3-70b"  # noqa: E501
  # noqa: E501
# Option 3: Qwen 2.5 72B - Strong multilingual, good for complex reasoning  # noqa: E501
# JUDGE_MODEL = "qwen-2.5-72b"  # noqa: E501
  # noqa: E501
# Option 4: QwQ 32B Preview - Good at reasoning through problems  # noqa: E501
# JUDGE_MODEL = "qwq-32b-preview"  # noqa: E501
  # noqa: E501
JUDGE_BASE_URL = "https://api.cerebras.ai/v1"  # noqa: E501
# Use paid API key for higher rate limits (1000 RPM vs 30 RPM)  # noqa: E501
JUDGE_API_KEY_ENV = "CEREBRAS_PAID_API_KEY"  # noqa: E501
  # noqa: E501
# Alternative Providers:  # noqa: E501
# Groq: JUDGE_MODEL = "llama-3.3-70b-versatile", JUDGE_BASE_URL = "https://api.groq.com/openai/v1"  # noqa: E501
# DeepInfra: Various models available  # noqa: E501
  # noqa: E501
  # noqa: E501
def get_judge_api_key() -> str | None:  # noqa: E501
    """Get the judge API key from environment.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        The API key string, or None if not configured.  # noqa: E501
    """  # noqa: E501
    key = os.environ.get(JUDGE_API_KEY_ENV, "")  # noqa: E501
    if not key or key == "your-key-here":  # noqa: E501
        return None  # noqa: E501
    return key  # noqa: E501
  # noqa: E501
  # noqa: E501
def get_judge_client():  # noqa: E501
    """Create OpenAI-compatible client for the judge model.  # noqa: E501
  # noqa: E501
    Returns None if the API key is not set (non-fatal for optional judge usage).  # noqa: E501
    """  # noqa: E501
    key = os.environ.get(JUDGE_API_KEY_ENV, "")  # noqa: E501
    if not key or key == "your-key-here":  # noqa: E501
        return None  # noqa: E501
    from openai import OpenAI  # noqa: E501
  # noqa: E501
    return OpenAI(base_url=JUDGE_BASE_URL, api_key=key)  # noqa: E501
