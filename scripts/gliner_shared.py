"""Shared utilities for GLiNER-related scripts.

Extracted from duplicated code across:
- run_extractor_bakeoff.py
- eval_gliner_candidates.py
- build_fact_filter_dataset.py
- build_gliner_candidate_goldset.py
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def parse_context_messages(raw: object) -> list[str]:
    """Parse context payloads from gold JSON (string blob or list) into message texts."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if not isinstance(raw, str):
        return []

    payload = raw.strip()
    if not payload:
        return []

    # CSV context format: "id|speaker|text || id|speaker|text".
    chunks = [c.strip() for c in payload.split("||") if c.strip()]
    messages: list[str] = []
    for chunk in chunks:
        parts = chunk.split("|", 2)
        if len(parts) == 3:
            text = parts[2].strip()
        else:
            text = chunk
        if text:
            messages.append(text)
    return messages


def _safe_major(version: str) -> int:
    """Best-effort major version parser."""
    try:
        return int(version.split(".", 1)[0])
    except (TypeError, ValueError):
        return -1


def enforce_runtime_stack(allow_unstable_stack: bool) -> None:
    """Fail fast on unsupported runtime unless explicitly overridden."""
    try:
        import huggingface_hub
        import transformers
    except Exception:
        return

    tver = getattr(transformers, "__version__", "unknown")
    hver = getattr(huggingface_hub, "__version__", "unknown")
    if _safe_major(str(tver)) >= 5:
        msg = (
            "Detected transformers=%s, huggingface_hub=%s. "
            "GLiNER quality may degrade on this stack."
        )
        if allow_unstable_stack:
            log.warning(
                msg + " Continuing because --allow-unstable-stack was set.",
                tver,
                hver,
            )
            return
        log.error(
            msg + " Re-run via scripts/run_gliner_compat.sh (recommended), "
            "or pass --allow-unstable-stack to proceed anyway.",
            tver,
            hver,
        )
        raise SystemExit(2)
