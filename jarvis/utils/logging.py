"""Shared logging setup for scripts and modules."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_script_logging(
    script_name: str,
    *,
    log_dir: Path | str | None = None,
    level: int = logging.INFO,
    mode: str = "w",
) -> logging.Logger:
    """Configure logging with both file and stream handlers.

    Provides a standard logging setup used across all scripts. Logs to both
    stdout and a file, with graceful fallback if the file can't be opened.

    Args:
        script_name: Name of the script (used for log filename).
        log_dir: Directory for log file. Defaults to ``<project_root>/logs/``.
        level: Logging level.
        mode: File open mode ("w" to overwrite, "a" to append).

    Returns:
        A logger named after the calling module.
    """
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{script_name}.log"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(log_file, mode=mode))
    except OSError as exc:
        print(f"Warning: could not open log file {log_file}: {exc}", flush=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )

    logger = logging.getLogger(script_name)
    logger.info("Logging to %s", log_file)
    return logger
