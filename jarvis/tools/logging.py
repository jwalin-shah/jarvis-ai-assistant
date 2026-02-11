"""Standardized logging for JARVIS tools.

Provides consistent log formatting, file/console handlers,
and per-tool log file isolation.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Default log directory
LOG_DIR = Path("logs/tools")
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_tool_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    """Get a configured logger for a tool.

    Creates a logger with both file and console handlers.
    Log files are written to logs/tools/{name}.log

    Args:
        name: Tool name (used for logger name and log file)
        log_dir: Override default log directory

    Returns:
        Configured logger instance

    Example:
        logger = get_tool_logger("train_category")
        logger.info("Starting training...")
        # Logs to: logs/tools/train_category.log
        # And to console (stdout)
    """
    logger = logging.getLogger(f"jarvis.tools.{name}")

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler (INFO and above)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(console)

    # File handler (DEBUG and above)
    log_dir = log_dir or LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{name}.log"
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(file_handler)

    return logger


def setup_root_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
    console: bool = True,
) -> logging.Logger:
    """Set up root logging for tool execution.

    Args:
        level: Logging level
        log_file: Optional file to log to
        console: Whether to log to console

    Returns:
        Configured root logger
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers
    root.handlers = []

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    return root


class ProgressLogger:
    """Logger with progress tracking for long-running operations.

    Example:
        progress = ProgressLogger(logger, total=1000, desc="Processing")
        for item in items:
            process(item)
            progress.update(1)
        progress.finish()
    """

    def __init__(
        self,
        logger: logging.Logger,
        total: int,
        desc: str = "Processing",
        log_interval: int = 100,
    ):
        self.logger = logger
        self.total = total
        self.desc = desc
        self.log_interval = log_interval
        self.current = 0
        self.start_time = __import__("time").time()

        self.logger.info(f"{desc}: Starting (total={total})")

    def update(self, n: int = 1) -> None:
        """Update progress by n steps."""
        self.current += n

        if self.current % self.log_interval == 0 or self.current == self.total:
            pct = 100 * self.current / self.total
            elapsed = __import__("time").time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0

            self.logger.info(
                f"{self.desc}: {self.current}/{self.total} ({pct:.1f}%) [{rate:.1f} items/s]"
            )

    def finish(self) -> None:
        """Mark progress as complete."""
        elapsed = __import__("time").time() - self.start_time
        self.logger.info(f"{self.desc}: Complete ({self.current} items in {elapsed:.1f}s)")
