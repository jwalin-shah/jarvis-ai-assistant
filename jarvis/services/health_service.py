"""Shared health check helpers used by both the HTTP API and socket handlers.

This module contains pure-function helpers with no dependency on FastAPI or the
api/ layer, so they can be imported safely from jarvis/handlers/ without
creating a backwards layer coupling.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from typing import Any

import psutil

logger = logging.getLogger(__name__)

BYTES_PER_MB = 1024 * 1024
BYTES_PER_GB = 1024**3


def get_memory_stats() -> tuple[float, float, float]:
    """Return (available_gb, used_gb, total_gb) using native macOS vm_stat.

    Falls back to psutil if vm_stat is unavailable.
    """
    try:
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode != 0:
            raise RuntimeError("vm_stat failed")

        lines = result.stdout.strip().split("\n")

        page_size = 4096
        if lines and "page size of" in lines[0]:
            match = re.search(r"page size of (\d+) bytes", lines[0])
            if match:
                page_size = int(match.group(1))

        stats: dict[str, int] = {}
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                try:
                    stats[key.strip()] = int(value.strip().rstrip("."))
                except ValueError:
                    continue

        free = stats.get("Pages free", 0) * page_size
        inactive = stats.get("Pages inactive", 0) * page_size
        speculative = stats.get("Pages speculative", 0) * page_size
        active = stats.get("Pages active", 0) * page_size
        wired = (stats.get("Pages wired down", 0) or stats.get("Pages wired", 0)) * page_size
        compressed = stats.get("Pages occupied by compressor", 0) * page_size

        total = free + inactive + speculative + active + wired + compressed
        available = free + inactive + speculative
        used = active + wired + compressed

        return available / BYTES_PER_GB, used / BYTES_PER_GB, total / BYTES_PER_GB
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        logger.warning("vm_stat failed, falling back to psutil: %s", e)
        memory = psutil.virtual_memory()
        return (
            memory.available / BYTES_PER_GB,
            memory.used / BYTES_PER_GB,
            memory.total / BYTES_PER_GB,
        )


def get_process_memory() -> tuple[float, float]:
    """Return (rss_mb, vms_mb) for the current process."""
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / BYTES_PER_MB, mem_info.vms / BYTES_PER_MB
    except (OSError, AttributeError) as e:
        logger.error("Failed to fetch process memory: %s", e)
        return 0.0, 0.0


def check_imessage_access() -> bool:
    """Return True if iMessage Full Disk Access is available."""
    try:
        from integrations.imessage import ChatDBReader

        reader = ChatDBReader()
        result = reader.check_access()
        reader.close()
        return result
    except (OSError, PermissionError) as e:
        logger.error("iMessage access check failed: %s", e)
        return False


def get_memory_mode(available_gb: float) -> str:
    """Map available memory to a named mode: FULL / LITE / MINIMAL."""
    if available_gb >= 4.0:
        return "FULL"
    if available_gb >= 2.0:
        return "LITE"
    return "MINIMAL"


def check_model_loaded() -> bool:
    """Return True if the MLX generator has a model loaded."""
    try:
        from models import get_generator

        generator = get_generator()
        return generator.is_loaded()
    except Exception:
        return False


def get_model_info() -> dict[str, Any] | None:
    """Return a dict of current model info, or None if unavailable."""
    try:
        from jarvis.metrics import get_model_info_cache

        cache = get_model_info_cache()
        found, cached = cache.get("model_info_dict")
        if found:
            return cached  # type: ignore[return-value]
    except Exception:
        pass

    try:
        from models import get_generator

        generator = get_generator()
        if generator is None:
            return None

        loader = getattr(generator, "_loader", None)
        if loader is None:
            return None

        info = loader.get_current_model_info()
        if info is None:
            return None

        result: dict[str, Any] = {
            "id": info.get("id"),
            "display_name": info.get("display_name", "Unknown"),
            "loaded": info.get("loaded", False),
            "memory_usage_mb": info.get("memory_usage_mb", 0.0),
            "quality_tier": info.get("quality_tier"),
        }

        try:
            from jarvis.metrics import get_model_info_cache

            get_model_info_cache().set("model_info_dict", result)
        except Exception:
            pass

        return result
    except (ImportError, AttributeError, KeyError, TypeError, RuntimeError):
        return None


def get_recommended_model(total_ram_gb: float) -> str | None:
    """Return the recommended model ID for the given total RAM."""
    try:
        from models import get_recommended_model

        spec = get_recommended_model(total_ram_gb)
        return spec.id
    except Exception:
        return None
