"""Unified model lifecycle management.

Tracks all loaded models (generators, embedders, classifiers) in one place.
Provides a single point to query total memory usage and coordinate
loading/unloading under the 8GB memory constraint.

Usage:
    from models.lifecycle import get_lifecycle_manager

    manager = get_lifecycle_manager()
    manager.register("llm", generator)
    manager.register("embedder", embedder)

    # Query total memory
    print(f"Total model memory: {manager.total_memory_mb():.0f}MB")

    # Check before loading a new model
    if manager.can_load(required_mb=1200):
        model.load()

    # Get status of all models
    for name, info in manager.status().items():
        print(f"{name}: loaded={info['loaded']}, memory={info['memory_mb']:.0f}MB")
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from contracts.models import ManagedModel

logger = logging.getLogger(__name__)

# Default memory budget: 6GB for models (out of 8GB total, 2GB for OS/apps)
DEFAULT_MEMORY_BUDGET_MB = 6 * 1024


@dataclass
class ModelInfo:
    """Snapshot of a registered model's state."""

    name: str
    loaded: bool
    memory_mb: float
    model_type: str  # e.g., "generator", "embedder", "classifier"


class ModelLifecycleManager:
    """Tracks all loaded models and coordinates memory budget.

    Thread-safe singleton that provides:
    - Registration of model instances
    - Total memory usage across all models
    - Memory budget enforcement (8GB constraint)
    - Status reporting for observability

    All model types that implement the ManagedModel protocol can be registered.
    """

    def __init__(self, memory_budget_mb: float = DEFAULT_MEMORY_BUDGET_MB) -> None:
        self._models: dict[str, tuple[ManagedModel, str]] = {}  # name -> (model, type)
        self._memory_budget_mb = memory_budget_mb
        self._lock = threading.RLock()

    def register(self, name: str, model: ManagedModel, model_type: str = "unknown") -> None:
        """Register a model for lifecycle tracking.

        Args:
            name: Unique name for this model (e.g., "llm", "embedder", "category_classifier").
            model: Model instance implementing ManagedModel protocol.
            model_type: Category string for grouping (e.g., "generator", "embedder").
        """
        with self._lock:
            if name in self._models:
                logger.debug("Re-registering model '%s' (replacing previous)", name)
            self._models[name] = (model, model_type)
            logger.debug(
                "Registered model '%s' (type=%s, loaded=%s)",
                name,
                model_type,
                model.is_loaded(),
            )

    def unregister(self, name: str) -> None:
        """Remove a model from lifecycle tracking.

        Args:
            name: Name of the model to unregister.
        """
        with self._lock:
            if name in self._models:
                del self._models[name]
                logger.debug("Unregistered model '%s'", name)

    def total_memory_mb(self) -> float:
        """Return total memory usage across all registered models.

        Returns:
            Total memory in MB used by all loaded models.
        """
        with self._lock:
            total = 0.0
            for model, _ in self._models.values():
                try:
                    total += model.get_memory_usage_mb()
                except Exception:
                    pass
            return total

    def can_load(self, required_mb: float) -> bool:
        """Check if loading a model with the given memory requirement is safe.

        Args:
            required_mb: Memory required by the model to load (in MB).

        Returns:
            True if loading the model would stay within the memory budget.
        """
        current = self.total_memory_mb()
        available = self._memory_budget_mb - current
        can = available >= required_mb
        if not can:
            logger.warning(
                "Cannot load model: requires %.0fMB but only %.0fMB available "
                "(current: %.0fMB, budget: %.0fMB)",
                required_mb,
                available,
                current,
                self._memory_budget_mb,
            )
        return can

    def loaded_models(self) -> list[str]:
        """Return names of all currently loaded models.

        Returns:
            List of model names that are currently loaded in memory.
        """
        with self._lock:
            return [
                name
                for name, (model, _) in self._models.items()
                if model.is_loaded()
            ]

    def loaded_count(self) -> int:
        """Return the number of currently loaded models."""
        return len(self.loaded_models())

    def status(self) -> dict[str, dict[str, Any]]:
        """Get status of all registered models.

        Returns:
            Dictionary mapping model name to status info dict with keys:
            - loaded: bool
            - memory_mb: float
            - model_type: str
        """
        with self._lock:
            result = {}
            for name, (model, model_type) in self._models.items():
                try:
                    loaded = model.is_loaded()
                    memory = model.get_memory_usage_mb()
                except Exception:
                    loaded = False
                    memory = 0.0
                result[name] = {
                    "loaded": loaded,
                    "memory_mb": memory,
                    "model_type": model_type,
                }
            return result

    def summary(self) -> dict[str, Any]:
        """Get a summary of the lifecycle manager state.

        Returns:
            Dictionary with total_memory_mb, loaded_count, registered_count,
            memory_budget_mb, and budget_remaining_mb.
        """
        with self._lock:
            total = self.total_memory_mb()
            return {
                "total_memory_mb": total,
                "loaded_count": self.loaded_count(),
                "registered_count": len(self._models),
                "memory_budget_mb": self._memory_budget_mb,
                "budget_remaining_mb": self._memory_budget_mb - total,
            }


# Singleton instance with thread-safe initialization
_manager: ModelLifecycleManager | None = None
_manager_lock = threading.Lock()


def get_lifecycle_manager() -> ModelLifecycleManager:
    """Get or create the singleton ModelLifecycleManager.

    Returns:
        The shared ModelLifecycleManager instance.
    """
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = ModelLifecycleManager()
    return _manager


def reset_lifecycle_manager() -> None:
    """Reset the singleton ModelLifecycleManager.

    Use for testing or full reinitialization.
    """
    global _manager
    with _manager_lock:
        _manager = None
