"""Centralized dependency checking for test skip decisions.

This module provides a centralized registry for test dependencies,
enabling intelligent skip decisions based on available resources.

Usage:
    from tests.dependencies import requires, skip_if_missing

    @requires("spacy", "lightgbm_model")
    def test_feature_extraction():
        ...
"""

from __future__ import annotations

import enum
import platform
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pytest


class DependencyTier(enum.Enum):
    """Dependency importance tiers."""

    CRITICAL = "critical"  # Required for core functionality (pytest, numpy)
    STANDARD = "standard"  # Required for standard tests (spacy, lightgbm)
    EXTENDED = "extended"  # Required for extended ML tests (transformers, torch)
    HARDWARE = "hardware"  # Hardware-dependent (MLX, GPU, 16GB+ RAM)


@dataclass(frozen=True)
class Dependency:
    """A test dependency with metadata."""

    name: str
    tier: DependencyTier
    check: Callable[[], tuple[bool, str]]  # (available, reason)

    def is_available(self) -> bool:
        """Check if dependency is available."""
        available, _ = self.check()
        return available

    def skip_reason(self) -> str:
        """Get human-readable skip reason."""
        _, reason = self.check()
        return reason


# =============================================================================
# Dependency Check Functions
# =============================================================================


def _check_spacy_model() -> tuple[bool, str]:
    """Check if spaCy English model is available."""
    try:
        import spacy

        spacy.load("en_core_web_sm")
        return True, "en_core_web_sm available"
    except ImportError:
        return False, "spacy not installed (pip install spacy)"
    except OSError:
        return (
            False,
            "en_core_web_sm model not downloaded (python -m spacy download en_core_web_sm)",
        )


def _check_lightgbm_model() -> tuple[bool, str]:
    """Check if LightGBM category model exists."""
    model_path = (
        Path(__file__).parent.parent / "models" / "category_multilabel_lightgbm_hardclass.joblib"
    )
    if model_path.exists():
        return True, f"Model found at {model_path}"
    return (
        False,
        f"LightGBM model not found at {model_path} (run training pipeline)",
    )


def _check_mlx() -> tuple[bool, str]:
    """Check if MLX is available (macOS Apple Silicon only)."""
    try:
        import mlx.core  # noqa: F401

        return True, "MLX available"
    except (ImportError, OSError):
        return False, "MLX not available (macOS Apple Silicon required)"


def _check_metal() -> tuple[bool, str]:
    """Check if Metal GPU is available."""
    try:
        import mlx.core as mx

        if mx.metal.is_available():
            return True, "Metal GPU available"
        return False, "Metal GPU not available"
    except Exception as e:
        return False, f"Cannot check Metal availability: {e}"


def _check_apple_silicon() -> tuple[bool, str]:
    """Check if running on Apple Silicon."""
    if platform.system() != "Darwin":
        return False, f"Not macOS (found {platform.system()})"
    if platform.machine() != "arm64":
        return False, f"Not Apple Silicon (found {platform.machine()})"
    return True, "Apple Silicon (arm64) detected"


def _check_memory_gb(min_gb: float) -> Callable[[], tuple[bool, str]]:
    """Create memory check for specified GB threshold."""

    def check() -> tuple[bool, str]:
        try:
            import psutil

            total_gb = psutil.virtual_memory().total / (1024**3)
            if total_gb >= min_gb:
                return True, f"{total_gb:.1f}GB RAM available"
            return False, f"Requires {min_gb}GB RAM, found {total_gb:.1f}GB"
        except ImportError:
            return False, "psutil not installed"

    return check


def _check_model_files(model_names: list[str]) -> Callable[[], tuple[bool, str]]:
    """Check if model files exist in ~/.jarvis/."""

    def check() -> tuple[bool, str]:
        jarvis_dir = Path.home() / ".jarvis"
        missing = []
        for name in model_names:
            model_path = jarvis_dir / name
            if not model_path.exists():
                missing.append(name)
        if not missing:
            return True, f"All models found: {', '.join(model_names)}"
        return False, f"Missing models in ~/.jarvis/: {', '.join(missing)}"

    return check


def _check_package(package_name: str) -> Callable[[], tuple[bool, str]]:
    """Check if a Python package is installed."""

    def check() -> tuple[bool, str]:
        try:
            __import__(package_name)
            return True, f"{package_name} installed"
        except ImportError:
            return False, f"{package_name} not installed (pip install {package_name})"

    return check


# =============================================================================
# Dependency Registry
# =============================================================================

DEPENDENCIES: dict[str, Dependency] = {
    # Standard ML dependencies
    "spacy": Dependency("spacy", DependencyTier.STANDARD, _check_spacy_model),
    "lightgbm_model": Dependency("lightgbm_model", DependencyTier.STANDARD, _check_lightgbm_model),
    "sentence_transformers": Dependency(
        "sentence_transformers",
        DependencyTier.EXTENDED,
        _check_package("sentence_transformers"),
    ),
    "torch": Dependency("torch", DependencyTier.EXTENDED, _check_package("torch")),
    "transformers": Dependency(
        "transformers", DependencyTier.EXTENDED, _check_package("transformers")
    ),
    # Hardware-dependent
    "mlx": Dependency("mlx", DependencyTier.HARDWARE, _check_mlx),
    "metal": Dependency("metal", DependencyTier.HARDWARE, _check_metal),
    "apple_silicon": Dependency("apple_silicon", DependencyTier.HARDWARE, _check_apple_silicon),
    "memory_16gb": Dependency("memory_16gb", DependencyTier.HARDWARE, _check_memory_gb(16)),
    "memory_8gb": Dependency("memory_8gb", DependencyTier.HARDWARE, _check_memory_gb(8)),
    # Model artifacts
    "svm_trigger_classifier": Dependency(
        "svm_trigger_classifier",
        DependencyTier.STANDARD,
        _check_model_files(["trigger_classifier_model"]),
    ),
    "svm_response_classifier": Dependency(
        "svm_response_classifier",
        DependencyTier.STANDARD,
        _check_model_files(["response_classifier_model"]),
    ),
    "lfm_model": Dependency(
        "lfm_model",
        DependencyTier.HARDWARE,
        _check_model_files(["lfm-1.2b-soc-fused"]),
    ),
}


# =============================================================================
# Public API
# =============================================================================


def requires(*dep_names: str) -> pytest.MarkDecorator:
    """Mark a test as requiring specific dependencies.

    Usage:
        @requires("spacy", "lightgbm_model")
        def test_feature_extraction():
            ...

    Args:
        *dep_names: Names of dependencies from DEPENDENCIES registry

    Returns:
        pytest.mark.skipif decorator if deps missing, else no-op marker

    Raises:
        ValueError: If unknown dependency name provided
    """
    missing = []
    for name in dep_names:
        if name not in DEPENDENCIES:
            raise ValueError(f"Unknown dependency: {name}. Available: {list(DEPENDENCIES.keys())}")
        dep = DEPENDENCIES[name]
        if not dep.is_available():
            missing.append(f"{name}: {dep.skip_reason()}")

    if missing:
        reason = "; ".join(missing)
        return pytest.mark.skip(reason=reason)

    return pytest.mark.skipif(False, reason="All dependencies satisfied")


def skip_if_missing(*dep_names: str) -> bool:
    """Check if any dependency is missing (for conditional skipping).

    Usage:
        @pytest.mark.skipif(skip_if_missing("spacy"), reason="spacy not available")
        def test_spacy_features():
            ...

    Args:
        *dep_names: Names of dependencies to check

    Returns:
        True if any dependency is missing, False if all satisfied
    """
    for name in dep_names:
        if name not in DEPENDENCIES:
            return True  # Unknown dependency = skip
        if not DEPENDENCIES[name].is_available():
            return True
    return False


def get_dependency_report() -> dict[str, dict]:
    """Generate report of all dependencies and their status.

    Returns:
        Dict mapping dependency names to their status info
    """
    return {
        name: {
            "tier": dep.tier.value,
            "available": dep.is_available(),
            "reason": dep.skip_reason(),
        }
        for name, dep in DEPENDENCIES.items()
    }


def print_dependency_report() -> None:
    """Print formatted dependency report to stdout."""
    report = get_dependency_report()
    print("\n" + "=" * 60)
    print("TEST DEPENDENCY REPORT")
    print("=" * 60)

    for tier in DependencyTier:
        tier_deps = {k: v for k, v in report.items() if v["tier"] == tier.value}
        if tier_deps:
            print(f"\n[{tier.value.upper()}]")
            for name, info in sorted(tier_deps.items()):
                status = "✓" if info["available"] else "✗"
                print(f"  {status} {name}: {info['reason']}")

    total = len(report)
    available = sum(1 for r in report.values() if r["available"])
    print("\n" + "-" * 60)
    print(f"Summary: {available}/{total} dependencies satisfied")
    print("=" * 60 + "\n")


@lru_cache(maxsize=1)
def get_available_deps_by_tier(tier: DependencyTier) -> set[str]:
    """Get set of available dependencies for a specific tier.

    Args:
        tier: The dependency tier to query

    Returns:
        Set of available dependency names in that tier
    """
    return {name for name, dep in DEPENDENCIES.items() if dep.tier == tier and dep.is_available()}


def has_tier(tier: DependencyTier) -> bool:
    """Check if all dependencies for a tier are available.

    Args:
        tier: The tier to check

    Returns:
        True if all deps in tier are available
    """
    tier_deps = [d for d in DEPENDENCIES.values() if d.tier == tier]
    return all(d.is_available() for d in tier_deps)
