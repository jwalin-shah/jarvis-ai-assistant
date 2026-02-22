from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SamplingParams:
    temperature: float
    min_p: float
    rep_penalty: float
    n_samples: int


@dataclass
class SoftBoNConfig:
    temperature: float
    min_score_threshold: float


@dataclass
class RuntimeConfig:
    user_name: str
    default_contact_name: str
    default_relationship: str
    classifier_confidence_threshold: float
    classifier_vote_samples: int
    classifier_vote_temperature: float
    candidate_retry_limit: int
    min_valid_candidates: int
    timestamp_format: str


@dataclass
class AppConfig:
    raw: dict[str, Any]
    config_path: Path

    @property
    def models(self) -> dict[str, Any]:
        return self.raw["models"]

    @property
    def categories(self) -> list[str]:
        return self.raw["categories"]

    @property
    def strategy_templates(self) -> dict[str, str]:
        return self.raw["strategy_templates"]

    @property
    def sampling(self) -> dict[str, SamplingParams]:
        items = self.raw["sampling"]
        return {k: SamplingParams(**v) for k, v in items.items()}

    @property
    def soft_bon(self) -> SoftBoNConfig:
        return SoftBoNConfig(**self.raw["soft_bon"])

    @property
    def training(self) -> dict[str, Any]:
        return self.raw["training"]

    @property
    def runtime(self) -> RuntimeConfig:
        return RuntimeConfig(**self.raw["runtime"])

    @property
    def model_registry_path(self) -> Path | None:
        rel = self.raw.get("model_registry")
        if not rel:
            return None
        path = Path(rel)
        if not path.is_absolute():
            path = self.config_path.parent / path
        return path

    def model_candidates(self, role: str) -> list[str]:
        # Preferred: centralized registry file.
        reg_path = self.model_registry_path
        if reg_path and reg_path.exists():
            try:
                import yaml

                with reg_path.open("r", encoding="utf-8") as f:
                    registry = yaml.safe_load(f) or {}
                roles = registry.get("roles", {})
                items = roles.get(role, [])
                if isinstance(items, list):
                    items = [str(x).strip() for x in items if str(x).strip()]
                    if items:
                        return items
            except Exception:
                pass

        # Backward-compatible fallback: single model ID in config.yaml.
        value = self.models.get(role)
        return [value] if isinstance(value, str) and value.strip() else []


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required. Install dependencies with: pip install -r requirements.txt"
        ) from exc
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AppConfig(raw=data, config_path=cfg_path.resolve())
