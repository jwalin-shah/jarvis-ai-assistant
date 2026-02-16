"""Persistent store for user-defined response templates."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from models.templates import ResponseTemplate

logger = logging.getLogger(__name__)


@dataclass
class CustomTemplate:
    """User-defined template for custom response patterns."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    template_text: str = ""
    trigger_phrases: list[str] = field(default_factory=list)
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    min_group_size: int | None = None
    max_group_size: int | None = None
    enabled: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "template_text": self.template_text,
            "trigger_phrases": self.trigger_phrases,
            "category": self.category,
            "tags": self.tags,
            "min_group_size": self.min_group_size,
            "max_group_size": self.max_group_size,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "usage_count": self.usage_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CustomTemplate:
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            template_text=data.get("template_text", ""),
            trigger_phrases=data.get("trigger_phrases", []),
            category=data.get("category", "general"),
            tags=data.get("tags", []),
            min_group_size=data.get("min_group_size"),
            max_group_size=data.get("max_group_size"),
            enabled=data.get("enabled", True),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            usage_count=data.get("usage_count", 0),
        )

    def to_response_template(self) -> ResponseTemplate:
        from models.templates import ResponseTemplate

        return ResponseTemplate(
            name=f"custom_{self.id}",
            patterns=self.trigger_phrases,
            response=self.template_text,
        )


CUSTOM_TEMPLATES_PATH = Path.home() / ".jarvis" / "custom_templates.json"
_custom_template_store: CustomTemplateStore | None = None
_custom_template_store_lock = threading.Lock()


class CustomTemplateStore:
    """Thread-safe storage for custom templates."""

    def __init__(self, storage_path: Path | None = None) -> None:
        self._storage_path = storage_path or CUSTOM_TEMPLATES_PATH
        self._templates: dict[str, CustomTemplate] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        with self._lock:
            if self._storage_path.exists():
                try:
                    with self._storage_path.open() as f:
                        data = json.load(f)
                        templates = data.get("templates", [])
                        self._templates = {t["id"]: CustomTemplate.from_dict(t) for t in templates}
                    logger.info("Loaded %d custom templates", len(self._templates))
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("Failed to load custom templates: %s", e)
                    self._templates = {}
            else:
                self._templates = {}

    def _save(self) -> bool:
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            with self._storage_path.open("w") as f:
                data = {
                    "version": 1,
                    "templates": [t.to_dict() for t in self._templates.values()],
                }
                json.dump(data, f, indent=2)
            return True
        except OSError as e:
            logger.error("Failed to save custom templates: %s", e)
            return False

    def get(self, template_id: str) -> CustomTemplate | None:
        with self._lock:
            return self._templates.get(template_id)

    def list_all(self) -> list[CustomTemplate]:
        with self._lock:
            return list(self._templates.values())

    def list_enabled(self) -> list[CustomTemplate]:
        with self._lock:
            return [t for t in self._templates.values() if t.enabled]

    def list_by_category(self, category: str) -> list[CustomTemplate]:
        with self._lock:
            return [t for t in self._templates.values() if t.category == category]

    def list_by_tag(self, tag: str) -> list[CustomTemplate]:
        with self._lock:
            return [t for t in self._templates.values() if tag in t.tags]

    def get_categories(self) -> list[str]:
        with self._lock:
            return sorted(set(t.category for t in self._templates.values()))

    def get_tags(self) -> list[str]:
        with self._lock:
            all_tags: set[str] = set()
            for t in self._templates.values():
                all_tags.update(t.tags)
            return sorted(all_tags)

    def create(self, template: CustomTemplate) -> CustomTemplate:
        with self._lock:
            if not template.id:
                template.id = str(uuid.uuid4())
            template.created_at = datetime.now().isoformat()
            template.updated_at = template.created_at
            self._templates[template.id] = template
            self._save()
            logger.info("Created custom template: %s", template.name)
            return template

    def update(self, template_id: str, updates: dict[str, Any]) -> CustomTemplate | None:
        with self._lock:
            if template_id not in self._templates:
                return None

            template = self._templates[template_id]
            for key, value in updates.items():
                if key not in ("id", "created_at") and hasattr(template, key):
                    setattr(template, key, value)

            template.updated_at = datetime.now().isoformat()
            self._save()
            logger.info("Updated custom template: %s", template.name)
            return template

    def delete(self, template_id: str) -> bool:
        with self._lock:
            if template_id not in self._templates:
                return False
            template = self._templates.pop(template_id)
            self._save()
            logger.info("Deleted custom template: %s", template.name)
            return True

    def increment_usage(self, template_id: str) -> None:
        with self._lock:
            if template_id in self._templates:
                self._templates[template_id].usage_count += 1
                self._save()

    def get_usage_stats(self) -> dict[str, Any]:
        with self._lock:
            total_usage = sum(t.usage_count for t in self._templates.values())
            by_category: dict[str, int] = {}
            for t in self._templates.values():
                by_category[t.category] = by_category.get(t.category, 0) + t.usage_count

            top_templates = sorted(
                self._templates.values(),
                key=lambda x: x.usage_count,
                reverse=True,
            )[:10]

            return {
                "total_templates": len(self._templates),
                "enabled_templates": len([t for t in self._templates.values() if t.enabled]),
                "total_usage": total_usage,
                "usage_by_category": by_category,
                "top_templates": [
                    {"id": t.id, "name": t.name, "usage_count": t.usage_count}
                    for t in top_templates
                ],
            }

    def export_templates(self, template_ids: list[str] | None = None) -> dict[str, Any]:
        with self._lock:
            if template_ids:
                templates = [
                    self._templates[tid].to_dict() for tid in template_ids if tid in self._templates
                ]
            else:
                templates = [t.to_dict() for t in self._templates.values()]

            return {
                "version": 1,
                "export_date": datetime.now().isoformat(),
                "template_count": len(templates),
                "templates": templates,
            }

    def import_templates(self, data: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
        imported = 0
        skipped = 0
        errors = 0
        templates = data.get("templates", [])

        with self._lock:
            for template_data in templates:
                try:
                    template = CustomTemplate.from_dict(template_data)
                    if template.id in self._templates and not overwrite:
                        skipped += 1
                        continue

                    template.created_at = datetime.now().isoformat()
                    template.updated_at = template.created_at
                    template.usage_count = 0
                    self._templates[template.id] = template
                    imported += 1
                except Exception as e:
                    logger.warning("Failed to import template: %s", e)
                    errors += 1

            self._save()

        logger.info("Imported %d templates (%d skipped, %d errors)", imported, skipped, errors)
        return {
            "imported": imported,
            "skipped": skipped,
            "errors": errors,
            "total_templates": len(self._templates),
        }

    def reload(self) -> None:
        self._load()


def get_custom_template_store() -> CustomTemplateStore:
    global _custom_template_store
    if _custom_template_store is None:
        with _custom_template_store_lock:
            if _custom_template_store is None:
                _custom_template_store = CustomTemplateStore()
    return _custom_template_store


def reset_custom_template_store() -> None:
    global _custom_template_store
    with _custom_template_store_lock:
        _custom_template_store = None
