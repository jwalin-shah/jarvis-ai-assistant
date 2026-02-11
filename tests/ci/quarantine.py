"""Test quarantine management for CI pipeline.

Quarantines flaky tests to prevent them from blocking builds
while maintaining visibility and tracking for fixes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class QuarantineEntry:
    """Entry in quarantine list."""

    test_id: str
    quarantined_at: datetime
    reason: str
    ticket_url: str | None = None
    max_retries: int = 3
    auto_unquarantine_date: datetime | None = None
    quarantined_by: str | None = None
    failure_pattern: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_id": self.test_id,
            "quarantined_at": self.quarantined_at.isoformat(),
            "reason": self.reason,
            "ticket_url": self.ticket_url,
            "max_retries": self.max_retries,
            "auto_unquarantine_date": (
                self.auto_unquarantine_date.isoformat() if self.auto_unquarantine_date else None
            ),
            "quarantined_by": self.quarantined_by,
            "failure_pattern": self.failure_pattern,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuarantineEntry:
        """Create from dictionary."""
        return cls(
            test_id=data["test_id"],
            quarantined_at=datetime.fromisoformat(data["quarantined_at"]),
            reason=data["reason"],
            ticket_url=data.get("ticket_url"),
            max_retries=data.get("max_retries", 3),
            auto_unquarantine_date=(
                datetime.fromisoformat(data["auto_unquarantine_date"])
                if data.get("auto_unquarantine_date")
                else None
            ),
            quarantined_by=data.get("quarantined_by"),
            failure_pattern=data.get("failure_pattern"),
            notes=data.get("notes"),
        )


class QuarantineManager:
    """Manage test quarantine list."""

    DEFAULT_AUTO_UNQUARANTINE_DAYS = 30

    def __init__(self, quarantine_file: Path = Path(".quarantine.json")):
        self.quarantine_file = quarantine_file
        self._entries: dict[str, QuarantineEntry] = {}
        self._load()

    def _load(self) -> None:
        """Load quarantine list from file."""
        if not self.quarantine_file.exists():
            return

        try:
            data = json.loads(self.quarantine_file.read_text())
            for entry_data in data.get("quarantined", []):
                try:
                    entry = QuarantineEntry.from_dict(entry_data)
                    self._entries[entry.test_id] = entry
                except (KeyError, ValueError) as e:
                    print(f"Warning: Skipping invalid quarantine entry: {e}")
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse quarantine file: {e}")

    def save(self) -> None:
        """Save quarantine list to file."""
        self.quarantine_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "updated_at": datetime.now().isoformat(),
            "version": 1,
            "quarantined": [entry.to_dict() for entry in self._entries.values()],
        }
        self.quarantine_file.write_text(json.dumps(data, indent=2))

    def add(
        self,
        test_id: str,
        reason: str,
        ticket_url: str | None = None,
        max_retries: int = 3,
        auto_unquarantine_days: int | None = None,
        quarantined_by: str | None = None,
        failure_pattern: str | None = None,
        notes: str | None = None,
    ) -> QuarantineEntry:
        """Add test to quarantine."""
        if auto_unquarantine_days is None:
            auto_unquarantine_days = self.DEFAULT_AUTO_UNQUARANTINE_DAYS

        auto_unquarantine = (
            datetime.now() + timedelta(days=auto_unquarantine_days)
            if auto_unquarantine_days > 0
            else None
        )

        entry = QuarantineEntry(
            test_id=test_id,
            quarantined_at=datetime.now(),
            reason=reason,
            ticket_url=ticket_url,
            max_retries=max_retries,
            auto_unquarantine_date=auto_unquarantine,
            quarantined_by=quarantined_by,
            failure_pattern=failure_pattern,
            notes=notes,
        )

        self._entries[test_id] = entry
        self.save()
        return entry

    def remove(self, test_id: str) -> bool:
        """Remove test from quarantine. Returns True if was quarantined."""
        if test_id in self._entries:
            del self._entries[test_id]
            self.save()
            return True
        return False

    def is_quarantined(self, test_id: str) -> bool:
        """Check if test is quarantined."""
        entry = self._entries.get(test_id)
        if not entry:
            return False

        # Check auto-unquarantine
        if entry.auto_unquarantine_date:
            if datetime.now() > entry.auto_unquarantine_date:
                self.remove(test_id)
                return False

        return True

    def get_entry(self, test_id: str) -> QuarantineEntry | None:
        """Get quarantine entry for a test."""
        entry = self._entries.get(test_id)
        if entry and self.is_quarantined(test_id):
            return entry
        return None

    def get_retry_count(self, test_id: str) -> int:
        """Get max retry count for quarantined test."""
        entry = self._entries.get(test_id)
        return entry.max_retries if entry else 0

    def get_quarantined_tests(self) -> list[QuarantineEntry]:
        """Get all quarantined tests (excluding auto-unquarantined)."""
        # Refresh to filter out expired entries
        return [entry for entry in self._entries.values() if self.is_quarantined(entry.test_id)]

    def get_quarantined_test_ids(self) -> list[str]:
        """Get list of quarantined test IDs."""
        return [entry.test_id for entry in self.get_quarantined_tests()]

    def update_notes(self, test_id: str, notes: str) -> bool:
        """Update notes for a quarantined test."""
        entry = self._entries.get(test_id)
        if entry:
            entry.notes = notes
            self.save()
            return True
        return False

    def extend_quarantine(
        self,
        test_id: str,
        additional_days: int,
    ) -> bool:
        """Extend quarantine period for a test."""
        entry = self._entries.get(test_id)
        if not entry:
            return False

        current_expiry = entry.auto_unquarantine_date or datetime.now()
        entry.auto_unquarantine_date = current_expiry + timedelta(days=additional_days)
        self.save()
        return True

    def get_summary(self) -> dict[str, Any]:
        """Get summary of quarantine status."""
        entries = self.get_quarantined_tests()

        expiring_soon = [
            e
            for e in entries
            if e.auto_unquarantine_date
            and e.auto_unquarantine_date - datetime.now() < timedelta(days=7)
        ]

        return {
            "total_quarantined": len(entries),
            "expiring_within_7_days": len(expiring_soon),
            "with_tickets": sum(1 for e in entries if e.ticket_url),
            "avg_retries": (sum(e.max_retries for e in entries) / len(entries) if entries else 0),
        }

    def merge(self, other: QuarantineManager) -> None:
        """Merge another quarantine list into this one."""
        for entry in other.get_quarantined_tests():
            if entry.test_id not in self._entries:
                self._entries[entry.test_id] = entry
        self.save()
