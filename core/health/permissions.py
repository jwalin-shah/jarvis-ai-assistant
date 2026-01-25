"""Permission monitoring for macOS TCC (Transparency, Consent, and Control).

Implements the PermissionMonitor protocol from contracts/health.py.
Provides checks for macOS permissions required by JARVIS.

Workstream 7 implementation.
"""

import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from contracts.health import Permission, PermissionStatus

logger = logging.getLogger(__name__)

# Default paths for permission checks
DEFAULT_MESSAGES_DB = Path.home() / "Library" / "Messages" / "chat.db"
DEFAULT_CONTACTS_PATH = Path.home() / "Library" / "Application Support" / "AddressBook"
DEFAULT_CALENDAR_PATH = Path.home() / "Library" / "Calendars"

# User-friendly fix instructions for each permission
FIX_INSTRUCTIONS = {
    Permission.FULL_DISK_ACCESS: (
        "To grant Full Disk Access:\n"
        "1. Open System Preferences (or System Settings on macOS 13+)\n"
        "2. Go to Privacy & Security > Full Disk Access\n"
        "3. Click the lock icon and authenticate\n"
        "4. Add your terminal application (Terminal.app, iTerm, etc.)\n"
        "5. Restart the terminal and try again"
    ),
    Permission.CONTACTS: (
        "To grant Contacts access:\n"
        "1. Open System Preferences (or System Settings on macOS 13+)\n"
        "2. Go to Privacy & Security > Contacts\n"
        "3. Click the lock icon and authenticate\n"
        "4. Add your terminal application\n"
        "5. Restart the terminal and try again"
    ),
    Permission.CALENDAR: (
        "To grant Calendar access:\n"
        "1. Open System Preferences (or System Settings on macOS 13+)\n"
        "2. Go to Privacy & Security > Calendars\n"
        "3. Click the lock icon and authenticate\n"
        "4. Add your terminal application\n"
        "5. Restart the terminal and try again"
    ),
    Permission.AUTOMATION: (
        "To grant Automation access:\n"
        "1. Open System Preferences (or System Settings on macOS 13+)\n"
        "2. Go to Privacy & Security > Automation\n"
        "3. Click the lock icon and authenticate\n"
        "4. Enable access for your terminal application to control other apps\n"
        "5. You may be prompted when first using automation features"
    ),
}


class TCCPermissionMonitor:
    """Thread-safe macOS TCC permission monitor.

    Checks and monitors macOS permissions required by JARVIS.
    Implements the PermissionMonitor protocol.
    """

    def __init__(
        self,
        messages_db_path: Path | None = None,
        contacts_path: Path | None = None,
        calendar_path: Path | None = None,
    ) -> None:
        """Initialize the permission monitor.

        Args:
            messages_db_path: Path to iMessage database (for Full Disk Access check)
            contacts_path: Path to Contacts directory
            calendar_path: Path to Calendar directory
        """
        self._messages_db_path = messages_db_path or DEFAULT_MESSAGES_DB
        self._contacts_path = contacts_path or DEFAULT_CONTACTS_PATH
        self._calendar_path = calendar_path or DEFAULT_CALENDAR_PATH
        self._lock = threading.Lock()
        logger.info("TCCPermissionMonitor initialized")

    def _get_timestamp(self) -> str:
        """Get current ISO 8601 timestamp.

        Returns:
            ISO 8601 formatted timestamp string
        """
        return datetime.now(UTC).isoformat()

    def _check_full_disk_access(self) -> bool:
        """Check if Full Disk Access is granted.

        Tests by checking if the iMessage chat.db is readable.

        Returns:
            True if Full Disk Access is granted, False otherwise
        """
        try:
            # Try to open the Messages database read-only
            if self._messages_db_path.exists():
                # Try to actually read from it - just checking existence isn't enough
                with open(self._messages_db_path, "rb") as f:
                    f.read(16)  # Read first 16 bytes (SQLite header)
                return True
            # If the file doesn't exist, we can't determine permission status
            # Assume permission is not granted to be safe
            return False
        except PermissionError:
            return False
        except OSError as e:
            # EACCES (13) or similar permission errors
            logger.debug("Full Disk Access check failed: %s", e)
            return False

    def _check_contacts_access(self) -> bool:
        """Check if Contacts access is granted.

        Tests by checking if the AddressBook directory is readable.

        Returns:
            True if Contacts access is granted, False otherwise
        """
        try:
            if self._contacts_path.exists():
                # Try to list the directory contents
                list(self._contacts_path.iterdir())
                return True
            return False
        except PermissionError:
            return False
        except OSError as e:
            logger.debug("Contacts access check failed: %s", e)
            return False

    def _check_calendar_access(self) -> bool:
        """Check if Calendar access is granted.

        Tests by checking if the Calendars directory is readable.

        Returns:
            True if Calendar access is granted, False otherwise
        """
        try:
            if self._calendar_path.exists():
                # Try to list the directory contents
                list(self._calendar_path.iterdir())
                return True
            return False
        except PermissionError:
            return False
        except OSError as e:
            logger.debug("Calendar access check failed: %s", e)
            return False

    def _check_automation_access(self) -> bool:
        """Check if Automation access is granted.

        Automation permissions are typically requested on-demand when
        automating other applications. We can't proactively check this
        without actually trying to automate something.

        Returns:
            True (assumes granted until proven otherwise at runtime)
        """
        # Automation permissions are checked at runtime when attempting
        # to automate specific applications. We return True as a default
        # since there's no way to check this without actually trying.
        return True

    def check_permission(self, permission: Permission) -> PermissionStatus:
        """Check if a specific permission is granted.

        Args:
            permission: The permission to check

        Returns:
            PermissionStatus with current grant status and fix instructions
        """
        with self._lock:
            granted = False
            check_funcs = {
                Permission.FULL_DISK_ACCESS: self._check_full_disk_access,
                Permission.CONTACTS: self._check_contacts_access,
                Permission.CALENDAR: self._check_calendar_access,
                Permission.AUTOMATION: self._check_automation_access,
            }

            check_func = check_funcs.get(permission)
            if check_func:
                granted = check_func()

            fix_instructions = "" if granted else FIX_INSTRUCTIONS.get(permission, "")

            status = PermissionStatus(
                permission=permission,
                granted=granted,
                last_checked=self._get_timestamp(),
                fix_instructions=fix_instructions,
            )

            logger.debug(
                "Permission check: %s = %s",
                permission.value,
                "granted" if granted else "denied",
            )

            return status

    def check_all(self) -> list[PermissionStatus]:
        """Check all required permissions.

        Returns:
            List of PermissionStatus for each permission type
        """
        statuses = []
        for permission in Permission:
            statuses.append(self.check_permission(permission))
        return statuses

    def wait_for_permission(
        self,
        permission: Permission,
        timeout_seconds: int,
    ) -> bool:
        """Block until permission granted or timeout.

        Polls the permission status at regular intervals until
        the permission is granted or the timeout expires.

        Args:
            permission: The permission to wait for
            timeout_seconds: Maximum time to wait in seconds

        Returns:
            True if permission was granted within timeout, False otherwise
        """
        poll_interval = 1.0  # Check every second
        elapsed = 0.0

        logger.info(
            "Waiting up to %d seconds for %s permission",
            timeout_seconds,
            permission.value,
        )

        while elapsed < timeout_seconds:
            status = self.check_permission(permission)
            if status.granted:
                logger.info(
                    "Permission %s granted after %.1f seconds",
                    permission.value,
                    elapsed,
                )
                return True

            time.sleep(poll_interval)
            elapsed += poll_interval

        logger.warning(
            "Timeout waiting for %s permission after %d seconds",
            permission.value,
            timeout_seconds,
        )
        return False

    def get_summary(self) -> dict[str, bool]:
        """Get a summary of all permission states.

        Returns:
            Dictionary mapping permission names to grant status
        """
        statuses = self.check_all()
        return {status.permission.value: status.granted for status in statuses}


# Module-level singleton
_monitor: TCCPermissionMonitor | None = None
_monitor_lock = threading.Lock()


def get_permission_monitor() -> TCCPermissionMonitor:
    """Get the singleton permission monitor instance.

    Returns:
        The shared TCCPermissionMonitor instance
    """
    global _monitor
    if _monitor is None:
        with _monitor_lock:
            if _monitor is None:
                _monitor = TCCPermissionMonitor()
    return _monitor


def reset_permission_monitor() -> None:
    """Reset the singleton permission monitor.

    Useful for testing or reinitializing the system.
    """
    global _monitor
    with _monitor_lock:
        _monitor = None
        logger.info("Permission monitor singleton reset")
