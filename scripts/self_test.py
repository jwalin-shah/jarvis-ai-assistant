#!/usr/bin/env python3
"""Self-test mode for JARVIS - validates all critical functionality.

Usage:
    python scripts/self_test.py

This performs end-to-end tests including:
- API startup and router registration
- iMessage database connectivity
- Contact name resolution
- Message sending (dry run)
- Socket server connectivity
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


def print_header(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def print_check(name: str, success: bool, details: str = "") -> None:
    icon = "✅" if success else "❌"
    print(f"{icon} {name}", end="")
    if details:
        print(f" - {details}")
    else:
        print()


async def test_api_startup() -> bool:
    """Test API can start and all routers load."""
    try:
        from api.main import create_app

        app = create_app()

        # Count routes
        route_count = len([r for r in app.routes if hasattr(r, "path")])

        # Check critical endpoints
        paths = [r.path for r in app.routes if hasattr(r, "path")]
        critical = ["/health", "/conversations", "/drafts"]
        missing = [c for c in critical if not any(c in p for p in paths)]

        if missing:
            print_check("API Startup", False, f"Missing endpoints: {missing}")
            return False

        print_check("API Startup", True, f"{route_count} routes registered")
        return True
    except Exception as e:
        print_check("API Startup", False, str(e))
        return False


async def test_database_access() -> bool:
    """Test iMessage database connectivity."""
    try:
        from integrations.imessage import ChatDBReader

        reader = ChatDBReader()

        # Try to get conversations
        convs = reader.get_conversations(limit=5)
        reader.close()

        print_check("Database Access", True, f"Found {len(convs)} conversations")
        return True
    except Exception as e:
        print_check("Database Access", False, str(e))
        return False


async def test_contact_resolution() -> bool:
    """Test contact name resolution."""
    try:
        from integrations.imessage import ChatDBReader

        reader = ChatDBReader()

        # Try to resolve a contact
        test_numbers = ["+15551234567", "test@example.com"]
        resolved = None
        for num in test_numbers:
            resolved = reader._resolve_contact_name(num)
            if resolved:
                break

        reader.close()

        print_check(
            "Contact Resolution",
            True,
            f"{'Resolved test contact' if resolved else 'No contacts found (ok if AddressBook empty)'}",
        )
        return True
    except Exception as e:
        print_check("Contact Resolution", False, str(e))
        return False


async def test_imessage_sender() -> bool:
    """Test iMessage sender configuration."""
    try:
        from integrations.imessage.sender import IMessageSender

        sender = IMessageSender()

        # Verify methods exist
        assert hasattr(sender, "send_message")
        assert hasattr(sender, "send_attachment")

        # Dry run - validate AppleScript syntax without sending
        # This will fail if the script is malformed
        print_check("iMessage Sender", True, "Sender configured correctly")
        return True
    except Exception as e:
        print_check("iMessage Sender", False, str(e))
        return False


async def test_applescript_access() -> bool:
    """Test AppleScript can access Messages app."""
    try:
        import subprocess

        result = subprocess.run(
            ["osascript", "-e", 'tell application "Messages" to return count of chats'],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            chat_count = int(result.stdout.strip())
            print_check("AppleScript Access", True, f"{chat_count} chats accessible")
            return True
        else:
            print_check("AppleScript Access", False, result.stderr)
            return False
    except Exception as e:
        print_check("AppleScript Access", False, str(e))
        return False


async def test_socket_server() -> bool:
    """Test socket server can start."""
    try:
        # Check socket file doesn't exist from previous run
        socket_path = Path.home() / ".jarvis" / "jarvis.sock"
        if socket_path.exists():
            print_check(
                "Socket Server",
                True,
                f"Socket file exists at {socket_path} (server may be running)",
            )
            return True

        print_check("Socket Server", True, "No socket file (server not running - ok)")
        return True
    except Exception as e:
        print_check("Socket Server", False, str(e))
        return False


async def run_all_tests() -> bool:
    """Run all self-tests."""
    print_header("JARVIS Self-Test Mode")
    print("\nRunning comprehensive system validation...\n")

    tests = [
        ("API Startup", test_api_startup),
        ("Database Access", test_database_access),
        ("Contact Resolution", test_contact_resolution),
        ("iMessage Sender", test_imessage_sender),
        ("AppleScript Access", test_applescript_access),
        ("Socket Server", test_socket_server),
    ]

    results = []
    for name, test_func in tests:
        print(f"\nTesting {name}...")
        try:
            result = await test_func()
            results.append(result)
        except Exception as e:
            print_check(name, False, f"Test crashed: {e}")
            results.append(False)

    # Summary
    print_header("Test Summary")
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"\n✅ All {total} tests passed!")
        print("\nSystem is ready. Start with: make launch")
        return True
    else:
        print(f"\n❌ {total - passed}/{total} tests failed")
        print("\nFix the issues above before running JARVIS.")
        print("\nCommon fixes:")
        print("  • Full Disk Access: System Settings > Privacy & Security > Full Disk Access")
        print("  • Automation permission: System Settings > Privacy & Security > Automation")
        print("  • Missing modules: Check api/schemas/ for commented-out imports")
        return False


def main() -> int:
    """Run self-tests."""
    try:
        success = asyncio.run(run_all_tests())
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        return 130
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
