#!/usr/bin/env python3
"""Database maintenance CLI for JARVIS.

Provides commands for backup, restore, health checks, migrations,
and routine maintenance tasks.

Usage:
    # Create a hot backup
    uv run python scripts/db_maintenance.py backup

    # Restore from latest backup
    uv run python scripts/db_maintenance.py restore

    # Run health check
    uv run python scripts/db_maintenance.py health

    # Daily maintenance (VACUUM, ANALYZE)
    uv run python scripts/db_maintenance.py maintain --daily

    # Full maintenance with backup
    uv run python scripts/db_maintenance.py maintain --full

    # Test all migrations
    uv run python scripts/db_maintenance.py test-migrations

    # Run backup/restore drill
    uv run python scripts/db_maintenance.py drill
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis.db.backup import BackupManager, get_latest_backup
from jarvis.db.migration import MigrationTester, print_migration_report
from jarvis.db.reliability import RecoveryManager, ReliabilityMonitor, run_health_report


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_backup(args: argparse.Namespace) -> int:
    """Create a database backup."""
    manager = BackupManager()

    if args.type == "hot":
        result = manager.create_hot_backup()
    elif args.type == "export":
        result = manager.create_sql_export(
            tables=args.tables,
            compress=not args.no_compress,
        )
    elif args.type == "migration":
        result = manager.create_migration_backup()
    else:
        print(f"Unknown backup type: {args.type}")
        return 1

    if result.success:
        print("✓ Backup created successfully")
        print(f"  Path: {result.backup_path}")
        print(f"  Size: {result.size_bytes:,} bytes")
        print(f"  Checksum: {result.checksum[:16]}...")
        print(f"  Duration: {result.duration_seconds:.2f}s")

        if result.tables_backed_up:
            print("\n  Tables:")
            for table, count in sorted(result.tables_backed_up.items()):
                print(f"    {table}: {count:,} rows")
        return 0
    else:
        print(f"✗ Backup failed: {result.error_message}")
        return 1


def cmd_restore(args: argparse.Namespace) -> int:
    """Restore database from backup."""
    manager = BackupManager()

    if args.backup:
        backup_path = Path(args.backup)
    else:
        backup_path = get_latest_backup()
        if backup_path is None:
            print("✗ No backups found")
            return 1
        print(f"Using latest backup: {backup_path}")

    if not args.force:
        print("\nThis will replace the current database at:")
        print(f"  {manager.db_path}")
        print("\nWith backup from:")
        print(f"  {backup_path}")
        print("\nType 'yes' to continue:", end=" ")
        response = input().strip().lower()
        if response != "yes":
            print("Restore cancelled")
            return 0

    result = manager.restore_from_backup(
        backup_path=backup_path,
        verify_integrity=not args.no_verify,
        create_safety_copy=not args.no_safety,
    )

    if result.success:
        print("✓ Restore completed successfully")
        print(f"  Duration: {result.duration_seconds:.2f}s")
        print(f"  Integrity check: {'PASS' if result.integrity_check_passed else 'FAIL'}")
        if result.schema_version:
            print(f"  Schema version: {result.schema_version}")
        return 0
    else:
        print(f"✗ Restore failed: {result.error_message}")
        return 1


def cmd_health(args: argparse.Namespace) -> int:
    """Run database health check."""
    if args.json:
        monitor = ReliabilityMonitor()
        report = monitor.check_health()
        print(report.to_json())
    else:
        run_health_report()
    return 0


def cmd_maintain(args: argparse.Namespace) -> int:
    """Run database maintenance tasks."""
    import sqlite3

    from jarvis.db.models import JARVIS_DB_PATH

    db_path = JARVIS_DB_PATH

    if not db_path.exists():
        print(f"✗ Database not found: {db_path}")
        return 1

    print(f"Running maintenance on {db_path}...")

    tasks_completed = []
    tasks_failed = []

    # Connect to database
    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.Error as e:
        print(f"✗ Failed to open database: {e}")
        return 1

    try:
        # 1. Integrity check
        print("\n1. Running integrity check...")
        cursor = conn.execute("PRAGMA integrity_check")
        result = cursor.fetchone()[0]
        if result == "ok":
            print("   ✓ Integrity check passed")
            tasks_completed.append("Integrity check")
        else:
            print(f"   ✗ Integrity check failed: {result}")
            tasks_failed.append("Integrity check")
            if not args.force:
                return 1

        # 2. Foreign key check
        print("\n2. Checking foreign key constraints...")
        cursor = conn.execute("PRAGMA foreign_key_check")
        violations = cursor.fetchall()
        if not violations:
            print("   ✓ No foreign key violations")
            tasks_completed.append("Foreign key check")
        else:
            print(f"   ✗ {len(violations)} foreign key violations")
            tasks_failed.append("Foreign key check")

        # 3. ANALYZE for query optimization
        if args.daily or args.full:
            print("\n3. Running ANALYZE...")
            try:
                conn.execute("ANALYZE")
                print("   ✓ ANALYZE completed")
                tasks_completed.append("ANALYZE")
            except sqlite3.Error as e:
                print(f"   ✗ ANALYZE failed: {e}")
                tasks_failed.append("ANALYZE")

        # 4. REINDEX
        if args.full:
            print("\n4. Rebuilding indices (REINDEX)...")
            try:
                conn.execute("REINDEX")
                print("   ✓ REINDEX completed")
                tasks_completed.append("REINDEX")
            except sqlite3.Error as e:
                print(f"   ✗ REINDEX failed: {e}")
                tasks_failed.append("REINDEX")

        # 5. VACUUM (only with --full due to time/space requirements)
        if args.full:
            print("\n5. Vacuuming database (this may take a while)...")
            try:
                size_before = db_path.stat().st_size
                conn.execute("VACUUM")
                size_after = db_path.stat().st_size
                reduction = (size_before - size_after) / size_before * 100
                print("   ✓ VACUUM completed")
                print(f"   Size reduction: {size_before - size_after:,} bytes ({reduction:.1f}%)")
                tasks_completed.append("VACUUM")
            except sqlite3.Error as e:
                print(f"   ✗ VACUUM failed: {e}")
                tasks_failed.append("VACUUM")

        # 6. WAL checkpoint
        print("\n6. Checkpointing WAL...")
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            print("   ✓ WAL checkpoint completed")
            tasks_completed.append("WAL checkpoint")
        except sqlite3.Error as e:
            print(f"   ✗ WAL checkpoint failed: {e}")
            tasks_failed.append("WAL checkpoint")

        # 7. Backup verification (if --full)
        if args.full and not args.no_backup:
            print("\n7. Creating backup...")
            manager = BackupManager()
            result = manager.create_hot_backup()
            if result.success:
                print(f"   ✓ Backup created: {result.backup_path}")
                print(f"   Size: {result.size_bytes:,} bytes")
                tasks_completed.append("Backup")
            else:
                print(f"   ✗ Backup failed: {result.error_message}")
                tasks_failed.append("Backup")

        # Summary
        print("\n" + "=" * 50)
        print("MAINTENANCE SUMMARY")
        print("=" * 50)
        print(f"Completed: {len(tasks_completed)} tasks")
        for task in tasks_completed:
            print(f"  ✓ {task}")

        if tasks_failed:
            print(f"\nFailed: {len(tasks_failed)} tasks")
            for task in tasks_failed:
                print(f"  ✗ {task}")
            return 1

        return 0

    finally:
        conn.close()


def cmd_test_migrations(args: argparse.Namespace) -> int:
    """Test database migrations."""
    tester = MigrationTester()

    if args.from_version:
        print(f"Testing migration from v{args.from_version}...")
        result = tester.test_migration(from_version=args.from_version)

        if result.success:
            print("✓ Migration test passed")
            print(f"  Duration: {result.duration_seconds:.2f}s")
            return 0
        else:
            print("✗ Migration test failed")
            for error in result.errors:
                print(f"  Error: {error}")
            return 1
    else:
        print("Running full migration test suite...")
        results = tester.run_full_test_suite()
        print_migration_report(results)

        failed = sum(1 for r in results.values() if not r.success)
        return 0 if failed == 0 else 1


def cmd_rollback_test(args: argparse.Namespace) -> int:
    """Test migration rollback."""
    tester = MigrationTester()

    from_version = args.from_version
    to_version = args.to_version

    print(f"Testing rollback from v{from_version} to v{to_version}...")
    result = tester.test_rollback(from_version=from_version, to_version=to_version)

    if result.success:
        print("✓ Rollback test passed")
        print(f"  Duration: {result.duration_seconds:.2f}s")
        return 0
    else:
        print("✗ Rollback test failed")
        for error in result.errors:
            print(f"  Error: {error}")
        return 1


def cmd_drill(args: argparse.Namespace) -> int:
    """Run backup/restore drill."""
    import tempfile
    import time

    print("=" * 60)
    print("BACKUP/RESTORE DRILL")
    print("=" * 60)

    manager = BackupManager()

    # Step 1: Create backup
    print("\n1. Creating hot backup...")
    start = time.time()
    result = manager.create_hot_backup()
    backup_time = time.time() - start

    if not result.success:
        print(f"   ✗ Backup failed: {result.error_message}")
        return 1

    print(f"   ✓ Backup created in {backup_time:.2f}s")
    print(f"   Path: {result.backup_path}")
    print(f"   Size: {result.size_bytes:,} bytes")

    # Step 2: Verify backup
    print("\n2. Verifying backup integrity...")
    report = manager.verify_backup(result.backup_path)

    if not report.integrity_check_passed:
        print("   ✗ Backup integrity check failed")
        return 1

    print("   ✓ Backup integrity verified")
    print(f"   Pages: {report.page_count}")
    print(f"   Fragmentation: {report.fragmentation_ratio:.1%}")

    # Step 3: Restore to temporary location
    print("\n3. Testing restore to temporary location...")
    with tempfile.TemporaryDirectory() as tmpdir:
        restore_path = Path(tmpdir) / "restored.db"

        # Temporarily override db_path for restore test
        original_path = manager.db_path
        manager.db_path = restore_path

        start = time.time()
        restore_result = manager.restore_from_backup(
            result.backup_path,
            verify_integrity=True,
            create_safety_copy=False,
        )
        restore_time = time.time() - start

        manager.db_path = original_path

        if not restore_result.success:
            print(f"   ✗ Restore failed: {restore_result.error_message}")
            return 1

        print(f"   ✓ Restore completed in {restore_time:.2f}s")

    # Step 4: Verify data
    print("\n4. Verifying restored data...")
    if result.tables_backed_up:
        for table, count in result.tables_backed_up.items():
            print(f"   {table}: {count:,} rows")

    # Summary
    print("\n" + "=" * 60)
    print("DRILL SUMMARY")
    print("=" * 60)
    print(f"Backup time:  {backup_time:.2f}s")
    print(f"Restore time: {restore_time:.2f}s")
    print(f"Backup size:  {result.size_bytes:,} bytes")
    print("\n✓ All drill steps completed successfully")

    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    """Attempt database recovery."""
    print("=" * 60)
    print("DATABASE RECOVERY")
    print("=" * 60)

    # First, run health check
    print("\n1. Running health check...")
    monitor = ReliabilityMonitor()
    report = monitor.check_health()

    print(f"   Status: {report.status.value}")

    if report.status.value in ("healthy", "degraded"):
        print("\n✓ Database is healthy, no recovery needed")
        return 0

    print("\n   Issues detected:")
    for warning in report.warnings:
        print(f"   - {warning}")

    # Check corruption
    print("\n2. Checking for corruption...")
    corruption = monitor.detect_corruption()

    if corruption.corruption_detected:
        print("   ✗ Corruption detected!")
        print(f"   Affected tables: {len(corruption.affected_tables)}")
        print(f"   Recoverable: {'Yes' if corruption.recoverable else 'No'}")
        print(f"   Suggested action: {corruption.suggested_action.value}")
    else:
        print("   ✓ No corruption detected")

    if not args.force:
        print("\nProceed with recovery? (yes/no):", end=" ")
        response = input().strip().lower()
        if response != "yes":
            print("Recovery cancelled")
            return 0

    # Attempt recovery
    print("\n3. Attempting recovery...")
    recovery = RecoveryManager()
    result = recovery.attempt_recovery()

    if result.success:
        print("\n✓ Recovery successful")
        print(f"   Level: {result.level.value}")
        print(f"   Duration: {result.duration_seconds:.2f}s")
        print(f"   Data loss: {result.data_loss_estimate}")
        print("\n   Actions taken:")
        for action in result.actions_taken:
            print(f"   - {action}")
        return 0
    else:
        print("\n✗ Recovery failed")
        print(f"   Error: {result.error_message}")
        return 1


def cmd_list_backups(args: argparse.Namespace) -> int:
    """List available backups."""
    manager = BackupManager()
    backups = manager.list_backups()

    if not backups:
        print("No backups found")
        return 0

    print(f"\nFound {len(backups)} backup(s):\n")
    print(f"{'#':<4} {'Date':<20} {'Size':<15} {'Path'}")
    print("-" * 70)

    for i, backup in enumerate(backups, 1):
        mtime = datetime.fromtimestamp(backup.stat().st_mtime)
        size = backup.stat().st_size
        size_str = f"{size / 1024 / 1024:.1f} MB"
        print(f"{i:<4} {mtime.strftime('%Y-%m-%d %H:%M:%S'):<20} {size_str:<15} {backup.name}")

    print()
    return 0


def cmd_cleanup(args: argparse.Namespace) -> int:
    """Clean up old backups."""
    manager = BackupManager()

    if args.dry_run:
        print("DRY RUN - No files will be deleted\n")

    removed = manager.cleanup_old_backups(
        max_age_days=args.max_age,
        dry_run=args.dry_run,
    )

    if removed:
        action = "Would remove" if args.dry_run else "Removed"
        print(f"{action} {len(removed)} old backup(s):")
        for backup in removed:
            print(f"  - {backup.name}")
    else:
        print("No old backups to remove")

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="JARVIS Database Maintenance CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a backup
  uv run python scripts/db_maintenance.py backup

  # Restore from latest backup
  uv run python scripts/db_maintenance.py restore

  # Run health check
  uv run python scripts/db_maintenance.py health

  # Daily maintenance
  uv run python scripts/db_maintenance.py maintain --daily

  # Test migrations
  uv run python scripts/db_maintenance.py test-migrations
        """,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # backup command
    backup_parser = subparsers.add_parser("backup", help="Create database backup")
    backup_parser.add_argument(
        "--type",
        choices=["hot", "export", "migration"],
        default="hot",
        help="Type of backup to create",
    )
    backup_parser.add_argument(
        "--tables",
        nargs="+",
        help="Tables to export (for export type)",
    )
    backup_parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Don't compress SQL export",
    )
    backup_parser.set_defaults(func=cmd_backup)

    # restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from backup")
    restore_parser.add_argument(
        "--backup",
        type=Path,
        help="Specific backup file to restore from",
    )
    restore_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )
    restore_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip integrity verification",
    )
    restore_parser.add_argument(
        "--no-safety",
        action="store_true",
        help="Don't create safety copy of current DB",
    )
    restore_parser.set_defaults(func=cmd_restore)

    # health command
    health_parser = subparsers.add_parser("health", help="Run health check")
    health_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    health_parser.set_defaults(func=cmd_health)

    # maintain command
    maintain_parser = subparsers.add_parser("maintain", help="Run maintenance tasks")
    maintain_parser.add_argument(
        "--daily",
        action="store_true",
        help="Run daily maintenance (ANALYZE, checkpoint)",
    )
    maintain_parser.add_argument(
        "--full",
        action="store_true",
        help="Run full maintenance (includes VACUUM, backup)",
    )
    maintain_parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backup during full maintenance",
    )
    maintain_parser.add_argument(
        "--force",
        action="store_true",
        help="Continue even if integrity check fails",
    )
    maintain_parser.set_defaults(func=cmd_maintain)

    # test-migrations command
    test_parser = subparsers.add_parser(
        "test-migrations",
        help="Test database migrations",
    )
    test_parser.add_argument(
        "--from-version",
        type=int,
        help="Test specific version migration",
    )
    test_parser.set_defaults(func=cmd_test_migrations)

    # rollback-test command
    rollback_parser = subparsers.add_parser(
        "rollback-test",
        help="Test migration rollback",
    )
    rollback_parser.add_argument(
        "--from-version",
        type=int,
        required=True,
        help="Source version",
    )
    rollback_parser.add_argument(
        "--to-version",
        type=int,
        required=True,
        help="Target version",
    )
    rollback_parser.set_defaults(func=cmd_rollback_test)

    # drill command
    drill_parser = subparsers.add_parser(
        "drill",
        help="Run backup/restore drill",
    )
    drill_parser.set_defaults(func=cmd_drill)

    # recover command
    recover_parser = subparsers.add_parser(
        "recover",
        help="Attempt database recovery",
    )
    recover_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )
    recover_parser.set_defaults(func=cmd_recover)

    # list command
    list_parser = subparsers.add_parser("list", help="List backups")
    list_parser.set_defaults(func=cmd_list_backups)

    # cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old backups")
    cleanup_parser.add_argument(
        "--max-age",
        type=int,
        default=7,
        help="Maximum age in days (default: 7)",
    )
    cleanup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting",
    )
    cleanup_parser.set_defaults(func=cmd_cleanup)

    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
