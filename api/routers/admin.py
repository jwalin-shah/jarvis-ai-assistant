"""Admin API endpoints for database health and backup management.

Provides administrative endpoints for monitoring database health
and managing database backups.
"""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api.ratelimit import RATE_LIMIT_READ, RATE_LIMIT_WRITE, limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get(
    "/health",
    summary="Database health check",
    response_description="Database health status including connection, disk space, and integrity",
    responses={
        200: {
            "description": "Health check successful",
            "content": {
                "application/json": {
                    "example": {
                        "healthy": True,
                        "db_exists": True,
                        "db_size_mb": 12.5,
                        "disk_free_mb": 50000.0,
                        "connection_ok": True,
                        "integrity_ok": True,
                        "issues": [],
                    }
                }
            },
        },
    },
)
@limiter.limit(RATE_LIMIT_READ)
async def admin_health(request: Request) -> dict:
    """Get database health status.

    Returns connection status, disk space, database size,
    and any detected issues.
    """
    from jarvis.db.reliability import get_reliability_monitor

    monitor = get_reliability_monitor()
    status = monitor.check_health()

    return {
        "healthy": status.healthy,
        "db_exists": status.db_exists,
        "db_size_mb": status.db_size_mb,
        "disk_free_mb": status.disk_free_mb,
        "connection_ok": status.connection_ok,
        "integrity_ok": status.integrity_ok,
        "issues": status.issues,
    }


@router.post(
    "/backup",
    summary="Create database backup",
    response_description="Backup creation result",
    responses={
        200: {
            "description": "Backup created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "backup_path": "~/.jarvis/backups/jarvis_20260210_120000.backup",
                    }
                }
            },
        },
        503: {
            "description": "Backup creation failed",
        },
    },
)
@limiter.limit(RATE_LIMIT_WRITE)
async def admin_backup(request: Request) -> JSONResponse:
    """Create a database backup.

    Uses SQLite's backup API for consistent snapshots.
    Old backups are automatically rotated (keeps last 5).
    """
    from jarvis.db.backup import get_backup_manager

    manager = get_backup_manager()
    backup_path = manager.create_backup()

    if backup_path is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "BackupFailed",
                "code": "BACKUP_FAILED",
                "detail": "Failed to create database backup.",
            },
            headers={"Retry-After": "60"},
        )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "backup_path": str(backup_path),
        },
    )


@router.get(
    "/backups",
    summary="List available backups",
    response_description="List of backup files",
)
@limiter.limit(RATE_LIMIT_READ)
async def admin_list_backups(request: Request) -> dict:
    """List available database backups, newest first."""
    from jarvis.db.backup import get_backup_manager

    manager = get_backup_manager()
    backups = manager.list_backups()

    return {
        "backups": [
            {
                "path": str(b),
                "size_mb": round(b.stat().st_size / (1024 * 1024), 2),
                "name": b.name,
            }
            for b in backups
        ],
        "count": len(backups),
    }
