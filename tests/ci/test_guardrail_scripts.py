from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_retired_import_guard_script_passes() -> None:
    proc = subprocess.run(
        ["python", "scripts/analysis/check_retired_imports.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_module_size_guard_script_passes() -> None:
    proc = subprocess.run(
        ["python", "scripts/analysis/check_module_size.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
