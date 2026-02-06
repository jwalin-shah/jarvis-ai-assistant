#!/usr/bin/env python3
"""Migrate MLX embedding service to proper venv structure.

This script migrates the current MLX embedding service from:
  ~/.jarvis/mlx-embed-service/ (uv run approach)

To:
  ~/.jarvis/venvs/embedding/ (proper venv)

This enables consistent environment management with other services.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Migrate MLX embedding service to venv structure."""
    jarvis_home = Path.home() / ".jarvis"
    old_service_dir = jarvis_home / "mlx-embed-service"
    new_venv_dir = jarvis_home / "venvs" / "embedding"
    new_service_dir = new_venv_dir

    print("🔄 JARVIS MLX Embedding Service Migration")
    print("=" * 50)

    # Check if old service exists
    if not old_service_dir.exists():
        print(f"❌ Old service directory not found: {old_service_dir}")
        print("This system may not have MLX embedding service installed.")
        return 1

    # Check if migration already done
    if new_venv_dir.exists():
        print(f"✅ New venv structure already exists: {new_venv_dir}")
        print("Migration appears to be already complete.")
        return 0

    print(f"📁 Old service location: {old_service_dir}")
    print(f"📁 New service location: {new_service_dir}")

    try:
        # Create new directory structure
        print("\n📁 Creating new directory structure...")
        new_service_dir.mkdir(parents=True, exist_ok=True)

        # Copy all files from old service
        print("📋 Copying service files...")
        skip_names = {".venv", ".pid", "server.log", "__pycache__", ".git", ".DS_Store"}
        for item in old_service_dir.iterdir():
            if item.name in skip_names:
                continue
            if item.is_file():
                shutil.copy2(item, new_service_dir / item.name)
                print(f"  ✓ {item.name}")
            elif item.is_dir():
                shutil.copytree(item, new_service_dir / item.name, dirs_exist_ok=True)
                print(f"  ✓ {item.name}/")

        # Create venv
        print("\n🐍 Creating Python virtual environment...")
        result = subprocess.run(
            [sys.executable, "-m", "venv", str(new_venv_dir)],
            check=True,
            capture_output=True,
            text=True,
        )

        # Install dependencies
        print("📦 Installing dependencies...")
        pip_path = new_venv_dir / "bin" / "pip"
        requirements_file = new_service_dir / "requirements.txt"

        if requirements_file.exists():
            print("  Using requirements.txt...")
            result = subprocess.run(
                [str(pip_path), "install", "-r", str(requirements_file)],
                cwd=new_service_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            # Try to infer dependencies from imports in server.py
            server_file = new_service_dir / "server.py"
            if server_file.exists():
                print("  Inferring dependencies from server.py...")
                with open(server_file) as f:
                    content = f.read()

                deps_to_install = []
                if "mlx" in content:
                    deps_to_install.append("mlx")
                if "fastapi" in content:
                    deps_to_install.append("fastapi")
                    deps_to_install.append("uvicorn")
                if "numpy" in content:
                    deps_to_install.append("numpy")

                if deps_to_install:
                    result = subprocess.run(
                        [str(pip_path), "install"] + deps_to_install,
                        cwd=new_service_dir,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    print(f"  Installed: {', '.join(deps_to_install)}")

        # Test the migration
        print("\n🧪 Testing migration...")
        python_path = new_venv_dir / "bin" / "python"
        result = subprocess.run(
            [str(python_path), "--version"], capture_output=True, text=True, check=True
        )
        print(f"  ✓ Python: {result.stdout.strip()}")

        # Check if server.py exists and is importable
        if (new_service_dir / "server.py").exists():
            print("  ✓ server.py found")
        else:
            print("  ⚠️  server.py not found - this may cause issues")

        print("\n✅ Migration completed successfully!")
        print("\nNext steps:")
        print("1. Test the service: uv run python -m jarvis services start-service embedding")
        print("2. If it works, you can remove the old directory:")
        print(f"   rm -rf {old_service_dir}")
        print("3. Update any scripts that reference the old path")

        return 0

    except subprocess.CalledProcessError as e:
        print(f"❌ Migration failed during subprocess call: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return 1
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
