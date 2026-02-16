from __future__ import annotations

from pathlib import Path

from api import routers


def test_router___all___matches_existing_modules() -> None:
    router_dir = Path("api/routers")
    existing_modules = {
        path.stem
        for path in router_dir.glob("*.py")
        if path.stem not in {"__init__"}
    }
    assert set(routers.__all__).issubset(existing_modules)
