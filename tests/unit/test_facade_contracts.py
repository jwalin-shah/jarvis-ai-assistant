from __future__ import annotations

import ast
from pathlib import Path


def _read_module(path: str) -> ast.Module:
    return ast.parse(Path(path).read_text(encoding="utf-8"))


def _module_all(path: str) -> set[str]:
    module = _read_module(path)
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        return {
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        }
    return set()


def test_retired_facade_modules_removed() -> None:
    retired = [
        Path("jarvis/socket_server.py"),
        Path("jarvis/errors.py"),
        Path("jarvis/router.py"),
        Path("jarvis/cache.py"),
    ]
    for path in retired:
        assert not path.exists()


def test_canonical_exception_surface_exists() -> None:
    exported = _module_all("jarvis/core/exceptions/__init__.py")
    assert "ErrorCode" in exported
    assert "JarvisError" in exported


def test_canonical_cache_surface_exists() -> None:
    exported = _module_all("jarvis/infrastructure/cache/__init__.py")
    assert "TTLCache" in exported
