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


def test_socket_server_facade_exports_expected_surface() -> None:
    exported = _module_all("jarvis/socket_server.py")
    assert "JarvisSocketServer" in exported
    assert "WebSocketWriter" in exported
    assert "RateLimiter" in exported
    assert "JsonRpcError" in exported


def test_errors_facade_exports_expected_surface() -> None:
    exported = _module_all("jarvis/errors.py")
    assert "ErrorCode" in exported
    assert "JarvisError" in exported
    assert "GraphContactNotFoundError" in exported
    assert "iMessageAccessError" in exported


def test_router_facade_retains_compat_methods() -> None:
    module = _read_module("jarvis/router.py")
    reply_router = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "ReplyRouter"
    )
    methods = {
        node.name
        for node in reply_router.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"route", "route_message", "get_routing_stats", "close"}.issubset(methods)

    module_functions = {
        node.name for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"get_reply_router", "reset_reply_router"}.issubset(module_functions)
