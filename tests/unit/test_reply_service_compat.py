from __future__ import annotations

import ast
from pathlib import Path


def _reply_service_method_names() -> set[str]:
    source = Path("jarvis/reply_service.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    reply_service_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "ReplyService"
    )
    return {
        node.name
        for node in reply_service_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_reply_service_keeps_legacy_logging_apis() -> None:
    """Refactors should preserve legacy callsites in worker/task flows."""
    methods = _reply_service_method_names()
    assert "log_custom_generation" in methods
    assert "_persist_reply_log" in methods
    assert "_compute_example_diversity" in methods
