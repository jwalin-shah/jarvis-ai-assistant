"""Functions that build and format prompts for various use cases.

This module is a facade that re-exports all functions from the modular prompt builders:
- jarvis.prompts.tone
- jarvis.prompts.reply
- jarvis.prompts.summary
- jarvis.prompts.search
- jarvis.prompts.rag
- jarvis.prompts.contact
- jarvis.prompts.classify
- jarvis.prompts.utils

Auto-generated exports ensure this stays in sync with the submodules.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any


def _import_all(module_name: str) -> dict[str, Any]:
    """Import all callables from a module."""
    module = importlib.import_module(module_name)
    return {
        name: obj
        for name, obj in module.__dict__.items()
        if inspect.isfunction(obj) and not name.startswith("__")
    }


# Auto-import all functions from submodules
_tone = _import_all("jarvis.prompts.tone")
_reply = _import_all("jarvis.prompts.reply")
_summary = _import_all("jarvis.prompts.summary")
_search = _import_all("jarvis.prompts.search")
_rag = _import_all("jarvis.prompts.rag")
_contact = _import_all("jarvis.prompts.contact")
_classify = _import_all("jarvis.prompts.classify")
_utils = _import_all("jarvis.prompts.utils")

# Export everything
__all__ = []

for _mod in (_tone, _reply, _summary, _search, _rag, _contact, _classify, _utils):
    for _name, _obj in _mod.items():
        globals()[_name] = _obj
        __all__.append(_name)

# Clean up private vars
del _tone, _reply, _summary, _search, _rag, _contact, _classify, _utils, _mod, _name, _obj, _import_all
