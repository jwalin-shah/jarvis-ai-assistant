"""Interface adapters for JARVIS.

Provides adapters for different interfaces (desktop, CLI, API).
"""

from jarvis.interfaces.desktop import JarvisSocketServer

__all__ = ["JarvisSocketServer"]
