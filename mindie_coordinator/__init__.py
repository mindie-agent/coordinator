"""Local MindIE coordinator: task stdio MCP and host NPU allocation."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mindie-coordinator")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.5.0.dev5"
