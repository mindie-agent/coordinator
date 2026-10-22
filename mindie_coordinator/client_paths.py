"""Known local path mapping for WSL clients using the Windows coordinator.

[Role] Translates WSL drive mounts to their Windows paths for local clients.
[Invariants] Only WSL's standard drive mounts are translated. This does not
infer UNC, Linux-only filesystem, remote host, or task identity mappings.
[Timeouts & Locks] Pure in-memory mapping with no waits or locks.
[Side-Effect Policy] Read-only; returns a new path string.
"""
from __future__ import annotations

import os
from pathlib import PureWindowsPath
import re


def client_path(value, *, platform: str | None = None) -> str:
    text = os.fspath(value)
    if (platform or os.name) == "nt":
        match = re.fullmatch(r"/mnt/([a-zA-Z])(?:/(.*))?", text)
        if match:
            return str(PureWindowsPath(match[1].upper() + ":/" + (match[2] or "")))
    return text
