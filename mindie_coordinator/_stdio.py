"""UTF-8 at native Windows command-line protocol boundaries.

[Role] Reconfigures stdio streams to UTF-8 on native Windows.
[Invariants] Non-Windows streams and streams without `reconfigure` are untouched.
[Timeouts & Locks] No waits or locks.
[Side-Effect Policy] Mutates only the process-local stdio encodings.
"""
import os
import sys


def configure_stdio():
    if os.name == "nt":
        for stream in (sys.stdin, sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")
