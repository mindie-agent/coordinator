"""UTF-8 at native Windows command-line protocol boundaries."""
import os
import sys


def configure_stdio():
    if os.name == "nt":
        for stream in (sys.stdin, sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")
