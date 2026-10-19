"""Native fixtures come from an installed wheel, not optional CPython tests."""
import importlib
from pathlib import Path
import sys

import pytest


@pytest.fixture
def native_extension():
    module_name = {
        "linux": "_psutil_linux",
        "darwin": "_psutil_osx",
        "win32": "_psutil_windows",
    }[sys.platform]
    module = importlib.import_module("psutil." + module_name)
    return module_name, Path(module.__file__)
