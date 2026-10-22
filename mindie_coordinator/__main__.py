"""Module entry point for `python -m mindie_coordinator`.

[Role] Delegates to the CLI main function.
[Invariants] No task state is created before CLI argument parsing.
[Timeouts & Locks] Delegated entirely to the CLI.
[Side-Effect Policy] Only the selected CLI command's effects.
"""
from __future__ import annotations

from mindie_coordinator.cli import main

raise SystemExit(main())
