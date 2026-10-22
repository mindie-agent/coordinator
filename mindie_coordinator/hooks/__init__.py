"""Native-client session attachment hooks. Local only.

[Role] Marks the local client hook adapters package.
[Invariants] Hooks remain local and never imply remote execution.
[Timeouts & Locks] No import-time waits or locks.
[Side-Effect Policy] Importing this package writes nothing.
"""
