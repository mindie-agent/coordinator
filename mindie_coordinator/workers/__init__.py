"""Generic process control lives in remote-dev, not here.

[Role] Marks the reserved local worker package boundary.
[Invariants] No process-control implementation is duplicated here.
[Timeouts & Locks] No import-time waits or locks.
[Side-Effect Policy] Importing this package writes nothing.
"""
