"""Host-side NPU coordination source, shipped to the physical host and executed there.

[Role] Packages the stdlib-only host protocol split across the four coordination modules.
[Invariants] The bundled stream concatenates probe, state, leases, and public modules in order.
[Timeouts & Locks] No import-time waits or locks.
[Side-Effect Policy] Importing this package writes nothing.
"""
