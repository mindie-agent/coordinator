# mindie-coordinator 0.7.13: Reversal of `AD-04/AD-05` Session-Binding Gates, Zero-Binding Pass-Through, and Control-Plane Hardening

[中文](coordinator-0.7.13.md)

`mindie-coordinator v0.7.13` eliminates legacy session-binding anti-patterns (`AD-04/AD-05`), completes control-plane modularization, and pins `mindie-diagnostics v0.5.16` and `remote-dev v0.11.13`:

- **Zero Pre-Call Session-Binding Gates on Deterministic Tools (`P2.7`)**: Standard MCP `tools/call` requests with zero `_meta`, zero prior hooks, and zero `context_file`/`request_nonce` execute `mindie_hosts`, `mindie_probe`, `mindie_container`, `mindie_lease`, and handle-addressed `task.*` operations directly, reserving `HostQueue` leases (`lease_id` + `fence_token`) exclusively for physical NPU HBM, compile-slot, and port exclusivity.
- **No Local Self-Bricking & Receipt Honesty**: Removes one-way local fault lockouts, preserves `isError=False` with `receipt_warning` + `receipt_unconfirmed=True` when local receipt accounting warns after successful remote operations, splits oversized modules under `< 800` lines, and propagates `TimeoutError` before `OSError`.
- **Dependency Pins & `control-plane` Governance**: Advances `mindie-diagnostics` to `v0.5.16` and `remote-dev` to `v0.11.13`, aligned with `--profile control-plane` PR governance.
