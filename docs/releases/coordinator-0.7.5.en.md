# coordinator 0.7.5

[中文](coordinator-0.7.5.md)

## Task continuity and first-time SSH access

- Reuse supplied credentials through shared `remote_auth` to authenticate, install a public key, and verify key access. Missing saved authentication profiles must not block otherwise authorized work.
- Optional knowledge, diagnostics, component setup, and local accounting failures affect their capability. Continue independent authorized native work while preserving task identity, resource constraints, and active jobs.
- Preserve successful primary remote outcomes with explicit warnings for local receipt failures. Reconcile uncertain external writes before retrying them.
- Validation covers local contracts, platform CI, real isolated SSH servers, and release benchmarks. No private deployment-host, NPU inference, or model accuracy acceptance is claimed.
