# mindie-coordinator 0.7.0

[中文](coordinator-0.7.0.md)

Host onboarding, credential persistence, public-key installation, and SSH recovery now use remote-dev. Machine inventories and endpoints retain the returned `auth_ref`, allowing new agent sessions to connect without asking for the password again. Password-only hosts remain usable when key installation or public-key authentication is unavailable.

Coordinator no longer maintains a separate askpass, password transport, or key installer. Passwords belong only to remote-dev's private local plaintext credential file; inventories and npu-top registration carry references. Plugin virtual environments under the same user share configuration without a system keychain or unlock prompt. A container's direct SSH port is a separate authentication scope and does not inherit another endpoint's reference.

The npu-top bridge now consumes actual per-server HTTP 207 receipts. Authentication that is not ready and local reference-recording failures remain visible as partial outcomes without erasing completed registration. Existing keys and inventory metadata are preserved. Authentication recovery never replays submitted remote work.

This component pins the formal remote-dev 0.11.0 release at Git commit `153813703a3cc64c695dd8e5f19e8a173e144dda`. Mindie-diagnostics remains pinned to 0.5.4 at `5ae643e19223f2181ecdbaa3edaf464137c4805e`. The workflow checks installed provenance, formal dependency releases, and tag/source alignment, and ships benchmark reports.

Shared authentication passed real SSH acceptance on all three platforms. macOS passed seven tests using installed wheels for the three components and the Kimi candidate (34.58 seconds). Native Windows PowerShell passed four tests (14.186 seconds), and real WSL2 Ubuntu 24.04 passed four tests (12.694 seconds), with no failures or skips on either. [Acceptance run 40774308378](https://github.com/mindie-agent/remote-dev/actions/runs/40774308378) tested the same source tree as remote-dev candidate `a7564f49b1ff106e59f72961c851b7b11eb81130`.

The real consumer path covers coordinator onboarding, two Kimi sessions and virtual environments, and npu-top connections and probes after reopening its database, all using the default `credentials.json` under an isolated `REMOTE_DEV_AUTH_DIR`. It also verifies automatic key installation, password access with public keys disabled, and revoked-key repair with business work executed once. Codex, Claude Code, and DSH coverage is through endpoint profiles, not complete native-host acceptance. HTTP 207 receipt and local consumer contracts passed. Both fixed placement and run-manifest benchmark workloads completed five local compatibility repetitions; the tag workflow produces the formal reports.
