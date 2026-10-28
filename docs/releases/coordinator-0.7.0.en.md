# mindie-coordinator 0.7.0

[中文](coordinator-0.7.0.md)

Host onboarding, credential persistence, public-key installation, and SSH recovery now use remote-dev. Machine inventories and endpoints retain the returned `auth_ref`, allowing new agent sessions to connect without asking for the password again. Password-only hosts remain usable when key installation or public-key authentication is unavailable.

Coordinator no longer maintains a separate askpass, password transport, or key installer. Passwords belong only to remote-dev's private local plaintext credential file; inventories and npu-top registration carry references. Plugin virtual environments under the same user share configuration without a system keychain or unlock prompt. A container's direct SSH port is a separate authentication scope and does not inherit another endpoint's reference.

The npu-top bridge now consumes actual per-server HTTP 207 receipts. Authentication that is not ready and local reference-recording failures remain visible as partial outcomes without erasing completed registration. Existing keys and inventory metadata are preserved. Authentication recovery never replays submitted remote work.

Release dependency closure remains required: after remote-dev 0.11.0 is formally released, replace the candidate version range with its exact release Git commit before publishing this component. Mindie-diagnostics remains pinned to 0.5.4 at `5ae643e19223f2181ecdbaa3edaf464137c4805e`. The workflow checks installed provenance, formal dependency releases, and tag/source alignment, and ships benchmark reports.

Acceptance status: shared-authentication, HTTP 207 receipt, and local consumer contracts have been verified. Both fixed placement and run-manifest benchmark workloads completed five local compatibility repetitions. Final native Windows PowerShell and real WSL acceptance remain pending. Final notes must identify tested platforms, source revisions, and results; candidate local checks do not establish three-platform acceptance.
