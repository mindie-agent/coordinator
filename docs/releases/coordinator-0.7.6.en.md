# coordinator 0.7.6

[简体中文](coordinator-0.7.6.md)

This release fixes business interpreter selection and propagation, and preserves task continuity when a preparation helper does not fit.

- Pin the formal remote-dev 0.11.7 source and share bounded interpreter discovery. Container readiness selects Torch / Torch-NPU metadata, with the framework required for MindIE-SD images, while preserving active environments and virtualenv identity.
- Propagate the selected interpreter into preparation. Reusing the same verified container endpoint preserves its business Python and SSH key; bindings do not transfer to another endpoint.
- Create task virtualenvs with the chosen donor and check required dependencies before building. Private packages in another virtualenv are not assumed to be inherited, and the donor remains unchanged.
- Clarify direct remote-directory development and managed source materialization. Failure responses retain execution observation routes and resource-release facts through `next`, allowing task scripts, CLIs, and native tools to continue. Observe and reconcile uncertain submissions first.
- Preserve distinct MindIE and MindIE-SD readiness requirements and compress readiness transport to fit Windows SSH command limits.

remote-dev 0.11.7 is published before this consumer. Clients update through independent formal component releases. CI covers Linux, macOS, and Windows local contracts and normal dependency installation; the release workflow tests and benchmarks the exact tag. A2/A3 SSH was unavailable, so native rebuilds, dynamic-library loading, and NPU canary acceptance remain unverified.
