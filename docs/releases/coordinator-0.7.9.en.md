# coordinator 0.7.9

[中文](coordinator-0.7.9.md)

- **Diagnostics 2.0 Internal Fault & Machine-Directory Contract-Violation Observability (`diagnostics.py`, `cli.py`, `ops.py`, `machine_directory.py`)**: Upgrades `mindie-diagnostics` to formal release `v0.5.11` (`171623584aa812677df907518dcb55b06493241b`) and `remote-dev` to formal release `v0.11.10` (`3556df571450dab481a25e0157167f1c3727dc6c`), adding `source_root` attribution and deduplicated reporting for confirmed internal faults and machine-directory schema/JSON contract violations while excluding caller context and SSH/remote environment errors.
