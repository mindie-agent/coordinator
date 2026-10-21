# A5 (Ascend950DT) occupancy captures

Read-only captures used by `tests/test_a5_npu_occupancy.py`. The live fleet
(`177.25.21.1`–`177.25.28.1`, rack06os1, 8× Ascend950DT 9582) runs
`npu-smi 25.1.rc3` / `Software Version 25.1.rc3`; the committed `a5-13.txt` /
`a5-23.txt` captures are `26.2.0.b007` and stay the mixed/idle goldens for that
firmware.

| Fixture | Source | Command |
| --- | --- | --- |
| `a5-25rc3-info-idle.txt` | `177.25.26.1` (all cards idle) | `npu-smi info` |
| `a5-25rc3-info-occupied.txt` | `177.25.21.1` (all 8 cards `reduce_test`) | `npu-smi info` |
| `a5-25rc3-info-mixed.txt` | `177.25.26.1` (NPU 0 `python3`, NPUs 1–7 idle) | `npu-smi info` |
| `a5-25rc3-proc-mem-idle.txt` | `177.25.26.1`, NPU 0 | `npu-smi info -t proc-mem -i 0` |
| `a5-25rc3-proc-mem-occupied.txt` | `177.25.21.1`, NPU 0 | `npu-smi info -t proc-mem -i 0` |
| `a5-25rc3-proc-mem-mixed.txt` | `177.25.26.1`, devices 0–7 concatenated with `=====DEVICE <n>` / `=====RC <n>` markers | `npu-smi info -t proc-mem -i N` |
| `a5-proc-mem-error-invalid-card.txt` | `177.25.26.1`, `-i 99` (exit 215) | captured stdout + `RC=` |
| `a5-proc-mem-error-unsupported-chip.txt` | `177.25.26.1`, `-i 0 -c 0` (exit 215) | captured stdout + `RC=` |
| `a5-info-proc-unsupported.txt` | `177.25.26.1` (exit 0) | `npu-smi info proc` |
| `a2-proc-mem-idle.txt` / `a3-proc-mem-idle.txt` | `173.125.1.2` / `192.168.13.165`, NPU 0 | `npu-smi info -t proc-mem -i 0` |
| `a2-proc-mem-idle-chip.txt` / `a3-proc-mem-idle-chip.txt` | same hosts | `... -i 0 -c 0` |

## Formats observed

- A5 25.1.rc3 idle: `\tNPU ID : 0\n\tNo process in device.\n\n` (exit 0).
- A5 occupied: `\tNPU ID : 0\n\tProcess id:<pid>  Process name:<name>  Process memory(MB):<n>\n\n` (exit 0).
- A5 `-c` (A2/A3 die selector): exit 215, `This device does not support input parameter of -c.`
- A5 absent/invalid card: exit 215, `Invalid card id.` + usage text.
- A2/A3 whole-card idle: `NPU ID`, `Chip Count`, then one
  `No process in device.` / `Chip ID : <n>` pair per chip (exit 0).
- A2/A3 with `-c <chip>` idle: bare `No process in device.` (exit 0).

## Caveat

`a5-26rc-proc-mem-idle-header-only.txt` is **not** a live capture: it records the
reported A5 26.x shape where an idle card prints only the `NPU ID : <n>` header
and no `No process in device.` sentinel. The header line is byte-identical to
the live 25.1.rc3 capture above; only the sentinel row is absent. The 25.1.rc3
fleet was polled for 5 minutes and always printed the sentinel, so the
header-only form could not be reproduced live on this firmware.
