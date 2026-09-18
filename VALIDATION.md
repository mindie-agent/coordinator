# Native preparation validation

Status: dated execution evidence, 2026-09-12.

The current package contract is in [README.md](README.md). The broader consumer
execution validation is recorded in
[consumer PR #145](https://github.com/mindie-agent/mindie-agent/pull/145).
This document records the native preparation experiment; it does not claim that
the entire package test suite or every supported environment was exercised.

## Final native sequence, 2026-09-12

The experiment used installed `mindie-coordinator` 0.4.0 at
`255b65d9b392fd92c0a8e1b8a0a142bbad647854` and `remote-dev` 0.7.0 at
`862e9ae4ab5e4bb8252a99cfee8629dbfeeb2597`. The loaded daemon identities were
checked before execution and remained unchanged throughout the sequence.
Subsequent task-finish and native-client attachment changes do not change the
native preparation path; their control-plane validation is separate from this
dated experiment.

Four sequential executions ran in one existing Linux aarch64 Ascend container
with CANN 9.1.0 and an `ascend910_9391` profile. Each execution received a fixed
source snapshot and a distinct source root, then ran an import and NPU smoke
command with one allocated NPU. The baseline used vLLM source
`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` and vLLM-Ascend source
`f69831343a1850a363f4892b1f5d5cfe8e4a051d`. Private test worktrees added a Python
marker, then a C++ module constant; the original baseline sources were preserved.

| Case | Preparation observed | Wall time (seconds) | Final result |
| --- | --- | ---: | --- |
| Baseline | Reused dependencies; created a venv and rebuilt native outputs | 832.058 | Succeeded; resources released |
| Python-only change | Reused the baseline interpreter and native outputs | 228.367 | Succeeded; resources released |
| C++ change after the Python change | Reused dependencies; created a separate venv and rebuilt native outputs | 840.821 | Succeeded; resources released |
| Switch back to baseline | Reused the baseline interpreter and native outputs | 220.215 | Succeeded; resources released |

All four commands imported vLLM, vLLM-Ascend and the C extension from their own
execution source root and performed an actual NPU tensor addition, producing
`[2.0, 4.0, 6.0]`. The Python marker appeared only in the two changed-source
executions. The loaded extension exposed the compiled constant `7` only after
the C++ change; it was absent again after switchback.

The dependency key stayed identical across all four executions. The native key
stayed identical for baseline, Python-only and switchback, and changed for the
C++ edit. Baseline, Python-only and switchback also had identical SHA256 values
for all 1,103 recorded native outputs and generated build metadata. Their stage
logs contained only `prepare-root`, `materialize`, `reuse-native` and
`verify-profile` for the two reuse cases: no venv creation or editable/native
installation occurred in those cases.

| Artifact identity | Baseline / Python-only / switchback | C++ change |
| --- | --- | --- |
| SHA256 of the sorted artifact-hash map | `7e7ee72eda45540324973a14d8986e6d7ad1345e1d5516981348af1d33975191` | `af6ddf4334cb246e6143f898f3219f75ad653f654299fa5d06efc1bf8f441716` |
| Primary C extension SHA256 | `31b788401f7d8b2062bb08eabf590369e0f5f2505346a8fa5c25227cd937f361` | `990227899f72d135fd92329e78e4028c070155c96a7d4e146ee279ef4c9d22d0` |

Source versions were checked independently of distribution versions. The vLLM
snapshot SCM version, profile source version and imported `vllm.__version__`
were all `0.27.1`. Distribution metadata was `0.27.1+empty`: the upstream
`get_vllm_version()` writes the SCM version file before appending the `+empty`
distribution suffix selected by the Ascend recipe's `VLLM_TARGET_DEVICE=empty`.
The suffix is build metadata, not a different SCM identity. vLLM-Ascend source
and distribution versions followed the actual candidate commits:

| Case | vLLM-Ascend SCM and distribution version |
| --- | --- |
| Baseline / switchback | `0.19.1rc2.dev1835+gf69831343` |
| Python-only change | `0.19.1rc2.dev1836+g34c19d6ad` |
| C++ change | `0.19.1rc2.dev1837+g5ce87af80` |

A separate preparation-cancellation execution waited for actual compiler
diagnostics before requesting stop. It ended cancelled with resources released.
An independent read of every retained preparation job confirmed `quiet=true`,
no remaining owned processes, and `descendants_drained=true`. The container boot
identity and all four pre-existing live process identities were preserved. One
previously completed zombie supervisor was reaped normally. The temporary source
comment used to force this cancellation build was restored afterward.

At the end of the four-case sequence, an independent audit read fresh receipts
for all 31 recorded preparation and business jobs. Every job was quiet and
drained, with no remaining owned processes or unknown outcomes. The validation
registry had no unresolved admitted execution. The original container identities
remained running, and the recorded init/SSH-daemon process identities were
unchanged. These checks observed existing jobs and containers without rerunning
business commands.

The retained private evidence includes `native-final-checks.json`, individual
admission/result/tail records, preparation logs and the independent cancellation
audit. The baseline's full stdout exceeded the usual tail window; it was read
from the original owned job using paged output cursors, without replaying the
command. The consolidated assertions passed for source-head bindings, versions,
artifact hashes, cache keys, interpreter reuse, import paths, markers, NPU output
and released resources.

These are single observed end-to-end wall times, including preparation,
transport, admission, command execution and release. They are not repeated
performance measurements. This final sequence covers one hardware profile and
an import/tensor smoke command; it does not establish model accuracy, serving
throughput, every custom operator's correctness, or portability to another
toolchain or SoC. No native rebuild was replayed to recover missing output.
