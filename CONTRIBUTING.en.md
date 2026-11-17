# Contributing Guide (`mindie-coordinator` — `control-plane`)

[中文](./CONTRIBUTING.md) · **English**

`mindie-coordinator` belongs to the MindIE Agent **`control-plane`** repository class, owning multi-host Ascend NPU/port lease coordination (`HostQueue`), A3 dual-Die topology placement, Git-commit parity materialization, and managed task execution.

## 1. Core Architectural Guardrails

1. **Single Physical Resource Authority (`CTRL-LEASE-OR-AUTH`)**:
   - Remote `HostQueue` (`lease_id` + `fence_token`) is the sole authority for physical NPU allocation (including A3 dual-Die `Phy-ID 0..15`) and service ports; never allocate cards locally or bypass fence tokens.
   - Keep `sessions.sqlite3`, local `coordinator.sqlite3`, and remote host SQLite databases separate; keep `machines.json` (+ `.lock`) and `endpoints.json` synchronized atomically while reusing `remote-dev` SSH credentials.
   - Never clobber intranet PyPI configuration (`PIP_CONFIG_FILE=/dev/null` or `unset PIP_EXTRA_INDEX_URL`) and never guess Python binaries with `sort -V` (use `mindie_coordinator/workers/python_selector.py`).
2. **Zero Pre-Call Binding & Explicit Handle Recovery (`CTRL-HANDLE`)**:
   - All 9 MCP tools must accept standard MCP `tools/call` requests with zero `_meta`, zero prior session hooks, and no required `context_file` (`P2.7`).
   - Explicit `execution_id` / `run_id` handles must resolve across reconnections and subagents; static operator AST checks (`op_preflight`) emit advisory `preflight_warnings` only and never abort submissions.
3. **Pure-Stdlib Remote Workers & Bounded Timeouts (`CTRL-STREAM-IO`)**:
   - Standalone remote scripts live under `mindie_coordinator/workers/`, use 100% Python stdlib (`python3 -I` safe, Windows SSH command line `< 8191` chars), and avoid relative package imports.
   - Enforce `outer_timeout >= sum(inner_budgets) + cleanup_grace` and keep core modules `< 800` lines with a top-30-line `[Role]` / `[Invariants]` / `[Timeouts & Locks]` / `[Side-Effect Policy]` docstring header.
4. **External-Effect & Receipt Honesty (`CTRL-RECEIPT`)**:
   - Always place `except TimeoutError: raise` before `except OSError:`.
   - Transition to `uncertain` only after `submit` is dispatched (`submit_dispatched=True`); pre-submit failures remain `pending` and refresh `epoch` on retry without one-way local self-bricking (`needs_repair` / `TaskRootBusy` are forbidden).
   - Once remote `provision`, `remove`, or `lease` succeeds, local registry/receipt write failures must preserve `outcome="success"` (`isError=False`) and attach `receipt_warning` + `receipt_unconfirmed=True`.

## 2. Isolated Testing & Out-of-Tree Wheel Smoke Check

Isolate host paths (`HOME`, `USERPROFILE`, `XDG_*`, `MINDIE_CONFIG_HOME`, `MINDIE_DATA_HOME`, `MINDIE_DIAGNOSTICS_ROOT`, `REMOTE_DEV_AUTH_DIR`, `REMOTE_DEV_STATE_DIR`, `UV_CACHE_DIR`, `TMPDIR`) in every test run:

```bash
# 1. Local Tier 0 / Tier 1 test suite
uv venv && uv pip install -e ".[test]"
.venv/bin/python -m pytest -q -m "not live_npu"

# 2. Out-of-tree wheel install smoke check
uv build
smoke_dir="$(mktemp -d)"
(cd "$smoke_dir" && uv venv .venv && VIRTUAL_ENV="$PWD/.venv" uv pip install /path/to/dist/*.whl && .venv/bin/mindie-coordinator --help)
rm -rf "$smoke_dir"

# 3. Non-destructive Tier 2 live NPU verification on standing A2 (173.125.1.2) / A3 (192.168.13.165) — never use .153
.venv/bin/python -m pytest -q -m live_npu
```

## 3. Pre-PR Gate (`control-plane` Profile)

1. Verify Unix `LF` (`\n`) line endings across tracked files:
   ```bash
   python3 .github/scripts/check_lf_eol.py
   ```
2. Complete all four checklist IDs (`CTRL-LEASE-OR-AUTH`, `CTRL-HANDLE`, `CTRL-STREAM-IO`, `CTRL-RECEIPT`) in [`.github/pull_request_template.md`](.github/pull_request_template.md) with concrete verification evidence (or `not-applicable: docs-only` for documentation-only PRs) and validate locally:
   ```bash
   python3 .github/scripts/check_pr_checklist.py --profile control-plane --body-file <pr_body.md>
   ```
