## 变更摘要 (Summary)

- 

## 设计原则 Review (Pre-PR Checklist)

> Profile: `control-plane` (`check_pr_checklist.py --profile control-plane`). Every item must be checked (`- [x]`) and backed by concrete evidence or `not-applicable: <specific reason>` (e.g., `not-applicable: docs-only`).

- [ ] **CTRL-LEASE-OR-AUTH**: `HostQueue` physical NPU/port authority (`lease_id` + `fence_token`), A3 dual-Die `Phy-ID 0..15` placement, and shared `remote-dev` SSH auth/credential profiles remain intact without local card allocation or silent fallbacks.
- [ ] **CTRL-HANDLE**: Standard zero-`_meta` MCP `tools/call` works without prior session hooks (`P2.7`), and explicit `execution_id` / `job_id` handles resolve across reconnections and subagents without cross-session `PermissionError` blocks.
- [ ] **CTRL-STREAM-IO**: Remote standalone worker scripts remain 100% pure Python stdlib (`python -I` safe, Windows SSH bootstrap `< 8191` chars), file/artifact hashes stream in bounded chunks (`<= 64 KiB`), and outer timeouts satisfy `outer >= sum(inner) + grace` without spin-polling.
- [ ] **CTRL-RECEIPT**: External mutations distinguish `completed` / `failed` / `unknown` (`TimeoutError` precedes `OSError`, pre-submit errors remain `pending` without wedging), and post-success local receipt/registry write failures preserve `isError=False` with `receipt_warning` + `receipt_unconfirmed=True`.

- **CTRL-LEASE-OR-AUTH**: 
- **CTRL-HANDLE**: 
- **CTRL-STREAM-IO**: 
- **CTRL-RECEIPT**: 

## 验证结果 (Verification)

- 
