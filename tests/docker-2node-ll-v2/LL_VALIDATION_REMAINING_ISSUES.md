# Internode Low-Latency (LL) Validation — Remaining Issues

## ✅ RESOLVED 2026-07-11 — root-caused to CODE, fixed, validated

**The "HW/driver ceiling" conclusion below was WRONG.** Debug-instrumented root-causing
(DEBUG xe built with `CONFIG_DRM_XE_DEBUG_VM` + a clean A/B) proved the LL
DEVICE_LOST / HANG124 / OOM39 are all the SAME software mechanism, and it is 100%
avoidable with no perf loss.

**Root cause.** IBGDA runs a *persistent long-running (LR) poll/quiet exec queue* on the
DeepEP process VM, which makes that VM a **preempt-fence VM**. Any BO
bind / rebind / eviction / migration on that VM *during the run* (a mid-run
`torch::empty`/`zeMemAllocDevice` VM_BIND — e.g. the fp8↔bf16 persistent-slot realloc,
per-iter test allocs, torch pool growth — or a BO eviction under VRAM pressure) triggers
xe's `preempt_rebind_work_func` (`drivers/gpu/drm/xe/xe_vm.c`). That worker must **suspend
the busy-spinning IBGDA LR queue** and migrate/rebind BOs via the **bcs copy engine**;
under contention it returns **`-16` (EBUSY)** and xe responds with **`xe_vm_kill(vm)`**,
which resets **all** exec queues on the VM (incl. bcs) → `engine_class=bcs` reset →
`UR_RESULT_ERROR_DEVICE_LOST`. It is intermittent because it needs a bind/evict to coincide
with a non-preemptible IBGDA busy-spin window. dmesg signature: `VM worker error: -16` →
`exec queue reset detected` → `Engine reset: engine_class=bcs`. There is **no** GPU
`Faulted Address` print — the failure is the VM_BIND worker, not an unmapped-VA access.

**Fix (shipped).** Run the DeepEP XPU process on a **fault-mode VM** (recoverable page
faults). Fault-mode binds pages on demand via the pagefault handler and *never* runs a
preempt-rebind (never suspends the LR IBGDA queue), so `xe_vm_kill` is unreachable.
`tests/docker-2node-ll-v2/node_wrapper.sh` now exports (default ON, opt out with
`DEEP_EP_XPU_FAULT_MODE=0`):

```
NEOReadDebugKeys=1
EnableRecoverablePageFaults=1
```

This is standard GPU on-demand paging (NOT a CPU proxy).

**Validation (H7168, `DEEP_EP_LL_RESET_DRIVER=1` clean reset per run):**
- Clean-reset A/B, recoverable page faults OFF→ON→OFF→ON: **FAIL→PASS→FAIL→PASS**
  (FAIL always with `VM worker -16` + bcs reset; PASS always with 0 resets).
- No-reset 8-run sweep, fault-mode ON: **8/8 PASS**, 0 resets.
- Clean-reset 6-run sweep, fault-mode default-on: **6/6 PASS**, 0 resets,
  avg_t ≈ 1165–1222 µs (== the good ~1140–1160 µs baseline; **no perf regression**).

**Defense-in-depth (optional, not required):** give fp8/bf16 separate persistent LL slots
and eagerly pre-bind all LL buffer variants at `configure_low_latency_layout` (while
quiescent) to remove mid-run VM_BINDs. This only reduces triggers (cannot cover BO
eviction/migrate rebinds), so fault-mode remains the systemic fix.

---

## Historical (pre-fix) analysis — superseded by the section above

# Internode Low-Latency (LL) Validation — Remaining Issues

_Generated: 2026-07-09. Ran the LL test myself (no subagent), 10× barrier + 10× flag,
fix default-on (`DEEP_EP_LL_PERSIST_BUFFERS=1 DEEP_EP_LL_ALLOC_RETRIES=8`), H7168._

Final tally: **barrier 3/10, flag 4/10** — all passes at the ~1140–1160 µs baseline
(no perf regression). HW left clean (containers down, igub loaded, 4 GPUs, shm clean).

## Validation results (fair warm protocol, H7168)

| Arm     | PASS               | min_t of passes | Failure classes                          |
|---------|--------------------|-----------------|------------------------------------------|
| barrier | 3/10 (runs 4,5,10) | 1142–1145 µs    | HANG124 ×4, DEVICE_LOST ×1, OOM39 ×1      |
| flag    | 4/10 (runs 1,4,5,8)| 1140–1146 µs    | HANG124 ×2, DEVICE_LOST ×4               |

The throwaway warm-up run itself took a DEVICE_LOST and cascaded into the barrier arm
(golden-rule-4), which is why this is below the historic ~6–7/10 pure-warm ceiling.

## Remaining issues

### 1. LL residual DEVICE_LOST / HANG124 (the dominant failure)
Root cause: driver-level **bcs/GT-engine wedge accumulation** on the BMG+mlx5 stack — once
one hard failure lands, subsequent runs cascade. A bare `torch.randn@randn` (no iSHMEM/DeepEP)
also DEVICE_LOSTs on an already-wedged GPU, proving it is the HW/driver ceiling, not our code.

- **Tried:** full igub reset+40s-drain+insmod between runs (cold reset → induces its own
  cold-QP init **HANG124** on run 1); pure-warm back-to-back (best streak but cascades after
  the first wedge); reset-on-failure (prevents cascade but the cold recovery-reset itself
  induces OOM39/DEVICE_LOST). No protocol is clean on this HW.

### 2. err-39 OUT_OF_DEVICE_MEMORY (OOM39)
Root cause: race between a fresh `zeMemAllocDevice` VM_BIND and concurrent bcs copy-engine /
IBGDA activity (`VM worker -16` EBUSY), pre-doorbell, with VRAM free.

- **Tried & shipped (commit 26dde53):** persistent shape-matched reused LL buffers (no fresh
  VM_BIND) + `ll_alloc` transient-retry (synchronize + 1→5 ms backoff ×8). This **absorbs the
  warm variant**. It does **not** absorb the OOM39 seen after a cold recovery-reset — that is a
  *persistent* wedge, not the transient race the retry targets.

### 3. Teardown segfault / `ishmemi_copy … ZE_RESULT_ERROR_UNINITIALIZED` flood
This is the "segment fault" being observed. It fires **after** `===== PASS =====` during
process teardown (seen in `barrier_run10`). Cause: DeepEP intentionally **skips
`ishmem_finalize`**, so the copy engine's command list is already torn down when late
`ishmemi_copy` calls run → UNINITIALIZED flood, which sometimes surfaces as a teardown SIGSEGV.

- **Tried:** classified it separately as a post-success artifact (the tally counts the run as
  PASS). Did **not** re-enable `ishmem_finalize` — per the repo's hard-won rule, calling it
  causes worse hangs/DEVICE_LOST. It is cosmetic on a PASS but should ideally be fixed by
  draining/quiescing the iSHMEM copy queue before teardown rather than by finalize.

## Verdict
The pass rate is HW-ceiling-bound, not code-bound. The buffer-reuse+retry fix is correct,
regression-free, and eliminates one genuine err-39 site — **keep default-on**. Issues 1 and 3
are driver/teardown-level, not in the DeepEP data path. Fix committed (26dde53); other test
artifacts uncommitted; HW clean (containers down, igub loaded, 4 GPUs).

### Open follow-up
Tackle the teardown segfault (issue 3) by adding an explicit iSHMEM copy-queue quiesce before
`destroy()` instead of relying on finalize-skip.
