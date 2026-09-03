# IBGDA QP scaling and the ordered commit gate: iSHMEM vs NVSHMEM

Companion to `SKILL.md`. Captures the root-cause analysis of the two remaining XPU
low-latency (LL) pathologies at high expert counts:

* an intermittent multi-millisecond stall that **amplifies** with QP count, and
* a hard hang above a token-count threshold.

Both trace to the same mechanism: the **IBGDA per-QP ordered commit gate**.
Reference: NVSHMEM public `main`, commit `b0d9d3dc08fc3ee0840fdb6f3a2c11932d85a2e2`.

---

## 1. DeepEP's QP model: one RC QP per local expert

Upstream DeepEP (`deep_ep/buffers/legacy.py`) sets, with **no clamp**:

```python
os.environ['NVSHMEM_IBGDA_NUM_RC_PER_PE'] = f'{num_qps_per_rank}'
```

and `buffer.py` documents that LL mode "requires that this number equals to the
number of local experts". The CUDA kernel passes the local expert index *directly*
as the qp_id, with no modulo and no bounds check:

```cpp
// csrc/cuda_kernels/internode_ll.cu
nvshmemi_ibgda_put_nbi_warp(..., dst_rank, dst_expert_local_idx, lane_id, slot_idx);
nvshmemi_ibgda_amo_nonfetch_add(..., dst_rank, dst_expert_local_idx);
```

QP lookup is a raw index (`rcs[id * npes + pe]`), so `qp_id >= num_rc_per_pe`
is an out-of-bounds read, not a wrap. **QPP must be >= num_local_experts.**

### NVSHMEM imposes no QP-count limit

| Knob | Bound | Source |
| --- | --- | --- |
| `NVSHMEM_IBGDA_NUM_RC_PER_PE` | **none** (default 2, type int) | `src/modules/transport/common/env_defs.h` |
| `NVSHMEM_QP_DEPTH` | `[128, 32768]` enforced | `nvshmem_common_ibgda.h`, `ibgda.cpp` |

There is no silent clamp: an over-large value simply fails at QP creation.
Per-RC-QP cost at depth 1024 is ~384-512 KB (64 KB WQ + 64 KB **dedicated** send CQ
+ ~256 KB fetch buffer, each 64 KB page-aligned). NVSHMEM shares nothing and
allocates eagerly; ~144 MB for 96 experts x 3 peers is simply accepted.

> **Consequence for iSHMEM:** the historical `qps_per_pe` clamp of 16 was *policy*,
> not structural. It is now `[1, 128]` (`src/ibgda.cpp:6098`). Note iSHMEM rounds
> **up** to a power of two, so 96 experts -> 128 QPs.

---

## 2. NVSHMEM has the SAME ordered commit gate

This is the central finding. NVSHMEM is **not** using a smarter lock-free protocol:

```cpp
// src/include/non_abi/device/pt-to-pt/ibgda_device.cuh  (ibgda_submit_requests)
if (need_strong_flush) {                       // QP shared across CTAs
    IBGDA_MEMBAR_NO_OPTIMIZATION();
    while (atomicCAS(ready_idx, base_wqe_idx, new_wqe_idx) != base_wqe_idx);
    IBGDA_MFENCE();
} else {                                       // QP private to one CTA
    IBGDA_MFENCE();
    while (atomicCAS_block(ready_idx, base_wqe_idx, new_wqe_idx) != base_wqe_idx);
    IBGDA_MFENCE();
}
```

Compare iSHMEM (`src/ibgda_device_impl.h`, ~`:1266` and ~`:1416`, `:1567`):

```cpp
uint32_t cur, spin_iters = 0;
while ((cur = commit_ref.load(sycl::memory_order::relaxed)) < wqe_idx) {
    if (++spin_iters > 64u) { spin_iters = 0;
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device); }
}
need_doorbell = (cur == wqe_idx);
```

Structurally identical: spin until the predecessor commits. Both deadlock if the
holder of slot `N` is not scheduled while the holder of `N+1` spins.

### The difference that matters: **atomic scope** — *investigated and disproven*

NVSHMEM selects `atomicAdd_block` / `atomicCAS_block` when `is_qp_shared_among_ctas`
is false, which looks like a safety mechanism iSHMEM lacks. **It is not available
here.** Verified in `ibgda_get_rc()` (`ibgda_device.cuh:1835-1901`): every RC path —
`NVSHMEMX_QP_DEFAULT`, `NVSHMEMX_QP_ANY`, and the explicit `qp_index` path DeepEP
uses — falls through to a single **unconditional** assignment:

```cpp
} else {              // explicit qp_index (DeepEP passes dst_expert_local_idx)
    idx = id + pe;
}
*out_shared_among_ctas = true;    // UNCONDITIONAL, line 1900
return &state->globalmem.rcs[idx];
```

So for RC QPs the `_block` variants are **dead code**; only `ibgda_get_dci()` can
produce `false`, and DeepEP does not use DCIs. DeepEP's own CUDA
`csrc/cuda_kernels/ibgda_device.cuh` goes further and drops the flag entirely,
hardcoding device-scope `atomicAdd`/`atomicCAS`.

**Conclusion: iSHMEM's device-scope reservation already matches NVSHMEM exactly.**
Reducing it to `memory_scope::work_group` would be a *divergence*, not parity, and
is unsafe for three independent reasons:

1. LL dispatch spreads tokens across work-groups and routes each to arbitrary
   experts, so many work-groups reserve on the same expert-indexed QP
   concurrently. `work_group`-scoped `fetch_add` is not mutually atomic across
   work-groups and would hand out duplicate send-queue slots.
2. The same `claimed` counter is reserved at `memory_scope::system` by the AMO
   path (`ishmemi_ibgda_device_rdma_atomic64`, `ibgda_device_impl.h:~2649`), which
   LL uses on the *same* QP for flag posts. System scope is required there because
   `qp_ctrl_block` falls back to host USM when the VRAM allocation fails
   (`ibgda.cpp:779-787`). Mixing in `work_group` scope breaks mutual atomicity.
3. Cross-work-group fence/quiet paths read `claimed`, and the companion `ready`
   counter is read with `ishmemi_ibgda_uc_load32` — an uncached load that bypasses
   L1 — so a `work_group`-scoped store could be missed entirely.

Doorbell ringing is *already* out of order in both stacks
(NVSHMEM: per-QP test-and-set `post_send_lock` + `atomicMax(prod_idx)`, last writer
wins; iSHMEM: CAS-max on `nic_wq_commit_addr`). The doorbell is **not** the problem.

> The reservation site in `put_nbi_batch` now carries an in-code comment recording
> all of the above, so this avenue is not re-explored.

---

## 3. Measured behaviour (XPU BMG, E=384, TOPK=6, H7168, 2 nodes)

QP count improves steady state monotonically but amplifies the tail monotonically:

| QPP | nt=32 avg/min | nt=64 avg/min | nt=128 avg/min/max |
| --- | --- | --- | --- |
| 2 | 2640.70/2471.66 | 3119.98/2855.11 | 4297.11/4120.53/4568.62 |
| 4 | 1655.22/1584.28 | 2207.20/2134.18 | **3802.07**/3339.23/4317.30 |
| 8 | 1396.20/1321.16 | 1957.03/1883.86 | 8296.19/3067.74/74299.94 |
| 16 | 1307.29/1249.09 | 1884.77/1790.20 | 11913.26/3019.33/96407.38 |
| 128 | **1265.38/1190.59** | **1847.41/1777.46** | 17269.34/**2971.07**/85897.66 |

Hang threshold **moves down** as QPP rises: nt>=224 at QPP=4/16, but **nt>=192 at
QPP=128** (nt=160 survives at min 3682 us / max 101136 us; nt=192 and nt=256 hang).

Interpretation: QP count does not *cause* the stall; it raises the number of
independent ordered chains that must all make progress, so each occurrence of a
non-co-resident predecessor is costlier and more likely.

Expert *count* dominates raw throughput independently: E=8/TOPK=6 gives
742.62 / 1399.65 / 3372.19 us, ~3.6x faster than E=384 at matched QPP.

`min_t` is the only trustworthy metric here; `avg_t` at nt=128 is dominated by the
outlier `max_t`. Treat deltas < 2% at low token counts as noise.

---

## 4. Why the XPU LL kernel still contends at QPP >= num_local_experts

`csrc/xpu/internode_ll.cpp` already routes correctly -- it passes `le` (local expert
index) as `qp_id`, matching CUDA:

```cpp
ishmemx_putmem_nbi_subgroup(dst, msg, used_bytes, dst_rank,
                            static_cast<unsigned int>(le), true, sg, false);
```

So with QPP >= 96 each expert owns a QP. Contention remains because a *single*
expert's payload is split across many producing sub-groups
(`tok_split=2`, `live put warps/WG=12 of 30`).

### The grid is NOT oversubscribed -- measured, not assumed

Device: **Intel Arc Pro B60 (BMG), `max_compute_units` = 160.** At E=384:

| quantity | formula | value |
| --- | --- | --- |
| `num_warp_groups` | `ceil(num_experts / 160)` | 3 |
| `num_warps_per_group` | `32 / nwg` | 10 |
| `num_warps` (sub-groups per WG) | `nwg * nwpg` | 30 (matches the `of 30` in the log) |
| `num_sms` = `send_wgs` (grid) | `ceil(num_experts / nwg)` | **128** |
| work-group size | `num_warps * 32` | 960 |

**128 work-groups on 160 Xe-cores** -- the grid fits. That is exactly what the
`num_warp_groups` multiplexing exists to guarantee, and it holds for every E that
passes the `nwg <= kLLMaxWarpGroups` check. So "more work-groups than SMs" is
**not** the failure mode.

### Sub-groups within one work-group are always co-resident

A work-group is dispatched as a unit to a single Xe-core and all of its threads
occupy execution slots simultaneously. That is precisely why
`sycl::group_barrier(group)` is legal -- and this kernel relies on it throughout.
If intra-work-group sub-groups could be scheduled apart, every `group_barrier` in
the file would deadlock. **The hazard is strictly INTER-work-group.**

### The real inter-work-group hazard: a deferred doorbell

The publish step is not "spin until my predecessor commits". It is a *scan*:

```cpp
target_pi = ready_ref.load(acquire);
if (target_pi < new_prod) { /* scan [target_pi, new_prod), break on first gap */ }
ready = CAS-max(ready, target_pi);
ring = (target_pi > 0) && (force_db || target_pi == new_prod ||
                           (batch > 0 && target_pi % batch == 0));
```

The scan breaks on the first unpublished slot, so nobody spins. The consequence is
that **the sub-group that writes a WQE is not necessarily the one that doorbells
it**. Worked example, two work-groups on the same QP:

* WG A reserves slot 10 (`new_prod` 11); WG B reserves slot 11 (`new_prod` 12).
* B finishes first, reads `ready` = 10, scans, finds slot 10 unwritten, breaks.
  `target_pi` stays 10, `10 != 12`, so **B does not ring**.
* A then writes slot 10, scans `[10, 11)`, gets `target_pi` = 11, `11 == 11`, rings
  with 11 -- which signals slots 0..10 only. **B's slot 11 is still not signalled.**

B's WQE is stranded until either the next put on that QP or a `fence_qp` / `quiet_qp`
scans it up. DeepEP does fence that QP by design (`internode_ll.cpp:1168`
`ishmemx_fence_qp(dst_rank, le)` before the flag post, and `fence_ready_wqes`
performs exactly this recovery scan), so this is *covered* -- but the recovery is
what turns a normal ~3 ms iteration into the observed `max_t` spikes of 74-101 ms.

**Honest status: the nt>=192 hang root cause is NOT verified.** Deferred doorbells
explain the latency spikes but have a designed recovery path. Remaining concrete
suspects, in order:

1. The **unbounded CQ-backpressure spin** in `put_nbi_batch` (`while (completed -
   target < 0)`) taken when `base + num_wqes - 1 >= num_wqebb`, i.e. on send-queue
   wrap (`nic_wq_slots` = 16384/QP). A producer blocked there cannot publish, which
   can strand a lower slot that another blocked producer is waiting on.
2. The `emit_direct_*` ordered commit gates (`:1266`, `:1416`, `:1567`), which *are*
   genuine unbounded predecessor spins, on whichever path the AMO/flag posts take.
3. Interaction with `DEEP_EP_LL_POLL_CAP` on the receive side giving up.

An E=160 / nwg=1 control on untouched code reproduces the pathology, so it is
**pre-existing**, not a regression from the warp-group feature or the WB/UC tuning.

---

## 5. Recommended fixes, in order of preference

1. ~~**Add a scope parameter to the iSHMEM subgroup put.**~~ **Disproven -- do not
   do this.** See section 2: NVSHMEM's block-scope path is dead code for RC QPs
   (`ibgda_get_rc` sets the flag `true` unconditionally), and the reduced scope is
   unsafe here. iSHMEM's device scope **already matches NVSHMEM**. This was
   implemented, analysed, and reverted; the reservation site now carries an
   in-code comment so it is not re-attempted.

2. **Bound the spin.** Convert the unbounded `while (cur < wqe_idx)` in the
   `emit_direct_*` commit gates into a retry-limited loop that backs off and
   re-checks, turning a hang into a recoverable stall. This is now the *primary*
   structural fix, and NVSHMEM offers no counter-example: it has the same
   unbounded spin and simply relies on GPU-wide forward progress.

3. **Keep the send grid co-residency-bounded**, and size QPP to
   `num_local_experts` rather than maximising it. At E=384, QPP=4 gives the best
   nt=128 average and the highest hang threshold; QPP=128 gives the best `min_t`
   but the worst tail and the *lowest* hang threshold (nt>=192 vs nt>=224). There
   is a genuine latency/stability trade-off here.

4. Do **not** raise `deep_ep/buffer.py`'s `min(num_qps_per_rank, 16)` clamp
   (lines ~254/305) until fix 2 lands -- raising it makes the
   worst-for-average configuration the default.

5. **Latent bug worth a separate look:** the `claimed` counter is reserved at
   `memory_scope::device` by the put path but at `memory_scope::system` by the AMO
   path (`rdma_atomic64`). Both are safe while `qp_ctrl_block` lives in VRAM, but
   if the host-USM fallback at `ibgda.cpp:783` ever triggers, the device-scope
   reservations are no longer guaranteed coherent with the system-scope ones.
   Making the scope follow the actual allocation would close this.

### Verification checklist

* `ISHMEM_DEBUG=1` prints `multi-QP mode: <N> QPs per PE pair` -- confirm N is what
  you asked for (remember the power-of-two round up).
* Confirm artifact ordering: iSHMEM sources < `libishmem.a` < `deep_ep_cpp*.so`,
  and `build/ishmem-sycl-dlink/.archive-stamp` matches the archive you intended.
* A frozen `[DeepEP] LL dispatch send:` log line with no forward progress for
  minutes is the commit-gate hang signature, not a NIC error.
