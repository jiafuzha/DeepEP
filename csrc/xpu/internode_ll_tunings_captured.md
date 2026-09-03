# DeepEP XPU LL Internode — Perf Tunings Captured (from session `e1a2ae34`)

Companion to `csrc/xpu/internode_ll_design.md`. Consolidates every performance
tuning that shipped in the LL (low-latency) internode kernel during the
"Tune Performance at E=384" session, with **mechanism, evidence, and gain** for
each, plus a **portability verdict** onto the internode HIGH-THROUGHPUT (normal)
kernel at `csrc/xpu/internode.cpp` (+ its fused `.inc` files).

Report generated: 2026-09-02.

---

## 1. Headline LL wins

- **E=384 / topk=2 / H7168**: 1246 µs → 575 µs (−54%, **2.17×**). E=8/topk=2 unregressed (317 → 321 µs).
- **E=384 / topk=6 / H7168 / nt=32**: 941 → 545 µs (−42%). nt=128: 3979 → 1568 µs (−61%, **2.53×**).
- **Combine reduce** at E=384/topk=6/nt=128: 1630 → 269 µs (~**6×**), bandwidth 3.27 → 10.4 GB/s.

## 2. Tunings that shipped (all now defaults)

### A. Work-group / grid geometry (occupancy, not wave count)

1. **`num_warp_groups = 1`**; over-subscribe the grid instead of subdividing WGs.
   XPU inverts CUDA's split — one WG per channel/expert; WG size shrinks when `E > SMs`.
2. **`DEEP_EP_LL_NUM_WARPS` heuristic** — 32 warps if `E ≤ SMs`, else 8 (256-item WG).
   Confirmed by falsification: cost linear in `E` with *no* step at 160/320 wave
   boundaries, so WG *size* is the lever, not wave count. **−30% at E=384**;
   *harmful* at E=8 (hence conditional).
3. **`send_wgs` clamped to `num_device_sms`.** A send grid wider than CUs adds
   only launch overhead and risks producer-not-resident wedge on iSHMEM commit
   gate → GuC watchdog → `DEVICE_LOST`.
4. **Reduce grid oversubscribed** — `ll_consume_wgs = min(4·CU, 512)`
   (`DEEP_EP_LL_REDUCE_WGS`). Only possible after B.

### B. Kernel split (replace GridBarrier with a kernel boundary)

5. **Split each direction into two sub-kernels** at what used to be
   `cg::this_grid().sync()`:
   - dispatch: `LLDispatchSendKernel → LLDispatchRecvKernel`
   - combine:  `LLCombineSendKernel  → LLCombineReduceKernel`
   BMG can only co-resident ~24 WGs of 1024 WI. A GridBarrier over `num_experts`
   WGs at E≥8 cannot be satisfied, and would wedge. A kernel boundary is a full
   device barrier with global visibility and **no co-residency requirement**.
6. **Two-pass channel restructure** in `LLDispatchRecvKernel` /
   `LLCombineSendKernel`: post-all-then-wait-all as two separate `for (ch...)`
   loops, not post-then-wait per channel. With `pack_channels > 1` the
   single-loop form *deadlocks*; at C=1 it stops each channel's poll latency
   from serializing behind the previous one. **−25% at E=384**, entirely in
   combine (866 → 580 µs).

### C. Fence removal (RC-ordering + doorbell publish)

7. **`DEEP_EP_LL_DROP_FENCE=1`** — drop the per-QP fence between payload put and
   flag AMO. Proven redundant: payload and flag share the same QP `qp=le`; RC
   strict SQ order guarantees delivery order; `ishmemx_long_atomic_add_qp`
   unconditionally rings the doorbell at `pi = wqe_idx+1` (publishing every
   earlier `force_db=false` WQE); and being a *fetching* atomic it blocks on
   its own completion — strictly stronger than a fence. **~−10% at E=384.**

### D. QP parallelism

8. **`ISHMEM_IBGDA_QPS_PER_PE` auto-defaulted to `num_local_experts`**
   (power-of-2, clamped [1,128]) — the primary throughput lever. Every expert
   keys QP as `qp_idx = le & (qps_per_pe-1)`; `=1` serializes all experts on
   QP0. Effective BW rose 2.07 → ~2.8 GB/s at E=8 high-token.
9. **QP clamp raised 16 → 128** in both `buffer.py` and `run.sh`. ~3%.

### E. Send-side warp parallelism (dispatch)

10. **`DEEP_EP_LL_SEND_TOK_SPLIT`** — original send loop wasted 30/32 warps
    (only warps `< num_topk` posted). Split the WG into `tok_split` casting
    teams; each team casts *and* posts its own token's top-k puts. Auto default
    `clamp(num_tokens/send_wgs, 1, num_warps/4)`. Correct because dispatch slots
    are handed out by an unordered `slot_counter.fetch_add`. **1.05–1.08×.**

### F. Combine reduce vectorization + acquire granularity

11. **`DEEP_EP_LL_REDUCE_VEC=8` (default)** — 8 BF16 elements per work-item =
    16 B vector load (`sycl::vec<uint32_t,4>`), with per-token `topk_idx (int64)`
    + `topk_weights (fp32)` hoisted out of the h-loop (amortized over 8 elems
    instead of re-read per element). Bit-exact via BF16↔float bit shift.
    **5.1× on the reduce kernel.** Falls back to 1 when `hidden % 8`.
12. **`DEEP_EP_LL_REDUCE_ACQ=1` (default)** — system-scope acquire granularity
    per work-item. Modes 2/3 are 13% faster on the kernel but unmask a
    pre-existing send-side skew tail at small shapes (bimodal `avg_t` 228→590 µs
    at E=8/nt=32). Opt-in only.

### G. Cross-rank flag-wait NBI AMO (latest, 2026-09)

13. **`DEEP_EP_LL_FLAG_NBI_AMO=1` (default)** — new iSHMEM entry point
    `ishmemx_long_atomic_add_nbi_qp`: blocking AMO minus the CQE poll and ibuf
    result-read. Slot claim / WQE build / ordered commit + doorbell are
    byte-identical, so unconditional-doorbell is preserved. Falsification:
    - raising `POLL_CAP` **failed correctness** ⇒ flags arrive, they're posted
      slowly (timeout theory dead);
    - collapsing 288→3 posts (`FLAG_AGG`) removed the stall ⇒ cost is per-post
      round-trips.
    At E=384/nt=1024 there were **96 × 3 = 288 serialized RDMA round-trips per
    iteration**. Correctness rests on: payload+flag on same QP, RC in-order,
    unconditional doorbell, receiver spin-poll. **−14.7% at E=384/topk=6/nt=32**;
    parity at E=8.
14. **Rejected: `DEEP_EP_LL_FLAG_AGG`** (collapse flags onto QP0). Destroys
    same-QP ordering → forces a mandatory quiet (~2.3 ms fixed cost) →
    regresses every small shape. Kept as experimental knob only.

### H. Symmetric-heap auto-sizer

15. **Auto-size `ISHMEM_SYMMETRIC_SIZE` to the RDMA layout** instead of blanket
    12 GiB. Oversized heap starves torch's working set → VRAM eviction →
    bimodal `max_t = 148 ms` vs `min_t = 5.6 ms`. Sizes: 256 MiB at nt≤256,
    then 448 / 896 / 1792 MiB for 1024 / 2048 / 4096.

### I. Diagnostics harness (essential for verifying the above)

- `DEEP_EP_SPLIT_DC` — dispatch/combine separately (no kineto on XPU).
- `DEEP_EP_LL_TIME_PHASES=1` — per-iter `send_us` / `reduce_us` for combine.
- `DEEP_EP_LL_PACK_CHANNELS` — diagnostic (C=1/3/6/12 all measured 929 µs at
  E=384), proved per-WG launch cost is not the bottleneck.

## 3. Method that made the session productive

- **Falsify, don't confirm.** Every root-cause claim was killed by a
  discriminating A/B before being trusted (wave-quantization, QP contention,
  per-token bandwidth, per-WG launch overhead all refuted before the
  WG-occupancy hypothesis was accepted).
- Run each candidate against **both** the target shape *and* an E=8/topk=2
  no-regression guard.
- Prefer `min_t` when `max_t` shows a VRAM-eviction / driver-reset tail; treat
  single FAILs as HW-wedge candidates and retry on a freshly-reset igub driver.
- Independently verify every subagent numeric claim.

---

## 4. Portability onto the internode HIGH-THROUGHPUT kernel

Source: code walk of `csrc/xpu/internode.cpp`,
`internode_dispatch_fused.inc`, `internode_combine_fused.inc`,
`internode_notify_fused.inc`, and `csrc/xpu/internode_normal_design.md`.

### 4.1 HT kernel shape (relevant differences vs LL)

- **Single fused warp-specialized kernel** per direction. **No grid barrier
  ever existed**; sync is warp-subset via SPIR-V named barriers (≤8 per kernel,
  IGC cliff).
- **Grid** = `num_channels·2` WGs (channels × {sender-SM, forwarder-SM}),
  `num_channels = num_sms/2`, capped by driver co-residency query = **20 on
  BMG**. Grid is a function of `num_sms`, **not of `E`** — E=384 does NOT
  over-subscribe the grid the way LL does.
- **WG size**: dispatch 512 WI (16 SGs); combine 800 WI (25 SGs at R=2).
  Warp roles hard-coded (`kRDMASender=7`, `kRDMAAndNVLForwarder=8`, etc.).
  Shrinking WG cannot raise co-residency cap on BMG (design §6.4 CLOSED).
- **QPs**: `qps_per_pe=16` is a correctness gate (~1 channel/QP); collapsing
  hangs on shared CQ (§6.3).
- **Combine reduce** (`combine_fused.inc:267–297`) already loads
  `int4_t = 8 bf16/lane/iter` via `ld_nc_global_v` with inline bit-op
  BF16↔float. Accumulator is a **plain sum over `topk_ranks`** — weights are
  applied *earlier* by `kNVLAndRDMAForwarder`, so there is **no per-element
  weight multiply to hoist**.
- **Send+flag fence pattern is the SAME as LL pre-fix**:
  - dispatch tail publish: `dispatch_fused.inc:770` put (`force_db=true`,
    `subgroup`) → `sg` barrier → `:791 ishmemx_fence_qp` → `:792
    ishmemx_long_atomic_add_qp` (same QP).
  - combine tail publish: `combine_fused.inc:970` put (`force_db=true`) →
    `sg` barrier → `:973 ishmemx_fence_qp` → `:974 ishmemx_long_atomic_add_qp`
    (same QP).
  - head-credit lone AMOs: `dispatch:1025`, `combine:1161`.
  - notify has 2× `ishmemx_fence_qp` (`notify_fused.inc:237,286`) — these are
    genuine transport fences (no paired same-QP AMO), must NOT be touched.
- **`ishmemx_long_atomic_add_nbi_qp` is present** in the target iSHMEM install
  (`ishmem_ibgda/src/amo.cpp:320`); no iSHMEM rebuild needed to try G.
- **`ishmem_quiet` in the fused kernels: none.** Host-side quiet only at
  teardown, so NBI-AMO swap does not leak in-flight ops into the outer loop.

### 4.2 Portability verdict table

| LL tuning | HT applicable? | Rationale |
|---|---|---|
| A. WG-size heuristic (`E > SMs`) | **No** | HT grid = `num_channels·2`, independent of `E`. §6.4 CLOSED proves WG size doesn't move BMG co-residency cap. Warp roles are hard-coded, shrinking breaks named-barrier arrive counts. |
| B. Kernel split at grid barrier + two-pass channels | **No** | HT never had a grid barrier (design §2.1). Channels already run on independent WGs on the device grid, so channels don't serialize inside one WG. |
| **C. Drop redundant fence between same-QP put and flag AMO** | **Yes — top candidate** | Byte-identical port at `dispatch_fused.inc:791` and `combine_fused.inc:973`. Payload put is `force_db=true` (already doorbell-rung) and shares the QP with the AMO; RC strict SQ order guarantees remote flag-after-payload. `ishmemx_long_atomic_add_qp` unconditionally re-rings as extra insurance. Notify fences (`:237,286`) must stay — they are lone fences not paired with a same-QP AMO on the sender side. |
| D. `QPS_PER_PE = num_local_experts` | **No** | HT already at correctness-mandated `QPS_PER_PE=16` (~1 channel/QP). Reducing hangs on collapsed CQ (§6.3). |
| E. Send-side token split | **Already shipped** in combine (`DEEP_EP_COMBINE_TOK_SPLIT`, §6.1.1, up to 2.09× on combine). Dispatch `fwd_split` tried, +1.3% (noise), broke correctness (§6.1.2) — reverted. Nothing left to port. |
| F. Vectorized reduce (`REDUCE_VEC=8`) + hoisted topk metadata | **No** | HT combine already loads 16 B per lane via `int4_t`, inline bf16 bit-ops. Accumulator has no per-element weight (weights applied upstream by forwarder), so there is nothing to hoist. C9 REJECTED note shows further unrolling regresses on register pressure. |
| **G. NBI flag AMO** | **Yes — second candidate** | `ishmemx_long_atomic_add_nbi_qp` present in target iSHMEM; 4 sites (2 tail-AMO after same-QP put; 2 lone head-credit). Tail sites keep same-QP RC ordering; head-credit sites are pure spin-poll counters — CQE-poll wait is pure overhead. LL saw −14.7% at E=384/topk=6/nt=32; HT expected smaller (fewer posts/iter) but positive-EV. |
| H. Auto-size `ISHMEM_SYMMETRIC_SIZE` | N/A | Already at harness level (`docker-2node-v2/run.sh` sets 1 GiB by default). |

### 4.3 HT-specific observations (no LL analogue)

- **Local (self-PE) tail AMO** uses `sycl::atomic_ref` acq_rel/system with
  `fetch_add` — already the fastest form; no change proposed.
- **Notify (`notify_fused.inc`) `ishmemx_fence_qp` at :237,286** are lone
  transport fences (no paired same-QP AMO) — must stay under any `DROP_FENCE`
  knob. Any C-style change must scope tightly to the 2 tail-AMO sites in
  dispatch and combine.
- **`DEEP_EP_RDMA_CHUNK=8`** (§6.3.2) is a known ~3.8% win at H7168/topk=2 but
  is unshipped because it *inverts* at H=1024. Orthogonal to LL.

### 4.4 iSHMEM archive/branch caveat — RESOLVED BY MEASUREMENT

`libishmem.a` is **statically linked** into `deep_ep_cpp*.so`, so only the
**build-time** archive matters. The HT harness (`tests/docker-2node-v2/run.sh`)
defaults to `ISHMEM_DIR=/root/jiafuzha/ishmem_ibgda/build/_install` — note this
is a *different* checkout from `/root/jiafuzha/code-repo/ishmem_ibgda`.

The concern was that **`workgroup_qp_issue_tuned`** @ `40bcd4b6` (tuned for LL,
26 commits ahead) might **regress HT** vs the last known-good HT branch
**`main-shared-qp-fix`**.

**Measured answer: NO regression — the two branches are statistically tied.**
Target shape `NT=32 H=7168 topk=6 E=384`, 3 runs/arm, rank-0 `min_t` (µs):

| arm | branch | barrier `.cpp.o` md5 | runs | mean | median | best |
|---|---|---|---|---|---|---|
| A | `main-shared-qp-fix` @ `4824fccc` | `37340a40…` | 1220.5 / 1278.7 / 1220.3 | 1239.8 | 1220.5 | 1220.3 |
| B | `workgroup_qp_issue_tuned` @ `40bcd4b6` | `40fa9a72…` | 1262.4 / 1232.2 / 1232.7 | 1242.4 | 1232.7 | 1232.2 |

Δ (LL-tuned vs known-good): **mean +0.2%, median +1.0%** — ~5× smaller than the
run-to-run noise. Component iso-min: dispatch 933.1 (A) vs 947.4 (B) µs;
combine 812.3 (A) vs 806.1 (B) µs. All 6 runs PASSed.

**Measured noise band: ±2.5% (≈4.2–4.8% peak-to-peak) at NT=32.** Consequence:
a 5% ship gate sits *at the edge of noise* — use **≥5 runs/arm and compare
medians**, not single bests.

**Baseline archive chosen: `workgroup_qp_issue_tuned`** — HT-tied with the
known-good branch, it is the harness default, and it is the only archive
carrying the NBI AMO entry point needed by tuning G.

**Correction to an earlier assumption: `40bcd4b6` is NOT purely additive.** It
also modifies the shared AMO ibuf slot allocator
(`ishmemi_ibgda_device_claim_ibuf_slot` / `…release_ibuf_slot`) to reserve slot
0 as the NBI result sink. Those functions had already been restructured on the
tuned branch by an earlier, unrelated commit `4d764d25` ("Fix workgroup QP WQE
ordering and addressing"), which changed `ibgda_types.h` from
`uint32_t ibuf_slot_bitmap` (32 slots) to `uint64_t ibuf_slot_bitmap[4]` (256
slots). A cherry-pick of `40bcd4b6` onto `main-shared-qp-fix` therefore
**conflicts** in `src/amo.cpp` and `src/ibgda_device_impl.h`, and landing G on
the known-good branch would require importing part of the very workgroup-QP
series we wanted to exclude. Moot given the tie above, but the "purely
additive" premise is false.

### 4.5 Plan (A/B once HW is available)

1. **C — `DEEP_EP_FUSED_DROP_FENCE`** (new env, default OFF). Guards the 2
   tail-AMO fences at `dispatch:791` and `combine:973`. Notify fences left
   alone. Ship default-ON if E=384/topk=6/H7168/nt=32 `min_t` improves ≥5%
   with 3-run stability AND E=8/topk=2 shows no `min_t` regression.
2. **G — `DEEP_EP_FUSED_FLAG_NBI_AMO`** (new env, default OFF). Swaps 4
   `ishmemx_long_atomic_add_qp` → `_nbi_qp`. Ship default-ON at ≥3% + no
   E=8 regression.

Below ship threshold, keep the knob but leave default-OFF.

### 4.6 Status (as of 2026-09-03 05:30Z)

**Step 0 COMPLETE** (§4.4): the LL-tuned iSHMEM branch does **not** regress HT.

**C and G are IMPLEMENTED and fully build-verified. The measurement campaign is
BLOCKED by the patched true-UC `xe` kernel** (§4.8) — not by these changes.

- **C — `DEEP_EP_FUSED_DROP_FENCE`**, gated at exactly the 2 tail-AMO fence
  sites: `internode_dispatch_fused.inc:793`, `internode_combine_fused.inc:975`.
  The trap was respected — `internode_notify_fused.inc:237` and `:286` remain
  **unconditional** (verified: they are the only ungated `ishmemx_fence_qp`
  calls left in the fused kernels).
- **G — `DEEP_EP_FUSED_FLAG_NBI_AMO`**, gated at exactly the 4 AMO sites:
  dispatch tail `:794` + head-credit `:1034`, combine tail `:976` +
  head-credit `:1169`.
- Both read the env once on the host (`fused_drop_fence()` /
  `fused_flag_nbi_amo()` in `internode.cpp`, above the `.inc` includes) and are
  captured by value into the kernel lambdas next to the existing `qps_per_pe`
  local. The `atoi("") == 0` trap is avoided via an explicit `env[0] == '1'`
  test, so unset *and* empty-string both read OFF; `_add_opt_genv` in
  `tests/docker-2node-v2/run.sh` additionally omits empty values.
- **Build gates all PASS** (rebuilt in-container, oneAPI 2025.3.3, torch
  2.14.0a0, after `rm -rf build/ishmem-sycl-dlink`):
  `barrier.cpp.o = 40fa9a725404a496a490b98ad30f5b1e` (exact match to the chosen
  baseline archive); `.archive-stamp` = `…libishmem.a:1788347267000000000:29995450`;
  AOT `-device pvc,bmg,arl-h,mtl-h,lnl-m,ptl-h,ptl-u` (never bare `bmg`);
  both knob strings present; `DEEP_EP_ISHMEM_NO_NBI_AMO` count = 0 (real NBI
  AMO, shim inactive); `ishmemx_long_atomic_add_nbi_qp` declared at
  `ishmemx.h:2371` with 8 syms in the archive.
- A verified copy of the knob build is saved at
  `/root/jiafuzha/deep_ep_cpp.so.knobs-sep3`.

**Remaining work once the substrate is fixed** (everything else is staged):
the matrix baseline / C / G / G∘C × {target, guard} at **≥5 runs each compared
on medians** (the ±2.5% noise band of §4.4 makes 3 runs under-powered, and C's
5% gate sits at the noise edge), plus the QPS A/B of §4.7 item 3.

### 4.7 HT-specific observations worth acting on

1. **The target shape is latency-bound, not bandwidth-bound — C and G are aimed
   correctly.** At NT=32/E=384 the RDMA payload is only 0.459 MB moving at
   ~0.47 GB/s (dispatch) / ~0.53 GB/s (combine), ~2 orders below link
   capability. Nearly all of the ~1232 µs is per-post round-trip latency, which
   is exactly what dropping a fence and de-blocking the AMO attack.
2. **Dispatch dominates combine** (947 vs 806 µs iso-min, ~54/46). If only one
   knob can ship, prioritize the dispatch-side sites.
3. **QP aliasing on channels 12..19 — hypothesis CONFIRMED empirically.** The
   kernel's own diagnostic prints
   `requested num_sms=24 -> using 20 (ISHMEM_IBGDA_QPS_PER_PE=32 is NOT the limiter here)`,
   so **iSHMEM provisions 32** (`deep_ep/buffer.py:308-310`
   `setdefault(clamp_pow2(num_qps_per_rank))`) while `fused_qps_per_pe_env()` in
   `internode.cpp` clamps the *kernel's* view to **16**, with **20 channels**.
   The tail QP is `channel_id % qps_per_pe` and the head-credit QP is
   `(channel_id + num_channels) % qps_per_pe`, so at 32 channels 12..19 wrap and
   **alias their head-credit QP onto another channel's tail QP** — which does
   not happen at 16. Benign for correctness (RC order per QP) but it silently
   couples head-credit AMOs behind tail payloads on 8 of 20 channels, and it
   interacts directly with tuning G's head-credit site. The 16-vs-32 A/B is
   free and may be worth more than C or G — but it is **unmeasurable today**
   (both arms hang, §4.8).
4. **The 5% ship gate for C is inside the noise band** — use ≥5 runs/arm and
   medians when the campaign resumes.

### 4.8 ROOT CAUSE of the HT hang: harness buffer under-provisioning

**Symptom.** Every internode-normal run hung in the first fused dispatch, `rc=124`,
with `ccs` Engine reset -> `Schedule disable failed to respond` ->
`xe_guc_exec_queue_lr_cleanup`, at BOTH the target (E=384/topk=6/H7168) and the
cheap guard (E=8/topk=2/H7168) shape.

**Actual cause.** `tests/docker-2node-v2/run.sh` defaulted to **128 MiB NVL /
64 MiB RDMA / 256 MiB symmetric**, which only fits the `HIDDEN=1024` smoke shape.
At `HIDDEN=7168` `launch_fused_dispatch` spins forever in its first chunk. No
error is raised, so the GuC watchdog fires and it surfaces as a timeout plus
`ccs` resets — **indistinguishable from a lost doorbell or a HW wedge.** Raising
the defaults to **512 MiB / 512 MiB / 2 GiB** makes both shapes pass with zero env
overrides.

Asymmetry worth knowing: **under-sizing hangs silently; over-sizing fails loudly**
(`ishmem_align failed` when the symmetric heap does not exceed RDMA+NVL).

**Control that proves it is not a code or archive regression.** The Aug-28
known-good commit `d5e7d3b`, built against the exact Aug-28 archive `f8af72cd`,
**hung identically** under the old buffer defaults. So neither the LL session's
commits nor the newer iSHMEM archive were responsible.

**GOLDEN RULE:** *an internode-normal hang at large HIDDEN is under-provisioned
`DEEP_EP_RDMA_BYTES`/`DEEP_EP_NVL_BYTES` until proven otherwise.* It bisects in one
run: drop to `HIDDEN=1024`; if that passes, it is a sizing problem, not code and
not hardware. A size check in `launch_fused_dispatch` would turn days of debugging
into one error message — recommended follow-up.

#### 4.8.1 Secondary real bug found on the way (fixed, correct hygiene, NOT the hang)

`deep_ep/buffer.py` has an LL branch and a normal branch. The LL session raised the
QP clamp `16 -> 128` in **both**, but for HT `num_qps_per_rank = max(num_sms,...) = 24`,
so `min(24,128) = 24 -> pow2 -> ISHMEM_IBGDA_QPS_PER_PE = 32` while the kernels clamp
their own view to `[1,16]` in `fused_qps_per_pe_env()` (`internode.cpp:543`) and
`fused_qps_per_pe()` (`internode_dispatch_fused.inc:175`). iSHMEM therefore provisioned
**32 QPs+CQs per PE that the kernel can never address**. Restored to 16 (LL stays 128),
with a comment recording the **lockstep invariant**: those three sites move together or
not at all. Measured afterwards: `QPS_PER_PE` 8/16/32 are all within noise and 32 does
**not** hang — confirming this was wasted resources, not the hang mechanism.

#### 4.8.2 Two harness bugs that manufactured fake "hangs"

1. **Unanchored container check.** `ensure_up()` used
   `docker ps | grep -q "$NODE0_CONTAINER"` — a **substring** match. A leftover
   `deepep-v2-node0-stuck-zombie` from a previous wedged run matches it, so `ensure_up`
   believed the stack was already running, skipped `up` entirely, and every downstream
   step then failed with the badly misleading `no IB devices visible` (the containers did
   not exist at all). Fixed to exact match (`grep -qx`) on **both** nodes.
2. **RDMA probe with no retry.** The `ibv_devices` probe sampled once; on a cold container
   it can read empty before the userspace RDMA stack is ready. Now retries for 30 s.

3. **Orphaned ranks are not reaped by `timeout`.** `timeout N` kills only the OUTER
   `mpirun` of the `docker exec`; the in-container `mpiexec.hydra` and the 4 Python ranks
   **survive** holding GPU contexts, IBGDA QPs and the symmetric heap. Stacking runs then
   hang and spawn more orphans — a self-inflicted cascade that once reached **21 live
   ranks (~5 concurrent jobs) across two repos**. **Gate every run on
   `pgrep -fc test_internode.py == 0` and no `deepep-*` containers, BEFORE and AFTER.**

#### 4.8.3 Three falsified theories — do not re-enter

**(a) "The patched true-UC `xe` wedges the HT path."** FALSE. Session `e1a2ae34`'s
`journalctl -b -5` shows **19,032 `NEEDS_UC` lines on Aug 28 — the day the tests passed
6/6**, plus 640 `ccs` resets in that same boot. Force-UC was active *while everything
worked*. The `ccs` resets seen on kill are a **consequence** of killing a spinning kernel,
not a cause. No reboot and no `xe.ko` swap is warranted.

**(b) "The pre-knob `.so` also hangs, so the code is exonerated."** INVALID: that binary
was also built from `internode_ll` @ `cb3261f`, so both arms carried the same shared-code
delta. It exonerated only knobs C and G.

**(c) "The QP clamp caused the hang."** FALSE (my own hypothesis). It was a real bug and
worth fixing, but `QPS_PER_PE=32` runs clean.

**Method note.** All three dead-ends share one failure mode: reasoning from a *plausible*
mechanism instead of a *discriminating* experiment. The decisive test was trivially cheap
and available the whole time — run the known-good Aug-28 tree + archive under the current
harness. It hung, which falsified every code-level theory in one run.

### 4.9 RESULT: tunings C and G are REVERTED to default OFF (they break at scale)

**Bottom line: do not enable these. They are a correctness bug, not a win.**

C (`DEEP_EP_FUSED_DROP_FENCE`) and G (`DEEP_EP_FUSED_FLAG_NBI_AMO`) measured
**-7% isolated dispatch** as a pair and were briefly shipped ON. That validation
was run **only at `num_tokens=32`**. A token sweep at E=384/topk=6/H=7168 then
showed:

| tokens | knobs ON | knobs OFF |
|---|---|---|
| 32 / 64 / 128 / 256 | PASS | PASS |
| **1024** | **HANG** | **PASS** |
| 2048 / 4096 | not reached | PASS |

Same shape, same build, same clean HW — only the knobs differ. The hang drives a
GuC watchdog timeout -> `ccs` engine reset, and twice that knocked GPU
`0000:1f:00.0` **off the PCI bus** entirely.

**Mechanism.** The flag AMOs are **flow control**, and the two sites are not
equivalent:

| site | QP | ordered behind payload? |
|---|---|---|
| tail AMO (`internode_dispatch_fused.inc:794`) | `channel_id % qps_per_pe` | **yes** — same QP as the put, RC strict order |
| head-credit AMO (`:1034`, combine `:1169`) | `(channel_id + num_channels) % qps_per_pe` | **no** — different QP, and it has no fence of its own |

The head-credit AMO is the signal telling the remote sender its receive slots
were freed. The **blocking** AMO's CQE poll was its ONLY completion guarantee;
the NBI variant posts the WQE and never confirms it. Below ~256 tokens the
receive buffer never fills, the sender never actually waits on credit, and the
missing guarantee is invisible. At >=1024 tokens the buffer wraps and credit
becomes load-bearing: a credit update that is posted but never completed leaves
the sender spinning forever.

So C's premise ("same QP, RC-ordered, fence redundant") is sound for the **tail**
AMO but **false for the head-credit AMO**, and G removed the completion guarantee
that was silently covering the gap.

**To re-enable**, the head-credit path needs either a completion/`quiet` or a
same-QP mapping — plus a full token sweep to 4096. Both knobs are retained,
default OFF, with a CORRECTNESS GATE comment at `csrc/xpu/internode.cpp:88`.

**METHOD LESSON (the important one).** A perf tuning validated at one shape is
not validated. `num_tokens=32` never exercises the flow-control path that the
tuning modifies, so the A/B was structurally incapable of detecting the bug — it
was measuring a code path whose correctness precondition it never triggered.
**Any change touching flow control, credit, or completion must be swept across
the token range before it ships**, because the failure mode is a hang that looks
exactly like a hardware fault (and can genuinely damage the PCI state).

### 4.10 Validated internode-normal token sweep (knobs OFF, E=384, topk=6, H=7168)

7/7 PASS, zero hangs, zero `DEVICE_LOST`, all 4 GPUs healthy afterwards. Times
are per-iteration minima, microseconds.

| tokens | dispatch(iso) | combine(iso) | round-trip |
|---|---|---|---|
| 32 | 961.7 | 947.9 | 1431.5 |
| 64 | 1105.2 | 1006.4 | 1716.5 |
| 128 | 1312.2 | 1532.7 | 2466.4 |
| 256 | 1687.9 | 1772.1 | 3039.1 |
| 1024 | 3355.0 | 3076.1 | 5877.0 |
| 2048 | 5356.8 | 4464.0 | 9294.1 |
| 4096 | 9786.3 | 6972.2 | 16290.1 |

Scaling: 32 -> 4096 is a 128x token increase for only ~10x round-trip, i.e. the
small shapes are **latency-bound** (a ~1.4 ms floor dominated by fixed
per-iteration cost) and only past ~1024 tokens does the kernel become
throughput-bound, where time grows roughly linearly (2048 -> 4096 is 1.75x for
2x the tokens). Dispatch overtakes combine at large shapes (9786 vs 6972 at
4096), so **dispatch is the right optimisation target — but only with a fix that
survives the whole sweep.**

Buffer sizing used: 512 MiB NVL / 512 MiB RDMA / 2 GiB symmetric up to 1024
tokens; 1 GiB / 1 GiB / 3 GiB for 2048 and 4096. Note a 4 GiB symmetric heap
aborted in NEO (`drm_neo.cpp:265`) — the heap must fit device memory alongside
the model tensors, so bigger is NOT always safer here.
