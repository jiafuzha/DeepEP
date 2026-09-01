# Internode Low-Latency (LL) Kernel — XPU/SYCL Design

Source: `csrc/xpu/internode_ll.cpp`
Reference: `csrc/cuda_kernels/internode_ll.cu`
Target: Intel BMG (Arc Pro B60, `max_compute_units == 160`), iSHMEM/IBGDA over mlx5.

This document describes the SYCL port of the DeepEP internode low-latency
dispatch/combine path, why it is structured as **four sub-kernels** (two per
direction) instead of two fused kernels, and which parameters govern its scaling
behaviour as `num_tokens`, `num_topk`, and `num_experts` grow.

---

## 1. What the LL path does

Low latency is the decode-time all-to-all used by MoE inference. Each rank holds
`num_tokens` tokens; every token selects `num_topk` experts out of `num_experts`
global experts (`num_local_experts = num_experts / num_ranks` live on each rank).

- **Dispatch** sends every token to the ranks that host its top-k experts, and on
  the receiving side lays the arriving tokens out per local expert.
- **Combine** does the reverse: each expert's output token is sent back to the
  rank that originally owned it, then reduced (weighted by the top-k gate weights)
  into one output row per original token.

The transport is **GPU-initiated RDMA (IBGDA)** via iSHMEM: the GPU kernel writes
WQEs and rings the NIC doorbell directly — no host proxy on the critical path.

### 1.1 Unified message layout and FP8-on-send

Dispatch casts **BF16 → FP8 on the send side** (when FP8 is requested), so only
`hidden` payload bytes + scales traverse the NIC instead of `2*hidden`. Every
dispatch message uses a single CUDA-parity layout, sized at the BF16 maximum so
one symmetric allocation serves both dtypes (`deep_ep_xpu.cpp::get_low_latency_buffer_layout`,
`internode_ll.cpp::make_layout`):

```
[ int4 header (16B) ][ payload: FP8 hidden bytes OR BF16 hidden bytes ][ FP8 scales (fp32) ]
  header[0] = source token index (src_idx)
```

Combine sends raw BF16 hidden rows (no cast); reduction happens after receipt.

### 1.2 Work-group geometry (`num_experts` may exceed the SM count)

The CUDA host code splits an oversized expert count across **warp groups** inside
a fixed 1024-work-item WG (`num_warp_groups = ceil(num_experts / num_device_sms)`,
`num_warps_per_group = 32 / num_warp_groups`). That shape is not reproducible on
BMG, because a warp-group-scoped `bar.sync` has no reliable SYCL equivalent (§4).

The XPU port therefore takes the opposite decomposition: **`num_warp_groups` is
forced to 1 and the grid is allowed to over-subscribe the SMs**, so one work-group
still owns exactly one channel/expert and the whole-WG `sycl::group_barrier`
remains the correct substitute for CUDA's `bar.sync`. `E > SMs` is supported by
shrinking the *work-group*, not by subdividing it:

| Quantity | Formula | E=8 (fits) | E=384 (over-subscribed) |
|---|---|---|---|
| `num_warp_groups` | forced `1` | `1` | `1` |
| `num_warps_per_group` | `ll_num_warps(num_topk, num_experts, num_device_sms)` | `32` | `8` |
| `num_warps` (sub-groups/WG) | `num_warp_groups * num_warps_per_group` | `32` | `8` |
| `wg_size` | `num_warps * 32` | `1024` | `256` |
| `num_sms` (commit-gated grid) | `num_experts` | `8` | `384` |
| `send_wgs` | `min(num_sms, num_device_sms)` | `8` | `160` |

`ll_num_warps()` keeps CUDA-parity 32 warps while the grid fits in the CUs and
drops to **8** once `num_experts > num_device_sms`, then rounds down to a power of
two and enforces `num_topk < num_warps <= 32`. `DEEP_EP_LL_NUM_WARPS` overrides.

**Why work-group *size*, not wave *count*, is the lever.** The grid is
`num_experts` work-groups of `num_warps*32` work-items. Once `E` exceeds the 160
CUs the WGs over-subscribe, and a 1024-work-item WG throttles occupancy: the
scheduler cannot pack enough of them per CU to hide the RDMA latency. Measured on
160 CUs at `topk=2, nt=32` (µs), the cost is flat in wave *count* but strongly
dependent on WG *size*:

| E | 32 warps | 16 warps | 8 warps | 4 warps |
|---|---|---|---|---|
| 384 | 820 | 718 | **578** | 590 |
| 320 | 741 | 744 | **553** | — |
| 8 | **323** | 325 | 348 | — |

An E-sweep showed cost rising **linearly in E with no step at the 160/320
wave boundaries**, which refutes wave-quantization/grid over-subscription as the
cause and points at per-WG occupancy. At `E <= SMs` the trend reverses (fewer
warps only costs parallelism), which is why the heuristic is conditional rather
than a flat "always 8".

`send_wgs` is additionally clamped to `num_device_sms`: a send grid wider than the
CUs only adds launch overhead, saving `(num_experts - num_device_sms)` work-groups
of pure overhead per dispatch at E=384.

---

## 2. Sub-kernels

Both directions are split at the point where the CUDA kernel calls
`cg::this_grid().sync()`. The **kernel boundary replaces the in-kernel grid
barrier** (see §3 for why). The four sub-kernels:

```
dispatch_bf16:  LLDispatchSendKernel  ──▶  LLDispatchRecvKernel
combine_bf16:   LLCombineSendKernel   ──▶  LLCombineReduceKernel
```

### 2.1 `LLDispatchSendKernel` — cast + put (commit-gated)

- **Grid:** `send_wgs` work-groups (default `min(num_experts, num_device_sms)`,
  tunable via `DEEP_EP_LL_SEND_WGS`), `wg_size = num_warps*32` (1024 at `E ≤ SMs`,
  256 once `E > SMs` — §1.2).
- Each WG strides over tokens (`t = sm_id; t < num_tokens; t += send_wgs`). For
  each token **all `num_warps` warps cast the hidden row cooperatively** (`cast_token_fp8_strided`
  for FP8, `coop_copy_bytes` for BF16) into the token's send-staging slot, write
  the `src_idx` header, then a **device-scope release fence** + whole-WG barrier
  makes the bytes NIC-visible.
- Warps `w < num_topk` then deliver the finished message to `topk_idx[t, w]`'s
  expert slot: `slot_counter[expert].fetch_add(1)` picks the destination slot, and
  `ishmemx_putmem_nbi_subgroup(..., force_db=false)` (self rank → `coop_copy_bytes_store_uc`).
- **No counter warp, no finish-counter, no grid barrier.** Counting is deferred to
  the recv kernel; the kernel boundary provides the ordering the finish-counter
  used to provide. `force_db=false` leaves the last doorbell batch deferred — with
  the fence now dropped by default (§5.2), the recv kernel's flag AMO
  (`ishmemx_long_atomic_add_qp`) is what flushes it: it rings the doorbell
  unconditionally at `pi = wqe_idx + 1`, publishing every earlier `force_db=false`
  WQE on that QP; RC in-order delivery still lands the count
  flag after the payloads on the target QP.
- Block 0 additionally zeroes the opposite-parity `rdma_recv_count` slots (CUDA
  `next_clean`).

### 2.2 `LLDispatchRecvKernel` — count flag + poll + copy

- **Grid:** `recv_wgs = ceil(num_experts / pack_channels)`, `wg_size = num_warps*32`.
  WG `sm_id` owns channels `responsible_expert_idx = sm_id * pack_channels + ch`.
- **Two-pass channel structure (required, not just an optimization):** the WG runs
  Phase A for *all* its channels, then Phase B for *all* of them — two separate
  `for (ch...)` loops rather than one loop doing post-then-wait per channel. With
  `pack_channels > 1` the single-loop form **deadlocks**: channel 0 would block in
  its flag-wait before channel 1 has posted its count flag, and the peer WG waiting
  on channel 1 would never be released. Keeping the passes separate also stops each
  channel's poll latency from serializing behind the previous channel's, which is
  where most of the E=384 win comes from (§5.6).
- **Phase A (count post):** warp 0 histograms `topk_idx == responsible_expert` and
  posts the count flag `(-count-1)` on QP `le`. Because the send kernel has fully
  exited, all payload puts are already committed. Historically
  `ishmemx_fence_qp(dst_rank, le)` established a **QP-scoped ordering fence** here;
  it is now **skipped by default** (`DEEP_EP_LL_DROP_FENCE=1`) because it is
  redundant by construction — see §5.2 — and the
  subsequent RC-ordered `ishmemx_long_atomic_add_qp` is guaranteed by RC in-order
  delivery to land the flag *after* every prior payload put on the same QP. The
  self channel (`dst_rank == rank`) uses a system-release fence + `uc_store`.
  Fence rather than quiet: the sender does **not** need to wait for the CQE
  completions of the payload puts — RC ordering makes the AMO land after them on
  the receiver regardless, so `fence_qp` (post-side ordering only, no local
  wait-for-drain) is strictly cheaper than the earlier `quiet_qp`, and was
  measured to reduce per-iteration latency without affecting correctness.
- A whole-WG `group_barrier` publishes the **self-channel** count (written by this
  same WG in Phase A) to Phase B. Cross-rank channels are remote (landed via NIC
  AMO) and need no barrier.
- **Phase B (poll + copy):** for each source rank, sub-warp 1 polls `rdma_recv_count`
  until the flag is non-zero, computes the receive offset via
  `packed_recv_count.fetch_add(count)`, and the warp group's sub-warps copy the
  arrived messages (payload + FP8 scales) into the packed receive buffers in
  parallel.

### 2.3 `LLCombineSendKernel` — per-token send + flag (commit-gated)

- **Grid:** `send_wgs_c = ceil(num_experts / pack_channels)`, `wg_size = num_warps*32`.
  WG owns channels `responsible_expert_idx = sm_id * pack_channels + ch`, and uses
  the same **two-pass** (post-all, then wait-all) structure as §2.2 for the same
  deadlock-avoidance reason.
- Block 0 zeroes the opposite-parity `combine_flag` slots (CUDA `next_clean`) and
  releases `atomic_clean_flag` (`clean_flag`, borrowed from the unused
  `slot_counter` cell) so flag posts wait for the clean.
- For its expert's received tokens, sub-warps copy each hidden row into per-token
  symmetric staging and `ishmemx_putmem_nbi_subgroup` it to the destination's original
  token slot (`src_idx`); self rank writes directly into local `combine_data`.
- After a warp-group barrier, sub-warp 1 posts the arrival flag `+1` on QP `le`
  (self → `uc_store`; remote → `atomic_add_qp`, with the preceding `fence_qp`
  dropped by default per §5.2 — RC in-order delivery
  guarantees the AMO lands after every prior payload put on the same QP without
  waiting for their CQE completions), and sub-warp 0 waits — **in the second
  channel pass** — on the incoming
  arrival flag for its channel. The kernel exit is the grid sync: once it
  returns, every flag has been observed and all `combine_data` is globally
  visible.

### 2.4 `LLCombineReduceKernel` — weighted top-k reduce (token-parallel)

- **Grid:** `reduce_wgs = ll_consume_wgs(...)` (large, oversubscribable —
  `min(4*CU, 512)` capped by work-item count, `DEEP_EP_LL_REDUCE_WGS` override),
  `wg_size = 512`.
- Grid-strides over `reduce_work = num_combined_tokens * hidden` elements. Each
  work-item accumulates `sum_k weight_k * combine_data[expert_k row][h]` over the
  token's top-k experts and writes BF16 output elements. Pure **cached local
  reads** — no IBGDA, no commit gate — so the grid scales freely with token count
  (mirrors CUDA sizing its combine grid to `num_combined_tokens / num_recv_per_sm`).
- **Vectorized (`DEEP_EP_LL_REDUCE_VEC=8`, DEFAULT — see §5.8).** Each work-item
  owns **8 contiguous BF16 elements (16 B)** instead of 1: one `uint32x4` load per
  top-k source per 8 outputs, one `uint32x4` store, and the per-token `topk_idx`
  (int64) / `topk_weights` (fp32) lookups amortized over 8 elements instead of
  re-read per element. BF16→float is a raw bit shift (`b<<16` / `b & 0xFFFF0000`),
  which is bit-exact, so the result is **bitwise identical** to the scalar path.
  `DEEP_EP_LL_REDUCE_VEC=1` restores the scalar path; it is also the automatic
  fallback when `hidden % 8 != 0`.

---

## 3. Why sub-kernels instead of one fused kernel

The original XPU port fused each direction into a single kernel of `num_experts`
big (1024-work-item) work-groups joined by an **in-kernel `GridBarrier`** plus, for
dispatch, a device-scope **finish-counter** to order payload-before-flag. Two hard
constraints made that fused design a scaling bottleneck.

### 3.1 The GridBarrier caps co-resident work-groups (~24)

A device-wide barrier inside a kernel only works if **every participating
work-group is simultaneously resident** on the GPU — otherwise a resident WG spins
forever waiting for one that has not been scheduled, and on BMG that trips the GuC
hang-check watchdog → GT reset → `DEVICE_LOST`. With `wg_size = 1024`, only a
small number of WGs (empirically ~24, hence the constraint the user called out) can
co-reside. The **token-parallel phases** (combine reduce over `num_combined_tokens
* hidden` elements; dispatch recv-copy) were therefore pinned to `num_experts`
(=8) work-groups and could not grow with the token count — the reduce ran ~224
grid-stride iterations per work-item at 256 tokens.

(The `E > SMs` path shrinks the WG to 256 work-items, which raises the co-residency
ceiling proportionally — but not to the `num_experts`=384 the grid would need, so
the conclusion below is unchanged.)

A **kernel boundary is a full device barrier with global memory visibility** that
imposes **no co-residency requirement**: the send kernel drains completely (all
WGs retire), then the consume kernel launches with a fresh, arbitrarily large grid.
That is exactly what CUDA does implicitly by sizing its combine grid to the token
count. Splitting lets the consume kernels oversubscribe (`ll_consume_wgs` →
`min(4*CU, 512)` WGs) while the commit-gated send stays small.

### 3.2 Sends are co-residency-bound anyway — splitting cannot help them

`ishmemx_putmem_nbi_subgroup` has a **per-QP ordered commit gate**: the sub-group
holding `commit == base` must be resident and publish before the next producer can
advance (a monotonic watermark, the analogue of CUDA's `ready_head` CAS). So the
**send grid must not exceed resident capacity** regardless of splitting — a
spinning producer would wedge the QP and trip the watchdog. Measurement confirms
this: `DEEP_EP_LL_SEND_WGS=16` vs `8` at 256 tokens is identical (~5250 µs) because
per-QP put throughput is fixed by the gate. Hence the send kernels default to
`num_sms` and the win comes entirely from the **consume** side.

> **Refinement (see §3.2.1).** "Splitting cannot help" applies to the send **grid**
> (more work-groups = more *non-co-resident* producers, which is exactly what the
> gate stalls on). It does **not** apply to adding more posting warps *inside* an
> already-resident work-group — those are precisely the producers the gate can
> drain. `DEEP_EP_LL_SEND_TOK_SPLIT` exploits that and is worth 1.05–1.08×.

### 3.2.1 Send-side token split (`DEEP_EP_LL_SEND_TOK_SPLIT`) — SHIPPED

**Problem.** `LLDispatchSendKernel` originally walked **one token per whole-work-group
iteration**:

```
for (t = sm_id; t < num_tokens; t += send_wgs) {
    <all 32 warps cooperatively cast token t into rdma_x[t]>
    group_barrier(whole WG);
    if (warp_id < num_topk) <put token t to topk_idx[t][warp_id]'s expert slot>
}
```

At the usual `num_topk = 2` that leaves **30 of 32 warps idle for the entire put
phase**, and the whole WG is serialized on one token at a time. This is the direct
analogue of the normal-path combine warp starvation (`internode_normal_design.md`
§6.1.1).

**Fix.** Split the WG into `tok_split` casting **teams** of `team_warps =
num_warps/tok_split` warps; team `g` casts token `base + g` and then **issues that same
token's top-k puts**, warp `team_warp` taking `k = team_warp, team_warp+team_warps, …`.
Live put warps rise from `num_topk` to `tok_split * min(team_warps, num_topk)`.

Because a team both produces and consumes its own token, teams are fully independent
and the cap is the warp count itself (`tok_split <= num_warps`) rather than
`num_warps/num_topk`. The barrier stays a **whole-WG** `group_barrier` — every warp
reaches it every iteration, since the loop bound depends only on the WG-uniform `base`
— so this does *not* need the warp-group-SUBSET barrier of §4; see §3.2.2 for why the
subset barrier is both unavailable *and* unnecessary here.

**Why it is correct.** Dispatch slots are handed out by an unordered
`slot_counter.fetch_add`, and the receiver recovers each token's identity from the
message header (`hdr[0] == src token index`) via `packed_recv_src_info`. Slot **order**
therefore carries no meaning — the send grid already stripes tokens
nondeterministically — so reordering puts within the WG is a no-op for correctness.
(Contrast the normal path, where the consumer uses a precomputed `combined_nvl_head`
slot map and slot assignment must be preserved exactly.)

**Auto default.** `tok_split = clamp(num_tokens / send_wgs, 1, num_warps / 4)`.
Two bounds: (a) splitting further than there are tokens to feed the teams leaves teams
idle; (b) a team must keep at least **4 sub-groups**, because a token's cast is a
hidden-sized strided quantise and casting it with too few sub-groups starves the load
pipeline and dominates the iteration (see the `tok_split=32` row in §3.2.2). At
`num_warps=32, send_wgs=8` this picks `4 / 8 / 8 / 8` for `nt = 32 / 64 / 128 / 256`,
matching the per-size optimum measured in §3.2.2.

**Measured** (2-node BMG, `H=7168 TOPK=2 E=8 NUM_PROCESSES=2`, `send_wgs=8`,
dispatch+combine round-trip `avg_t` µs; `s1` == pre-change baseline):

| tokens | s1 (base) | s2 | s4 | s8 | s16 | **auto** | speedup |
|---|---|---|---|---|---|---|---|
| 32  | 339.1  | 330.4  | **319.9** | 337.7  | 355.0  | 323.3 (split=4)  | **1.05×** |
| 64  | 584.3  | 559.4  | 549.0  | **545.6** | 561.9  | 544.5 (split=8)  | **1.07×** |
| 128 | 1057.6 | 1008.9 | 982.8  | 985.8  | **980.0** | 979.3 (split=16) | **1.08×** |
| 256 | 2240.5 | 2147.9 | 2114.9 | 2107.7 | **2101.6** | 2103.7 (split=16)| **1.06×** |

Bandwidth over the same points: 3.46→3.63, 4.39→4.71, 5.06→5.46, 4.87→5.19 GB/s.
`s1` reproduces the pre-change baseline to <0.5%, confirming the `tok_split == 1` path
is a faithful no-op. Correctness (in-test hash + `calc_diff` asserts) passed on all 26
runs, including 3 independent repeats at 128/256.

**Stacking with `ISHMEM_IBGDA_DB_BATCH_SIZE=8`.** The token split removes *cast/put*
serialization; batching doorbells removes *per-put doorbell* overhead. They are
independent levers and compose additively (avg_t µs, same shapes):

| tokens | split=1 dbb=0 (base) | split=1 dbb=8 | auto dbb=0 | **auto dbb=8** | combined |
|---|---|---|---|---|---|
| 32  | 339.1  | 332.7  | 323.3  | **312.1**  | **1.09×** |
| 64  | 584.3  | 538.0  | 544.5  | **516.9**  | **1.13×** |
| 128 | 1057.6 | 984.8  | 979.3  | **935.1**  | **1.13×** |
| 256 | 2240.5 | 2031.0 | 2103.7 | **1932.2** | **1.16×** |

Bandwidth at `auto dbb=8`: 3.76 / 4.96 / 5.72 / 5.65 GB/s (vs 3.46 / 4.39 / 5.06 /
4.87 baseline). Isolated contributions are −4.7…−7.4% (split) and −1.9…−9.3% (dbb=8),
summing to −8.0…−13.8% together — i.e. neither lever masks the other, consistent with
them attacking different serialization points. All runs passed the in-test correctness
asserts.

`ISHMEM_IBGDA_DB_BATCH_SIZE` is a **harness/runtime** env var (default `0` in
`tests/docker-2node-ll-v2/run.sh`), not a DeepEP knob. `8` is recommended for the
32–256-token LL regime. Note the separate scale constraint recorded in the repo
instructions: at ≥2048 tokens `0` deadlocks and `64` is required — so the optimum is
token-count dependent and the harness default was left at `0` rather than changed
globally.

**Large token counts (1024 / 2048 / 4096).** Both levers keep working at scale; the
auto split saturates at `tok_split=16` (all 32 put warps live) for every size ≥128.
Run-to-run `avg_t` variance is severe at 1024 (a repeated baseline gave 14066 vs 10056 µs
while `min_t` reproduced to 1.3%), so `min_t` is the reliable metric here and both are
listed. Values are rank-0 µs; baseline = `split=1 dbb=0`, best of two repeats.

| tokens | metric | base | auto dbb=0 | **auto dbb=8** | auto dbb=64 | total |
|---|---|---|---|---|---|---|
| 1024 | `min_t` | 8835.3  | 8205.2  | **7644.9**  | 7529.8  | **1.16×** |
| 1024 | `avg_t` | 10056.4 | 9092.5  | **8561.5**  | 11411.1 | **1.18×** |
| 2048 | `min_t` | 17746.9 | 16564.6 | **15221.5** | 15075.7 | **1.17×** |
| 2048 | `avg_t` | 19470.5 | 17923.0 | **15879.6** | 16468.2 | **1.23×** |
| 4096 | `min_t` | 36250.2 | 33806.7 | **30226.3** | 30209.8 | **1.20×** |
| 4096 | `avg_t` | 36600.7 | 35320.8 | **31099.1** | 34041.6 | **1.18×** |

Bandwidth: 4.41 / 4.56 / 4.86 GB/s baseline → **5.18 / 5.59 / 5.72 GB/s** at `auto dbb=8`.
The token split alone is worth a consistent ~1.07–1.08× on `min_t` at all three sizes,
so it generalizes beyond the 32–256 regime.

Two corrections to the older guidance recorded in the repo instructions:

* **`ISHMEM_IBGDA_DB_BATCH_SIZE=0` does not deadlock at ≥2048 tokens** on this build —
  all runs passed. What it *does* show is a very long tail (`max_t` 90390 µs vs
  `min_t` 33807 µs at 4096). The old "0 deadlocks at scale" note appears to describe a
  superseded iSHMEM archive.
* **`8` beats `64` at every size**, contrary to "64 is required at ≥2048". The two tie on
  `min_t` (within 0.1–1.5%), but `64` has markedly worse tail/average behaviour
  (`avg_t` 11411 vs 8562 µs at 1024, 34042 vs 31099 µs at 4096). So `8` is the single
  best setting across the whole 32–4096 range, which removes the token-count-dependence
  that was the reason for not changing the harness default.

**Buffer sizing at ≥1024 tokens.** The harness default `ISHMEM_SYMMETRIC_SIZE=268435456`
(256 MiB) is too small: at `nt=1024, H=7168, E=8` the LL buffer alone is 477 MB and
allocation fails with `RuntimeError: ishmem_align failed for 477102336 bytes`. Demand is
linear in `num_max_dispatch_tokens_per_rank`; `ISHMEM_SYMMETRIC_SIZE=4294967296` (4 GiB)
covers 1024–4096. This is a pre-existing harness default, unrelated to the token split.

**Why the split gain is only ~6% (vs 32–55% on the normal path).** LL at these sizes is
dominated by RDMA/QP serialization, not by warp throughput: with `E=8` over 4 ranks
`num_local_experts = 2`, so `ISHMEM_IBGDA_QPS_PER_PE = 2` (§5.4) — only two QPs carry
all traffic, and the QP count is structurally capped because a combine payload must
ride its expert's own QP for RC flag-after-payload ordering. Filling the idle put warps
removes the *cast/put serialization* but cannot widen the QP bottleneck. Round-trip time
still scales ~linearly with token count with roughly flat bandwidth, which is the
signature of a serialization limit rather than a fixed-latency floor.

**Not applicable to the other sub-kernels.** `LLCombineSendKernel` already stripes
tokens across all 32 sub-warps (`token_idx = begin + sub_warp_id; += num_warps_per_group`),
and `LLDispatchRecvKernel` / `LLCombineReduceKernel` are likewise fully token-parallel
(the reduce additionally oversubscribes to `min(4*CU, 512)` WGs). `LLDispatchSendKernel`
was the only sub-kernel with idle warp slots.

### 3.2.2 Per-team sub-group-SUBSET barrier — ATTEMPTED, DOES NOT WORK (negative result)

BMG *can* express a sub-group-SUBSET barrier via SPIR-V NamedBarrier (see
`named_barrier_usage.md`, and the working use in `internode_combine_fused.inc`), so the
obvious refinement to §3.2.1 is to replace the whole-WG `group_barrier` between cast and
put with a **per-team** barrier of `team_warps` sub-groups. Motivation: a whole-WG
barrier couples every team to the *slowest* team each iteration, and put latency is
highly variable because `ishmemx_putmem_nbi_subgroup` waits on the IBGDA per-QP ordered
commit gate.

This was implemented and measured. **It does not work, and it would not have helped
anyway.** Two independent findings:

**(a) The named barriers cannot be instantiated in this kernel — it faults at launch.**
Merely materialising the `named_barrier_init()` handles makes `LLDispatchSendKernel`
die with `SIGSEGV` on the very first dispatch, *even when the barrier is never taken*
(`DEEP_EP_LL_SEND_TEAM_BARRIER=0`) — the crash is immediately after the kernel-config
log line, before any progress. This was verified to be caused by the handles alone:
compiling the identical kernel with the handles `#if`-ed out (everything else unchanged)
passes and performs normally. Both a runtime arrive-count and a compile-time literal
arrive-count fault identically, and the build emits no `Stack call has been detected`
warning, so this is *not* the bf16-conversion collision documented in
`named_barrier_usage.md` — that one was fixed here by `ll_bf16_to_float()` (a
bit-manipulation `bit_cast<float>(uint32_t(bits) << 16)` instead of the
`sycl::ext::oneapi::bfloat16` conversion operator, which outlines to
`__devicelib_ConvertBF16ToFINTEL`). Something else in this kernel — most likely the
FP8 (`c10::Float8_e4m3fn`) conversion or the iSHMEM put path — still outlines and
collides with the `NBarrierCnt` kernel attribute. The code is kept behind
`DEEP_EP_LL_TEAM_NB` (off) for a future retry against a newer IGC.

**(b) The barrier is not the bottleneck — the cast width is.** The cleanest possible
version of the same idea needs *no* barrier at all: at `tok_split = num_warps` each team
is a **single sub-group**, which is lock-stepped, so the same warp casts and puts its
token with zero cross-warp synchronisation and teams are completely decoupled. That
configuration is by far the **worst** measured:

| `tok_split` | `team_warps` | cast→put barrier | nt=32 | nt=64 | nt=128 | nt=256 |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 8 | whole-WG | **314.3** | **510.1** | 939.8 | 1952.7 |
| 8 | 4 | whole-WG | 324.3 | 516.4 | **927.3** | **1927.2** |
| 16 | 2 | whole-WG | 351.5 | 539.1 | 943.0 | 1948.0 |
| 32 | 1 | **none** | 551.2 | 768.9 | 1198.6 | 2128.8 |
| *auto* | *4/8/8/8* | whole-WG | 315.4 | 516.8 | 929.7 | 1921.3 |

(`avg_t` µs, round trip, 2-node BMG, `H=7168 TOPK=2 E=8 NUM_PROCESSES=2`,
`ISHMEM_IBGDA_DB_BATCH_SIZE=8`.)

Removing the barrier entirely costs **1.1–1.75×**, i.e. narrowing the cast team hurts
far more than any barrier saving could recover. Since a per-team barrier is only
*possible* for `tok_split <= 8` (the IGC named-barrier budget is ~8 handles) and the
optimum already sits at `tok_split = 4…8` where teams are 4–8 warps wide and the
whole-WG barrier is cheap, there is no configuration in which the subset barrier could
pay for itself.

**Decision:** keep the whole-WG `group_barrier`. Record this as closed.

### 3.3 The kernel boundary also removes the finish-counter

The fused dispatch used a per-expert finish-counter (each send bumped it, the
count-send waited for `2*TAG`) purely to guarantee *all payloads for expert E are
posted before E's count flag*. The kernel boundary provides that for free: when
`LLDispatchSendKernel` exits, **every** put is committed, so `LLDispatchRecvKernel`
can `fence_qp` + post the flag with no finish-counter. This uses the iSHMEM
**fence_qp** primitive (`ibgda(fence_qp): publish full committed prefix on shared
QP without waiting for CQE`): fence rings the doorbell to the ordered
`nic_wq_commit` watermark caught up to the current claim counter — flushing the
`force_db=false` deferred doorbells the send kernel left, without ever
doorbelling an unwritten slot — but, unlike `quiet_qp`, does **not** locally spin
on the completion queue for the outstanding puts to retire. RC in-order
delivery on the target QP guarantees the subsequent RC AMO (count flag) lands
after every prior payload put anyway, so waiting for CQE completion is
unnecessary. Using `fence_qp` instead of `quiet_qp` therefore preserves
correctness while removing the drain wait from the flag-post critical path
(measured net win, no correctness regression).

### 3.4 Net effect

Measured 2-node BMG, H7168 / topk2 / exp8, all PASS, tight tails:

| tokens | fused | split (dispatch+combine) | win |
|---|---|---|---|
| 32 | 933 µs | 756 µs | −19% |
| 128 | 3072 µs | 2350 µs | −23% |
| 256 | 7030 µs | 5275 µs | −25% |

---

## 4. BMG barrier limitation (context for the split)

CUDA uses **warp-group-scoped named barriers** (`bar.sync warp_group_id+N,
num_warps_per_group*32`) so only the warps of one warp group rendezvous. On BMG
neither the SPIR-V `NamedBarrier` nor an SLM subset barrier nor an ESIMD hardware
named barrier is usable in this kernel: they either deadlock (sub-groups only have
guaranteed concurrent forward progress at hardware barriers) or `DEVICE_LOST` when
coexisting with the non-inlined iSHMEM RDC device library. The port therefore uses
the whole-WG `sycl::group_barrier`, which is correct **only at `num_warp_groups ==
1`** — and `num_warp_groups` is therefore forced to `1` unconditionally.

`num_experts > num_device_sms` is consequently **not** supported by reintroducing
warp groups. Instead the grid over-subscribes the SMs (one WG per expert, as
always) and the work-group is *shrunk* by `ll_num_warps` to protect occupancy
(§1.2, §5.6). This sidesteps the missing warp-group barrier entirely, so the
"make each warp group its own work-group" phase-split is no longer needed for
large `E`; it remains only a hypothetical route to sub-expert parallelism.

Note: `sycl::group_barrier` only fences at **work-group scope**, so every place the
NIC or another WG must observe a write still needs an explicit
`sycl::atomic_fence` — device scope for intra-GPU cross-WG rendezvous, **system**
scope only for genuine cross-PE/NIC paths (flag flush, `rdma_recv_count` /
`combine_flag` AMOs). This scope discipline is the F1–F5 alignment.

---

## 5. Configurable parameters affecting scaling

### 5.1 Problem-size inputs (set by the caller / model)

| Parameter | Effect as it grows |
|---|---|
| `num_tokens` (`num_max_dispatch_tokens_per_rank`) | Dominant cost driver. Lengthens the send kernels' token loop (commit-gate bound — does **not** parallelize away) and the recv-copy / reduce work (token-parallel — scales with the consume grid). Also grows every symmetric buffer linearly. |
| `num_topk` | More puts per token in dispatch (warps `0..num_topk-1` send) and more source rows per output in the combine reduce (`num_topk` reads/element). `TORCH_CHECK(num_topk + 1 <= num_warps)`. |
| `num_experts` (`num_local_experts`) | Sets `num_sms` = the commit-gated send/flag grid width and the number of QPs (one per global expert). More experts ⇒ more parallel QPs (helps send concurrency up to the co-residency limit) but also more channels to poll and more per-expert histogram/flag work. **`num_experts > num_device_sms` is supported** (§1.2): the grid over-subscribes the SMs and `ll_num_warps` shrinks the work-group to 8 warps to protect occupancy. Cost is ~linear in `num_experts`. |
| `hidden` | Bytes per message and per reduce element. Must be a multiple of 128 for FP8 scales. Larger `hidden` shifts the balance toward payload-copy/NIC bandwidth. |
| `use_fp8` / `round_scale` / `use_ue8m0` | FP8-on-send halves NIC payload vs BF16; adds cast cost on send and scale handling on recv. |

### 5.2 Environment tunables (perf knobs)

| Env var | Default | Scope | Guidance |
|---|---|---|---|
| `DEEP_EP_LL_NUM_WARPS` | `32` if `num_experts <= num_device_sms`, else `8` | dispatch/combine WG size | The **primary knob for `E > SMs`** (§1.2). Overrides the occupancy heuristic; rounded down to a power of two and clamped to `(num_topk, 32]`. Leave on AUTO — verify an override actually took effect, since an empty string parses as `0`. |
| `DEEP_EP_LL_DROP_FENCE` | `1` (fence dropped) | dispatch/combine flag AMO | Drops the per-QP fence between the payload put and the flag AMO. Redundant by construction — same QP + strict RC SQ ordering, and `ishmemx_long_atomic_add_qp` rings the doorbell unconditionally *and* blocks on its own completion, which is strictly stronger than the fence. Worth ~10% at E=384. `0` restores it. |
| `DEEP_EP_LL_PACK_CHANNELS` | auto (`ceil(num_experts / num_device_sms)`) | dispatch-recv / combine-send | Packs C channels into one WG, cutting the grid to `ceil(num_experts/C)` for identical per-channel work. Measured to have **no effect** (C=1/3/6/12 all ≈929 µs at E=384), which is the evidence that per-WG *launch* cost is not the bottleneck — WG *size* is. Retained as a diagnostic; C=1 is bit-identical to the unpacked path. |
| `DEEP_EP_LL_SEND_WGS` | `min(num_experts, num_device_sms)` | dispatch send grid | Raising it rarely helps (commit-gate bound) and **must not exceed resident WG capacity** — a spinning producer trips the GuC watchdog → `DEVICE_LOST`. Keep at default unless profiling shows cast-bound headroom. |
| `DEEP_EP_LL_SEND_TOK_SPLIT` | `clamp(num_tokens/send_wgs, 1, num_warps/num_topk)` | dispatch send: tokens in flight per WG | Number of casting teams the send WG is split into; raises live put warps from `num_topk` to `tok_split*num_topk`. See §3.2.1. The auto default is optimal at 32–256 tokens; override only to A/B. `1` restores the pre-split behaviour. |
| `DEEP_EP_LL_REDUCE_WGS` | `min(4*CU, 512)` capped by work-items | combine reduce grid (`ll_consume_wgs`) | The main scaling lever for combine. Increase toward the cap as `num_tokens`/`hidden` grow so the reduce is fully token-parallel; too small ⇒ long grid-stride loops. |
| `DEEP_EP_LL_REDUCE_VEC` | `8` (falls back to `1` if `hidden % 8`) | combine reduce | BF16 elements per work-item. `8` = 16-byte vector loads + amortized top-k metadata (bitwise identical to `1`, **5.1x faster**, §5.8). `1` = legacy scalar path. |
| `DEEP_EP_LL_REDUCE_ACQ` | `1` | combine reduce | Granularity of the system-scope acquire (cache invalidate) before reading NIC-written `combine_data`: `1` per work-item (default, tail-free), `2` per sub-group, `3` per work-group (13% faster kernel but unmasks a send-side skew tail at E=8/nt=32 — §5.8), `0` none (debug). |
| `DEEP_EP_LL_TIME_PHASES` | `0` | diagnostics | `1` prints `[LLPHASE] send_us=.. reduce_us=..` per combine call (adds a `queue.wait()` between the two sub-kernels). The only way to attribute the `DEEP_EP_SPLIT_DC` combine number. |
| `DEEP_EP_LL_PUT_WGS` | device CU count | `ll_put_wgs` base for send-grid sizing | Rarely changed; underlies `ll_send_wgs`. |
| `DEEP_EP_LL_POLL_CAP` | large (`ll_poll_cap`) | flag-wait spin cap | Bounds spins on a missing cross-PE flag; on timeout the slot is treated as 0 tokens (graceful). A too-large cap masks a wedge instead of failing fast; a too-small cap risks undercount. |
| `DEEP_EP_LL_FLAG_SENDER_FENCE` | `1` | flag flush | Flush `uc_store`-d flag bytes to the NIC domain before the AMO. Leave on; `0` only to reproduce the stale-flag race. |
| `DEEP_EP_LL_FLAG_RECV_ACQ` | tuned | receiver acquire mode | System-scope acquire vs uncached flag reads on receive. |
| `DEEP_EP_LL_FLAG_LSC` | `0` | flag-read primitive | `1` reads the per-expert flag via explicit LSC uncached load instead of hint-based `uc_load`. |
| `DEEP_EP_LL_MAX_PUT_KB` | tuned | chunked-put size | Max bytes per NIC put on the fast path. |
| `DEEP_EP_LL_RESET_DRIVER` | `0` | harness only | Reset the igub driver before a run to clear accumulated HW/NIC/QP wedge. Not a perf knob. |

### 5.3 Compile-time constants (in `internode_ll.cpp`)

| Constant | Value | Meaning |
|---|---|---|
| `kLLConsumeWGSize` | `512` | Work-group size of the token-parallel consume (reduce) kernel. |
| `kLLConsumeMaxWGs` | `512` | Cap on the consume grid width. |
| `NUM_MAX_NVL_PEERS` | `8` | Local ranks per node (`rdma_rank = rank/8`, `nvl_rank = rank%8`). |

### 5.4 iSHMEM/IBGDA runtime environment (transport)

Governs the NIC path rather than the kernel shape but strongly affects LL latency:
`ISHMEM_IB_ENABLE_IBGDA=1`, `ISHMEM_IBGDA_DIRECT_DOORBELL=1`,
`ISHMEM_IBGDA_BAR_BACKEND=igub`,
`ISHMEM_IBGDA_DB_BATCH_SIZE`, `ISHMEM_IBGDA_NIC` (pin each rank to the NIC under
its GPU's PCIe switch), `ISHMEM_SYMMETRIC_SIZE`, `ZE_AFFINITY_MASK`. The build-time
`ISHMEM_DIR` (the statically linked `libishmem.a`) determines the barrier/RDMA
implementation baked into `deep_ep_cpp.so` — a mismatched archive silently
degrades the barrier (≈32 ms/iter) or crashes the device.

**`ISHMEM_IBGDA_QPS_PER_PE` — the primary throughput lever.** Both LL send
kernels key the destination QP by the LOCAL expert index (`qp_idx = le &
(qps_per_pe - 1)`), so with `QPS_PER_PE=1` every expert's RDMA serializes through
QP 0. Setting it to **`num_local_experts` (= `num_experts / num_ranks`, rounded up
to a power of 2, clamped [1,128])** gives each expert an independent QP and lets the
NIC drive them in parallel. DeepEP now auto-defaults this: `deep_ep/buffer.py`
`os.environ.setdefault`s it from `num_qps_per_rank` for LL buffers, and the
`docker-2node-ll-v2` harness derives it from `NUM_EXPERTS`/`NUM_PROCESSES`. A
user-set value always wins. Beyond `num_local_experts` the extra QPs sit idle
(combine payloads must ride their expert's own QP for RC flag-after-payload
ordering), so that is the structural ceiling for a given expert count.

### 5.5 Tuned high-token config & measured scaling (2-node BMG, H7168, TOPK2, 8 experts)

Best config: `QPS_PER_PE = num_local_experts` (auto), `DEEP_EP_LL_REDUCE_WGS=2048`,
`ISHMEM_IBGDA_DB_BATCH_SIZE=64` (required at ≥2048 tokens; `0` deadlocks at scale),
`DEEP_EP_LL_POLL_CAP=500M`, `ISHMEM_SYMMETRIC_SIZE` sized to the RDMA hint.

| tokens | QPS=1 baseline avg | QPS=`num_local_experts` avg | speedup |
|---|---|---|---|
| 512  | 11588 µs | 8148 µs  | −30% |
| 1024 | 22241 µs | 15965 µs | −28% |
| 2048 | 42600 µs | 31580 µs | −26% |
| 4096 | 86257 µs | 63000 µs | −27% |

Effective BW rises 2.07 → ~2.8 GB/s and tails tighten. Scaling is then a clean
~1.98× per token-doubling (bandwidth-bound linear at fixed hidden); the remaining
constant is capped by the `num_local_experts` combine-QP count — more experts (or
a multi-QP combine with cross-QP quiet) would raise it further. `DEEP_EP_LL_SEND_WGS`
and `DB_BATCH_SIZE>64` gave no further gain (dispatch-send and doorbell-rate are not
the bottleneck at this scale).

### 5.6 Large expert counts (`E > SMs`): root cause and measured gains

At `E=384` on a 160-CU B60, LL was ~4× slower than `E=8` at the same token count.
The cause was isolated by falsification rather than by tuning:

| Hypothesis | Discriminating experiment | Verdict |
|---|---|---|
| Grid over-subscription / wave quantization | E-sweep across the 160 and 320 wave boundaries | **Refuted** — cost rises linearly in `E` with **no step** at either boundary |
| QP contention | `QPS_PER_PE` sweep 1→128, run in **both** directions to cancel HW-state drift | **Partial** — explains only 1→16, saturates at ~3% |
| Per-token/bandwidth cost | token sweep | **Refuted** — 8× tokens costs only +35%, i.e. a ~1050 µs token-independent floor |
| Per-WG launch overhead | `DEEP_EP_LL_PACK_CHANNELS` C=1/3/6/12 | **Refuted** — all ≈929 µs |
| **Work-group occupancy** | `DEEP_EP_LL_NUM_WARPS` sweep at fixed grid | **Confirmed** — 32w→8w is worth −30% at E=384 and is *negative* at E=8 |

Four stacking fixes, all now defaults:

| Fix | Mechanism | Gain at E=384 |
|---|---|---|
| `ll_num_warps` occupancy heuristic (§1.2) | 1024- → 256-work-item WG once `E > SMs` | −30% |
| Two-pass channel restructure (§2.2/§2.3) | post-all-then-wait-all; stops poll latency serializing | −25% |
| `ll_drop_fence` default ON (§5.2) | removes a provably redundant per-QP fence | −10% |
| `QPS_PER_PE` clamp 16 → 128 | more experts get independent QPs | ~3% |

**Net: E=384 / topk=2 / H7168 went 1246 µs → 575 µs (−54%, 2.17×)**, stable over a
3-run soak, with `E=8` unregressed (317 → 321 µs).

**Measured bandwidth, E=384 / topk=6 / H7168 / 4 ranks** (FP8 dispatch, BF16
combine; per-phase figures come from `DEEP_EP_SPLIT_DC=1`, since the upstream
`bench_kineto` per-phase path does not exist on XPU):

| tokens | fused avg | fused peak (`min_t`) | dispatch | combine | avg_t / min_t |
|---|---|---|---|---|---|
| 32  | 4.20 GB/s | 4.31 GB/s | 5.96 GB/s | 3.83 GB/s | 942 / 918 µs |
| 64  | 5.39 GB/s | 5.51 GB/s | 8.98 GB/s | 4.69 GB/s | 1508 / 1476 µs |
| 128 | 5.49 GB/s | 5.62 GB/s | 11.06 GB/s | 4.71 GB/s | 3001 / 2935 µs |
| 256 | 1.32 GB/s¹ | 5.92 GB/s | 12.55 GB/s | 4.92 GB/s | 25226 / 5608 µs |

¹ The `avg` at 256 tokens is a **measurement artifact**, not kernel behaviour: the
test walks 8 dtype/shape combos at 1.31 GiB per `packed_recv_x`, and the resulting
VRAM eviction produces `max_t` = 148 ms against `min_t` = 5.6 ms. The `min_t`-based
figure sits exactly on the 64→256 trend.

**Combine is now the bottleneck, and it is an efficiency problem, not a volume
one.** At 128 tokens combine moves only 1.94× dispatch's bytes but takes **4.55×**
the time (~2.3× worse per byte) and accounts for 79% of fused latency. Dispatch
amortizes its fixed cost cleanly (5.96 → 12.55 GB/s across the sweep) while combine
plateaus at ~4.7–4.9 GB/s from 64 tokens on — a per-message rather than per-byte
limit. Further LL tuning should target combine.

### 5.8 Combine is REDUCE-bound, not send-bound (2026-09) — vectorized reduce

`DEEP_EP_SPLIT_DC=1` only measures combine **end-to-end**. Instrumenting the two
sub-kernels separately (`DEEP_EP_LL_TIME_PHASES=1`, host timers + a `queue.wait()`
between the two submits) settled where the time actually goes. At
**E=384 / topk=6 / H7168 / nt=128 / 4 ranks** (per-iteration µs, `min` / `median`
over 928 samples):

| build | `LLCombineSendKernel` | `LLCombineReduceKernel` | combine-only avg |
|---|---|---|---|
| baseline (scalar reduce) | 424 / 734 | **1630 / 1710** | 3325 µs |
| `REDUCE_VEC=8` | 435 / 738 | **314 / 336** | 1884 µs |
| `REDUCE_VEC=8`, `REDUCE_ACQ=3` | 420 / 734 | **269 / 292** | 1608 µs |

**The reduce was ~70–80% of combine, and the send kernel was never the problem.**
That refutes the whole family of send-side hypotheses (token-split port, staging-copy
pipelining, put granularity) as the primary lever — the send kernel is unchanged
across all rows above.

**Root cause of the slow reduce: per-element access granularity, not bandwidth.**
The scalar path read 2 bytes per lane per top-k source, filling only 64 B of a 512 B
sub-group memory request, and re-read the int64 `topk_idx` + fp32 weight for *every*
output element (at topk=6 that is 6×8 B of index traffic per 2 B of payload). Moving
to 16 B per lane (`sycl::vec<uint32_t,4>`) and hoisting the top-k metadata is worth
**5.1×** on the kernel; the cache-invalidate granularity fix below adds ~13% on top.

Two knobs, both in §5.2: `DEEP_EP_LL_REDUCE_VEC` (default 8) and
`DEEP_EP_LL_REDUCE_ACQ` (default 1).

**Why `REDUCE_ACQ` defaults to 1 even though 3 is faster.** The reduce issues a
system-scope acquire (a cache invalidate) before reading NIC-written `combine_data`;
mode 1 issues it per work-item (262144 of them), modes 2/3 once per sub-group /
work-group behind a barrier. Mode 3 is 13% faster on the kernel, but at
**E=8 / topk=2 / nt=32** modes 2 and 3 reproducibly (4/4 runs) turn a tight
distribution into a bimodal one: `min_t` improves 216 → 173 µs while `avg_t`
degrades 228 → ~590 µs with `max_t` 7.7–23 ms. Phase instrumentation localizes those
outliers **entirely to the SEND kernel** (26/928 iterations > 1 ms, max 318 ms) with
the reduce never exceeding 119 µs — i.e. it is a pre-existing cross-rank
combine-flag-wait skew stall that the slow reduce used to *mask* by pacing the ranks,
not a cost of the acquire change. (The same stall is already visible in the
`DISPATCH-only` numbers of *every* arm, old and new: avg 560–615 µs against a
min of 84 µs.) Until that skew stall is addressed independently, the default stays
at the lower-tail-rate mode 1; 2/3 remain opt-in.

**Mode 1 is lower-tail, NOT tail-free (independently verified).** A 5-run A/B of the
shipping default against `REDUCE_VEC=1` on the same clean HW:

| arm | runs | `avg_t` | `min_t` | `max_t` |
|---|---|---|---|---|
| `REDUCE_VEC=1` (legacy) | 4/4 | 320.4–322.7 µs | 309.8–312.4 | 333.5–339.2 (tight) |
| `REDUCE_VEC=8` (default) | 4/5 | 227.0–230.5 µs | 212.7–219.2 | 244.7–248.9 (tight) |
| `REDUCE_VEC=8` (default) | **1/5** | **555.5 µs** | 217.8 | **14862 µs** |

So the same skew tail fires at roughly **1 run in 5 even at `ACQ=1`**, which the
mechanism above predicts: *any* reduce speedup removes inter-rank pacing, and mode 3
merely makes it near-certain rather than occasional. The change is still a clear win —
`min_t` improves in **every** run (213–219 vs 310–312 µs), so the vectorized reduce is
never slower; and even charging the outlier to the average, the expected 293 µs beats
legacy's 321 µs. But the guard's `avg_t` is **not** reliably 1.4× better; it is 1.4×
better ~80% of the time and ~1.7× worse otherwise. **Fixing the send-side skew stall
is a prerequisite for making combine's tail trustworthy**, and would unlock `ACQ=3`.

**Measured end-to-end (fused dispatch+combine `avg_t`, rank 0, correctness checking
ON, `DEEP_EP_SPLIT_DC=1`, defaults otherwise):**

| E | topk | tokens | before | after | speedup |
|---|---|---|---|---|---|
| 8   | 2 | 32  | 321 µs  | **229 µs**  | 1.40× |
| 8   | 2 | 128 | 1002 µs | **582 µs**  | 1.72× |
| 384 | 2 | 32  | 576 µs  | **489 µs**  | 1.18× |
| 384 | 2 | 128 | 1142 µs | **724 µs**  | 1.58× |
| 384 | 6 | 128 | 3979 µs | **1616 µs** | 2.46× |

Combine-only at E=384/topk=6/nt=128: 3325 → 1043 µs (3.19×); combine bandwidth
3.27 → 10.4 GB/s. `max_t` stays within ~5% of `avg_t` on all five shapes (no tail).

**Consequence for future work: after this change the combine SEND kernel is again the
larger half of combine**, and its dominant remaining cost is the cross-rank
flag-wait skew documented above — a pacing/skew problem, not a per-byte one.

### 5.7 Pitfall: an oversized `ISHMEM_SYMMETRIC_SIZE` is actively harmful

The iSHMEM symmetric heap is reserved **in full at init**. Sizing it generously
(e.g. a blanket 12 GiB on a 22.7 GiB card) starves torch's working set, and xe then
**silently evicts buffers to system RAM over the copy (bcs) engine with no error
returned**. Kernels keep producing correct results ~100× slower, which is
indistinguishable from a stalled RDMA flag — this masqueraded as a "token-count
cliff" until the cliff was shown to *move with heap size at constant token count*
(nt=160 fails at 12 GiB but passes at 3 GiB; nt=136 passes at 12 GiB but fails at
15 GiB). `dmesg` shows `Engine reset engine_class=bcs`.

Size the heap from the `xpu_layout_bytes` value reported at startup (layout + 5–10%).
`get_low_latency_rdma_size_hint()` now returns exactly the aligned XPU layout size
instead of the 26–35% larger CUDA-era `legacy_hint`, and
`ll_report_vram_budget()` prints total/free VRAM, heap size, layout size and
`packed_recv_x` size, warning with a suggested `ISHMEM_SYMMETRIC_SIZE` on
over-commit.

The `docker-2node-ll-v2/run.sh` auto-sizer computes this shape-aware. The true
requirement is dominated by three `num_experts * tokens`-scaled regions
(`dispatch_data`, `send_data`, `combine_data`), i.e.

```
bytes ~= num_experts * tokens * (msg_bytes + 2*hidden_bytes)   [~ E*nt*(6*hidden + 16)]
```

The original formula was `tokens*hidden*2*48`, whose magic `48` is exactly
`6*num_experts` at `E=8` — it **ignored the expert count** and so undersized by up to
48× at `E=384` (`ishmem_align failed for 529156992 bytes` at nt=32). It was then
`tokens*hidden*8*num_experts`, which fixed the expert-count blindness but kept a flat
1.28–1.40× headroom — harmless at small shapes, but at `E=384`/nt=1024 that reserves
**21 GiB for a 15.77 GiB need**, i.e. the sizer itself triggers the over-commit trap
this very section warns about.

The sizer now evaluates the **exact layout formula above with a 12% margin**, so
headroom is a consistent 1.13–1.14× at every shape (`E=384`/nt=1024: 21 → 17.75 GiB), and
rounds up to a **64 MiB multiple rather than the next power of two** — iSHMEM does not
require a power-of-two heap, and the pow2 ramp doubled an already-oversized
reservation. Validate it against the runtime's own `xpu_layout_bytes:` line, which it
should exceed by only the margin plus alignment padding. `E=8` sizing is unchanged in
practice (nt=32 → the 256 MiB floor; nt=4096 → 1792 MiB, down from 4096 MiB).

### 5.9 Current token-scaling baseline (E=8, topk=2, H7168, 4 ranks, 2026-09)

Post-vectorized-reduce, auto-sized heap, correctness checking ON. All 7 points PASS
with 0 `DEVICE_LOST`. `peak BW` is `min_t`-based and is the trustworthy column.

| tokens | heap | `avg_t` | `min_t` | avg BW | peak BW | `min_t` scaling |
|---|---|---|---|---|---|---|
| 32   | 256 MiB  | 227 µs¹   | 218 µs   | 5.17 GB/s | 5.38 GB/s | — |
| 64   | 256 MiB  | 381 µs    | 348 µs   | 6.74 GB/s | 7.37 GB/s | 1.60× |
| 128  | 256 MiB  | 579 µs    | 562 µs   | 9.25 GB/s | 9.52 GB/s | 1.61× |
| 256  | 256 MiB  | 1129 µs   | 1094 µs  | 9.67 GB/s | 9.98 GB/s | 1.95× |
| 1024 | 448 MiB  | 4360 µs²  | 4201 µs  | 10.16 GB/s | 10.54 GB/s | 3.84× (4× tok) |
| 2048 | 896 MiB  | 10082 µs³ | 8282 µs  | 8.81 GB/s | 10.73 GB/s | 1.97× |
| 4096 | 1792 MiB | 17077 µs  | 16388 µs | 10.42 GB/s | **10.86 GB/s** | 1.98× |

Scaling is **~1.95–1.98× per token doubling from nt=128 up** (bandwidth-bound linear at
fixed hidden), with peak BW saturating at **~10.9 GB/s**. The sub-linear 1.60× at
32→64→128 is the token-independent fixed cost being amortized. Combine BW (min-based)
now runs 5.8 → 12.2 GB/s across the sweep, versus the ~4.7–4.9 GB/s plateau before §5.8.

¹/²/³ **The skew tail (§5.8) fired at 3 of these 7 points**, inflating `avg_t` only:
¹ nt=32 measured 620 µs avg / 5.6 ms max in the sweep run; the quoted 227 µs is from a
5-run A/B where 4/5 landed at 227–230 µs (`min_t` 218 µs agrees in both).
² nt=1024 first gave 5104 µs avg with a 23 ms outlier; a re-run was tight at 4360 µs.
³ nt=2048 tails in **both** runs (max 65–98 ms) — its `avg_t` is the least trustworthy
number in the table, while its `min_t` is stable across runs (8282 / 8412 µs).

This is the single strongest argument for fixing the cross-rank flag-wait skew stall:
it is not a small-shape curiosity, it perturbs the average at nearly half the sweep.

### 5.10 Token-scaling at a large expert count (E=384, topk=2, H7168, 4 ranks, 2026-09)

Same build/config as §5.9, only `NUM_EXPERTS` raised 8 → 384 (i.e. `E > SMs`, §5.6).
All feasible points PASS with 0 `DEVICE_LOST`.

| tokens | heap | `avg_t` | `min_t` | `max_t` | avg BW | peak BW | dispatch `min` | combine `min` |
|---|---|---|---|---|---|---|---|---|
| 32  | 576 MiB  | 488 µs  | 476 µs  | 507 µs  | 2.41 GB/s | 2.47 GB/s | 184 µs | 287 µs |
| 64  | 1152 MiB | 568 µs  | 542 µs  | 598 µs  | 4.52 GB/s | 4.73 GB/s | 202 µs | 333 µs |
| 128 | 2304 MiB | 722 µs  | 701 µs  | 755 µs  | 7.41 GB/s | 7.63 GB/s | 253 µs | 425 µs |
| 256 | 4544 MiB | 1176 µs | 1149 µs | 1248 µs | 9.29 GB/s | **9.50 GB/s** | 376 µs | 693 µs |

Two observations that contrast sharply with the E=8 baseline in §5.9:

1. **Scaling is sub-linear (1.14× → 1.29× → 1.64× per token doubling)**, the opposite of
   E=8's clean ~1.97×. This is expected and healthy: at E=384 the kernel carries a large
   token-**independent** floor (384 channels of per-expert fixed cost — flag waits, count
   exchange, channel setup), so the ~476 µs at nt=32 is dominated by that floor and extra
   tokens are increasingly close to free. Peak BW consequently climbs 2.47 → 9.50 GB/s as
   the floor amortizes, approaching the ~10.9 GB/s ceiling E=8 reaches at nt=4096.
2. **Every point is tight** (`max_t` within 6% of `avg_t`) — the skew tail did **not** fire
   anywhere in this sweep, whereas it perturbed 3 of 7 points at E=8. Plausibly the larger
   per-expert fixed cost paces the ranks and hides the skew, the same masking effect the
   slow scalar reduce used to provide (§5.8).

**nt ≥ 1024 is infeasible at E=384 on a 22.7 GiB card — an arithmetic ceiling, not a bug.**
The LL layout is `E*nt*(msg + 2*hb) + nt*msg` (§5.7) and is *topk-independent*, so:

| tokens | LL layout | verdict |
|---|---|---|
| 1024 | 15.77 GiB | over-commits: leaves 3.55 GiB free vs **5.25 GiB** `packed_recv_x` per live copy |
| 2048 | 31.5 GiB  | layout **alone** exceeds the 22.71 GiB card |
| 4096 | 63.1 GiB  | layout **alone** exceeds the 22.71 GiB card |

nt=1024 does not fail loudly — it exhibits exactly the §5.7 signature: the run keeps
"working" while xe evicts to system RAM over the bcs engine, ~100× slower, and looks
indistinguishable from a hang. The `LL VRAM budget` warning added for §5.7 is what
identifies it; **trust that warning rather than debugging it as an RDMA stall.** No heap
size rescues nt=1024: even at the minimum viable 16.40 GiB heap the free VRAM is short of
what several live `packed_recv_x` copies need.

---

### 5.11 Token-scaling at E=384, **topk=6** (H7168, 4 ranks, 2026-09)

Same build/config as §5.10, only `NUM_TOPK` raised 2 → 6. The layout is
topk-independent, so heaps are identical to §5.10 and all four points fit. All PASS,
0 `DEVICE_LOST`.

| tokens | heap | `avg_t` | `min_t` | `max_t` | avg BW | peak BW | dispatch `min` | combine `min` |
|---|---|---|---|---|---|---|---|---|
| 32  | 576 MiB  | 645 µs  | 622 µs  | 669 µs   | 6.13 GB/s | 6.36 GB/s | 226 µs | 394 µs |
| 64  | 1152 MiB | 902 µs  | 875 µs  | 942 µs   | 9.01 GB/s | 9.30 GB/s | 302 µs | 530 µs |
| 128 | 2304 MiB | 1616 µs | 1574 µs | 1665 µs  | 10.20 GB/s | 10.47 GB/s | 516 µs | 1012 µs |
| 256 | 4544 MiB | **24498 µs**¹ | 2618 µs | 100365 µs | 1.35 GB/s | **12.67 GB/s** | 900 µs | 1419 µs |

nt=32 lands at 645 µs against the 644 µs measured independently in §5.8 — a useful
confirmation that these numbers are reproducible run-to-run.

`min_t` scales cleanly (1.41× → 1.80× → 1.66× per doubling) and peak BW rises
monotonically to **12.67 GB/s**, the highest figure recorded on this stack (above the
~10.9 GB/s E=8 ceiling of §5.9 — more topk means more payload amortizing the same
per-expert floor). **Every `min_t` in this table is healthy; the problem is confined to
`avg_t` at nt=256.**

¹ **The nt=256 combine stall — a distinct, reproducible, and much more severe
manifestation than the §5.9 tail.** Combine `avg`/`min` = **14.6×** (20243 / 1419 µs),
versus 1.03–1.04× at topk=6/nt=128 and at topk=2/nt=256. Dispatch is untouched
(avg 916 vs min 900 µs — rock solid across every run). What has been ruled out, with
evidence, so this is not re-litigated:

| Hypothesis | Test | Result |
|---|---|---|
| VRAM over-commit / eviction (§5.7) | compare budget lines | **Ruled out** — 16.90 GiB *free*, and the footprint is byte-identical to topk=2/nt=256, which is tight. Only topk differs. |
| Hardware wedge | `DEVICE_LOST`, `dmesg` | **Ruled out** — 0 and 0; both runs PASS. |
| Send-queue depth exhaustion | `QPS_PER_PE` | **Ruled out** — already 128 (96 local experts), so ~16 sends/QP. |
| Caused by the vectorized reduce (§5.8) | `DEEP_EP_LL_REDUCE_VEC=1` | **Ruled out** — stall persists on the legacy scalar path (avg 14869 / min 4439 µs). |
| Random tail | re-run | **Ruled out** — reproducible; combine `avg` 20243 then 19870 µs, `min` 1419 then 1436 µs. |

Two further clues for whoever fixes this. First, the *absolute* stall is ~10–19 ms in
both reduce modes, i.e. a fixed additive cost rather than a multiple of the work — so
it is a wait, not slow compute. Second, in the fused runs **all four ranks report
`avg_t` agreeing to within 2 µs** (40769.61 / 40769.89 / 40770.96 / 40769.02), which
means the ranks enter and leave the stall *together*: a genuine collective wait, not
per-rank jitter. That is the signature of the cross-rank flag-wait skew stall, and
nt=256/topk=6 is by far the best reproducer found so far — **use it as the test case**,
since it fires on nearly every iteration instead of ~1 in 5.

---

### 5.12 Root cause of the cross-rank flag-wait stall, and the NBI fix (2026-09)

The §5.11 stall is now **explained and largely fixed**. Root cause: the combine posted
each arrival flag with the *blocking* `ishmemx_long_atomic_add_qp`, which polls a CQE
before returning. One flag is posted per `(local_expert, dst_rank)`, so at E=384 on 4
ranks that is **96 × 3 = 288 serialized RDMA round-trips per rank per iteration**. That
serialization *is* the 10–20 ms stall, and it explains both the magnitude and the
"all ranks agree to within 2 µs" signature (every rank pays the same queue).

Two independent falsifications pin it down — neither is a confirmation test:

| Probe | Expected if theory X | Observed | Conclusion |
|---|---|---|---|
| Raise `DEEP_EP_LL_POLL_CAP` to 1e6 | if the stall were a receiver *timeout*, correctness holds and time drops | **correctness FAILED** | flags *do* arrive; they are merely posted slowly. Timeout theory dead. |
| Collapse 288 posts → 3 (`FLAG_AGG`) | if cost scales with flag *count*, stall vanishes | stall vanished (avg/min 10.3× → 1.03×) | cost is per-post round-trips, confirming the root cause. |

**The fix: `DEEP_EP_LL_FLAG_NBI_AMO`, now ON by default.** A new iSHMEM entry point
`ishmemx_long_atomic_add_nbi_qp` is the blocking atomic *minus* the CQE poll and ibuf
result read; slot claim, WQE build, and ordered commit + doorbell are byte-identical, so
the unconditional-doorbell property is preserved. No flush, barrier, or quiet is needed:

1. Payload (`ishmemx_putmem_nbi_subgroup`) and flag both target **the same QP** (`qp=le`),
   and an RC QP consumes WQEs in strict order — so flag-after-payload is free.
2. The NBI AMO rings the doorbell unconditionally, publishing all earlier WQEs on that QP.
3. The receiver spin-polls the flag, so the sender never needs to learn when it landed.
4. The two-parity slot scheme leaves a full iteration of slack before any slot is reused.

Measured on the rebuilt canonical archive (all PASS, 0 `DEVICE_LOST`):

| Shape | blocking (`=0`) | **NBI (default)** | Δ |
|---|---|---|---|
| E=8, topk=2, nt=32 (regression guard) | 233.4 µs | 232.7 µs | parity |
| E=384, topk=6, nt=32 | 638.8 µs | **545.0 µs** | **−14.7 %** |
| E=384, topk=6, nt=128 | 1616 µs | **1568 µs** | −3 % |

**Why flag *aggregation* was rejected**, despite being the only thing that fully removes
the stall: it collapses the flags onto QP 0 while payloads still span QPs 0–95, which
destroys the same-QP ordering argument above and therefore forces a mandatory quiet. That
costs a fixed ~2.3 ms, regressing every small shape (E=8/nt=32 229 → 775 µs;
E=384/topk=6/nt=32 645 → 2967 µs). It is kept only as an experimental knob
(`DEEP_EP_LL_FLAG_AGG`, default OFF); do not enable it as a general setting.

Two notes for future work. (a) A *silent-corruption* hazard was found and fixed in the
iSHMEM ibuf allocator while adding the NBI path: the NBI AMO parks its discarded result
in ibuf slot 0, but `claim_ibuf_slot()` still handed slot 0 out to fetch-type AMOs, which
could then read another operation's value. Slot 0 is now reserved (allocation starts at
1, release refuses 0, `num_slots >= 2` required). (b) The **nt=256 stall is reduced but
not eliminated** — `min_t` improves (combine 1436 → 1052 µs) yet `avg/min` is still
~10–16×. With NBI on, `DEEP_EP_LL_TIME_PHASES` shows `reduce_us` steady at 680–800 µs
while `send_us` spikes to ~978 ms **in pairs of consecutive iterations**. Queue-depth
exhaustion is ruled out: `peer_ctx` is indexed per-`(pe, qp_idx)`, so with
`num_qps_per_pe = 128` (confirmed at runtime) there are only ~17 WQEs per QP. The paired
pattern points at the two-parity slot scheme — the `clean_flag` wait or the recv-flag
wait — rather than the flag post. That is the open question. Concretely, nt=256 with the
default ON still PASSes with 0 `DEVICE_LOST` and improves end-to-end (avg 24498 →
**20645 µs**, −16 %; peak BW 12.67 → **13.14 GB/s**, the best recorded on this stack),
but `avg/min` remains ~8× (min 2526 µs, max 194 ms) — so the residual wait is real.

---

## 6. Buffer layout (symmetric heap)

`make_layout` / `get_low_latency_buffer_layout` allocate, in this exact order (both
files must agree):

```
dispatch_data (rdma_recv_x)   num_dispatch_slots * msg_bytes
rdma_x (send staging)         num_max_dispatch_tokens_per_rank * msg_bytes
dispatch_count                num_local_experts * num_ranks * 2 * int64   (2 parity slots)
send_data (combine staging)   num_send_slots * hidden_bytes
send_count                    num_ranks * num_local_experts * int
combine_data                  num_combined_slots * hidden_bytes
combine_flag                  num_experts * 2 * int64                     (2 parity slots)
mask / sync                   num_ranks * int each
barrier (GridBarrier scratch) 2 * uint32     (legacy; unused by the split)
slot_counter                  num_experts * int   (also reused as combine clean_flag)
finish_counter / finish_ready num_experts * int each  (legacy; unused by the split)
```

The `barrier`, `finish_counter`, and `finish_ready` regions are retained for layout
stability but are no longer written by the phase-split kernels.

---

## 7. Kernel fusion attempt (rolled back on XPU — kept as a design record)

**Status: attempted, measured, and rolled back.** The current XPU LL path
remains **phase-split**: dispatch is two kernels
(`LLDispatchSendKernel` → `LLDispatchRecvKernel`) and combine is two kernels
(`LLCombineSendKernel` → `LLCombineReduceKernel`). This section documents the
CUDA-parity single-kernel fusion that was implemented on branch
`namedbarrier_cooperative_launch` (commit `3e51216`), the empirical result on
BMG, and why it was reverted.

### 7.1 What was attempted

The CUDA reference in `csrc/cuda_kernels/internode_ll.cu` runs each direction as
**one** kernel: warp-specialized send/count/recv phases inside a single WG,
joined across WGs by `cg::this_grid().sync()`. Two SYCL primitives already
present in `xpu_kernels.hpp` make a direct port possible:

1. **`NamedBarrier` (SPIR-V `cl_khr_subgroup_named_barrier`)** — sub-group-subset
   barrier replacing CUDA `bar.sync <id>, <count>`. See `named_barrier_usage.md`
   for the class and the JIT-repro test. **Requires** the iSHMEM archive to be built
   with `-DISHMEMI_IBGDA_BNXT_NOINLINE=OFF`; otherwise it fails vISA finalization with
   `More than 1 kernel attribute defined NBarrierCnt`. The previously documented
   `IGC_SelectiveFunctionControl=1` workaround is OBSOLETE and must not be used.
2. **`sycl_ext_oneapi_root_group` + `use_root_sync` + `nd_launch`** — device-wide
   barrier `sycl::group_barrier(root_group)` replacing CUDA
   `cg::this_grid().sync()`. See `root_group_cooperative_launch.md`.

Fused kernels `LowLatencyDispatchFusedKernel` and `LowLatencyCombineFusedKernel`
were built on top of these, with the CUDA `atomic_finish_counter_per_expert`
handshake (caster `+1`, counter warp `+TAG` on WG0 and `+(TAG - count)` per
responsible expert, count-sender polls `2*TAG`) routing the per-expert send
count through global scratch (`finish_ready_offset`) because `nd_launch` does
not accept a `sycl::handler` — SLM is unavailable on this path.

### 7.2 CUDA → XPU sync mapping used by the fused kernels

| CUDA source | XPU/SYCL equivalent | Role in the fused LL kernel |
|---|---|---|
| `bar.sync 1, num_threads` (dispatch, `internode_ll.cu:252`) | `NamedBarrier::init(num_threads/32)` + `.sync(kNamedBarrierGlobalFence)` | Sync ALL caster warps between "row cast into `rdma_x`" and "IBGDA put to top-k experts" |
| `bar.sync warp_group_id+1, num_warps_per_group*32` (combine, `internode_ll.cu:918`) | Per-warp-group `NamedBarrier` | Warp-group-scoped rendezvous between the per-token send warps and the flag-post sub-warp |
| `cg::this_grid().sync()` (`internode_ll.cu:360, 977`) | `sycl::group_barrier(item.ext_oneapi_get_root_group())` via `nd_launch_root_sync<>` | Device-wide barrier between send/count and recv/reduce phases inside one launch |

### 7.3 Empirical result on BMG (why the fusion is rolled back)

Test: `tests/docker-2node-ll-v2/run.sh` on a 2×BMG docker harness, H7168, topk=2,
8 experts, 32 tokens, 10-iter warmup + timed loop; average `dispatch+combine`
per-iter across all 4 ranks.

| Configuration | avg_t | Δ vs split |
|---|---|---|
| **Split** (`LL{Dispatch,Combine}{Send,Recv/Reduce}Kernel` — current code) | **~337 µs** | baseline |
| Dispatch fused (`nd_launch_root_sync` or plain `parallel_for`) + combine split | ~394 µs | **+57 µs (+17 %)** |
| Both fused (`nd_launch_root_sync<>` + `group_barrier(root_group)` on both) | ~525–533 µs | **+188–196 µs (+56–58 %)** |

Both dispatch and combine fusion regress the split baseline. Root cause is XPU
architectural, not a bug in the fused code:

1. **`cg::this_grid().sync()` on CUDA is essentially free** (cooperative-launch
   HW barrier). On BMG, `sycl::group_barrier(item.ext_oneapi_get_root_group())`
   under `use_root_sync` combines (a) a co-residency constraint that limits
   scheduler freedom, and (b) a runtime cost that is measurably higher than the
   split-path's implicit kernel-boundary sync (which the SYCL runtime already
   overlaps with the next `queue.submit`).
2. **Fused dispatch pays the `atomic_finish_counter_per_expert` cost** the split
   path avoids. Split-recv-kernel just re-scans `topk_idx` for the count (which
   the caster kernel already produced in `slot_counter[dst_expert]` via a
   `fetch_add(1)` per top-k pick — no per-token `finish_counter` add-1 needed
   because the kernel boundary already publishes `slot_counter`). Fused-dispatch
   has to reproduce CUDA's `2*TAG` acq-rel handshake so the count-sender knows
   all casters are done — that is `num_topk` extra system-scope acq-rel
   atomics per token on the caster hot path.
3. **Combine reduce reads from ALL experts' `combine_data`**, so it genuinely
   needs a grid barrier before starting; there is no `finish_counter`-style
   shortcut that removes it. The `group_barrier(root_group)` therefore stays
   on the critical path in the fused version.
4. **`nd_launch` bans SLM** (no handler), forcing intra-WG communication
   through global memory — `finish_ready_scratch` in dispatch,
   `packed_recv_layout_range` re-read in the recv copy phase — which adds
   loads on the critical path that the split-kernel version does via
   `sycl::local_accessor`.

The fused kernels are correct (both configs PASS `test_low_latency.py` on
32 and 256 tokens with no `DEVICE_LOST`), so this is a pure performance
finding, not a correctness one.

### 7.4 Decision

Keep the **split** dispatch/combine on XPU as the production LL path. The
`NamedBarrier` and `nd_launch_root_sync` infrastructure in `xpu_kernels.hpp` is
retained: `NamedBarrier` is still the right primitive if `num_warp_groups > 1`
lands (see §4), and `nd_launch_root_sync` is retained for possible future
kernels where the grid-barrier cost is amortized over a larger critical path
(e.g. a fused dispatch→compute→combine kernel, or when the CUDA
`atomic_finish_counter_per_expert` cost is offset by removing more than one
kernel boundary).

If revisiting fusion later, the two levers to close the gap are:

- **Avoid the CUDA finish-counter handshake on XPU.** In an XPU-native fused
  design, the count-sender warp can `sycl::group_barrier(work_group)` on the
  whole WG (single WG per responsible expert) and then read `slot_counter`
  directly — same semantics as the split path, no per-token system-scope
  atomics. This drops the +57 µs dispatch-fusion overhead.
- **Amortize the grid barrier.** `group_barrier(root_group)` cost is largely
  fixed-per-launch, so it becomes proportionally cheaper as the per-iter work
  grows (bigger hidden, more tokens, or a fused kernel with more phases).
  A three-way fused dispatch→GEMM→combine on the same launch would be the
  obvious payoff shape.

Historical reference: the fused implementation lives at git commit `3e51216`
on branch `namedbarrier_cooperative_launch`; the pre-fusion split baseline is
`ba6332a` on the same branch.
