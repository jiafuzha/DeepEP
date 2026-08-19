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

### 1.2 Work-group geometry (CUDA parity)

Derived in both `dispatch_bf16` and `combine_bf16`, identical to the CUDA host code:

| Quantity | Formula | Value on B60 (exp ≤ 160) |
|---|---|---|
| `num_warp_groups` | `ceil(num_experts / num_device_sms)` | `1` |
| `num_warps_per_group` | `32 / num_warp_groups` | `32` |
| `num_warps` (sub-groups/WG) | `num_warp_groups * num_warps_per_group` | `32` |
| `wg_size` | `num_warps * 32` | `1024` |
| `num_sms` (commit-gated grid) | `ceil(num_experts / num_warp_groups)` | `num_experts` |

A `TORCH_CHECK` enforces `num_warp_groups == 1`: with one warp group per WG the
whole-WG `sycl::group_barrier` is the correct substitute for CUDA's warp-group-
scoped `bar.sync`. `num_warp_groups > 1` would require the per-warp-group barrier
that BMG cannot reliably express (see §4), and is the future reason to push the
phase-split further (each warp group becomes its own WG).

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

- **Grid:** `send_wgs` work-groups (default `num_sms == num_experts`, tunable via
  `DEEP_EP_LL_SEND_WGS`), `wg_size = 1024`.
- Each WG strides over tokens (`t = sm_id; t < num_tokens; t += send_wgs`). For
  each token **all 32 warps cast the hidden row cooperatively** (`cast_token_fp8_strided`
  for FP8, `coop_copy_bytes` for BF16) into the token's send-staging slot, write
  the `src_idx` header, then a **device-scope release fence** + whole-WG barrier
  makes the bytes NIC-visible.
- Warps `w < num_topk` then deliver the finished message to `topk_idx[t, w]`'s
  expert slot: `slot_counter[expert].fetch_add(1)` picks the destination slot, and
  `ishmemx_putmem_nbi_subgroup(..., force_db=false)` (self rank → `coop_copy_bytes_store_uc`).
- **No counter warp, no finish-counter, no grid barrier.** Counting is deferred to
  the recv kernel; the kernel boundary provides the ordering the finish-counter
  used to provide. `force_db=false` leaves the last doorbell batch deferred — the
  recv kernel's `fence_qp` flushes it (advances `nic_wq_commit` and rings the
  doorbell without waiting for CQEs; RC in-order delivery still lands the count
  flag after the payloads on the target QP).
- Block 0 additionally zeroes the opposite-parity `rdma_recv_count` slots (CUDA
  `next_clean`).

### 2.2 `LLDispatchRecvKernel` — count flag + poll + copy

- **Grid:** `num_sms == num_experts`, `wg_size = 1024`. WG `sm_id` owns the channel
  `responsible_expert_idx = sm_id`.
- **Phase A (count post):** warp 0 histograms `topk_idx == responsible_expert` and
  posts the count flag `(-count-1)` on QP `le`. Because the send kernel has fully
  exited, all payload puts are already committed; `ishmemx_fence_qp(dst_rank, le)`
  establishes a **QP-scoped ordering fence** (flushing any deferred doorbells and
  publishing the ordered-commit watermark up to the current claim, §3.3), and the
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

- **Grid:** `num_sms == num_experts`, `wg_size = 1024`. WG owns
  `responsible_expert_idx`.
- Block 0 zeroes the opposite-parity `combine_flag` slots (CUDA `next_clean`) and
  releases `atomic_clean_flag` (`clean_flag`, borrowed from the unused
  `slot_counter` cell) so flag posts wait for the clean.
- For its expert's received tokens, sub-warps copy each hidden row into per-token
  symmetric staging and `ishmemx_putmem_nbi_subgroup` it to the destination's original
  token slot (`src_idx`); self rank writes directly into local `combine_data`.
- After a warp-group barrier, sub-warp 1 posts the arrival flag `+1` on QP `le`
  (self → `uc_store`; remote → `fence_qp` + `atomic_add_qp` — RC in-order delivery
  guarantees the AMO lands after every prior payload put on the same QP without
  waiting for their CQE completions), and sub-warp 0 waits on the incoming
  arrival flag for its channel. The kernel exit is the grid sync: once it
  returns, every flag has been observed and all `combine_data` is globally
  visible.

### 2.4 `LLCombineReduceKernel` — weighted top-k reduce (token-parallel)

- **Grid:** `reduce_wgs = ll_consume_wgs(...)` (large, oversubscribable —
  `min(4*CU, 512)` capped by work-item count, `DEEP_EP_LL_REDUCE_WGS` override),
  `wg_size = 512`.
- Grid-strides over `reduce_work = num_combined_tokens * hidden` elements. Each
  work-item accumulates `sum_k weight_k * combine_data[expert_k row][h]` over the
  token's top-k experts and writes one BF16 output element. Pure **cached local
  reads** — no IBGDA, no commit gate — so the grid scales freely with token count
  (mirrors CUDA sizing its combine grid to `num_combined_tokens / num_recv_per_sm`).

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
1`**. Making each warp group its own work-group (the natural next step of the
phase-split) is how `num_warp_groups > 1` (i.e. `num_experts > num_device_sms`)
would eventually be supported.

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
| `num_experts` (`num_local_experts`) | Sets `num_sms` = the commit-gated send/flag grid width and the number of QPs (one per global expert). More experts ⇒ more parallel QPs (helps send concurrency up to the co-residency limit) but also more channels to poll and more per-expert histogram/flag work. Constrained to `num_experts <= num_device_sms` (`num_warp_groups == 1`). |
| `hidden` | Bytes per message and per reduce element. Must be a multiple of 128 for FP8 scales. Larger `hidden` shifts the balance toward payload-copy/NIC bandwidth. |
| `use_fp8` / `round_scale` / `use_ue8m0` | FP8-on-send halves NIC payload vs BF16; adds cast cost on send and scale handling on recv. |

### 5.2 Environment tunables (perf knobs)

| Env var | Default | Scope | Guidance |
|---|---|---|---|
| `DEEP_EP_LL_SEND_WGS` | `num_sms` (= `num_experts`) | dispatch send grid | Raising it rarely helps (commit-gate bound) and **must not exceed resident WG capacity** — a spinning producer trips the GuC watchdog → `DEVICE_LOST`. Keep at default unless profiling shows cast-bound headroom. |
| `DEEP_EP_LL_REDUCE_WGS` | `min(4*CU, 512)` capped by work-items | combine reduce grid (`ll_consume_wgs`) | The main scaling lever for combine. Increase toward the cap as `num_tokens`/`hidden` grow so the reduce is fully token-parallel; too small ⇒ long grid-stride loops. |
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
to a power of 2, clamped [1,16])** gives each expert an independent QP and lets the
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
   for the class, the iSHMEM `NBarrierCnt` coexistence requirement
   (`IGC_SelectiveFunctionControl=1`), and the JIT-repro test.
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
