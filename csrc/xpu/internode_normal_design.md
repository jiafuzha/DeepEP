# Internode NORMAL (high-throughput) dispatch/combine on XPU — design & implementation

**Scope.** The SYCL/XPU port of DeepEP's *normal* (high-throughput) internode path:
`csrc/xpu/internode.cpp` + `csrc/xpu/internode_dispatch_fused.inc` +
`csrc/xpu/internode_combine_fused.inc`. Source of truth for semantics is
`csrc/cuda_kernels/internode.cu` (`dispatch` L447, `combine` L1716,
`notify_dispatch` L93, `cached_notify` L1311).

This is **not** the low-latency path (`internode_ll.cpp`) and **not** intranode
(`intranode.cpp`). Companion documents:
- `csrc/xpu/named_barrier_usage.md` — named-barrier budget and IGC constraints.
- `.github/agents/cuda-to-xpu-internode-normal-migration.agent.md` — the migration
  playbook, including the falsified-hypothesis log (read §26 before proposing a
  perf theory; six plausible ones are already dead).

Written for a performance review. §6 is the part that matters for that; §1–§5 exist
so the numbers in §6 can be interpreted.

---

## 1. Topology and the two-level hierarchy

Ranks are partitioned exactly as in CUDA:

```
rdma_rank = rank / NUM_MAX_NVL_PEERS      // which node
nvl_rank  = rank % NUM_MAX_NVL_PEERS      // which GPU within the node
NUM_MAX_NVL_PEERS = 8
```

Traffic therefore has two legs:

| Leg | Transport (CUDA) | Transport (XPU) |
|---|---|---|
| Between nodes | NVSHMEM/IBGDA RDMA | iSHMEM/IBGDA RDMA (`ishmemx_putmem_nbi_subgroup`) |
| Within a node | NVLink over CUDA IPC | Level-Zero IPC peer-mapped pointers |

A token going from `(src_rdma, src_nvl)` to `(dst_rdma, dst_nvl)` never takes two
RDMA hops. It is RDMA'd **once** to `(dst_rdma, src_nvl)` and then forwarded over
the intra-node IPC leg to `(dst_rdma, dst_nvl)`. That is the whole reason the
kernels have a distinct *forwarder* warp role.

**PE-space deviation (D4).** CUDA's NVSHMEM is initialized on a same-GPU-index
team, so `translate_dst_rdma_rank<false>()` returns the bare `rdma_rank`. iSHMEM
has a single flat PE space, so everywhere in the XPU port:

```
dst_pe = dst_rdma_rank * num_nvl_ranks + nvl_rank
```

## 2. Kernel structure

Both directions are **single fused warp-specialized kernels**, matching CUDA. The
earlier XPU baseline split them into phase micro-kernels because SPIR-V named
barriers were not yet working; that split is gone. There is no grid-wide sync in
either kernel — and there never was one in the CUDA original either. All
synchronization is *warp-subset* synchronization inside one work-group.

### 2.1 Grid / block shape

| | dispatch | combine |
|---|---|---|
| Grid | `num_channels * 2` work-groups | `num_channels * 2` work-groups |
| SM parity | even `sm_id` = forwarder, odd = sender/receiver | odd `sm_id` = forwarder, even = NVL-sender / RDMA-receiver |
| Block | `(7 + 1 + 8) * 32 = 512` WI = **16 sub-groups** | `(kForwarders + 1) * 32` |
| Roles | `kRDMASender`, `kRDMASenderCoordinator`, `kRDMAAndNVLForwarder`, `kForwarderCoordinator`, `kNVLReceivers` | `kNVLSender`, `kNVLAndRDMAForwarder`, `kRDMAReceiver`, `kCoordinator` |

Combine's shape is templated on the RDMA-rank count `R`
(`FusedCombineShape<R>`, `internode_combine_fused.inc:62`):

```
kWarpsPerForwarder = max(1, 24 / R)
kForwarders        = R * kWarpsPerForwarder      // 24 at R=2 and R=4
kRDMAReceivers     = kForwarders - 8             // 16
```

So at `R=2` combine is `(24+1)*32 = 800` WI = 25 sub-groups.

`num_channels = num_sms / 2`.

### 2.2 Named barriers (the CUDA `barrier.sync <id>, <count>` port)

SPIR-V `cl_khr_subgroup_named_barrier`. **`count` is expressed in SUB-GROUPS, not
work-items** — this is the single most common porting mistake.

| CUDA site | CUDA count (WI) | XPU `named_barrier_init` (sub-groups) | Participants |
|---|---|---|---|
| dispatch `:563` `sync_rdma_sender_smem` | `(7+1)*32 = 256` | `8` | warps 0–6 senders + warp 7 coordinator |
| dispatch `:580` `sync_forwarder_smem` | `(8+1)*32 = 288` | `9` | warps 0–7 forwarders + warp 8 coordinator(`target_rank==0`) |
| combine `:1952` `sync_forwarder_smem` | `(kForwarders+1)*32` | `25` | forwarder-SM warps 0–23 + coordinator 24 |
| combine `:1953` `sync_rdma_receiver_smem` | `(kRDMAReceivers+1)*32` | `17` | receiver-SM warps 8–23 + coordinator 24 |
| combine `:1966` `sync_large_warp` | `kWarpsPerForwarder*32`, id `dst_rdma_rank+2` | one handle **per destination RDMA rank** | forwarder sub-warps |

Two hard IGC constraints, both load-bearing:

1. The handle returned by `named_barrier_init()` must be a **plain SSA local of the
   function that calls `work_group_named_barrier()`** — no struct members, no
   function arguments, no PHI nodes, no GEPs, no SLM round-trip. Handles are
   therefore materialized unconditionally at the very top of the kernel lambda and
   used only through *literal* `if/else`. This is why `sync_large_warp` is an
   if/else chain on `dst_rdma_rank` and not an array index.
2. Every kernel that links IBGDA code must be built with
   `-DISHMEMI_IBGDA_BNXT_NOINLINE=` (empty). Otherwise IBGDA's
   `__attribute__((noinline))` functions get their own `NBarrierCnt` kernel
   attribute and IGC fails with *"More than 1 kernel attribute defined
   NBarrierCnt"*. iSHMEM itself must be built with
   `-DISHMEMI_IBGDA_BNXT_NOINLINE=OFF` (see `_build_ishmem.sh`).

The shape is regression-tested standalone by
`csrc/xpu/tools/test_nbarrier_dispatch_shape.cpp`.

**Budget — this is a *complexity* limit, not an architectural one.** There is no
hard cap at ~5–9: a *trivial* standalone BMG kernel compiles and runs with 10 named
barriers. But inside the real fused combine kernel, **8 named barriers JIT fine
(full matrix passes) and 9 raises `error: IGC: internal compiler error`** (dies
between `push_analysis` and `codegen`). Combine declares `2 + num_rdma_ranks`
barriers, so **`num_rdma_ranks <= 4` is supported and `R=8` (10 barriers) ICEs** —
`R=8` exists for dispatch only.

Consequence for review: the barrier budget must be **re-measured by bisection after
any significant kernel growth**. An unrelated addition can push combine over the
cliff. Related: *any* external/outlined device function in a NamedBarrier kernel
triggers the same `NBarrierCnt` failure (hence §2.2 constraint 2); symptom is the
same IGC ICE accompanied by `warning: ... Stack call has been detected`. Also: **SLM
subset barriers deadlock** — only hardware barriers give guaranteed cross-sub-group
forward progress; do not hand-roll SLM arrival-counter spin barriers.

### 2.3 Deliberate deviations from CUDA

| ID | Deviation | Why |
|---|---|---|
| D1/C1 | TMA (`tma_load_1d`/`tma_store_1d`/mbarrier) → sub-group-cooperative 16 B (`int4`) LSC copies | No Xe equivalent. This is exactly what CUDA did pre-TMA (`combine_token` non-TMA branch, `internode.cu:1646-1691`). |
| D2/C2 | Work-group keeps all 16 sub-groups even when `num_nvl_ranks < 8` | Keeps named-barrier arrive-counts compile-time constants. Idle *forwarder* warps must still enter `sync_forwarder_smem` once or the 9-way barrier deadlocks; idle *receiver*/*sender* warps enter no barrier and return. `(warp_id + channel_id) % 8` is a bijection so exactly `num_nvl_ranks` warps stay live. |
| D3 | Per-lane `acquire_lock(rdma_send_channel_lock + lane_id)` serialized over `dst_rdma_rank` | CUDA relies on Volta+ independent thread scheduling. On Xe a sub-group is one lock-stepped EU thread: a lane holding lock[a] cannot reach its release while a sibling lane spins on lock[b] → deadlock. Serializing means a sub-group holds at most one lock, so no cycle can form. Window/tail release ordering is unchanged. |
| D4 | Flat iSHMEM PE space (§1) | No NVSHMEM team concept. |
| D5/C5 | Queue counters via `uc_load`/`uc_store`; payloads via `lsc_fence_sysacq()` + `ld_nc_global_v` (`lsc_load.ugm.uc.ca`) | NIC- and P2P-delivered bytes are **not** coherent with the Xe cache hierarchy. |
| D6 | QP index = `channel_id % qps_per_pe` (payload/tail), `(channel_id + num_channels) % qps_per_pe` (head credit) | No iSHMEM equivalent of `ibgda_get_state()->num_rc_per_pe`. |
| C3 | All cross-rank NVL traffic is a **PUSH** | Sender writes x/tail into the *destination's* buffer, reads head from its *own*; coordinator writes head into the *peer's* buffer. Remote IPC **read** is unstable on this stack; remote IPC **write** is stable. There is no IPC pull-read anywhere in combine. |
| C4 | `nvshmemi_ibgda_put_nbi_warp` → `ishmemx_putmem_nbi_subgroup`; `nvshmemi_ibgda_amo_nonfetch_add` → `ishmemx_fence_qp` + `ishmemx_long_atomic_add_qp` on the **same QP** (RC in-order therefore orders the tail AMO behind the payload), or a local atomic when the destination is this RDMA rank. **Never `ishmem_quiet()`.** | |
| C6 | Spin loops bounded by `kFusedSpinCap = 2e8` | CUDA uses `clock64()` timeout + `trap()`. A lost NIC completion surfaces as a diagnostic rather than an unkillable hang. |
| — | `hidden` is a **runtime** value (`hidden_int4`) | CUDA uses compile-time `SWITCH_HIDDEN`. New hidden sizes need no new instantiation on XPU. |

## 3. Data flow and queue protocol

### 3.1 Buffers

RDMA-side (symmetric heap, `FusedSymBuffer`, per `(dst_rdma_rank, channel)`):

| Buffer | Elements | Purpose |
|---|---|---|
| `rdma_channel_data` | `num_max_rdma_chunked_recv_tokens` slots | token payload ring |
| `rdma_channel_meta` | `num_nvl_ranks*2 + 2` ints | per-channel prefix start/end metadata |
| `rdma_channel_head` | 1 × `uint64` | consumer credit |
| `rdma_channel_tail` | 1 × `uint64` | producer cursor |

NVL-side (IPC peer-mapped, `FusedAsymBuffer`, per `(peer, channel)`) — note the
asymmetry that encodes the push protocol: `nvl_channel_x` / `nvl_channel_tail` /
`nvl_channel_prefix_*` live in the **write-remote** pointer, `nvl_channel_head` in
the **read-self** pointer.

Both rings are classic head/tail credit rings: the producer spins until
`tail - head <= capacity - chunk`, writes payload, fences, then bumps the tail.

### 3.2 Dispatch

1. **Layout** (host + `layout.cpp`): `get_dispatch_layout()` produces
   `num_tokens_per_rank`, `num_tokens_per_rdma_rank`, `num_tokens_per_expert`,
   `is_token_in_rank`.
2. **`notify_dispatch`**: exchanges per-channel counts, produces
   `gbl_channel_prefix_matrix` / `rdma_channel_prefix_matrix`, i.e. each
   (rank, channel) pair's contiguous token block.
3. **Fused kernel:**
   - `kRDMASender` (warps 0–6) reads its token block, and for each destination
     RDMA rank stages tokens into the symmetric send window under the
     per-`dst_rdma_rank` lock (D3).
   - `kRDMASenderCoordinator` (warp 7) issues `ishmemx_putmem_nbi_subgroup` for
     each ready chunk, then `ishmemx_fence_qp` + `ishmemx_long_atomic_add_qp` on
     the same QP to publish the tail.
   - `kRDMAAndNVLForwarder` (warps 0–7 of the forwarder SM) polls the RDMA tail,
     `lsc_fence_sysacq()`, reads payload with `ld_nc_global_v`, and **pushes** it
     over IPC into the destination NVL peer's `nvl_channel_x`.
   - `kForwarderCoordinator` (warp 8) publishes NVL tails / recycles RDMA head
     credit.
   - `kNVLReceivers` (warps 8–15) drain `nvl_channel_x` into `recv_x` and write
     back head credit.

The forwarder's payload copy loop (`internode_dispatch_fused.inc:823-825`) is a
plain lane-strided `int4` copy with **no unroll**, followed by a per-token
`group_barrier`. This is the fastest form measured; see §6.

### 3.3 Combine

Combine is dispatch reversed, plus the head-transform half of CUDA's
`cached_notify` (a separate kernel in CUDA too).

- `kNVLSender` pushes reduced tokens into the forwarder peer's NVL ring.
- `kNVLAndRDMAForwarder` drains the NVL ring, reduces, and RDMAs back.
- `kRDMAReceiver` drains the RDMA ring into the final output.
- `kCoordinator` (warp 24) handles credits and participates in both named barriers.

**Token-count rule (was a real hang).** `x.size(0)` is an *allocation* size. On XPU
`dispatch()` is always called with `num_worst_tokens`, so `recv_x` is **padded**.
Combine's NVL sender must end the last block at
`gbl_rank_prefix_sum[num_ranks-1]`, not at `x.size(0)`, or it pushes hundreds of
pad rows the receiver never expects and the ring never drains. Both ends are
`min()`-clamped. CUDA never hits this because there `x.size(0)` *is* the received
count.

> **Rule: never use a tensor's row count as a logical token count on XPU — the
> logical count lives in the prefix sums.**

## 4. Configuration surface

### 4.1 Python (`deep_ep/buffer.py`)

- **`Buffer.num_sms`** — default `20` → 10 channels. Must be even.
  `Buffer.set_num_sms()` asserts evenness.
- **`Config(num_sms, nvl_send, nvl_recv, rdma_send, rdma_recv)`** — chunk sizes,
  from `get_dispatch_config()` / `get_combine_config()`. Invariants enforced in
  `csrc/xpu/xpu_runtime.hpp:53`: `send < recv`, `rdma_send <= rdma_recv/2`, and
  `rdma_recv` is aligned up to a multiple of `rdma_send`.
- **`ISHMEM_IBGDA_QPS_PER_PE`** — set via `setdefault` from `num_qps_per_rank`,
  clamped to `[1,16]` and rounded to a power of two (→ **16** in practice).
  **This is a correctness requirement, not a tuning knob**, for the normal path.

  The in-tree comment block documenting `C=1` as optimal is a *low-latency* /
  perf-only study and remains valid there. For internode-normal, one QP per PE
  means every channel's RDMA sender drives the same send queue; once ~12 channels
  share it the run intermittently hangs or silently drops whole tokens
  (measured 4/6 hangs at `num_sms=24`, 2048 tok, hidden 7168; 6 ch/QP 0/6;
  4 ch/QP 0/16). One QP per channel removes it **and** is faster, because the grid
  clamp that `C=1` forces costs far more than the extra per-QP flags.
  The LL branch of `buffer.py` is deliberately untouched.

### 4.2 Native env knobs

| Var | Default | Meaning |
|---|---|---|
| `DEEP_EP_FUSED_MAX_SMS` | device-derived | Overrides the co-residency/QP clamp (§4.3). Rounded down to even. Logs to stderr when used. |
| `DEEP_EP_COMBINE_SND_SPLIT` | `min(kNumRDMARanks, 8/num_nvl_ranks)` | Combine NVL-sender sub-warps per destination (§6.1). |
| `DEEP_EP_COMBINE_TOK_SPLIT` | `8/(num_nvl_ranks*snd_split)` (**ON**) | Combine NVL-sender sub-warps per *token stream* within one queue (§6.1.1). `=1` disables. |
| `DEEP_EP_COMBINE_SND_PLAIN` / `DEEP_EP_DISP_SND_PLAIN` | off | Diagnostic: plain C++ copy loop instead of the LSC-intrinsic one. Measured **43 % worse**; diagnostic only. |
| `DEEP_EP_FUSED_PAD_WGS` / `DEEP_EP_FUSED_PAD_CYCLES` | off | Diagnostic occupancy padding. |
| `DEEP_EP_NVL_BYTES` / `DEEP_EP_RDMA_BYTES` | — | Buffer sizing for the harness. |
| `DEEP_EP_COMBINE_TELEMETRY` | compile-time | Per-warp-role cycle attribution (stalled-on-credit vs copying). Never in production builds. |

Plus the standard iSHMEM/BMG matrix (`ISHMEM_IB_ENABLE_IBGDA=1`,
`ISHMEM_IBGDA_DIRECT_DOORBELL=1`, `ISHMEM_IBGDA_BAR_BACKEND=igub`,
`ISHMEM_ENABLE_GPU_IPC=0`, `ISHMEM_SYMMETRIC_SIZE`, per-rank `ISHMEM_IBGDA_NIC`,
`ZE_ENABLE_PCI_ID_DEVICE_ORDER=1`, per-rank `ZE_AFFINITY_MASK`).

### 4.3 The grid clamp (`fused_max_coresident_sms`, `internode.cpp:524`)

```
sms       = min(dispatch_cap, combine_cap)     // driver max-coresident-WG query
qp_safe   = 2 * kSafeChannelsPerQp(=2) * qps_per_pe
safe_sms  = max(kEmpiricalSafeSms(=8), qp_safe)
sms       = min(sms, safe_sms); sms -= sms % 2; sms = max(sms, 2)
```

The `kEmpiricalSafeSms = 8` floor is a **legacy of the QP-contention era**. The
driver's own answer on Arc Pro B60 (160 EU / 20 Xe-cores) is 20 WGs for both fused
kernels; at `QPS_PER_PE=1` that hung, which is what motivated the cap. The
mechanism is now understood to be per-QP contention, not co-residency — with
`QPS_PER_PE=16`, `num_sms=24` measured 16/16 @2048 tok and 8/8 @4096 tok. The
`qp_safe` term already encodes that. **Whether the clamp can now simply be the
driver bound is an open review question (§6.4).**

## 5. Current performance

Shipped default (`num_sms=20` / 10 channels / `QPS_PER_PE=16`), rank 0,
hidden 7168, topk 2, experts 8, 4 ranks / 2 nodes, post-`snd_split`:

| tokens | round-trip µs | dispatch iso µs | combine iso µs | comb/disp | nvl_send GB/s | rdma_recv GB/s |
|---|---|---|---|---|---|---|
| 32 | 1361.6 | 949.3 | 818.8 | 0.86 | 0.88 | 0.56 |
| 64 | 1607.3 | 1092.2 | 943.2 | 0.86 | 1.67 | 0.97 |
| 128 | 2296.9 | 1252.9 | 1458.6 | 1.16 | 2.20 | 1.26 |
| 256 | 3173.0 | 1537.8 | 2015.4 | 1.31 | 3.10 | 1.82 |
| 512 | 4714.6 | 2224.6 | 2959.6 | 1.33 | 3.99 | 2.48 |
| 1024 | 7378.6 | 3038.4 | 4761.8 | 1.57 | 4.97 | 3.08 |
| 2048 | 10649.2 | 4485.0 | 6400.5 | 1.43 | 7.58 | 4.59 |
| 4096 | 19055.2 | 7958.8 | 11595.8 | 1.46 | 8.28 | 5.06 |

Correctness: **31 clean full-matrix runs** (64/64 configs each) at 2048 and 4096
tokens, hidden 7168.

### 5.1 MoE-realistic shapes, and the `tok_split` A/B (2026-08-26)

The table above is `num_experts=8`. At a realistic MoE fan-out
(**NT=4096, hidden 7168, num_experts=384**, 4 ranks / 2 nodes, `num_nvl_ranks=2`,
`kNumRDMARanks=2`) combine dominated the round-trip — 57 % at topk=2 and 63 % at
topk=6. Same-session A/B of the shipped default against the immediately-preceding
build (only `DEEP_EP_COMBINE_TOK_SPLIT`, §6.1.1, differs), rank 0, µs:

| shape | metric | before | after | speedup |
|---|---|---|---|---|
| topk=2 | round-trip | 16 536.5 | **13 432.7** | **1.23×** |
| topk=2 | round-trip (min) | 16 248.1 | 13 047.5 | 1.25× |
| topk=2 | dispatch iso | 7 347.1 | 7 303.1 | 1.01× (unchanged, as intended) |
| topk=2 | combine iso | 9 431.4 | **6 275.3** | **1.50×** |
| topk=6 | round-trip | 24 121.5 | **16 408.4** | **1.47×** |
| topk=6 | round-trip (min) | 23 782.6 | 15 962.1 | 1.49× |
| topk=6 | dispatch iso | 9 350.4 | 9 311.1 | 1.00× |
| topk=6 | combine iso | 15 265.4 | **7 310.5** | **2.09×** |

Combine bandwidth at topk=2: `nvl_send` 9.38 → **14.26 GB/s**, `rdma_recv`
6.16 → **9.36 GB/s**. Combine's share of round-trip drops 57 %→47 % (topk=2) and
63 %→45 % (topk=6); **dispatch is now the larger half at topk=6.**

Correctness of the shipped default at these shapes: **6/6 clean full 64-config
matrix runs** (3 at topk=2, 3 at topk=6, NT=4096 hidden 7168 experts 384, no
`DEEP_EP_PERF_TOKENS`), plus 3/3 more with the knob forced explicitly.

> ⚠️ Run-to-run noise at these shapes is **6–13 %**, far wider than the ±1.3 % band
> quoted elsewhere in this document for hidden 1024. **Always A/B within one
> session**; a cross-session comparison at NT=4096 is not interpretable.

#### 5.2 Chunk knobs at NT=4096 / hidden 7168 (not shipped)

`DEEP_EP_RDMA_CHUNK=8 DEEP_EP_NVL_CHUNK=4` is worth a further ~2.4 % on top
(rt 13 061, dispatch 6 790) by helping **dispatch** only; combine is flat under
every chunk setting tried. Left unshipped: §6.3.2 shows the optimum inverts at
hidden 1024, so this would have to be a shape-conditioned default and the gain
does not justify the conditioning risk yet.

| label (topk=2) | rt | dispatch | combine |
|---|---|---|---|
| shipped (rdma16 / nvl8) | 13 378 | 7 317 | 6 303 |
| rdma8 | 13 370 | 6 910 | 6 576 |
| **rdma8 nvl4** | **13 061** | **6 790** | 6 479 |
| rdma8 nvl16 | 13 721 | 6 875 | 6 989 |
| rdma4 | 16 235 | 7 773 | 8 622 |

`num_sms` remains optimal at the driver bound of 20 **after** the change
(16 → rt 14 763, 24 → rt 15 533), i.e. §6.4 still holds.

#### 5.3 Why topk=6 costs more than topk=2 — not pathological

topk=6 moves ~1.9× the combine traffic of topk=2 (higher fan-in: ~24 576 vs
~6 144 received tokens per rank). Post-`tok_split` combine scales 6 275 → 7 311 µs,
i.e. **sub-linearly** in that traffic, so there is no topk-specific pathology left
to chase. Before `tok_split` the same ratio was 9 431 → 15 265 µs (1.62×), which is
what made topk=6 look like a separate regression — it was the warp deficit biting
harder under more traffic. The per-token `uc_store` of `topk_weights`
(`num_topk` uncached stores per token) was the suspected culprit and is **not**
implicated by these numbers; it was left alone.

## 6. Known bottlenecks and open questions for review

### 6.1 The structural asymmetry — dispatch partitions by token, combine by destination

**This is the single most perf-relevant design fact.**

- Dispatch's RDMA sender partitions the token block across 7 sender warps × 10
  channels → ~**70 live warps**.
- Combine's `kNVLSender` originally mapped `dst_nvl_rank = warp_id` one-for-one
  (mirroring `internode.cu:1849`). With `num_nvl_ranks = 2` that leaves **6 of 8
  warp slots idle** → ~20 live warps.

The shipped mitigation (`internode_combine_fused.inc:342`) adds `snd_split`
sub-warps per destination, each owning a disjoint subset of RDMA lanes:

```
snd_split   = clamp(NUM_MAX_NVL_PEERS / num_nvl_ranks, 1, kNumRDMARanks)
dst_nvl_rank= warp_id % num_nvl_ranks
snd_sub     = warp_id / num_nvl_ranks
lane_owned  = (lane_id < kNumRDMARanks) && (lane_id % snd_split == snd_sub)
```

This needs **no new synchronization and no new named barrier**, because every
queue resource is already per-RDMA-lane. Measured **3.56×–3.61×** (combine
22 850 → 6 423 µs @2048; 41 900 → 11 611 µs @4096); dispatch unchanged.

> ⚠️ **`snd_split` is a no-op at `num_nvl_ranks == 8`** (it evaluates to 1). It is a
> small-NVL win. At the production 8-peer shape the warp deficit does not exist in
> this form, and CUDA has the same 1:1 mapping — so this is not a port-fidelity
> bug. But it also means the fix does **not** generalize.

**Open:** scaling *within* one `(dst, rdma)` queue requires token-range splitting
plus cooperative slot claiming. Not attempted. This is the main structural item
for review.

#### 6.1.1 CLOSED (2026-08-26): token-range split — `DEEP_EP_COMBINE_TOK_SPLIT`, ON by default

The "not attempted" item above is now implemented and **shipped ON**.

`snd_split` can only partition by RDMA lane, so it saturates at
`min(NUM_MAX_NVL_PEERS/num_nvl_ranks, kNumRDMARanks)` = 2 at the 2×2 shape,
leaving 4 of 8 sender warp slots idle. `tok_split` recovers them by partitioning
the **token stream** of one `(dst_nvl_rank, rdma_lane, channel)` queue:

```
tok_split_max = NUM_MAX_NVL_PEERS / (num_nvl_ranks * snd_split)   // = 2 here
snd_w    = warp_id / num_nvl_ranks
snd_sub  = snd_w % snd_split         // which RDMA lanes I own
tok_sub  = snd_w / snd_split         // which token chunks I own
live warps/channel = num_nvl_ranks * snd_split * tok_split        // 4 -> 8
```

**Slot mapping is unchanged**, which is mandatory: the consumer locates every
token through the precomputed `combined_nvl_head` map, so token at queue position
`p` must land in slot `p % capacity`. Sub-warp `t` simply takes the chunks
`c` with `c % tok_split == t` and writes them at their *fixed* positions
(`token_start_idx += tok_sub*chunk` initially, then `+= tok_split*chunk`).

The only new coordination is the **tail publish**, which must expose only the
contiguous completed prefix:

- New SLM board `smem_snd_progress[NUM_MAX_NVL_PEERS * tok_split][kNumRDMARanks]`.
- Each sub-warp publishes *the start of its next unprocessed chunk* (publishing
  "end of last completed chunk" is **wrong** — verified by hand-tracing
  `tok_split=2, CH=8, N=20`).
- Only `tok_sub == 0` writes `ch_tail`, with `min` over the `tok_split` entries,
  monotone-guarded by `last_published_tail`.
- Every sub-warp must publish **both** a pre-loop and a post-loop value. Without
  the pre-loop publish, a sub-warp whose range is empty (`tok_sub*CH >= N`) never
  enters the loop, leaves 0 on the board, and the `min` pins the tail at 0 → hang.
- `tok_sub == 0` must **outlive its own token range**: after its loop it runs a
  bounded drain loop republishing the min until it reaches `lane_num_tokens`,
  otherwise the other sub-warps' final tokens are never announced.
- The board needs an **unconditional whole-work-group `item.barrier()` at kernel
  entry** to zero it. Combine's NVL-sender warps participate in *no* named barrier
  (deviation C2), so they have no other way to agree it is initialised. Reading a
  stale 0 is safe (the tail merely does not advance yet), so the barrier only has
  to establish "not garbage".

Flow control is unchanged in form: `capacity - (position - head) >= chunk`, with
`position` re-derived from the token cursor rather than incremented monotonically.
No deadlock — a sub-warp holding a higher chunk index waits on `head`, which is
advanced by the consumer once the lower-index chunks (strictly smaller position
bounds) are drained.

`DEEP_EP_COMBINE_TOK_SPLIT=1` restores the previous behaviour bit-for-bit.

**Also measured:** `snd_split=1, tok_split=4` (same 8 live warps, purely token
partitioned) is **worse** than `snd_split=2, tok_split=2` — rt 13 974 vs 13 378 µs.
Prefer lane partitioning where it is available; use token splitting only to fill
what is left.

#### 6.1.2 NEGATIVE RESULT: the same trick on DISPATCH's forwarder does NOT pay

Dispatch telemetry (4096 tok, hidden 7168, 10 channels) shows
`kRDMAAndNVLForwarder` is the **longest-lived** dispatch role — 11.5e6 cyc/warp vs
7.2e6 for the RDMA sender, i.e. it spans the whole kernel — is 66% copy / 17% wait,
and has the *identical* idle-slot pathology (`target_rank = (warp_id+channel_id) %
NUM_MAX_NVL_PEERS`, active only when `< num_nvl_ranks` → **2 of 8 slots live**).
The naive projection was dispatch 7.5 ms → ~4.7 ms.

It was implemented (sub-warp 0 runs the control loop alone and publishes the
chunk's `(rdma_slot, nvl_slot)` pairs through SLM; the group then copies them
`fwd_split`-strided, two named-barrier arrivals per *chunk*) and measured:

| config | rt | dispatch(iso) | combine(iso) |
|---|---|---|---|
| `fwd_split=1` | 13 414.8 | 7 297.7 | 6 352.2 |
| `fwd_split=4` (4× the live forwarder warps) | 13 308.1 | **7 203.8** | 6 265.8 |

**1.3% — inside noise.** Quadrupling the forwarder warps did essentially nothing,
so the dispatch forwarder is **not warp-count-bound**; its long lifetime is
waiting on RDMA arrival rate, not on copy throughput. The telemetry "66% copy"
figure therefore includes stalls on data that has not landed yet and must not be
read as copy-bound.

Worse, the restructure **broke correctness even at `fwd_split=1`** (the supposedly
identical path): `topk_weights diff=1.15e-04` on the `with top-k` config of the
full matrix, reproducible, absent from both the pristine baseline and a build
carrying only the combine change. **The change was reverted in full.** Do not
re-attempt it without a decisive reason — the payoff ceiling is ~1%.

> Method note: the bisect that established this was worth its cost — pristine
> baseline PASS 64/64, combine-change-only PASS 64/64, combine+dispatch FAIL. A
> "should be bit-identical" claim about a restructured loop is not evidence.


### 6.2 Per-warp streaming ceiling ~0.115 GB/s — **the "6.8× gap" was an artifact**

Per-warp payload streaming is pinned around 0.115–0.19 GB/s essentially regardless of
code form, which is why aggregate bandwidth tracks live-warp count so closely.
**Do not re-open the following — all measured, all dead:**

- Unroll depth: 2→8 gave only −8 %; dispatch uses **no unroll** and was 5× faster.
- Per-token `group_barrier` in the copy loop: dispatch has the identical barrier.
- Peer-IPC write being slow: combine 163 k (self) vs 155 k (peer) cyc/tok;
  dispatch 23.9 k vs 23.7 k. **Destination type is irrelevant in both.**
- Plain C++ copy loop: 226 k cyc/tok, **43 % worse**.
- Queue/chunk depth: sender wait measured **0.0 %**.
- Work-group co-residency (128/128 concurrent WGs achieved at 64 KiB SLM).
- The three uncapped iSHMEM SQ spins — breaking all three left the hang rate
  unchanged, with a working positive control.
- **L1-cacheable source load (NEW, 2026-08).** The NVL sender reads `x` with
  `ld_nc_global_v` = `lsc_load.ugm.uc.ca`, i.e. L1 bypass. Unlike every other payload
  read in these kernels, `x` is an ordinary *local, coherent* torch tensor (never
  NIC-delivered, never peer-written), so a fully-cached `.ca.ca` load is legal there and
  was expected to restore L1 coalescing/prefetch. **Measured neutral**: combine
  6397.0 µs (`.ca.ca`) vs 6458.7 µs (`.uc.ca`) @2048 tok / hidden 7168 — ~1 %, inside the
  6369–6459 µs run-to-run band. The L1 hint is not the lever; the code was reverted.

#### The dispatch-vs-combine cycle gap does not exist per-warp

The previous edition of this document called the 23.9 k vs 163 k cyc/tok ratio "the most
valuable unexplained number in this document". **It is a telemetry normalization artifact
and is now retracted.** Normalising the *same* shipped run by live warp count instead:

| kernel | aggregate | live streaming warps | **GB/s per warp** |
|---|---|---|---|
| dispatch (`nvl_recv`) | 10.77 GB/s | 8/WG × 10 ch = 80 | **0.135** |
| combine (`nvl_send`) | 7.62 GB/s | 4/WG × 10 ch = 40 | **0.191** |

Per-warp throughput is the same to within measurement noise — combine is *not* less
efficient per warp, it simply runs **half as many streaming warps**. Confirmed directly by
sweeping `DEEP_EP_COMBINE_SND_SPLIT` (2048 tok / hidden 7168):

| `snd_split` | live sender warps | `nvl_send` GB/s | GB/s per warp |
|---|---|---|---|
| 1 | 20 | 2.65 | 0.132 |
| 2 (shipped) | 40 | 7.63 | 0.191 |

Doubling the warps gave **2.9×** the bandwidth — slightly *superlinear*, because more
concurrent warps also hide more memory latency.

> **Model: aggregate NVL bandwidth ≈ live_warp_count × ~0.15 GB/s, and the only lever is
> the warp count.** The ~0.15 GB/s/warp figure is a genuine per-warp memory-level-
> parallelism ceiling common to *both* kernels, not a combine-specific defect.

This makes §6.2 and §6.1 the same open item, and bounds the remaining upside: at the
production `num_nvl_ranks = 8` shape `snd_split` is already 1 and there are **no idle
sender warp slots left**, so the idle-slot opportunity exists only at small NVL widths.
Going further requires token-range splitting *within* one `(dst, rdma)` queue, which needs
ordered tail publication across warps — and the obvious implementation (a hand-rolled SLM
arrival counter) is exactly the construct §2.2 records as deadlocking on this stack. A
named barrier would be required, and the barrier budget is already near the IGC ICE cliff.

### 6.3 iSHMEM per-QP stall site — **root cause CONFIRMED; CLOSED as a perf item (§6.3.1)**

Previously four candidates, "none confirmed". The mechanism is now traced end-to-end, and
it is candidate (1); candidates (2) and (3) are *consequences* of the same path, not
independent bugs.

**The chain.** DeepEP publishes every RDMA queue tail with
`ishmemx_fence_qp` + `ishmemx_long_atomic_add_qp` (deviation C4). In iSHMEM:

```
ishmemx_long_atomic_add_qp                       (src/amo.cpp:339)
  -> ishmemi_ibgda_device_amo_nonfetch<long, AMO_ADD>
  -> ishmemi_ibgda_device_amo_fetch<long, AMO_FETCH_ADD>   (ibgda_device_impl.h:2923)
  -> ishmemi_ibgda_device_rdma_atomic64(ATOMIC_FA, ...)    (ibgda_device_impl.h:2460)
```

`amo_nonfetch` has **no non-fetching implementation at all** — every "non-fetch" op is
unconditionally rewritten to its fetching equivalent and the result is written into a
`dummy` the caller discards. `rdma_atomic64` then:

1. claims an ibuf slot for the NIC to DMA the fetched result into (→ candidate 2:
   `claim_ibuf_slot` exhaustion returns `false` and silently falls back to the host proxy —
   but that slot only exists *because* of the fetch);
2. posts the WQE and rings the doorbell;
3. **Step 4: blocks polling the collapsed CQ with an explicitly uncapped spin** —
   the in-source comment reads *"No spin cap -- a bounded cap silently drops flags"*;
4. reads the result out of ibuf — the value DeepEP throws away.

So **each tail publish is a synchronous full network round-trip** (post → NIC → remote →
CQE → poll) where DeepEP only ever needed a fire-and-forget increment. That is the
557 µs vs 37 µs on the wire.

**Why it degrades with channels-per-QP.** The CQ polled in step 3 is a *collapsed* CQ,
one per QP: a single `wc_counter` at `nic_cq_buf + 0x3C`. Every channel sharing that QP
blocks on the same counter waiting for its own `target_wc`. Because the counter only ever
reports the most recent completion, a waiter whose CQE is collapsed past never observes
its own target and — with no spin cap — **spins forever**. That is precisely the measured
signature: clean at 4 and 6 channels/QP, 4/6 hangs at ~12 channels/QP. It is a hang, not a
slowdown, exactly as reported.

**Why the shipped config is safe.** `ISHMEM_IBGDA_QPS_PER_PE=16` (§4.1) gives ~1 channel
per QP at the shipped 10 channels, so no two channels share a collapsed CQ. This is why
that setting is documented as a *correctness requirement rather than a tuning knob* — the
reason is now known.

**The fix, and why it is not landed here.** DeepEP does not need an atomic at all:
`rdma_channel_tail.buffer(rdma_rank)` is indexed by the **source** RDMA rank, so each
`(dst_pe, channel, src_rdma_rank)` tail has exactly **one writer**, and that writer already
tracks the absolute value in `last_issued_tail`. A plain 8-byte RDMA **write** of the
absolute tail, on the same QP, is sufficient — RC in-order gives the same
"flag-after-payload" guarantee the AMO relied on, with no ibuf slot, no CQ poll and no
collapsed-CQ hazard. It would address candidates 1–3 at once and is a DeepEP-side change,
so it needs no iSHMEM rebuild (preserving archive parity, §2.2).

The two iSHMEM entry points that would make this a one-line change are **declared but
never implemented** — `ishmemx_putmem_nbi_qp` and `ishmemx_putmem_signal_qp` exist only as
prototypes in `src/ishmemx.h` (2348, 2352) with no definition in any `.cpp`. The only
implemented QP-pinned put is the sub-group collective `ishmemx_putmem_nbi_subgroup`, so a
DeepEP-side fix must (a) restructure the single-lane publish into a sub-group-collective
call, and (b) add an 8-byte staging slot in the symmetric heap, since an RDMA write source
must be NIC-registered memory and `last_issued_tail` is a register. That is a protocol
change to the most hang-prone path in the port and needs a full intermittency campaign
(k/N over many runs) to land safely — deliberately **not** attempted in the same pass as
the measurements above.

A parked design for a non-fetching `ishmemx_long_atomic_add_qp_nbi`
(`ishmem-nbi-amo-patch-design.md`) attacks the same root cause from the iSHMEM side, but
requires rebuilding `libishmem.a` and therefore re-validating archive parity for the LL
path as well.

#### 6.3.1 Is the blocking AMO actually a throughput bottleneck? — **NO (measured)**

The fix above was scoped but deliberately not landed, because a direct measurement shows
the blocking AMO **is not on the throughput critical path** at the shipped configuration.

The test issues one tail-publish AMO per RDMA chunk, and chunks are counted in *tokens*,
so `num_max_rdma_chunked_send_tokens` is a direct control on the **AMO count** at fixed
byte volume. `Config` now honours `DEEP_EP_RDMA_CHUNK` / `DEEP_EP_NVL_CHUNK`
(`xpu_runtime.hpp`) purely so this can be varied without editing a caller — the frozen
`tests/test_internode.py` hardcodes `Config(num_sms, 8, nvl, 16, rdma)`.

If the blocking AMO dominated, **halving the AMO count should reduce dispatch time**.
It does the opposite. Dispatch(iso), 2048 tokens, hidden 1024 (chosen because the
byte-independent floor is ~80% of dispatch there, so the effect is amplified):

| `rdma_chunk` | 4 | 8 | **16 (shipped)** | 32 | 64 |
| --- | --- | --- | --- | --- | --- |
| dispatch(iso) µs | 6980.9 | 3381.1 | **2030.1** | 2434.1 | 3430.2 |

A clean unimodal curve peaking **exactly at the shipped value**, with 4× fewer AMOs
(`64`) being **69% slower**. The AMO count is therefore not what sets the floor;
**pipelining granularity** is. Larger chunks make the sender accumulate longer before
publishing, starving the forwarder; smaller chunks lose per-chunk efficiency.

This also explains the "557 µs vs 37 µs" figure that motivated §6.3: it is an aggregate
over a microbenchmark, not a per-op cost on this path. At the test shape there are only
~7 AMOs per channel and channels run concurrently, so even a pessimistic 10 µs round-trip
contributes ~70 µs to a ~4500 µs dispatch (~1.5%).

**Conclusion — §6.3 is CLOSED as a performance item.** The root-cause analysis above
remains correct and is still the reason `ISHMEM_IBGDA_QPS_PER_PE=16` is a *correctness*
requirement (it keeps ~1 channel/QP, avoiding the collapsed-CQ hang). But replacing the
AMO with an RDMA write is a **robustness / configuration-simplification** change, not a
throughput win, and must not be justified on perf grounds. Given it is a protocol change
to the most hang-prone path in the port, the cost/benefit does not support landing it.

#### 6.3.2 Side result: the chunk-size optimum is shape-dependent

The same sweep at the headline shape (2048 tokens, hidden 7168) puts the optimum at
`rdma_chunk=8`, not 16. Two independent samples each, run-to-run spread ±9 µs:

| | round-trip µs | dispatch(iso) µs | combine(iso) µs |
| --- | --- | --- | --- |
| `rdma_chunk=16` (shipped) | 10644.8 / 10659.0 | 4513.4 / 4506.8 | 6369.2 / 6409.3 |
| `rdma_chunk=8` | 10444.7 / 10426.0 | **4334.9 / 4344.0** | 6361.1 / 6353.1 |
| `rdma_chunk=32` | 12096.7 | 5194.1 | 7004.0 |

`nvl_chunk` is flatter and the shipped value is already near-optimal (combine, hidden
7168): `4` → 6381.8 µs, **`8` (shipped)** → ~6389 µs, `16` → 7088.7 µs.

The two knobs compose. Best measured configuration at hidden 7168,
`rdma_chunk=8 nvl_chunk=4`: round-trip **10242.5 µs (-3.8%)**, dispatch **4301.0 µs
(-4.6%)**, combine 6326.8 µs (-1.0%). Verified 64/64 on the correctness gate.

**The shipped defaults are deliberately left unchanged.** The optimum inverts with shape
(16 wins at hidden 1024, 8 wins at hidden 7168), and every number here comes from a
`num_nvl_ranks=2, kNumRDMARanks=2` rig — exactly the configuration §6.2 warns does not
generalise to the production `nvl=8` shape. These are tuning data and a knob, not a new
default; re-run the sweep on the target shape before changing anything.

### 6.4 Can the SM clamp be raised? — **CLOSED: no, 20 is optimal**

`kEmpiricalSafeSms = 8` predates the QP-contention diagnosis and is dead at the shipped
`QPS_PER_PE=16` (`qp_safe = 2·2·16 = 64` dominates the `max()`), so the binding
constraint is the **driver co-residency query**, which answers **20** on Arc Pro B60
(= its Xe-core count). The open question was whether to push past it, on the strength of
`num_sms=24` having been measured *clean*.

**Measured, and the answer is no.** `num_sms=24` is correct but 32 % SLOWER. Sweep at
2048 tok / hidden 7168 / `QPS_PER_PE=16`, rank 0, via `DEEP_EP_FUSED_MAX_SMS`:

| num_sms | round-trip µs | dispatch µs | combine µs | vs. 20 |
|---|---|---|---|---|
| 12 | 15541.2 | 5332.6 | 10823.1 | +46 % |
| 16 | 12065.7 | 4630.5 | 7579.4 | +13 % |
| **20** (driver bound, shipped) | **10644.8** | **4513.4** | **6369.2** | — |
| 24 | 14097.5 | 4765.7 | 9577.8 | **+32 %** |

The curve is unimodal and peaks **exactly** at the driver bound. Mechanism: the fused path
runs **two mutually-spinning work-groups per channel** (sender ↔ forwarder). Once the grid
exceeds what the device holds resident, some channel has one WG scheduled and its partner
not, so the resident WG burns its Xe-core spinning on a partner that cannot run until it
yields — a scheduling deadlock survivable only because of `kFusedSpinCap`. Below the bound
the kernel is warp-starved instead (§6.1/§6.2), which is why 16 and 12 also lose.

> **CLOSED SUB-QUESTION (2026-08-26): can the bound be raised by SHRINKING the work-group?**
> No. The obvious follow-on — since `num_sms` is capped by a *per-kernel occupancy* query,
> shrink `DEEP_EP_COMBINE_FWD_WARPS` (24 → 16 → 12 → 8) so the combine WG shrinks from 800
> work-items and the driver hands back a bigger number — was falsified without a single
> rebuild sweep, by printing both caps:
> ```
> [DeepEP] fused co-residency query: dispatch_wg=512 cap=20 | combine_wg=800 cap=20
> ```
> The query returns **20 for both**, a 512-work-item kernel and an 800-work-item one. It is
> reporting the **Xe-core count** (160 EU / 8 per subslice = 20), not a work-group-size-
> sensitive occupancy. Shrinking the work-group cannot raise it, so this whole avenue is
> dead. `DEEP_EP_COMBINE_FWD_WARPS` was left `#ifndef`-wrappable (harmless) but there is no
> reason to sweep it. **Print both caps before theorising about the clamp.**

> **The earlier "24 is clean 16/16" result was a *correctness* observation and was
> mis-read as a throughput opportunity.** Correct and fast are different questions.

**Resolution.** Ship the driver bound unchanged. `fused_max_coresident_sms()` now emits an
explicit WARNING when `DEEP_EP_FUSED_MAX_SMS` is pushed above the device-derived value.
`kEmpiricalSafeSms` is retained because it is *not* dead at low `QPS_PER_PE` (at
`QPS_PER_PE=1` it is what clamps the grid to 8).

### 6.5 Reduced-precision residuals

Combine output `x` shows BF16 errors of 0.0078/0.0156 (1–2 BF16 ULPs) and FP8
~0.11–0.14. Consistent with intrinsic rounding but **not proven benign** (see §7).

## 7. Testing and validation guidance

### 7.1 Running the tests

**Perf sweep** (what produced §5). `DEEP_EP_PERF_TOKENS` reuses ONE iSHMEM init and
buffer across the whole token list, avoiding one relaunch per size and the
associated `DEVICE_LOST`-accumulation risk:

```bash
cd tests/docker-2node-v2
docker rm -f deepep-v2-node0 deepep-v2-node1
rm -f /dev/shm/sem.psm3* /dev/shm/psm3_* /dev/shm/sem.ishmem* /dev/shm/*ishmem* \
      /tmp/deep_ep_xpu_ipc_*.sock
env ISHMEM_DIR=/root/jiafuzha/ishmem_ibgda/build/_install SKIP_BUILD=1 TIMEOUT_SEC=900 \
    NUM_PROCESSES=2 NUM_TOKENS=$T HIDDEN=7168 NUM_TOPK=2 NUM_EXPERTS=8 \
    DEEP_EP_PERF_TOKENS=$T \
    DEEP_EP_NVL_BYTES=536870912 DEEP_EP_RDMA_BYTES=536870912 \
    ISHMEM_SYMMETRIC_SIZE=1073741824 \
    bash run.sh
```

**Correctness gate** — the same command with `DEEP_EP_PERF_TOKENS` **removed**, so
the full 64-config matrix runs. Expect 64 `passed` lines.

Traps, each learned the expensive way:

- **`DEEP_EP_PERF_TOKENS` sets `DEEP_EP_MIN=1` and `DEEP_EP_PERF=1`**
  (`tests/test_internode.py:632-634`) → **2** `passed` lines, not 64.
  **Perf mode is blind to correctness. Never validate a fix in it.**
- **The 64-config matrix at 32 tok / hidden 1024 is PROVEN BLIND** to both bugs
  fixed in this port. Gates must run at **2048+ tokens, hidden 7168**.
- Healthy full-matrix wall clock at 2048/7168 is **~80 s** (~40 s of that is
  first-launch SPIR-V JIT; the other 63 configs are ~0.5 s each). Judge
  hang-vs-slow by **log mtime advancing**, not by return code.
- `run.sh` forwards env only via an explicit `_add_opt_genv` whitelist (~:324) — a
  new debug variable silently does nothing until it is added there.
- Containers share GPUs and NICs → **only one 2-node sim at a time**.
- Report intermittent results as `k/N`, never as "works".

### 7.2 ⚠️ The `x` tolerance is effectively vacuous

`tests/test_internode.py:430` uses `tol = max(5e-6, 6e-4*(N/32)^2)` on XPU
→ **2.46 at 2048 tokens, 9.83 at 4096**. The assert cannot fail. A probe showed the
test's own diagnostic fires on 32/32 configs with **0 "x OK"**. `topk_weights` *is*
checked strictly (`tw_diff < 1e-9`, unscaled), which is currently the only real
numerical gate. Tightening this requires touching `tests/`, which has been kept
byte-identical to pre-migration `dc2f2bd` and is off-limits without explicit
sign-off.

### 7.3 Build

```bash
source /opt/intel/oneapi/setvars.sh --force && conda activate <env>
export DEEP_EP_TARGET=xpu ISHMEM_DIR=/root/jiafuzha/ishmem_ibgda/build/_install
unset TORCH_XPU_ARCH_LIST XPU_AOT_TARGETS     # bmg-only AOT is unstable -> DEVICE_LOST
rm -rf build/ishmem-sycl-dlink                # force iSHMEM archive re-extraction
touch csrc/xpu/*.cpp                          # setuptools does NOT track .inc deps
python setup.py build_ext --inplace
```

Verify both, every time:
- `strings deep_ep_cpp*.so | grep -m1 -- '-device '` →
  `-device pvc,bmg,arl-h,mtl-h,lnl-m,ptl-h,ptl-u` (**not** `-device bmg`).
- iSHMEM `barrier.cpp.o` md5 matches the known-good archive. A ~32 ms/iter LL
  result means the wrong archive was linked.
- `strings deep_ep_cpp*.so | grep <a-string-you-just-added>` to confirm the `.inc`
  edit actually made it in.

Build **inside the container** if the host oneAPI differs from the container's
(2026.0 host vs 2025.3 container → `llvm-link: linked module is broken`).

### 7.4 Untested envelope

Nothing below has evidence either way — treat as unknown, not as working:

- `num_rdma_ranks == 4` (permitted at `internode.cpp:434`, never run — needs 4 nodes).
- **`num_nvl_ranks > 2`**, i.e. the production `NUM_MAX_NVL_PEERS = 8` shape
  (needs 8 GPUs; only 4 available). Note this is exactly where `snd_split` becomes
  a no-op (§6.1).
- tokens > 4096; experts/topk beyond 8/2 at large N.
- `ISHMEM_IBGDA_QPS_PER_PE=1` is **known-hanging** for this path.
