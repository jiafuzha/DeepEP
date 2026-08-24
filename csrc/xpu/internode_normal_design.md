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

### 6.2 Per-warp streaming ceiling ~0.115 GB/s

Per-warp payload streaming is pinned around 0.115 GB/s essentially regardless of
code form, which is why aggregate bandwidth tracks live-warp count so closely and
plateaus around 2.3 GB/s. **Do not re-open the following — all measured, all dead:**

- Unroll depth: 2→8 gave only −8 %; dispatch uses **no unroll** and was 5× faster.
- Per-token `group_barrier` in the copy loop: dispatch has the identical barrier.
- Peer-IPC write being slow: combine 163 k (self) vs 155 k (peer) cyc/tok;
  dispatch 23.9 k vs 23.7 k. **Destination type is irrelevant in both.**
- Plain C++ copy loop: 226 k cyc/tok, **43 % worse**.
- Queue/chunk depth: sender wait measured **0.0 %**.
- Work-group co-residency (128/128 concurrent WGs achieved at 64 KiB SLM).
- The three uncapped iSHMEM SQ spins — breaking all three left the hang rate
  unchanged, with a working positive control.

The dispatch-vs-combine per-token cycle gap (23.9 k vs 163 k) is ~6.8× and is
**not** explained by any of the above. That gap is the most valuable unexplained
number in this document.

### 6.3 Open iSHMEM stall site (per-QP contention)

Mechanism established: onset at ~12 channels/QP; 4 and 6 channels/QP clean.
Remaining candidates, none confirmed:

1. Uncapped CQ poll at `ibgda_device_impl.h:2624-2645`.
   `ishmemx_long_atomic_add_qp` silently maps to a **fetching** AMO (`~:2923`):
   557 µs vs 37 µs on the wire.
2. `claim_ibuf_slot` exhaustion → silent host-proxy fallback.
3. Shared `ibuf_base_addr` bug at `ibgda.cpp:1064-1075`.
4. Possibly DeepEP's own spin.

Needs `gdb-oneapi`; device `printf` is lost when a kernel hangs.
A parked design for a non-fetching `ishmemx_long_atomic_add_qp_nbi` exists
(`ishmem-nbi-amo-patch-design.md`) and would address candidate (1) directly.

### 6.4 Can the SM clamp be raised?

`kEmpiricalSafeSms = 8` predates the QP-contention diagnosis. With `QPS_PER_PE=16`
the `qp_safe` term dominates anyway, but the shipped default is `num_sms=20` while
`num_sms=24` has been measured clean 16/16 @2048 and 8/8 @4096. Raising the default
is the cheapest available throughput knob and should be decided in review.

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
