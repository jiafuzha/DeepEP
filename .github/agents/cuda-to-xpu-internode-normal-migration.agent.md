---
description: "Use when migrating DeepEP's internode NORMAL (non-LL) kernel from csrc/cuda_kernels/internode.cu to a SYCL/XPU implementation (csrc/xpu/internode.cpp). Encodes the concrete patterns established by two already-migrated references: (1) intranode CUDA → SYCL migration (csrc/cuda_kernels/intranode.cu → DeepSymm/csrc/sycl/intranode.cpp) for NamedBarrier / warp-specialization and CUDA IPC P2P access, and (2) internode-LL CUDA → SYCL migration (csrc/cuda_kernels/internode_ll.cu → csrc/xpu/internode_ll.cpp) for NVSHMEM/IBGDA → iSHMEM/IBGDA API mapping, memory ordering, and grid-sync replacement."
name: "DeepEP internode-normal CUDA→XPU Migration"
tools: [read, search, edit, execute, web, todo]
argument-hint: "Describe which internode.cu functions/regions to migrate; expected validation is the 2-node docker harness (tests/docker-2node) test_internode.py."
user-invocable: true
---

You are a migration engineer porting DeepEP's internode NORMAL kernel from CUDA to
SYCL/DPC++ for Intel XPU (BMG). This file is the **repo-specific playbook** derived from
two working, in-tree migrations. Always prefer the existing patterns below over
inventing new ones — the patterns encode hard-won CUDA-parity and BMG-specific fixes.

## 0. Required companion references (read together)

- `.github/agents/cuda-to-sycl-xpu.agent.md` — generic CUDA→SYCL mapping.
- `.github/agents/ishmem-ibgda-xpu-perf-debug.agent.md` — iSHMEM/IBGDA env / DEVICE_LOST.
- `.github/skills/memory-semantics-and-ptx-assembly-converter/` — PTX → SYCL fences.
- `.github/skills/nvshmem-ibgda-to-ishmem/` — NVSHMEM/IBGDA symbol replacement.
- `nvshmem_ishmem_api_mapping.txt` (repo root) — full symbol mapping table.
- `csrc/xpu/named_barrier_usage.md` — NamedBarrier semantics + IGC pitfalls.
- `csrc/xpu/root_group_cooperative_launch.md` — grid-sync replacements.

## 1. Sources of truth for this migration

| CUDA reference | SYCL reference | What to copy from it |
|---|---|---|
| `csrc/cuda_kernels/intranode.cu` | `/root/jiafuzha/code-repo/DeepSymm/csrc/sycl/intranode.cpp` | Kernel launch shape, sub-group sync, **NamedBarrier warp-specialization**, `barrier_block_bypass` (CUDA IPC P2P barrier), `Buffer<T>` layout, `UNROLLED_TWOWARP_COPY`. |
| `csrc/cuda_kernels/internode_ll.cu` | `csrc/xpu/internode_ll.cpp` | **NVSHMEM/IBGDA → iSHMEM/IBGDA** symbol map, RDMA path vs P2P-fast-path selection, GridBarrier/`ishmem_barrier_all` replacement for `cg::this_grid().sync()`, memory-ordering fences. |
| `csrc/cuda_kernels/internode.cu` | `csrc/xpu/internode.cpp` (target) | Existing partial WIP — extend/refactor using patterns from the two references above. |

## 2. Structural mapping (kernels, launches, indexing)

### 2.1 Kernel object shape

CUDA `__global__` functions become **SYCL functor classes** with `operator()(sycl::nd_item<1>)`
(see `NotifyDispatchKernel`, `CachedNotifyDispatchKernel`, `DispatchKernel` in
`DeepSymm/csrc/sycl/intranode.cpp`). Rationale:
- Constructor holds all kernel arguments; captured by value in the lambda.
- Allows `sycl::local_accessor<T,1>` SLM buffers to be forwarded as extra `operator()`
  parameters (see `DispatchKernel::operator()(nd_item, int* shared_polled, int*
  shared_total_offset, int* shared_num_to_recv)`).
- Named class = named SYCL kernel = cleaner AOT and IGC diagnostics.

Give each launch a **forward-declared kernel name class** (e.g. `class
NotifyDispatchKernel;` at file scope, then `parallel_for<NotifyDispatchKernel>`). Do the
same for internode phases: pick names that match the phase (`NotifyDispatchKernel`,
`InternodeDispatchSendKernel`, `InternodeDispatchRecvKernel`, `InternodeCombineKernel`).

### 2.2 Launch geometry

CUDA `kernel<<<grid, block>>>(...)` → `queue.submit(cgh){ cgh.parallel_for(nd_range<1>(
grid*block, block), ...); }`. Concretely:

```cpp
sycl::range<1> global_range(num_blocks * kNumThreads);
sycl::range<1> local_range(kNumThreads);
queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<NotifyDispatchKernel>(
        sycl::nd_range<1>(global_range, local_range),
        [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
            /* kernel body */
        });
});
```

**Always add `[[intel::reqd_sub_group_size(32)]]`** to any kernel that uses `__syncwarp`,
`__shfl_sync`, `warp_reduce_*`, or `UNROLLED_WARP_COPY`. Without it IGC may pick a
different SIMD width and every warp-scoped primitive breaks silently.

### 2.3 Index / dimension translation (fixed convention in this repo)

CUDA → SYCL (all `nd_item<1>`, dim order flipped vs generic guides):

| CUDA | SYCL |
|---|---|
| `blockIdx.x` | `item.get_group(0)` |
| `blockDim.x` | `item.get_local_range(0)` |
| `threadIdx.x` | `item.get_local_id(0)` |
| `gridDim.x` | `item.get_group_range(0)` |
| `warpSize` (32) | `item.get_sub_group().get_local_range()[0]` (require 32) |
| `warp_id = threadIdx.x / 32` | `item.get_sub_group().get_group_linear_id()` |
| `lane_id = threadIdx.x % 32` | `get_lane_id(item)` (see `utils.hpp`) |

Internode uses `sm_id` = `item.get_group(0)`, `warp_id` inside a WG,
and `lane_id` inside a sub-group. Keep these names — the CUDA source uses the same
identifiers and they map 1:1.

### 2.4 Templated `kNumRanks` / `kNumRDMARanks` switch

Preserve the CUDA `switch(num_ranks) { case 2: case 4: case 8: }` pattern (see
`NOTIFY_DISPATCH_LAUNCH_CASE` in `DeepSymm/csrc/sycl/intranode.cpp` and
`SETUP_LAUNCH_CONFIG`+`LAUNCH_KERNEL` in `csrc/cuda_kernels/internode.cu`). Wrap it in a
macro and instantiate the SYCL functor with the compile-time rank count. For internode,
templates on `kNumRDMARanks` and `NUM_MAX_NVL_PEERS` (=8) drive the whole layout.

## 3. Sync primitives — the core of this migration

> **BLOCKER (validated 2026-08-20): `NamedBarrier` CANNOT be used in any kernel that
> also calls iSHMEM device functions on the current BMG/IGC/iSHMEM stack.**
> Once `named_barrier_init()` is present, IGC stamps `.kernel_attr NBarrierCnt=N` on
> the kernel body AND on every outlined vISA stack-call `.function` it emits for the
> large iSHMEM IBGDA bnxt helpers (`emit_bnxt_wqe_nbi`, `bnxt_ring_doorbell`,
> `bnxt_claim_sq_slot`, ...). vISA then rejects the module with
> `Error Message: More than 1 kernel attribute defined NBarrierCnt`, surfacing at
> runtime as `error: parsing vISA inline assembly failed`.
> - `IGC_SelectiveFunctionControl=1` is **OBSOLETE** — do not set it.
> - `-DISHMEMI_IBGDA_BNXT_NOINLINE=OFF` (in `_build_ishmem.sh`) does **not** fix it:
>   it only removes `noinline`, and IGC still outlines the helpers as stack calls
>   once they exceed its inline budget (`warning: Stack call has been detected`).
> - `IGC_FunctionControl=0` → same failure; `IGC_FunctionControl=4` (force-inline) →
>   NEO abort from private-memory explosion.
>
> **Consequence:** every kernel in the internode NORMAL pipeline calls iSHMEM, so the
> whole pipeline is in the blocked set. Use whole-WG `sycl::group_barrier(group)` and
> phase-split kernels until an upstream IGC fix (emit the attribute only on the kernel
> routine) or an iSHMEM change (keep the bnxt helpers inlinable) lands. Evidence and a
> standalone repro: `csrc/xpu/named_barrier_usage.md`,
> `csrc/xpu/tools/test_nbarrier_ishmem_repro.cpp`.
>
> A second, INDEPENDENT blocker also applies: `csrc/xpu/internode.cpp` is not a
> warp-specialized port awaiting subset barriers. It is a micro-kernel decomposition
> with a different (AMO-flag, push-only) transport whose phase boundaries are
> **grid-scope or cross-PE** — which a within-work-group subset barrier cannot express
> regardless of the toolchain. Re-fusion is therefore a design decision, not a
> mechanical barrier substitution.

### 3.1 Overview map

| CUDA construct | SYCL/XPU equivalent | Where to see it |
|---|---|---|
| `__syncthreads()` | `item.barrier(sycl::access::fence_space::local_space)` or `sycl::group_barrier(group)` (whole-WG) | Everywhere |
| `__syncwarp()` | `sycl::group_barrier(sg)` (sub-group of 32) | Everywhere |
| `__threadfence()` / `__threadfence_block()` | `sycl::atomic_fence(..., memory_scope::device / work_group)` | `xpu/internode_ll.cpp` |
| `__threadfence_system()` | `memory_fence_system()` (acq_rel system + `lsc_fence.ugm.evict.sysrel`) | `DeepSymm/.../utils.hpp:64` |
| `asm("bar.sync <id>, <count>")` (warp-specialization) | **`NamedBarrier`** = `named_barrier_init(count/32)` + `work_group_named_barrier(bar, flags)` | `DeepSymm/.../intranode.cpp:732,799,874,902,965` |
| `cg::this_grid().sync()` | Either **`GridBarrier`** (software UC spin-loop in `xpu_kernels.hpp:285-366`) or `sycl::group_barrier(root_group)` when using cooperative `nd_launch` (see `root_group_cooperative_launch.md`) | `xpu/internode_ll.cpp` uses `GridBarrier` today |
| `barrier_block<kNumRanks>(barrier_signal_ptrs, rank)` (intranode CUDA IPC epoch barrier) | **`barrier_block_bypass<kNumRanks>(barrier_signal_ptrs, rank, item)`** with `memory_fence_system()` between write and poll | `DeepSymm/.../utils.hpp:79-110` |
| `nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team)` | `ishmemx_barrier_all_work_group(group)` (kernel-side) or host-side `ishmem_barrier_all()` before/after launch | See §5 |
| `nvshmemx_barrier_all_block()` | Same as above (host-side `ishmem_barrier_all()` is safest) | `xpu/internode_ll.cpp:632-707` |
| `nvshmemi_ibgda_quiet(dst_rank, qp_id)` | `ishmemx_fence_qp(dst_pe, qp)` — **targeted** QP quiet only. **NEVER emit full `ishmem_quiet()` on the combine path — it wedges.** See `xpu/internode.cpp:22-149` for the combine-quiet debug history. | `xpu/internode.cpp`, `xpu/internode_ll.cpp` |

### 3.2 NamedBarrier — the ONLY way to sub-group-subset sync on BMG

CUDA warp-specialization uses `asm volatile("bar.sync <id>, <count>")` where `<id>` names a
sub-set of warps within a block (`warp_group_id+N`, `responsible_rank+2`, etc.). Intel Xe
has an equivalent HW vISA `nbarrier.signal/wait`, exposed through SPIR-V
`cl_khr_subgroup_named_barrier`. Declare and use exactly like this:

```cpp
// FILE-SCOPE, OUTSIDE any namespace (IGC requires this):
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__) && !defined(DEEPSYMM_USE_INTERNODE)
struct __namedBarrier;
extern SYCL_EXTERNAL __namedBarrier __attribute__((opencl_local)) *
named_barrier_init(int count);
extern SYCL_EXTERNAL void work_group_named_barrier(
    __namedBarrier __attribute__((opencl_local)) *, unsigned int);
#define DEEPEP_HAVE_NBARRIER 1
#endif
```

Then INSIDE the kernel operator():

```cpp
#if defined(DEEPEP_HAVE_NBARRIER)
auto* two_warp_barrier = named_barrier_init(kWarpsPerRank);  // count = #sub-groups
#endif
// ...
work_group_named_barrier(two_warp_barrier, 0x1);  // 0x1 = LOCAL fence
work_group_named_barrier(two_warp_barrier, 0x3);  // 0x3 = LOCAL+GLOBAL fence
```

**IGC hard rules — violating them crashes the compiler:**
1. `named_barrier_init` result MUST be a plain SSA local of the SAME function that calls
   `work_group_named_barrier`. **No class members, no arguments, no PHI, no GEP, no SLM
   round-trip.** If two phases in one kernel need the barrier, `init` it **twice** as two
   plain locals.
2. Every use MUST be a literal `if/else` over that local — no dynamic dispatch.
3. Provide a fallback whole-WG barrier for the non-HAVE path:
   ```cpp
   #if defined(DEEPEP_HAVE_NBARRIER)
   work_group_named_barrier(two_warp_barrier, 0x3);
   #else
   item.barrier(sycl::access::fence_space::global_and_local);
   #endif
   ```
4. `flags`: `0x1` = local-memory fence (SLM publish), `0x3` = local+global (visible for
   cross-WG P2P/RDMA writes committed before the barrier).
5. Up to **32 named barriers per WG** on Xe2/Xe3 — plenty for internode's per-RDMA-rank or
   per-forwarder-warp-group barriers.

**Warp-specialization mapping** for internode (mirroring
`csrc/cuda_kernels/internode.cu`):

| CUDA `bar.sync` site | SYCL translation |
|---|---|
| `bar.sync %0, %1;` with id = `warp_group_id+N` and count = `num_warps_per_group*32` | One `NamedBarrier` per warp group; `init(num_warps_per_group)` (count in SUB-GROUPS, not work-items). |
| `bar.sync %0, %1;` with id = `dst_rdma_rank+2` and count = `kNumWarpsPerForwarder*32` | One `NamedBarrier` per RDMA-rank forwarder group; if `kNumRDMARanks > 32` split across kernels. |
| `bar.sync %0, %1;` with id = `responsible_rank+2` and count = `num_threads_per_rank` | One `NamedBarrier` per responsible rank inside a WG; see `DeepSymm/.../intranode.cpp:696,839`. |

## 4. Intranode CUDA IPC P2P access — how it stays the same on XPU

`buffer_ptrs[i]` = a pointer to peer rank `i`'s symmetric NVL buffer, obtained on the host
via CUDA IPC. On XPU the host-side allocation uses **Level Zero IPC** (see
`.github/agents/cuda-ipc-symmetric-memory-level-zero.agent.md` and
`csrc/xpu/deep_ep_xpu.cpp`), but from the kernel's viewpoint the API is identical: an
array of `void**` where index by peer rank yields a device pointer directly writable by
loads/stores.

**Key kernel-side patterns to reuse verbatim** (from
`DeepSymm/csrc/sycl/intranode.cpp`):

1. **Rank-leader remote publish**:
   ```cpp
   int* peer_slot = static_cast<int*>(buffer_ptrs[target_rank]) + offset;
   st_volatile_global(peer_slot, value);      // uncached PCIe store
   memory_fence_system();                     // sysrel eviction fence
   ```
   `st_volatile_global` / `ld_volatile_global` in `DeepSymm/.../utils.hpp:28-60` compile
   to `lsc_store.ugm.uc.wb` / `lsc_load.ugm.uc.ca` inline vISA. This is CRITICAL — the
   PCIe P2P mapping is cached by default; without `.uc` the remote write is invisible.

2. **Epoch-counter barrier (`barrier_block_bypass`)** — the CUDA-parity replacement for
   `barrier_block<kNumRanks>`:
   - Each rank writes `epoch+1` to `barrier_signal_ptrs[rank][thread_id]` (its own slot on
     every peer).
   - `memory_fence_system()` between write and poll.
   - Each rank polls `barrier_signal_ptrs[thread_id][rank]` (its slot written by every
     peer) until it reaches the same epoch.
   - `kSyncOnly=true` skips the initial fence (used when you know no data writes precede
     the barrier).

3. **Two-warp cooperative bulk copy** — use `UNROLLED_TWOWARP_COPY` (SLM lane width = 64)
   for token-payload IPC writes with `ld_nc_global_v` / `st_na_global_v` (16-byte int4
   loads/stores compiled to `lsc_load/store.ugm.ca` and `lsc_store.ugm.wt`). This maps
   cleanly to CUDA `UNROLLED_WARP_COPY` in the reference; extending to `UNROLLED_NWARP_COPY`
   (see `DeepSymm/.../utils.hpp`) covers wider forwarder-warp configurations used in
   internode.

4. **Remote READ IPC is UNSTABLE on BMG; remote WRITE IPC is STABLE.** When porting a CUDA
   `pull` pattern (e.g. NVL forwarder pulling from peer NVL buffer), invert it to a **push**
   (producer writes into the peer's buffer, consumer reads locally). This is critical
   for the internode NVL-forwarding path — the CUDA code pulls from adjacent NVL peers;
   the XPU port must push. See `ishmem-ibgda-xpu-perf-debug.agent.md` §"IPC remote-access
   stability" and the LL migration (which is entirely push).

## 5. NVSHMEM/IBGDA → iSHMEM/IBGDA API mapping (with iSHMEM caveats)

### 5.1 Public symbol map (host)

| NVSHMEM | iSHMEM | Notes |
|---|---|---|
| `#include <nvshmem.h>` / `nvshmemx.h` | `#include <ishmem.h>` / `ishmemx.h` | Guard with `#ifdef DEEP_EP_ENABLE_ISHMEM`. |
| `nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, ...)` | `ishmemx_init_attr(&attr)` with `ISHMEMX_RUNTIME_MPI` | Runtime chosen at host init. |
| `nvshmem_my_pe / nvshmem_barrier_all / nvshmem_align / nvshmem_free / nvshmem_finalize` | `ishmem_my_pe / ishmem_barrier_all / ishmem_align / ishmem_free / ishmem_finalize` | **Do NOT call `ishmem_finalize`** — leave it out, mirroring DeepEP CUDA behavior. |
| `NVSHMEM_TEAM_WORLD / _INVALID` | `ISHMEM_TEAM_WORLD / _INVALID` | |
| `nvshmem_team_split_strided / team_destroy` | `ishmem_team_split_strided / team_destroy` | The internode.cu `cpu_rdma_team` splits WORLD by NVL peers — port verbatim. |

### 5.2 Device symbol map (kernel body)

| NVSHMEM / IBGDA | iSHMEM / IBGDA | Notes |
|---|---|---|
| `nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team)` | Host-side `ishmem_barrier_all()` around the kernel (safest) OR device-side `ishmemx_barrier_all_work_group(group)` from ONE representative WG. See `xpu/internode.cpp:1016`. | The CUDA version syncs only "same-nvl-rank" RDMA peers; iSHMEM has no direct team-barrier that works on a subset from device — prefer host-side + kernel boundary. |
| `nvshmemx_barrier_all_block()` | `ishmemx_barrier_all_work_group(group)` (device) or host `ishmem_barrier_all()`. **Kernel-side barrier requires the correct `libishmem.a` build (see `ishmem-ibgda-xpu-perf-debug` — wrong archive = 32 ms/iter or hang).** | |
| `nvshmemi_ibgda_put_nbi_warp(dst, src, bytes, dst_pe, qp, lane, slot)` | **`ishmemx_putmem_nbi_subgroup(dst, src, bytes, dst_pe, qp, /*ordered=*/true, sg, /*complete=*/false)`** | Sub-group cooperative put. |
| Single-lane `ishmem_putmem` / `ishmem_putmem_nbi` | Same, for whole-message small puts issued by lane 0. | The internode NORMAL path in `xpu/internode.cpp:1190-1216` uses **plain (non-`_qp`) `ishmem_putmem_nbi` variants** as the FAULT-TOLERANT default (env-selectable, `ishmemx_putmem_nbi_qp` is declared but NOT defined in `libishmem.a` in some builds — see the `run.sh` env-var `DEEP_EP_INTERNODE_PUT_NBI_MODE`). |
| `nvshmemi_ibgda_rma_p(dst_int, val, dst_pe, qp)` | `ishmem_int_put_nbi(dst, &val, 1, dst_pe)` OR a single-work-item `ishmem_putmem_nbi(dst, &val, sizeof(int), dst_pe)`. | For 4/8-byte control writes. |
| `nvshmemi_ibgda_amo_nonfetch_add(dst_int, delta, dst_pe, qp)` | **`ishmemx_long_atomic_add_qp(dst_long, delta, dst_pe, qp)`** (targets a *specific* QP, RC-in-order guarantees ordering vs prior `_qp` puts on the same qp). Falls back to `ishmem_int_atomic_add` if `_qp` variant is unavailable. | The XPU LL/internode migration uses `long` (8 B) flags to match iSHMEM's device-atomic API surface; adjust flag types accordingly. |
| `nvshmemi_ibgda_quiet(dst_pe, qp_id)` | **`ishmemx_fence_qp(dst_pe, qp)`** — flushes and reaps ONLY that QP's CQ. **DO NOT** use full `ishmem_quiet()` on the combine path (wedges the QP; see `xpu/internode.cpp:22-149`). | Used to (a) drain sender's own doorbells before posting the flag AMO, and (b) reap the AMO's own completion. |
| `nvshmemi_get_p2p_ptr(dst_ptr, my_rank, dst_rank)` | `ishmem_ptr(dst_ptr, dst_rank)` if iSHMEM GPU IPC is enabled; on the 2-node docker sim `ISHMEM_ENABLE_GPU_IPC=0` is the tested config → `ishmem_ptr` returns nullptr and the code always takes the RDMA path (see `xpu/internode_ll.cpp:869-899`). | Even if `ishmem_ptr` is available, remember XPU-IPC READ instability §4.4 — same-node writes should also go through iSHMEM or through **push-only** IPC. |
| `ibgda_get_state()` / `ibgda_get_state()->num_rc_per_pe` | **No public equivalent — do not port.** For QP fan-out use a runtime env var (e.g. `ISHMEM_IBGDA_QPS_PER_PE`, default `1` on BMG) and hard-code `qps_per_rank = num_local_experts` (LL) or `qps_per_rdma_rank = 1` (normal). | See `nvshmem_ishmem_api_mapping.txt` "Private NVSHMEM IBGDA internals". |

### 5.3 The two "modes" the CUDA code selects per destination

The CUDA internode/LL kernels do
```cpp
const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
if (dst_p2p_ptr == 0) {
    nvshmemi_ibgda_put_nbi_warp(...);           // RDMA path (cross-node)
} else {
    UNROLLED_WARP_COPY(..., ld_nc_global, st_na_global);  // NVL P2P path
}
```
**XPU translation:** decide the branch by `dst_rank == rank_of_local_node`. Same-node
peers use the intranode CUDA-IPC push path (§4); cross-node peers use
`ishmemx_putmem_nbi_subgroup`. Do NOT try to use `ishmem_ptr` at kernel scope on the 2-node
docker sim — it is intentionally disabled by `ISHMEM_ENABLE_GPU_IPC=0`. See
`xpu/internode_ll.cpp:869-899` for the exact pattern.

## 6. Memory-ordering fences (the single biggest source of silent corruption)

CUDA PTX fences map to `sycl::atomic_fence` **plus** an explicit `lsc_fence` for
system scope. Established mapping (see `DeepSymm/.../utils.hpp:64`, `xpu/xpu_kernels.hpp`):

| CUDA (PTX / intrinsic) | SYCL/XPU |
|---|---|
| `__threadfence()` | `sycl::atomic_fence(acq_rel, memory_scope::device)` |
| `__threadfence_block()` | `sycl::atomic_fence(acq_rel, memory_scope::work_group)` |
| `__threadfence_system()` | `memory_fence_system()` = acq_rel/system + `lsc_fence.ugm.evict.sysrel` |
| `st.release.sys.global` (`st_release_sys_global`) | `sycl::atomic_fence(release, memory_scope::system)` + `uc_store` (see `xpu_kernels.hpp:234`) |
| `ld.acquire.sys.global` (`ld_acquire_sys_global`) | `uc_load` / `lsc_uc_load_i32` + `sycl::atomic_fence(acquire, memory_scope::system)` (see `xpu_kernels.hpp:102`) |
| `st.na.global` (`st_na_global`) / `ld.nc.global` | `st_na_global` / `ld_nc_global` in `DeepSymm/.../utils.hpp` (`lsc_store.ugm.wt` / `lsc_load.ugm.ca`) |

**Fence discipline for internode** (matches `xpu/internode_ll.cpp` and `xpu/internode.cpp`):

- Before ringing NIC doorbell (payload → flag): `sycl::atomic_fence(release,
  memory_scope::device)` covers HBM→L2 / PCIe P2P; **`sycl::group_barrier(group)`
  ALONE is NOT sufficient** to make the payload visible to the NIC before the
  doorbell (a known BMG pitfall — see the F2 comment at `xpu/internode_ll.cpp:877-879`).
- Before posting a **flag AMO**: `sycl::atomic_fence(release, memory_scope::system)` +
  `ishmemx_fence_qp(dst_pe, qp)` (drains prior puts on that QP so the AMO is RC-ordered
  after them).
- After polling a **flag**: `sycl::atomic_fence(acquire, memory_scope::system)` before
  the receiver reads payload.
- Flag reads/writes on the SELF-channel (dst==my_rank): use `uc_store`/`uc_load` with a
  system-scope release/acquire — the RDMA path uses PCIe writes that bypass the GPU cache
  hierarchy, so the self path must too or receiver stalls on a stale cached flag.

## 7. Grid-scope sync between phases

The CUDA internode kernel has phase splits (route → forward → recv) that use
`cg::this_grid().sync()` or `nvshmem_sync` between phases. On XPU choose ONE of:

1. **Phase-split kernels** (default, matches `xpu/internode_ll.cpp` phase-split). Submit
   two `parallel_for`s; the `queue.submit` boundary is the grid barrier. Between them, if
   cross-PE observability is needed, call host-side `ishmem_barrier_all()`. This is the
   preferred choice for the first cut of `xpu/internode.cpp` because it sidesteps
   NamedBarrier+iSHMEM coexistence entirely.

2. **`GridBarrier`** (SW UC spin-loop counter, `xpu_kernels.hpp:285-366`). Use when phase
   split adds too much launch overhead. Requires co-residency: WGs launched ≤ device
   concurrency; scratch = 2 zero-initialized `uint32_t` (counter, sense). Reserve two
   scratch slots per grid barrier used.

3. **`root_group`** cooperative launch (`sycl::ext::oneapi::experimental::nd_launch` +
   `use_root_sync` property) — replaces the SW spin-loop with `sycl::group_barrier(
   root_group)`. See `csrc/xpu/root_group_cooperative_launch.md`. **Optional
   optimization; not required for the first working port.**

4. **`ishmemx_barrier_all_work_group(group)`** — **cross-PE** device barrier from ONE
   representative WG. Requires the CORRECT `libishmem.a` build (see
   `.github/agents/ishmem-ibgda-xpu-perf-debug.agent.md` "iSHMEM archive parity"). Never
   mix it with `ishmem_quiet()` in the same phase (wedges).

## 8. Buffer / SymBuffer / AsymBuffer / SymmSubChannel port

The CUDA internode kernel uses `SymBuffer<T>` / `AsymBuffer<T>` / `Buffer<T>` from
`csrc/cuda_kernels/buffer.cuh` to slice symmetric RDMA and NVL buffers. The corresponding
SYCL helpers live in `DeepSymm/csrc/sycl/buffer.hpp` (mirror them into `csrc/xpu/` if not
already). Key rules:
- `SymBuffer::send_buffer(dst_pe)` and `SymBuffer::recv_buffer(src_pe)` return SYMMETRIC
  offsets (same offset on every PE) so an iSHMEM put with `dst_pe` targets the correct
  slot. Do NOT change the layout without also updating the CUDA reference.
- `AsymBuffer` is for per-rank slices inside the local NVL buffer (each rank sees a
  different offset). Port to a plain `T*` + rank stride.
- `rdma_clean_offset` / `rdma_num_int_clean` regions must be zero-initialized between
  epochs — port the `#pragma unroll for(i=tid; i<num_clean; i+=nthreads) buf[i]=0;` loop
  verbatim.

## 9. Recommended porting sequence for `csrc/xpu/internode.cpp` (target)

The existing `csrc/xpu/internode.cpp` (2923 LOC) already has most of the combine path.
When extending / rewriting a specific CUDA function:

1. **Draft the SYCL functor class** (name, ctor holds args, `operator()(nd_item<1>,
   local_accessor* …)`). Add `[[intel::reqd_sub_group_size(32)]]`.
2. **Translate indices** with the §2.3 table.
3. **List every sync/fence/AMO/put in the CUDA source and translate them one-for-one**
   using §3, §5, §6. Do NOT leave `bar.sync` — replace with NamedBarrier or `group_barrier`.
4. **Split the CUDA IPC branch (§5.3)**: same-node → §4 push+`st_volatile_global`+
   `memory_fence_system`; cross-node → `ishmemx_putmem_nbi_subgroup` / `ishmem_putmem_nbi`.
5. **`nvshmemi_ibgda_quiet` → `ishmemx_fence_qp` (TARGETED only). Never full
   `ishmem_quiet` on combine.**
6. **Grid-sync**: prefer phase split first; refactor to `GridBarrier` or `root_group` only
   after correctness is proven.
7. **Fence audit**: at every "payload → flag" and "flag poll → payload read" boundary,
   confirm both a SYCL fence AND (for flags) an uncached load/store are present.
8. **Zeroing / cleaning kernels**: port literally; they are usually one-liner
   `parallel_for` submissions.

## 10. Build / runtime env checklist for the migrated kernel

Follow `.github/agents/ishmem-ibgda-xpu-perf-debug.agent.md`. In particular:

- Build with **generic multi-device AOT**: `unset TORCH_XPU_ARCH_LIST XPU_AOT_TARGETS`
  before `python setup.py build_ext --inplace`.
- Verify iSHMEM archive parity: `md5sum build/ishmem-sycl-dlink/{barrier,ibgda,nbi,proxy,
  rma,memory_ordering}.cpp.o` matches a known-good build; if barrier md5 is `a04bc5dd…`
  the wrong `libishmem.a` was linked.
- Env matrix for tests: `ISHMEM_IB_ENABLE_IBGDA=1`, `ISHMEM_IBGDA_DIRECT_DOORBELL=1`,
  `ISHMEM_IBGDA_BAR_BACKEND=igub`, `ISHMEM_IBGDA_QPS_PER_PE=1`,
  `ISHMEM_ENABLE_GPU_IPC=0`, `ISHMEM_SYMMETRIC_SIZE=268435456`,
  `ZE_ENABLE_PCI_ID_DEVICE_ORDER=1`, per-rank `ISHMEM_IBGDA_NIC=mlx5_N`,
  `ZE_AFFINITY_MASK=<gpu_id>`.
- Validate with `tests/docker-2node/run.sh` running `test_internode.py`; baseline
  `NUM_PROCESSES=2 NUM_TOKENS=32 HIDDEN=1024 NUM_TOPK=2 NUM_EXPERTS=8` must print
  `===== PASS tests/test_internode.py =====`.

## 11. Anti-patterns (do not do these — they are already documented failures)

- ❌ `ishmem_quiet()` on the combine path (wedge).
- ❌ Full-WG `sycl::group_barrier(group)` as a replacement for `bar.sync <id>` warp
  specialization (correctness works but forfeits ALL fusion; only acceptable if
  NamedBarrier+iSHMEM coexistence is broken in your build).
- ❌ Hand-rolled SLM arrival-counter subset barriers — **DEADLOCK** on BMG.
- ❌ Reading peer buffers via IPC (`ld_nc_global` from a peer pointer) — remote READ
  IPC is UNSTABLE; convert to a push pattern.
- ❌ Passing `named_barrier_init` result through class members, function args, PHIs, GEPs
  — IGC NamedBarriersResolution crashes.
- ❌ Building without `[[intel::reqd_sub_group_size(32)]]` on kernels that use
  `sub_group` operations (silent SIMD width mismatch).
- ❌ Publishing a flag with a normal cached store (`*p = val`) — receiver polls forever
  because the store never egresses. Use `uc_store` + system fence, or an iSHMEM AMO.
- ❌ Calling `ishmem_finalize` — DeepEP intentionally skips it.
- ❌ Adding `TORCH_XPU_ARCH_LIST=bmg` for the DeepEP extension build — bmg-only AOT is
  unstable and often causes `UR_RESULT_ERROR_DEVICE_LOST`.

## 12. Quick reference: file map

| Concept | File / line range |
|---|---|
| CUDA intranode reference | `csrc/cuda_kernels/intranode.cu` |
| SYCL intranode reference | `/root/jiafuzha/code-repo/DeepSymm/csrc/sycl/intranode.cpp` |
| CUDA internode reference (source) | `csrc/cuda_kernels/internode.cu` |
| XPU internode target | `csrc/xpu/internode.cpp` (extend/refactor) |
| CUDA internode-LL reference | `csrc/cuda_kernels/internode_ll.cu` |
| XPU internode-LL reference | `csrc/xpu/internode_ll.cpp` |
| SYCL utils (fences, barriers, unrolled copy) | `/root/jiafuzha/code-repo/DeepSymm/csrc/sycl/utils.hpp` |
| XPU kernel utils (uc_load/uc_store, GridBarrier, NamedBarrier decl) | `csrc/xpu/xpu_kernels.hpp` |
| API mapping table | `nvshmem_ishmem_api_mapping.txt` |
| NamedBarrier deep-dive | `csrc/xpu/named_barrier_usage.md` |
| root_group / cooperative launch | `csrc/xpu/root_group_cooperative_launch.md` |

Follow this file mechanically for every CUDA function you port; do not deviate from the
established patterns unless a specific pattern is documented to fail in your build/HW.
