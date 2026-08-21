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
| `csrc/cuda_kernels/internode.cu` | `csrc/xpu/internode.cpp` (+ `internode_{dispatch,combine,notify}_fused.inc`) | **COMPLETE** — see §1.1. |

## 1.1 Migration status: COMPLETE (2026-08-20)

The internode NORMAL path is now a fused, warp-specialized, CUDA-faithful port and is the
**default** (the `DEEP_EP_INTERNODE_FUSED` env gate and all legacy phase-split micro-kernels
were deleted; `internode.cpp` went 3090 → 447 lines with the kernels in three `.inc` files).
`tests/docker-2node-v2` passes the full matrix (64 configs, BF16/FP8 × with/without top-k ×
async × previous-event).

| CUDA kernel | XPU kernel | Barriers |
|---|---|---|
| `dispatch` (`internode.cu:447`) | fused, 5 warp roles, 512 WI / 16 sub-groups | `named_barrier_init(8)` (`:563`), `named_barrier_init(9)` (`:580`) |
| `combine` (`internode.cu:1716`) | fused, 4 warp roles, 800 WI / 25 sub-groups | `init(kForwarders+1)`=25 (`:1952`), `init(kRDMAReceivers+1)`=17 (`:1953`), one per RDMA destination (`:1966`) |
| `notify_dispatch` (`internode.cu:93`) | separate kernel (as in CUDA) | cross-PE iSHMEM barriers — legitimate |
| `cached_notify` (`internode.cu:1311`) | `launch_fused_cached_notify_heads` | head-negation transform, `:1375-1465` |

> ⚠️ **KNOWN LIMITATION: `num_rdma_ranks ∈ {2, 4}` only** (host `TORCH_CHECK`). Combine declares
> `2 + num_rdma_ranks` named barriers and 9 barriers ICE in that kernel (§11), so `R=8` is not
> supported. **R=4 is compile/JIT-validated but NOT runtime-validated** — the harness is
> 2 nodes × 2 GPUs, so only `R=2` is exercised numerically. 8-node deployments will hit the check.

**Deviations from CUDA** (each forced, with the reason):
1. `poll_load` dual uncached+cached-atomic read instead of `ld_acquire_sys_global` (§6.0).
2. Inline bit-manipulation bf16↔float (RNE) — the devicelib converts are stack calls (§11).
3. Fully-uncached 16-byte payload path (`ld_uc_global_v` / `st_uc_global_v`).
4. `ishmemx_fence_qp` instead of `nvshmemi_ibgda_quiet`; never a global `ishmem_quiet()`.
5. `send_rdma_head` / `send_nvl_head` pre-filled with `-1` — they are *sparse* handles that
   `cached_notify` rewrites (CUDA parity, `deep_ep/buffer.py:874-875`).
6. `recv_x` / `recv_x_scales` zero-filled. The XPU dispatch always runs in `num_worst_tokens`
   mode and a *cached* dispatch returns the padded rows to the caller; CUDA returns exactly
   `num_recv_tokens` rows so it can leave padding uninitialised (`internode.cu:1194-1207` only
   cleans `recv_topk_idx`). `torch::empty` garbage there shows up as `NaN`/non-uniform rows.
7. CUDA's per-lane `acquire_lock(... + lane_id)` serialized over `dst_rdma_rank` (§11).

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

> **TOOLCHAIN REQUIREMENT (validated 2026-08-20): to use SPIR-V `NamedBarrier` in a kernel
> that also calls iSHMEM device functions, `libishmem.a` MUST be built with
> `-DISHMEMI_IBGDA_BNXT_NOINLINE=OFF`.**
> With the CMake default (`ON`) the bnxt IBGDA helpers keep `__attribute__((noinline))`, so
> IGC outlines them as vISA stack calls and stamps `.kernel_attr NBarrierCnt=N` on the kernel
> body AND on every outlined `.function`. vISA then rejects the module with
> `More than 1 kernel attribute defined NBarrierCnt`, surfacing at runtime as
> `error: parsing vISA inline assembly failed`.
> - Build it with `bash _build_ishmem.sh`, then **verify the cache, not the script**:
>   `grep ISHMEMI_IBGDA_BNXT_NOINLINE <ishmem>/build/CMakeCache.txt` → must be `OFF`.
>   `setup.py` (`check_ishmem_bnxt_inlinable`) also warns loudly when it is `ON`.
> - After switching archives: `rm -rf build/ishmem-sycl-dlink` before rebuilding DeepEP.
> - Passing `-DISHMEMI_IBGDA_BNXT_NOINLINE=` in DeepEP's own flags does NOTHING: the macro is
>   `PRIVATE` to `ishmem-objects` and `ibgda_device_impl.h` is not installed, so the bnxt code
>   reaches DeepEP only as pre-compiled device bitcode inside `libishmem.a`.
> - `IGC_SelectiveFunctionControl=1` is obsolete; do not set it.
> Evidence + standalone repro: `csrc/xpu/named_barrier_usage.md`,
> `csrc/xpu/tools/test_nbarrier_ishmem_repro.cpp`.
>
> **RETRACTION (2026-08-20): the former "grid-scope phase boundary" objection was WRONG.**
> An earlier revision of this file argued that the XPU internode NORMAL phase boundaries are
> inherently grid-scope/cross-PE, so NamedBarrier was structurally the wrong tool. That
> reasoned from the *workaround's* structure back to the algorithm — circular. Verified:
> `csrc/cuda_kernels/internode.cu` has **no** `cooperative_groups`, **no** `this_grid()`, and
> **no** grid sync at all. `dispatch` (line 447) and `combine` (line 1716) are each **one fused
> warp-specialized kernel** synchronized purely by intra-block `barrier.sync`/`bar.sync`
> (lines 563, 580, 1952, 1953, 1966). Only `notify_dispatch` (line 93) and `cached_notify`
> (line 1311) are separate kernels in CUDA — and those stay separate in the XPU port.
> The grid-scope barriers exist ONLY in the phase-split workaround. **The correct target is one
> fused kernel per direction**, with each CUDA `barrier.sync <id>, N*32` becoming a NamedBarrier
> of count `N` sub-groups.

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

> ### 6.0 SPLIT PRODUCER PATHS: never poll a queue counter with `uc_load` alone
>
> **This was the single hardest bug in the internode migration — it silently dropped tokens
> and produced `NaN` rows rather than failing loudly.**
>
> CUDA polls queue heads/tails with `ld_acquire_sys_global`, a **cached** system-scope acquire
> load. On BMG the two producers of those very same counters write through **different routes**:
>
> | Producer | Route | Visible to |
> |---|---|---|
> | Remote peer | IBGDA AMO | lands directly in memory → only an **uncached** load sees it |
> | Self / local path | device `atomic_ref::fetch_add` | lands in the **cache hierarchy** → an uncached load never sees it |
>
> So `uc_load` alone misses every local increment, and a cached load alone misses every remote
> AMO. Decisive evidence captured at a receiver spin-out: `uc_load` returned `0` while a cached
> system-scope acquire `atomic_ref::load` **on the same address** returned `8`. Every RDMA-receiver
> poll therefore ran to the spin cap (`max_rcv_spins == 200000001`) and then `break`-ed, silently
> dropping tokens.
>
> **Fix — `poll_load<T>()` in `csrc/xpu/xpu_kernels.hpp`: read BOTH and take the max.** These
> counters are monotonically increasing, so the max is always a valid observation. After the fix
> `max_rcv_spins` fell from 2×10⁸ to 131–359.
>
> **Rule: use `poll_load` at every queue-counter polling site.** A silent spin-cap `break` is the
> signature of this bug — always instrument the observed spin count when tokens go missing.

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
  IPC is UNSTABLE; convert to a push pattern. **Note (verified):** CUDA's internode NVL
  protocol is *already* push-only in both dispatch and combine — the sender writes `x`/`tail`
  into the destination's buffer and reads only its own `head`; the coordinator writes `head`
  into the peer (`internode.cu:529-556`, `:2044-2085`, `:2216-2280`). So nothing in the
  internode path needs converting; keep CUDA's data flow verbatim.
- ❌ Passing `named_barrier_init` result through class members, function args, PHIs, GEPs
  — IGC NamedBarriersResolution crashes.
- ❌ Building without `[[intel::reqd_sub_group_size(32)]]` on kernels that use
  `sub_group` operations (silent SIMD width mismatch).
- ❌ Publishing a flag with a normal cached store (`*p = val`) — receiver polls forever
  because the store never egresses. Use `uc_store` + system fence, or an iSHMEM AMO.
- ❌ Calling `ishmem_finalize` — DeepEP intentionally skips it.
- ❌ Adding `TORCH_XPU_ARCH_LIST=bmg` for the DeepEP extension build — bmg-only AOT is
  unstable and often causes `UR_RESULT_ERROR_DEVICE_LOST`.
- ❌ **`sycl::ext::oneapi::bfloat16` conversion operators inside a NamedBarrier kernel.**
  They lower to the external devicelib symbols `__devicelib_ConvertBF16ToFINTEL` /
  `__devicelib_ConvertFToBF16INTEL`, which IGC materializes as vISA **stack calls** →
  `error: IGC: internal compiler error` (same NBarrierCnt-on-outlined-function class of
  failure as the iSHMEM bnxt helpers). Convert inline with bit manipulation (RNE rounding).
  Generalize: **any** outlined/external device function in a NamedBarrier kernel breaks it.
- ❌ **Declaring ~10 distinct named barriers in one kernel** — ICEs in IGC codegen (between
  `push_analysis` and `codegen`) even with zero stack calls. 4 is known-good; the true limit
  is somewhere in 5..9 and is uncharacterised. Alias handles when you need more.
- ❌ **Porting CUDA per-lane divergent lock/spin idioms verbatim.** A SYCL sub-group is one
  lock-stepped EU thread, so patterns that depend on Volta+ independent thread scheduling —
  e.g. `acquire_lock(rdma_send_channel_lock + lane_id)`, where each lane holds a *different*
  lock and progress requires lanes to advance independently — deadlock on Xe. Serialize the
  critical section (e.g. over `dst_rdma_rank`), or have one lane evaluate a sub-group-wide
  condition and broadcast it. Preserve the original release ordering.
- ❌ Rebuilding after editing only a `.inc` file. `setup.py` does **not** track `.inc` files as
  dependencies, so the `.so` relinks from **stale** objects and you silently test the OLD code.
  Always `touch csrc/xpu/internode.cpp` first.

### Fast IGC ICE triage (no 2-node harness needed, ~3 min)

The runtime JIT-compiles from embedded SPIR-V (`-device pvc,bmg,...` is passed to the runtime;
the `.so` is not a finished AOT image). So dump and replay offline:

```bash
IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=$PWD/igcdump  <run once>
ocloc compile -file <dump>.spv -spirv_input -device bmg
```

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

## 13. Measured perf after fusion (A/B baseline)

Config: 2 nodes x 2 ranks (`R=2`), BF16, `num_experts=8`, `topk=2`, `hidden=7168`,
`ISHMEM_IBGDA_DB_BATCH_SIZE=8`. Harness `tests/docker-2node-v2`, sweep driven by
`DEEP_EP_PERF_TOKENS`. "legacy" = commit `b231f1c` (phase-split micro-kernels default),
"fused" = commit `c804991` (warp-specialized NamedBarrier kernels default).
Both legs measured on the same clean hardware, back to back.

Round-trip latency (us):

| tokens | legacy | fused | speedup |
|---|---|---|---|
| 32 | 3897.3 | 1927.2 | 2.02x |
| 64 | 6877.6 | 2615.8 | 2.63x |
| 128 | 12994.3 | 5782.7 | 2.25x |
| 512 | 52471.7 | 20227.4 | 2.59x |
| 1024 | 103301.8 | 35922.7 | 2.88x |
| 2048 | 204436.5 | 70359.9 | 2.91x |
| 4096 | 414178.2 | 132573.9 | 3.12x |

Dispatch, isolated (us) — this is where fusion pays off, and the win grows with size:

| tokens | legacy | fused | speedup |
|---|---|---|---|
| 32 | 2922 | 1180 | 2.5x |
| 64 | 5099 | 1289 | 4.0x |
| 128 | 9825 | 1412 | 7.0x |
| 512 | 36624 | 2365 | 15.5x |
| 1024 | 72307 | 3442 | 21.0x |
| 2048 | 143941 | 5487 | 26.2x |
| 4096 | 287055 | 9319 | 30.8x |

Fused dispatch RDMA bandwidth scales 0.39 -> 6.30 GB/s across the sweep (NVL 10.30 GB/s).

Combine, isolated (us) — essentially unchanged by fusion:

| tokens | legacy | fused | ratio |
|---|---|---|---|
| 32 | 1551 | 1230 | 1.26x |
| 64 | 2360 | 1872 | 1.26x |
| 128 | 4027 | 4935 | 0.82x |
| 512 | 15177 | 17698 | 0.86x |
| 1024 | 30194 | 32832 | 0.92x |
| 2048 | 61306 | 64420 | 0.95x |
| 4096 | 127833 | 122462 | 1.04x |

### Interpretation / open bottleneck

Combine bandwidth is flat at ~0.46-0.49 GB/s in **both** implementations, i.e. combine
was already this slow before fusion — fusion did not regress it. After fusion, combine
dominates: ~92% of round-trip time at 4096 tokens. This is a **pre-existing**, still
uninvestigated bottleneck, not a fusion artifact.

Suspects investigated (see section 14 for the resolution):
1. per-destination NamedBarrier serialization (one barrier per `dst_rdma_rank`)
   — **ruled out** at `R=2` (only `nb_d0`/`nb_d1` live),
2. ~~the serialized critical section over `dst_rdma_rank`~~ — **THIS SUSPECT WAS WRONG.**
   Combine has no such critical section: the forwarder assigns
   `dst_rdma_rank = warp_id / Shape::kWarpsPerForwarder`, so every forwarder warp owns
   its own destination and they run in parallel, exactly matching CUDA. The serialized
   `dst_rdma_rank` critical section was a **dispatch** issue, not a combine one.
3. forwarder spin loops — a **symptom**, not the cause (see 14),
4. the uncached 16-byte payload store path — **this was the root cause.**

Do **not** attribute a future combine regression to fusion without re-running this A/B.

## 14. Combine root cause + fix: payload cache policy and MLP (RESOLVED)

### Evidence (measure first)

A build-time-gated device telemetry harness was added to
`csrc/xpu/internode_combine_fused.inc` (`#ifdef DEEP_EP_COMBINE_TELEMETRY`, enabled with
`DEEP_EP_COMBINE_TELEMETRY=1 python3 setup.py build_ext --inplace`; the gate lives in
`setup.py`). It uses `dev_clock()` (`__spirv_ReadClockKHR(0)`, added to
`csrc/xpu/xpu_kernels.hpp`) to attribute cycles per warp role into a device counter
buffer that the host dumps after the kernel. `ReadClockKHR` is an intrinsic, not an
outlined call, so it does **not** trip the NamedBarrier ICE.

At 4096 tokens the attribution was unambiguous:

| role (warps) | total cycles | wait | copy |
|---|---|---|---|
| NVL sender (24) | 3.59e9 | 1.3e5 (**0.004%**) | 3.55e9 (**98.7%**) |
| forwarder (288) | 43.4e9 | 4% space + **40% tail** | 25% |
| RDMA receiver (192) | 30.3e9 | **78%** | 22% |

Per-warp totals (~150e6 cycles) / 122 ms => ~1.3 GHz, i.e. per-warp total ~= kernel wall
time: every role is resident for the whole kernel. **The NVL sender never stalls and
spends ~100% of the kernel inside its payload push loop** (~18 900 cycles ~= 14.5 us per
512-byte sub-group store). The forwarder `waittail` and receiver `wait` are downstream
consequences of that producer, not independent bottlenecks.

### Root cause

Combine's payload path used **fully uncached** LSC accesses — `ld_uc_global_v` /
`st_uc_global_v` (`lsc_{load,store}.ugm.uc.uc`) — which bypass both L1 and L3, so each
16 B/lane access is a full-latency memory transaction. Because those helpers are
`asm volatile`, the compiler cannot software-pipeline them either, so exactly **one**
512 B sub-group message is in flight per warp at a time.

Dispatch — on the same hardware, doing the *same* operations (NVL P2P push into a peer's
IPC buffer, RDMA staging read by the NIC) — uses `ld_nc_global_v` (`.uc.ca`, cached
non-coherent) + `st_na_global_v` (`.wb.wb`, cached write-back) throughout
(`internode_dispatch_fused.inc:509-511, 795-797, 945`) and reaches 10.3 GB/s NVL /
6.3 GB/s RDMA. That is a direct in-tree A/B precedent.

The header comment C5 already *claimed* payload reads went through `ld_nc_global_v` while
the code did the opposite — the uncached accesses were an un-reverted debugging shotgun.

### Why cached payload is correct

Verified against the actual fence placement (do not assume this — re-verify if you move
fences). The producer writes payload, then issues
`sycl::atomic_fence(release, system)` + `lsc_fence_sysrel()`
(`lsc_fence.ugm.evict.sysrel`, an L3 flush to the system domain), and only then publishes
the tail via `uc_store`. The consumer `poll_load`s the tail, then does
`sycl::atomic_fence(acquire, system)` + `lsc_fence_sysacq()`
(`lsc_fence.ugm.invalidate.sysacq`) before touching payload. That release/acquire pairing
was already present everywhere in combine (sender before the `ch_tail` store; forwarder
per token and before `ishmemx_putmem_nbi`; receiver per token), so the uncached payload
accesses were pure redundancy.

`poll_load()` (uncached load max'd with a cached atomic load) remains **required for
counters** — that is the split-producer bug of section 6.0 (remote IBGDA AMO lands in
memory, local `atomic_ref::fetch_add` lands in cache) and is untouched by this work.

### The fix (three changes, measured one at a time)

1. NVL sender payload store `st_uc_global_v` -> `st_na_global_v`.
2. `fused_combine_token` payload load `ld_uc_global_v` -> `ld_nc_global_v` and output
   store `st_uc_global_v` -> `st_na_global_v`.
3. Memory-level parallelism in the NVL sender copy:
   `UNROLLED_GROUP_COPY(2, lane_id, 32, hidden_int4, dst, src, ld_nc_global_v, st_na_global_v)`
   — 2 loads issued before the first store, so 2 messages are in flight per warp.

Isolated combine (us), same hardware, clean between legs:

| tokens | before (`a0dce51`) | +#1 | +#2 | +#3 (final) | total |
|---|---|---|---|---|---|
| 32 | 1230 | 1021 | 913 | 893 | 1.38x |
| 64 | 1872 | 1544 | 1326 | 1264 | 1.48x |
| 128 | 4935 | 3783 | 3160 | 2599 | 1.90x |
| 512 | 17698 | 13748 | 11794 | 8672 | 2.04x |
| 1024 | 32832 | 25296 | 20029 | 14157 | 2.32x |
| 2048 | 64420 | 49188 | 37441 | 25906 | 2.49x |
| 4096 | 122096 | 92206 | 68313 | 46479 | **2.63x** |

Round-trip at 4096 tokens: 132588 -> 56166 us (**2.36x**). Combine RDMA BW at 4096:
0.481 -> 1.263 GB/s (NVL send 2.065 GB/s). Dispatch is unchanged (~9300 us @ 6.3 GB/s) — the changes are
combine-local. Correctness: full 64-config matrix `===== PASS =====`.

### Rejected optimizations (do not retry blindly)

- **NVL sender `UNROLLED_GROUP_COPY(4, ...)`**: faster (512 tokens 7301 us vs 8672; 1024 11994 vs 14157) but
  **hung twice at 4096 tokens** with `dmesg` `Engine reset: engine_class=ccs` on two GPUs.
  U=2 is the conservative safe point. The exact mechanism (spill-induced watchdog trip vs
  P2P write-queue overrun) was not isolated.
- **Unrolling `fused_combine_token`'s reduce loop by 2**: correctness passed but the bench
  never completed in 2400 s with **no** engine reset in `dmesg` — a massive slowdown, i.e.
  register spilling. The reduce loop needs `values[2][8]` + `v[2]` + `b[2]` of extra live
  state in a kernel already built with `-ze-intel-enable-auto-large-GRF-mode`. Reverted;
  see the "C9 (REJECTED)" comment in the `.inc`.

**Register pressure is the hard ceiling on further unrolling in this kernel.** Unroll the
straight copy (no accumulators), never the reduce.

### Not verified

- `R=4` is compile-validated only (the harness is 2 nodes x 2 GPUs).
- The residual forwarder `waittail` / receiver `wait` fractions were not re-profiled after
  the fix; the remaining ~5x gap vs dispatch is still open.

## 15. Line-by-line logic diff: XPU combine vs CUDA combine

Full audit of `csrc/xpu/internode_combine_fused.inc` against
`csrc/cuda_kernels/internode.cu` `combine` (L1716-2277) plus the host path
(`csrc/xpu/internode.cpp` `combine_nvl_rdma`, `csrc/xpu/xpu_runtime.hpp` `Config`,
`tests/test_internode.py:231`). Motivation: rule out a semantic/algorithmic
divergence as the cause of the flat combine bandwidth. **Result: no algorithmic
divergence found. Every sizing, partitioning and cadence parameter is bit-identical
to CUDA.** The differences that do exist are all memory-model / platform mappings.

### 15.1 Sizing and partitioning: VERIFIED EQUIVALENT

| # | CUDA | XPU | Consequence | Verdict |
|---|---|---|---|---|
| 1 | `Config(num_sms, nvl_send, nvl_recv, rdma_send, rdma_recv)` from `test_internode.py:231` = `(24, 8, 512, 16, 128)` | identical struct in `xpu_runtime.hpp:53`, same `align_up(rdma_recv, rdma_send)` normalisation and the same 4 `TORCH_CHECK`s | none | equivalent |
| 2 | `num_channels = gridDim.x / 2`, grid `= num_channels * 2` | `num_sms = num_channels * 2`, `channel_id = sm_id / 2`, `is_forwarder_sm = sm_id % 2` | none | equivalent |
| 3 | `kNumCombineForwarderWarps = 24` (:2307) | `DEEP_EP_COMBINE_FWD_WARPS 24` | none | equivalent |
| 4 | `kNumWarpsPerForwarder = max(24/R,1)`, `kNumForwarders = R*that`, `kNumRDMAReceivers = kNumForwarders - 8` (:1713-1715) | `FusedCombineShape` with the identical three expressions | at R=2: 12/24/16 both sides | equivalent |
| 5 | block `= (kNumForwarders+1)*32` = 800 | same | none | equivalent |
| 6 | `get_num_bytes_per_token(hidden_int4,0,0,num_topk)` (:42) | `fused_combine_num_bytes_per_token` = same `align_up(hidden_int4*16 + sizeof(SourceMeta) + num_topk*4, 16)` | no padding inflation | equivalent |
| 7 | `num_max_nvl_chunked_recv_tokens_per_rdma = nvl_recv / kNumRDMARanks` (:1781) | identical | 256 slots both | equivalent |
| 8 | sender chunk `min(num_max_nvl_chunked_send_tokens, end-start)` = 8 tokens (:1877) | identical | equivalent chunk size | equivalent |
| 9 | forwarder chunk loop `token_start_idx += num_max_rdma_chunked_send_tokens` = 16 tokens; RDMA msg = `num_chunked_tokens * num_bytes_per_token` (~229 KB) (:2009/:2117) | identical | RDMA ops are NOT undersized | equivalent |
| 10 | forwarder sub-warp stride `token_idx += kNumWarpsPerForwarder` (:2044) | `+= Shape::kWarpsPerForwarder` | equivalent | equivalent |
| 11 | receiver stride `token_idx += kNumRDMAReceivers` (:2004) | `+= Shape::kRDMAReceivers` | equivalent | equivalent |
| 12 | payload lane stride `i += 32` in `combine_token` | same | equivalent | equivalent |
| 13 | `dst_rdma_rank = warp_id / kNumWarpsPerForwarder` | same | **each forwarder warp owns its own destination; there is NO serialized critical section** (section 13's suspect #2 was wrong) | equivalent |
| 14 | warp-role shuffle `(warp_id + channel_id) % N` | same for both sender and forwarder | equivalent | equivalent |
| 15 | `get_channel_task_range` = `ceil_div` + two `min`s (`utils.cuh:434`) | identical (`xpu_kernels.hpp:326`) | equivalent | equivalent |
| 16 | head-release cadence: coordinator RDMA head AMO only when `min_head >= last + num_max_rdma_chunked_send_tokens`; NVL head store whenever `min_head > last` | identical predicates | credit-release cadence matches | equivalent |
| 17 | tail publish cadence: sender publishes `ch_tail` once per outer chunk iteration; forwarder AMOs the RDMA tail once per 16-token chunk | identical | equivalent | equivalent |

### 15.2 Divergences (all memory-model / platform, none algorithmic)

| # | CUDA does | XPU does | Perf consequence | Verdict |
|---|---|---|---|---|
| D1 | multi-stage async TMA (`kNumStages=2`, `mbarrier`) for both sender and forwarder payload | lane-strided 16 B LSC copies (conversion note C1), now 2-way unrolled in the sender | **the real cost**: only 2 messages in flight per warp vs TMA's decoupled pipeline. Unroll 4 was faster but caused `ccs` engine resets | **divergent, dominant, partially mitigated** |
| D2 | one TMA store publishes the whole token (payload + SourceMeta + topk weights) | three separate stores: 896x16 B payload, 1 SourceMeta, `num_topk` scalars | more messages per token | divergent, measured **null** (see 15.3 C12) |
| D3 | `ld_acquire_sys_global` for queue counters (an L2 hit) | `poll_load` = uncached LSC load **+** system-scope acquire atomic (2 memory transactions) | required by the split-producer bug (section 6.0); doubles spin cost | divergent, **necessary** |
| D4 | acquire *load* orders one thread; no cache maintenance | `atomic_fence(acquire,system)` + `lsc_fence.ugm.invalidate.sysacq` per token in forwarder and receiver | a full L1/L3 invalidate per token that CUDA never performs | divergent, measured **null** (15.3 C10) |
| D5 | `__nanosleep(NUM_WAIT_NANOSECONDS)` (500 ns) at the end of every coordinator iteration (:2278) | no backoff anywhere | unthrottled polling shares the work-group (= one Xe-core) with the copying warps | divergent, measured **null** (15.3 C11) |
| D6 | `st_relaxed_sys_global` for the NVL head credit | `uc_store` + `atomic_fence(release,system)` + `lsc_fence.ugm.evict.sysrel`, inside the `for i < R` loop | an L3 evict per head update where CUDA has none; coordinator-only, off the critical path | divergent, harmless (untested) |
| D7 | `clock64()` timeout + `trap()` | bounded spins (`kFusedSpinCap = 2e8`), `break` on overrun | a wedge degrades to wrong results instead of a trap; the coordinator's counter is cumulative for the whole kernel, not per wait | divergent, correctness-visibility only |
| D8 | `volatile __shared__` for `forwarder_nvl_head` / `retired` | `sycl::atomic_ref<work_group, acq_rel>` on SLM | slightly heavier SLM ops in the coordinator scan | divergent, harmless |
| D9 | 9 barriers usable (`kNumRDMARanks + 2 <= 16`) | 8 named barriers max on BMG; `nb_d4..nb_d7` aliased to `nb_d0`, R capped at 4 | none at R<=4; blocks R=8 | divergent, documented limit |
| D10 | `bfloat16` HW convert instructions | inline RNE bit manipulation (note C7) | avoids the vISA stack-call ICE; a few ALU ops per element | divergent, required |
| D11 | `cached_notify` launched async on the same stream | `queue.wait()` + `ishmem_barrier_all()` **twice** around clean/notify | two host-side 4-PE collectives per combine call; a fixed cost, significant only at small token counts (combine at 32 tokens is 893 us total) | divergent, dispatch does the same, not size-scaling |

### 15.3 Follow-up experiments after the section-14 fix: three consecutive NULL results

Each built and measured separately on freshly-reset hardware. Noise band established
by re-running the unchanged committed build: combine(iso) @512 tokens
= 8672 / 8642 us, @1024 = 14157 / 14240 us, i.e. **+/-1.3%**.

| id | change | 512 tok | 1024 tok | verdict |
|---|---|---|---|---|
| — | committed baseline (section 14) | 8672 / 8642 | 14157 / 14240 | reference |
| C10 | issue the acquire fence + `lsc_fence_sysacq` only when the tail was actually re-read (D4) | 8630 | (leg hung) | **null**, reverted |
| C11 | restore CUDA's `__nanosleep` as `dev_backoff()` in the coordinator + throttle the forwarder/receiver poll loops (D5) | 8562 | (leg hung) | **null**, reverted |
| C12 | sender's topk-weight `uc_store` -> plain cached store (D2) | 8646 | 14207 | **null**, reverted |

All three are inside the noise band. C10 and C11 each saw the 1024-token leg hang
(rc=124) at the same point; the restored baseline then passed the same leg, so the
hangs are the usual accumulated-HW flakiness, not the changes — but neither change
earned the risk.

### 15.4 Conclusion

The flat-then-fixed combine bandwidth was **entirely** a memory-hint problem
(section 14), not a semantic one. After that fix the residual gap is D1: the sender
warps push 16 B/lane with only 2 messages in flight, where CUDA overlaps a
multi-stage TMA pipeline. Evidence that this — and not waiting — is what is left:

- telemetry: NVL sender warps stall **0.004%** of their cycles and copy **98.7%**;
- the three "stall-side" fixes above (fence frequency, poll throttling) moved nothing;
- removing residual uncached scalar stores moved nothing;
- the win that *did* land (2.63x) came entirely from cache policy + 2-way unroll,
  i.e. from making each message cheaper and doubling the messages in flight.

Next avenue, in order of expected value:
1. more messages in flight per sender warp without more registers — wider LSC
   messages (e.g. 32 B/lane `d32x8`) rather than deeper unrolling, since unroll 4
   destabilised the GPU while unroll 2 was a clean 1.3-1.5x;
2. more sender warps: at `num_nvl_ranks=2` six of the eight NVL-sender warps exit
   immediately (`dst_nvl_rank >= num_nvl_ranks`), so only 2 warps per even SM do all
   the NVL payload work. Splitting each destination across a "large warp" of idle
   sender warps (as the forwarder already does for RDMA destinations) would give up
   to 4x more copy parallelism. Needs care: the queue-slot bookkeeping is currently
   per-warp, and the named-barrier budget is 8.

## 16. Post-fix telemetry: the four forwarder counters (2048 tokens)

Requested to discriminate between the remaining hypotheses in one run. Built with
`DEEP_EP_COMBINE_TELEMETRY=1`, single perf size, `combine(iso) ~= 25.9 ms`.
Counters are sums over all warps of a role (24 senders / 288 forwarders /
192 receivers); per-warp totals are 31.3-34.4e6 cycles, and 34.3e6 / 25.9 ms
=> ~1.32 GHz, i.e. **every role is resident for the whole kernel**.

| role | total | breakdown |
|---|---|---|
| NVL sender (24) | 7.51e8 | **wait 0.01%**, **copy 97.70%** |
| forwarder (288) | 9.58e9 | waitspace 7.26%, **waittail 39.72%**, copy 23.91%, **send 0.68%** |
| RDMA receiver (192) | 6.61e9 | **wait 80.50%**, copy 19.30% |

### What this settles

- **`FwdSend` = 0.68%.** The entire iSHMEM issue path -- `ishmemx_putmem_nbi_subgroup`
  (~230 KB), `ishmemx_fence_qp`, the tail `ishmemx_long_atomic_add_qp`, and both
  bracketing `group_barrier`s -- costs **under 1%** of forwarder time. **iSHMEM is
  exonerated as a perf factor in combine**, including `fence_qp` even if it turns out
  to be a completion wait, and including doorbell batching (`ISHMEM_IBGDA_DB_BATCH_SIZE`
  cannot matter at 0.68%).
- **`FwdWaitSpace` = 7.26%.** RDMA credit starvation is minor, so the
  "no send/produce decoupling => at most one RDMA write in flight per (channel,
  dst_rdma_rank) with its latency exposed" hypothesis is **not** the dominant term.
  It is a real structural difference from dispatch's `kRDMASenderCoordinator`, but it
  is also CUDA-faithful (CUDA combine likewise issues the put inline in the forwarder
  chunk loop, internode.cu:2107-2140) and it accounts for at most 7% here.
- **`FwdWaitTail` = 39.72%** and **`RcvWait` = 80.50%** are starvation, and
  **`SndWait` = 0.01% / `SndCopy` = 97.70%** identifies the single source. The causal
  chain is unambiguous and one-directional:
  `NVL sender copy (100% busy) -> forwarder waits on the NVL tail (40%) -> receiver
  waits on the RDMA tail (80%)`.
- Consistent with the three null experiments of section 15.3: fence frequency (C10),
  poll throttling (C11) and residual uncached scalar stores (C12) all target the
  stall side or per-access overhead, and none of them can move a producer that is
  already 97.7% busy.

### The one open number

At 2048 tokens the 24 sender warps push 48.542 MB in 25.9 ms = **1.87 GB/s**
(78 MB/s per warp; ~7.7e3 cycles per 512 B sub-group message with 2 in flight).
Dispatch's forwarder performs the *same* operation -- lane-strided 16 B
`ld_nc_global_v`/`st_na_global_v` into a peer's IPC buffer -- with the *same* 24 warps
and a *non-unrolled* loop, and moves 48.542 MB in 5.48 ms = **8.9 GB/s**. So the P2P
write path itself sustains ~9 GB/s with this warp count; combine's sender is 4.7x
below that. Excluded so far: queue starvation (0.01% wait), cache-maintenance
overhead (C10), co-resident poll contention (C11), residual uncached scalars (C12),
and every sizing/partitioning parameter (section 15.1).

Remaining candidates, untested:
1. concurrency of the memory system: unlike dispatch, combine's NVL push runs
   *simultaneously* with the forwarder's NVL read-back, the RDMA send and the RDMA
   receive -- ~184 MB of traffic in 25.9 ms, of which ~78 MB crosses PCIe in both
   directions on a link the NIC also uses;
2. messages in flight per warp: the fix took this from 1 to 2 and bought 1.3-1.5x;
   going to 4 destabilised the GPU. Wider LSC messages (32 B/lane) rather than deeper
   unrolling is the way to buy more without registers;
3. sender warp count: at `num_nvl_ranks=2`, six of eight NVL-sender warps exit
   immediately, so only 2 warps per even SM do all NVL payload work (section 15.4).

## §17. The blocking-AMO hypothesis: mechanism REAL, magnitude BOUNDED (~5%), and empirically falsified as the bottleneck

Hypothesis under test (from iSHMEM source analysis): `ishmemx_long_atomic_add_qp` is implemented as a
*fetching* atomic (`ibgda_device_impl.h` maps `AMO_ADD` → `AMO_FETCH_ADD`) that polls the collapsed CQ
`wc_counter` with no spin cap. Because the AMO's `wqe_idx` is claimed after the 230 KB payload put on the
*same* QP, waiting for its CQE implicitly waits for the whole put to land — a de-facto per-chunk `quiet`
on combine's producer critical path.

### 17.1 Telemetry at 4096 tokens (244 rank-reports, 99.78% accounted)

Two new counters split the send region (`put` = `putmem_nbi_subgroup` + subgroup barrier;
`amo` = `fence_qp` + `long_atomic_add_qp` + barrier), plus `bar2` closes the previously-unaccounted 28%.

| forwarder region | share of forwarder time |
|---|---|
| `waittail` (NVL producer starvation) | **34.32%** |
| `bar2` (2nd `sync_large_warp`, sibling imbalance) | **29.52%** |
| `copy` | 25.93% |
| `waitspace` | 9.18% |
| `send` | 0.84% |
| accounted | 99.78% |

NVL sender: wait **0.010%**, copy **97.74%**. RDMA receiver: wait 76.92%, copy 22.84%.

### 17.2 Correcting the earlier 0.68% figure — the mechanism IS real

`kDbgFwdSend` is summed over all 288 forwarder warps but only 1-in-12 (`sub_warp_id ==
kWarpsPerForwarder-1`) issues a send. Undiluted, the send region is **10.09% of the *sending* warp's
life**. Per chunk-send (134 chunk-sends/rank, clock 1.34 GHz):

- `put` = **245 µs**
- `amo` = **557 µs**  (66.2% of the send region)
- 230 KB at dispatch's measured 6.23 GB/s wire rate = **37 µs**

So the AMO costs ~15× the payload wire time — exactly the blocking-completion signature predicted.
**The mechanism is confirmed.**

Further, `waitspace` is NOT RDMA credit starvation: the predicted sibling stall
(11 warps × 24 groups × per-sending-warp send time) = 1.65e9 cycles vs measured `waitspace` = 1.64e9.
`waitspace` is almost entirely the 11 sibling sub-warps blocked behind their group's send.

### 17.3 Why fixing it cannot pay: total attributable cost is ~10%, and it is off the critical path

Total send-attributable forwarder time = `send` + `waitspace` = **10.02%**, of which the AMO share is
**~6.5%**. That is a hard ceiling, set by arithmetic: at 4096 tokens there are only ~134 chunk-sends per
rank (16 tokens/chunk across 24 (channel,dst) groups), so even an infinitely fast AMO removes a bounded
amount of time.

More decisively, **the NVL sender never waits (0.010%)** and is copy-busy 97.74% of its life. The sender
lifetime is **44.8 ms of the 46.5 ms combine**; the remaining 1.7 ms is drain. Forwarder stalls therefore
do not feed back into the sender, which is the critical path. Removing the AMO cost entirely can only
attack part of that 1.7 ms drain tail.

### 17.4 Empirical falsification: `rdma_chunk_size` sweep (zero code change)

If the blocking AMO were the bottleneck, halving the number of chunk-sends should roughly halve its cost.
Driver: `tests/perf_combine_chunk.py` (a copy of `test_internode.py` reading `DEEP_EP_RDMA_CHUNK`;
`test_internode.py` itself is byte-identical/untouched), plus a `DEEP_EP_RDMA_CHUNK` passthrough in
`tests/docker-2node-v2/run.sh`. 4096 tokens, hidden 7168, R=2, production build:

| rdma_chunk_size | chunk-sends | combine(iso) µs | RDMA BW | vs 16 |
|---|---|---|---|---|
| 16 (default) | ~134 | **46316** | 1.2678 GB/s | — |
| 32 | ~67 | 47391 | 1.2391 GB/s | **+2.3% slower** |
| 64 | ~34 | 49284 | 1.1915 GB/s | **+6.4% slower** |

Quartering the AMO count made combine **worse**, not better (larger chunks coarsen credit granularity and
increase `waitspace`/`bar2` imbalance). Combined with the ±1.3% noise band, this rules out the blocking
AMO as the bottleneck.

### 17.5 Verdict

The iSHMEM fetching-AMO mechanism is **real and worth fixing in iSHMEM on its own merits** (557 µs of
blocking per chunk is indefensible for a fire-and-forget credit update, and it would matter for any
workload with a high chunk-send rate — LL, or normal combine at much larger token counts). But for
*this* bottleneck it is bounded at ~6.5% of forwarder time, sits off the critical path, and a chunk-count
sweep falsifies it directly. **Do not spend hardware time on the iSHMEM AMO patch for the combine
regression.**

The bottleneck remains where §16 put it: **the NVL sender's peer-IPC copy**, 97.74% busy at ~1.9 GB/s
while dispatch's structurally identical forwarder copy achieves ~8.9 GB/s. That 4.7× gap is the only
remaining lead worth pursuing (candidates: wider LSC messages, engaging the 6 idle sender warps at
`num_nvl_ranks=2`, traffic concurrency across the PCIe link).

## §18. `ISHMEM_IBGDA_QPS_PER_PE` sweep at 4096 tokens — combine is QP-count-INSENSITIVE (third falsification), but dispatch gains ~14%

Motivation: `deep_ep/buffer.py:260-291` sets `ISHMEM_IBGDA_QPS_PER_PE = 1` (setdefault) on the argument
that the path is "NIC-latency-bound, not bandwidth-bound" and each extra QP channel adds "a full extra
NIC round-trip on the critical path" (per-QP `fence_qp` + tail `long_atomic_add_qp`), with a measured
monotonic round-trip penalty at NT=32 (3875.7 µs at C=1 → 4667.1 at C=16). If that mechanism dominated
combine at 4096 tokens, combine should degrade monotonically in C.

Because `QPS_PER_PE` is a `setdefault`, an explicit env override wins — pure env, no rebuild.
All legs below: 4096 tokens, hidden 7168, R=2, production build, **igub reset + 4-GPU health gate before
each leg**, `===== PASS =====`.

| `QPS_PER_PE` | combine(iso) µs | combine RDMA BW | dispatch(iso) µs | dispatch RDMA BW |
|---|---|---|---|---|
| 1 | 46270 | 1.2691 GB/s | 9324 | 6.2977 GB/s |
| 2 | 46900 | 1.2520 GB/s | **7983** | **7.3562 GB/s** |
| 4 | 46299 | 1.2683 GB/s | **8040** | **7.3039 GB/s** |

### 18.1 Combine: completely flat in C — the round-trip-per-chunk mechanism does not dominate

46270 / 46900 / 46299 µs all sit inside the ±1.3% noise band. Adding QP channels — which by `buffer.py`'s
own stated model adds a full extra NIC round trip per chunk to the critical path — moved combine by
**nothing**. This is a third independent falsification of the blocking-AMO-dominates hypothesis, joining
the telemetry bound (§17.2-17.3) and the `rdma_chunk_size` sweep (§17.4).

Corollary for the arithmetic: the entire send region (put + fence + AMO) is 10.02% of forwarder time and
sits off the critical path (NVL sender waits 0.010%). Doubling the AMO count therefore cannot exceed
~+6.5%, and in fact registers as zero because the forwarder is not the critical path at all.

### 18.2 The `buffer.py` C=1 conclusion is token-count-specific and is now WRONG for dispatch at 4096

`buffer.py`'s study was tuned at small token counts (NT≤1024, latency-dominated). At 4096 tokens the
trade inverts for **dispatch**: C=2 and C=4 are ~14% faster than C=1 (9324 → 7983/8040 µs, 6.30 → 7.36
GB/s), reproducibly and well outside noise. Dispatch is bandwidth-bound at this size and genuinely
benefits from QP striping, exactly as one would expect once the path stops being latency-dominated.

Note this happens **today, with the blocking AMO still in place** — so it is not evidence for the AMO
theory either; it is ordinary bandwidth striping.

**Not changing the `buffer.py` default.** The C=1 choice remains correct at the small token counts it was
tuned for, the default is load-bearing for the LL path, and `deep_ep/` is out of scope for this work. The
actionable finding is that the optimal C is token-count-dependent and a future auto-tune could pick C≥2
for large-token normal dispatch. That is a **dispatch** opportunity, not a combine one.

### 18.3 Methodology warning — a wedged-HW artifact nearly produced a false positive

The first C=2 measurement read **89336 µs (+93%)**, which looked like dramatic confirmation of the
monotonic-penalty model. It was an artifact: that leg ran immediately after a C=1 leg that had hung
(rc=124) inside the same loop, with no igub reset in between (golden rule 4 cascade). Re-run on freshly
reset + health-gated hardware it read 46900 µs. C=4 measuring identical to C=1 is what exposed the
non-monotonicity and prompted the re-test. **Every env-sweep leg must get its own reset + health gate;
a single hung leg poisons every subsequent leg in the loop.**

### 18.4 Patch-design constraint for any future iSHMEM AMO work

`buffer.py` forces `qps_per_pe = 1` while the harness runs `kNumRDMARanks = 2`, so at the coordinator
head-credit site `internode_combine_fused.inc:931` (`lane_id < kNumRDMARanks`) **both lanes target the
same QP**. Any non-blocking-AMO patch must handle that collision; a naive per-lane watermark advance can
livelock there. The tail-credit site `:748` is single-lane (`lane_id == 0`) and is safe. Telemetry also
shows the coordinator is cold (`coord tot` ≈ 1.5e9 vs forwarder 18e9 cycles), so this site is a
correctness hazard rather than a perf opportunity.

## §19. Measured run durations — use them as timeouts, and as a fast wedge detector

I had been sizing `TIMEOUT_SEC` by guesswork (2400/3000 s) and polling with blind 600 s waits. Measured
on freshly reset + health-gated hardware:

| run | wall clock |
|---|---|
| production `.so`, full 64-config matrix + 4096-token perf | **68 s** |
| same run on wedged HW | never finishes (hits whatever `TIMEOUT_SEC` is set) |
| telemetry `.so` (`DEEP_EP_COMBINE_TELEMETRY=1`, per-call `queue.wait()` + D2H + `fprintf`) | ~30-50 min |
| igub reset cycle (`--down` + `rmmod` + 45 s drain + `--up`) | ~60-75 s |

### 19.1 The operational consequence: a long run is a WEDGE, not slow work

A healthy production run is **68 s**. So `TIMEOUT_SEC=3000` never bought tolerance for a slow machine —
it bought a **50-minute wait to discover the GPU was wedged**. Every rc=124 I hit cost ~40-50 min of
dead time that a 300 s budget would have surfaced in 5.

Recommended budgets:

| scenario | `TIMEOUT_SEC` | agent `initial_wait` |
|---|---|---|
| production perf/correctness run | **300** (4.4× margin) | 120, then poll at 60 |
| telemetry build | 3000 | 600 |
| reset cycle | – | 180 |

**Anything past ~150 s on a production build means the hardware is wedged.** Kill it, run the reset
recipe + 4-GPU health gate, and re-run — do not wait it out.

### 19.2 Retro-explanation of every rc=124 in this campaign

- The first `rdma_chunk_size` sweep timed out at `TIMEOUT_SEC=2400` because it was accidentally running
  the **telemetry** `.so` (reverted the `.inc` without rebuilding). Genuinely slow, not wedged.
- The `QPS_PER_PE=1` leg and the `TIMEOUT_SEC=600` timing probe both ran on **non-reset** hardware and
  wedged. On clean HW the identical run takes 68 s.

Both classes are diagnosable in ~2 min with a 300 s budget. Neither justified a 40-minute wait.

### 19.3 Rule

Always run a production leg as: reset → 4-GPU health gate → `TIMEOUT_SEC=300`. If it exceeds that,
treat it as a hardware wedge (golden rule 4), not as a measurement.

## §20. Where the 68 s actually goes — only ~3.5 s is measurement, and there is a REAL intermittent hang

Challenged on "it should not take so long", I instrumented the harness (wall-clock stamps on every log
line) and my own driver copy (`tests/perf_combine_chunk.py`). Measured phase budget of a healthy
4096-token run:

| phase | wall clock |
|---|---|
| driver load, NIC gate, MPI/container start, buffer create | 13 s |
| `[layout]` bench | 2 s |
| **test-side dispatch + `check_data` region (pre-combine)** | **43.6 s** |
| `buffer.combine()` kernel call | **0.10 s** |
| validation math (`calc_diff`, per-token err, asserts) | **0.12 s** |
| `DEEP_EP_PERF` benches (round_trip + dispatch + combine, 30 iters each) | ~3.4 s |
| teardown | ~6 s |

**Only ~3.5 s of the 68 s is actual measurement.** The single dominant cost (43.6 s, 64%) is the test's
own dispatch + `check_data` section at 4096 tokens — 8192×7168 reductions, per-rank `.item()`
synchronisations, and dtype conversions. It is **test overhead, not kernel time**: the combine kernel
call itself is 0.10 s and the validation math 0.12 s.

Two hypotheses I formed and **disproved** by measurement: that the cost was `calc_diff`'s
double-precision 470 MB reductions (it is 0.115 s), and that it was the combine kernel (0.10 s). What
remains unlocalised inside the 43.6 s is the dispatch test + `check_data` loop specifically; I did not
narrow further.

### 20.1 `DEEP_EP_PERF_TOKENS` runs are NOT the 64-config matrix — my "PASS" claims were weaker than stated

`test_internode.py:633`: `DEEP_EP_PERF_TOKENS` does `os.environ.setdefault('DEEP_EP_MIN', '1')`, which
shrinks the 32-combination sweep to ONE deterministic sanity config. So every perf run in §17/§18 emits
exactly **2 `passed` lines (one config, printed by both node-local rank 0s), not 64**. Those runs are a
sanity gate, not full validation. The genuine 64-config matrix was run separately at `c0f322c`.

### 20.2 A real intermittent hang in the perf-bench phase — do NOT keep filing this as "HW wedge"

Across four consecutive runs of the *identical* command on **freshly igub-reset, health-gated** hardware
with an unchanged `.so`: **PASS (68 s), HANG, HANG, PASS (68 s)**. All hangs occur *after* the
correctness config prints ` passed`, i.e. inside the `DEEP_EP_PERF` bench loop at 4096 tokens
(round_trip / dispatch / combine, 30 iterations each).

This materially weakens the golden-rule-4 "accumulated hardware wedge" attribution I had been applying:
a reset immediately preceding the run does not prevent it, and a healthy run can directly follow a hung
one. **~50% failure rate at 4096 tokens is a real robustness problem in the measured path and should be
treated as a live suspect (code or iSHMEM), not written off as environmental.** It also means every
perf number in §17/§18 came from the surviving ~50% of runs — the numbers are reproducible and mutually
consistent, but they are conditioned on non-hanging runs.

### 20.3 Consequence for the tolerance question

`test_internode.py:428-431` applies an XPU-only tolerance `tol = max(5e-6, 6e-4 * scale)` with
`scale = max(1, (num_tokens/32)**2)` — at N=32 that is 6e-4 vs CUDA's 5e-6 (~120× looser), and it grows
quadratically, so at large N the `x_diff` check is effectively vacuous. The per-token `[x ALSO WRONG]`
prints (errs 0.0078/0.0156 = 1-2 BF16 ULPs) appear in **passing** runs too and are diagnostics, not
failures. This is **pre-existing** and out of scope (the file must stay byte-identical), but it means the
x-correctness gate is much weaker on XPU than the CUDA reference and should not be leaned on.
