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

> # ✅ RESOLVED (2026-08-21) — was: "THE CURRENT DEFAULT IS NOT SAFE"
>
> **History (do not delete — this is the hard-won rule).** The fused internode-normal path shipped as
> the default at HEAD and intermittently HUNG and SILENTLY DROPPED TOKENS at large token counts.
> Controlled A/B (identical `tests/`+harness, only `csrc/` swapped; full igub reset + 4-GPU health gate
> before every launch; `NUM_TOKENS=2048 HIDDEN=7168`):
>
> | arm | PASS | HANG | fail rate |
> |---|---|---|---|
> | fused (HEAD, pre-fix) | 4 | 6 | **6/10 = 60%** |
> | legacy (`b231f1c`) | 10 | 0 | **0/10** |
>
> Fisher exact one-sided **p = 0.0054** ⇒ a regression introduced by the migration. One launch lost
> **exactly 65 whole tokens** (zeroed rows, surviving rows bit-correct). See §21–§22.
>
> **Root cause: the fused grid was not clamped to the device's work-group co-residency limit** (§23).
> Both fused kernels split every channel across two work-groups (`channel_id = sm_id/2`,
> `is_forwarder = sm_id%2`) that spin on each other's queue counters, so the grid only makes forward
> progress if **every work-group is simultaneously resident**. Intel GPUs give no such guarantee and do
> not preempt a spinning work-group.
>
> **Fix (§24): `Config::num_sms` is now a REQUEST, clamped in `Buffer::fused_num_channels()`** →
> `internode::fused_max_coresident_sms()`, which loudly logs whenever it reduces the requested grid.
> Post-fix, reset + health gate before every launch:
>
> | regime | PASS | FAIL | N |
> |---|---|---|---|
> | 2048 tok / hidden 7168 | 16 | 0 | **16** |
> | 4096 tok / hidden 7168 | 10 | 0 | **10** |
> | 64-config matrix (32/1024) | 64/64 ×2 runs | 0 | 2 |
>
> **The 64-config matrix at its default size is blind to this class of bug** — always validate at
> 2048+ tokens / hidden 7168 and report `k/N`.
>
> **⚠️ Perf cost is real and NOT a wash:** the §13 tables were measured on the unsafe, over-subscribed
> grid. Post-clamp round-trip is ~2.8× slower at 2048/4096 tokens. See §24.3.

## 1.1 Migration status: COMPLETE (2026-08-20) — SEE KNOWN ISSUE ABOVE

The internode NORMAL path is now a fused, warp-specialized, CUDA-faithful port and is the
**default** (the `DEEP_EP_INTERNODE_FUSED` env gate and all legacy phase-split micro-kernels
were deleted; `internode.cpp` went 3090 → 447 lines with the kernels in three `.inc` files).
`tests/docker-2node-v2` passes the full matrix (64 configs, BF16/FP8 × with/without top-k ×
async × previous-event) **at the matrix's default size only (32 tokens / hidden 1024), which is
proven blind to the KNOWN ISSUE above.** At 2048 tokens / hidden 7168 this default fails 6/10.

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

## 13. Measured perf after fusion (A/B baseline) — ⚠️ SUPERSEDED, measured on an UNSHIPPABLE grid

> **All numbers in this section are SUPERSEDED by §24.4.** They were measured at `num_sms=24`, the
> pre-fix default, which §22 proves fails 6/10 at 2048 tokens / hidden 7168 and which §23 shows
> over-subscribes the device's work-group co-residency limit. Every figure here is therefore
> **conditioned on the surviving non-hanging runs of a configuration we no longer ship**, and each is
> a single sample. The "all PASS" validation referenced below came from the 64-config matrix at
> 32 tokens / hidden 1024, which is proven blind to the defect.
>
> **Use §24.4 for current numbers.** The clamp costs ~1.7x round-trip at 2048 tokens and ~2.8x at
> 4096 versus the table below. The fused path is still ~4.0x faster than legacy at 4096, but the
> headline "3.1x faster" below was measured on a configuration that hangs 60% of the time.

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

## §21. PRIORITY-1 RESULT: the instability is in DISPATCH, not combine — and it includes silent data loss

Bisected with a bench selector (`DEEP_EP_BENCH_SEL` = `rt|disp|comb|all`) plus progress markers, so the
last marker before a failure localises it. Every leg: full igub reset + 4-GPU health gate before launch,
`TIMEOUT_SEC=300` per §19.

| arm | tokens | result | failure marker |
|---|---|---|---|
| combine only | 4096 | **6/6 PASS** | – |
| dispatch only | 4096 | **3/6 HANG** | always `entering dispatch` |
| dispatch only | 2048 | 5/6 PASS, **1 CORRECTNESS FAILURE** | – |

**The combine kernel is not the unstable one.** All my combine perf numbers (§17/§18) stand. The
instability I had been attributing to hardware wedge lives in the **fused dispatch** path.

### 21.1 Silent data loss at 2048 tokens — 65 whole tokens dropped

The 2048 leg failed with `AssertionError: segment src_rank=3 values mismatch` from `check_data`:

```
[check_data FAIL rank=0] segment for src_rank=3 rows [2560,3386) err_sum=-1397760
                         first_col_vals=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
```

Decoding: the segment is 826 rows × 7168 hidden. Expected value 3 everywhere. If N elements are **zero**
instead of 3, `err_sum = -3N` ⇒ N = 465 920 elements ⇒ **465 920 / 7168 = exactly 65.0 rows**.

So the failure is **65 entire tokens never written** (left as allocator zeros), not numerical drift, not
a partial/torn row. `first_col_vals` being all 3 confirms the surviving rows are perfectly correct — this
is dropped whole tokens, i.e. a lost queue slot / lost credit, precisely the failure mode the
`poll_load` split-producer bug produced and that a capped spin (`kFusedSpinCap`) `break`ing instead of
failing would produce.

### 21.2 Why this matters more than the remaining perf gap

- The dispatch path intermittently **drops tokens without failing loudly** — at 4096 the sanity config
  passed and the *bench* hung; at 2048 the sanity config caught it. Between those two, the 30.8× dispatch
  speedup (`b231f1c` → `c804991`) was measured on a path that is not reliably correct.
- Because `DEEP_EP_PERF_TOKENS` collapses the matrix to ONE config (§20.1), the routine perf runs have
  very little chance of catching this. The single time it *was* caught, it was caught by the sanity
  config, at the smaller token count.
- Onset is token-count dependent (0/6 hangs at 2048 vs 3/6 at 4096), which fits a **wraparound/capacity
  threshold** rather than a pure timing race.

### 21.3 Next steps (not yet done)

1. Re-run the full 64-config matrix several times to establish the true dispatch corruption rate — the
   matrix, not the 2-config perf gate, is the instrument that catches this.
2. Instrument `kFusedSpinCap` exhaustion in the dispatch kernel: any spin that hits its cap and `break`s
   is a candidate for the lost credit. A cap-hit counter would confirm or exclude it in one run.
3. Check queue head/tail wraparound arithmetic in the dispatch RDMA/NVL receivers at the 4096-token
   capacity, and the decoupled `kRDMASenderCoordinator`'s `processed_tail` vs `last_issued_tail`
   accounting (`internode_dispatch_fused.inc:625-631`) — it keeps multiple puts in flight, so a missed
   credit there loses a whole chunk.
4. 65 = 64+1 is suspiciously close to 4×16 (rdma chunk) and 8×8 (nvl chunk); worth checking whether the
   loss is exactly one chunk plus one token.

**Correction to a standing lead:** the Priority-2 suggestion to compare combine's `st_uc_global_v` /
`ld_uc_global_v` against dispatch is **stale** — those were the `c0f322c` fix. Combine now uses
`ld_nc_global_v` + `st_na_global_v` everywhere on the payload path (`:259, :264, :276, :287, :537-539`),
identical hints to dispatch (`:509-511, :796-797, :945`), and combine's copy is additionally unrolled ×2
where dispatch's is not. The 4.7× sender gap therefore persists *with identical, already-cached hints* —
the cache-hint explanation for the residual gap is closed.

## §22. DECISION-RELEVANT: the instability is a REGRESSION INTRODUCED BY THE MIGRATION

Controlled A/B, identical harness/driver/test files (only `csrc/` swapped), full igub reset + 4-GPU
health gate before **every** launch, N=10 per arm, `NUM_TOKENS=2048 HIDDEN=7168` (the regime that
reproduces), classification PASS / HANG / CORRUPT.

| arm | build | PASS | HANG | CORRUPT | fail rate |
|---|---|---|---|---|---|
| **fused (HEAD `194cf80`)** | fused warp-specialized, default | 4 | **6** | 0 | **6/10 = 60%** |
| **legacy (`b231f1c`)** | phase-split, default | **10** | 0 | 0 | **0/10** |

**Fisher exact, one-sided: p = 0.0054.** The difference is significant; this is not run-to-run variance.

At `b231f1c` the fused kernels are still behind the `DEEP_EP_INTERNODE_FUSED` env gate, so the legacy arm
is simply that build's default — no source edits were needed to select it, and `tests/` stayed at HEAD
for both arms, isolating the variable to `csrc/`.

### 22.1 Verdict

The fused warp-specialized internode path introduced an intermittent failure that the legacy phase-split
path does not exhibit: **60% hang rate at 2048 tokens / hidden 7168**, plus the separately observed
silent 65-token data loss (§21.1). The legacy path was clean in 10/10 runs under identical conditions.

**The fused default (`f494d97`) should be treated as not shippable in its current state.** The
engineering options are (a) restore the `DEEP_EP_INTERNODE_FUSED` gate and default it OFF until fixed,
or (b) revert the default flip. Either keeps a silently-corrupting kernel out of the default path while
root-causing continues. This is a call for the humans; recording the evidence and the recommendation.

### 22.2 Scope of what the regression invalidates

- The **dispatch** 30.8× speedup and the fused round-trip table (§13) were measured on the unreliable
  path. The numbers are reproducible but conditioned on non-hanging runs.
- The **combine** work (§14, the 2.63× cache-policy fix) is unaffected in its own right — combine-only
  benching was **6/6 clean** (§21) — but it ships inside the same fused default, so it inherits the gate
  decision.
- All prior "64/64 PASS" claims are **single samples of an intermittent path** and must not be read as
  proof of correctness. At the default matrix size (32 tokens / hidden 1024) a full matrix run takes
  64 s and passed 64/64 — i.e. the standard gate does **not** exercise the failing regime at all.

### 22.3 Rate caveats (report k/N, never "passes")

- Hang rate is strongly regime-dependent: 0/6 (dispatch-only bench) vs 6/10 (all three benches) at 2048;
  3/6 at 4096 dispatch-only. More benching = more exposure, consistent with an accumulation/wraparound
  threshold rather than a fixed per-launch probability.
- CORRUPT is rarer than HANG: 1 observed corruption across ~30 fused launches vs many hangs. The single
  corruption (65 tokens zeroed) is the more dangerous manifestation because it is **silent**.
- The default-size 64-config matrix has **zero observed sensitivity** to this bug. Any future validation
  of a fix must run at 2048+/7168 with N large enough to distinguish from a 60% base rate.

---

## 23. Root cause of the fused-dispatch hang: WORK-GROUP CO-RESIDENCY (occupancy), not queue wrap

Two hypotheses were tested head-to-head on freshly-reset HW, one launch per reset, 4-GPU health gate
before every launch, `TIMEOUT_SEC=300`, `tests/perf_combine_chunk.py` (byte-identical `tests/` otherwise),
2 nodes x 2 ranks, num_tokens=2048, hidden=7168, BF16.

### 23.1 DISPROVEN: the RDMA receive-queue wrap / half-depth sign-flip theory

The theory: 65 lost tokens = 64+1 = half of `num_max_rdma_chunked_recv_tokens=128` + 1, i.e. a wrapped
index comparison flipping sign at half depth. Probe: raise the RDMA recv queue to 256 via a new
`DEEP_EP_RDMA_RECV` env (driver + `run.sh` `_add_opt_genv` whitelist + campaign passthrough). The
`[queue-probe] rdma_chunk=16 rdma_recv_tokens=256` line was verified in every log.

| `num_max_rdma_chunked_recv_tokens` | PASS | HANG | N |
| --- | --- | --- | --- |
| 128 (control, this session) | 3 | 1 | 4 |
| 128 (control, §22 arm) | 4 | 6 | 10 |
| **256** | **4** | **4** | **8** |

Doubling the queue does not move the failure. **Hypothesis rejected.** A full line-by-line re-audit of
the fused dispatch against CUDA `internode.cu` also found the queue arithmetic faithful: the producer
flow-control gate (`.inc:477` vs CUDA `:644`), the 32-bit SLM release window (`.inc:563-583` vs CUDA
`:725-748`), the coordinator's `processed_tail`/`last_issued_tail`/`% num_max_rdma_chunked_recv_tokens`
accounting (`.inc:620-641` vs CUDA `:800-812`), the forwarder (`.inc:735-812` vs CUDA `:903-1005`) and
the NVL receiver (`.inc:915-981` vs CUDA `:1098-1188`) are all equivalent.

### 23.2 CONFIRMED: failure rate is monotone in `num_sms` (grid size)

`num_sms` was made overridable (`DEEP_EP_NUM_SMS`, driver-side only; `tests/test_internode.py` untouched).
Everything else held constant.

| `num_sms` | work-groups | PASS | HANG | N | hang rate |
| --- | --- | --- | --- | --- | --- |
| 8  | 8  | 8 | 0 | 8  | **0%** |
| 16 | 16 | 6 | 2 | 8  | 25% |
| 24 (default) | 24 | 4 | 6 | 10 | **60%** |
| 32 | 32 | 2 | 4 | 6  | **67%** |

The relationship is monotone in both directions from the default: shrinking the grid removes the
failure entirely, over-subscribing it makes the failure worse. Fisher exact, `num_sms=8` vs the `num_sms=24` default: one-sided **p = 0.011**. Env application was
verified independently of the classifier: at `num_sms=8` dispatch(iso) is 7635 us / 3.85 GB/s vs
5494 us / 5.34 GB/s at 24, so the grid really changed.

### 23.3 Mechanism

The fused dispatch kernel launches `num_sms` work-groups of `(7 + 1 + 8) * 32 = 512` work-items and
splits each channel across **two different work-groups**: `channel_id = sm_id / 2`, and
`is_forwarder = (sm_id % 2 == 0)`. So the forwarder half of a channel lives in an even-numbered
work-group and the RDMA-sender/NVL-receiver half in the odd-numbered one, and they **spin on each
other** (`rdma_channel_tail`, `nvl_channel_tail`, `nvl_channel_head`). That makes the whole grid a
producer/consumer network with **hard forward-progress dependencies across work-groups**, which is only
correct if **all `num_sms` work-groups are simultaneously resident**. Intel GPUs give no such guarantee
and do not preempt a spinning work-group, so once the grid exceeds what the device can co-schedule, a
resident consumer spins on a producer that has not been dispatched. Depending on which spin is starved,
it surfaces as either:
- a **HANG** (the spin outlasts the 300 s harness timeout), or
- **silent CORRUPTION** (`kFusedSpinCap = 2e8` is exceeded and the loop `break`s, leaving whole tokens
  never copied - exactly the observed 65 zeroed rows, with all surviving rows bit-correct).

This also explains everything the wrap theory could not: monotonicity in `num_sms`, token-count
dependence (longer occupancy of each work-group widens the window), intermittency (co-residency is
marginal, not deterministic - it depends on register/SLM-driven occupancy and on what else is on the
device), and why the **legacy phase-split path at `b231f1c` is 0/10** - separate kernel launches per
phase have no cross-work-group spin dependency, so occupancy can never deadlock them.

CUDA gets away with the fused form because DeepEP sizes `num_sms` to the SM count of a 108/132-SM
datacenter GPU. The BMG part here reports far fewer independent work-group slots for a 512-work-item,
high-register kernel. The codebase **already knows this rule** for the LL path: `csrc/xpu/internode_ll.cpp:264-277`
(`ll_put_wgs`) states "the ordered commit gate ... requires the producing sub-groups to be CO-RESIDENT
(CUDA sizes its grid to num_sms for the same reason)" and clamps the grid to `max_compute_units`. The
fused internode-normal port did not inherit that clamp.

### 23.4 Direction of the fix

Clamp the fused internode dispatch/combine grid to the device's co-residency capacity in
`csrc/xpu/deep_ep_xpu.cpp` (where `num_channels = config.num_sms / 2` is computed), mirroring
`ll_put_wgs()`: derive a maximum work-group count from
`device.get_info<sycl::info::device::max_compute_units>()` and the actual occupancy of the fused kernel,
and cap `num_channels` at half of it, with an env override. This keeps `Config.num_sms` as a request
rather than a mandate. Validation must be at 2048+/7168 with N large enough to separate from a 60% base
rate - the default-size 64-config matrix is proven blind (§22.3).

### 23.5 Reproducer plumbing added

- `tests/perf_combine_chunk.py` (driver copy only): `DEEP_EP_RDMA_RECV`, `DEEP_EP_NUM_SMS`.
- `tests/docker-2node-v2/run.sh`: both added to the `_add_opt_genv` whitelist (a var not on that
  whitelist is silently dropped - this has cost a whole sweep before).
- `/tmp/corrcamp.sh`: passes both through; `TIMEOUT_SEC=300`.

---

## 24. THE FIX: clamp the fused grid to work-group co-residency

### 24.1 What was implemented

`Config::num_sms` is now a **request**, not a mandate.

- `csrc/xpu/internode.cpp` → `internode::fused_max_coresident_sms(num_rdma_ranks, queue)`
  (declared in `csrc/xpu/xpu_runtime.hpp`). Result cached per `num_rdma_ranks`.
- `csrc/xpu/deep_ep_xpu.cpp` → `Buffer::fused_num_channels(config)`, called at BOTH internode
  entry points (`internode_dispatch` and `internode_combine`). It **must** be the same value in
  both: `num_channels` is baked into the shared RDMA/NVL buffer layout and into the tensors
  dispatch hands to combine.
- **Loud logging**: whenever the clamp actually reduces the requested grid, rank 0 prints a
  `[DeepEP] WARNING: internode fused grid CLAMPED for work-group co-residency: requested
  num_sms=24 -> using 8 ...` line to stderr. This bug once looked like a correctness bug; it must
  never be silent again.
- **Override**: `DEEP_EP_FUSED_MAX_SMS` (also logs, and prints the device-derived value it replaced).

The limit is **derived, not guessed**, from the driver's own per-kernel occupancy answer:
`kernel.ext_oneapi_get_info<syclex::info::kernel_queue_specific::max_num_work_groups>(queue, wg_size, 0)`
on `FusedDispatchKernel<R>` (512 work-items) and `FusedCombineKernel<R>` (`(kForwarders+1)*32` = 800
work-items at R=2), taking the min. Fallback if the query throws: Xe-core arithmetic from
`ext::intel::info::device::gpu_eu_count / gpu_eu_count_per_subslice / gpu_hw_threads_per_eu`.

### 24.2 ⚠️ The derived value is NECESSARY BUT NOT SUFFICIENT — read this before tuning

On **Arc Pro B60** (`max_compute_units=160`, `gpu_eu_count=160`, `eu_per_subslice=8`,
`hw_threads_per_eu=8`, 5 slices × 4 subslices = **20 Xe-cores**, built with
`-ze-intel-enable-auto-large-GRF-mode`) the driver answers **20 work-groups for BOTH fused kernels**
(`dispatch=20 combine=20`, i.e. 1 work-group per Xe-core).

**But 20 is not safe.** Measured on this hardware:

| `num_sms` | PASS | FAIL | N | source |
| --- | --- | --- | --- | --- |
| 8  | 8 | 0 | 8  | `DEEP_EP_NUM_SMS=8` |
| 8  | 8 | 0 | 8  | `DEEP_EP_FUSED_MAX_SMS=8`, post-fix build |
| 16 | 6 | 2 | 8  | `DEEP_EP_NUM_SMS=16` |
| 20 | 0 | 1 | 1  | first build of the clamp, driver-derived default |
| 24 | 4 | 6 | 10 | shipped default (pre-fix) |
| 32 | 2 | 4 | 6  | `DEEP_EP_NUM_SMS=32` |

So the driver's co-residency bound is an **upper** bound only; something beyond raw co-residency
also scales with the grid. **RESOLVED — see §24.2.1: the mechanism is NOT co-residency but single-QP IBGDA contention.** Until it is explained, `fused_max_coresident_sms()` applies an explicit, commented
`kEmpiricalSafeSms = 8` cap on top of the derived value. Candidate explanations not yet tested:
Level Zero may not actually dispatch all "co-resident-capable" work-groups concurrently without a
cooperative launch; the two ranks per node share a GPU-adjacent NIC/proxy; the query may not model
SLM + named-barrier resources; or an additional per-grid resource (named barriers are a per-Xe-core
resource) is exhausted. Note also that `DEEP_EP_NUM_SMS` changes the channel count as well as the
grid, so the dose-response is not a pure grid-size sweep.

### 24.2.1 ⚠️ RESOLVED (2026-08): it is **NOT co-residency** — it is **single-QP IBGDA contention**

The §24.2 open question is answered, and the answer **contradicts the co-residency story**. Three
independent experiments, all on the same build, reset + 4-GPU health gate before every launch:

**(a) Grid size alone does NOT trigger it (pad-work-group discriminator).**
`DEEP_EP_FUSED_PAD_WGS=N` / `DEEP_EP_FUSED_PAD_CYCLES=C` (`internode_dispatch_fused.inc`) append N
work-groups that take the **LOW** work-group ids (so the hardware gives them their slots FIRST),
touch no memory, spin `C` GPU cycles and retire. They add pure co-residency pressure at a **fixed
channel count**. Real work-groups are then re-indexed `sm_id = raw_sm_id - pad_wgs`.

| arm | grid | channels | pad cycles | PASS | FAIL | N |
| --- | --- | --- | --- | --- | --- | --- |
| clamp 8 + PAD=16 | 24 WGs | 4 | 3e7 (~12 ms) | 8 | 0 | 8 |
| clamp 8 + PAD=16 | 24 WGs | 4 | 6e8 (~250 ms) | 2 | 0 | 2 |
| clamp 8 + PAD=19 | 27 WGs | 4 | 6e8 (~250 ms) | 3 | 0 | 3 |
| **clamp 24 + PAD=0 (control)** | 24 WGs | **12** | — | 2 | **4** | 6 |

At PAD=19 at most **one** real work-group can be resident and the denial lasts ~250 ms, i.e. **3×
longer than `kFusedSpinCap` = 2e8 cycles (~83 ms)** — and it still never hangs or corrupts. A grid
of 24-27 work-groups is clean at 4 channels and fails 4/6 at 12 channels. **Failures track channel
count, not grid size.**

**(b) The runtime co-schedules far more work-groups than the driver's `max_num_work_groups`.**
Standalone probe `/tmp/probe_res.cpp` (every WG bumps a global counter then spins until it sees all
of them or ~80 ms elapses; the max value observed is a lower bound on true concurrency): a 512-WI
kernel reaches **128/128 concurrent** at 0, 8 KiB, 32 KiB and 64 KiB of SLM per WG. So
`max_num_work_groups = 20` is a **capability/occupancy estimate, not a scheduling limit**, and
raw co-residency was never the binding constraint at 24 WGs.

**(c) The actual trigger: N channels sharing ONE IBGDA QP.** `deep_ep/buffer.py` pins
`ISHMEM_IBGDA_QPS_PER_PE=1` (setdefault) for the normal internode path — so **every channel's RDMA
sender drives the same QP**, and pressure on that QP's send queue scales with the channel count.
Forcing one QP per channel removes the failure entirely at the previously-failing grid:

| config (2048 tok / hidden 7168) | PASS | HANG | N |
| --- | --- | --- | --- |
| `FUSED_MAX_SMS=24` (12 ch), `QPS_PER_PE=1` (default) | 2 | **4** | 6 |
| `FUSED_MAX_SMS=24` (12 ch), `QPS_PER_PE=16` | **6** | 0 | 6 |

Fisher exact, same build, one-sided **p = 0.030**; pooled with the pre-fix 6/10 at the same config,
**10/16 vs 0/6, p = 0.012**.

**Consequences.**
1. `kEmpiricalSafeSms = 8` works, but for the wrong reason: it limits the grid, which limits the
   channel count, which limits per-QP pressure. It is a **proxy fix**, not the root fix.
2. The mechanism to audit next is the **uncapped SQ-wrap backpressure spin** in
   `ishmem_ibgda/src/ibgda_device_impl.h` (~:2481-2491, fires only once `wqe_idx >= nic_wq_slots`)
   and the cross-work-group WQE-index reservation on a shared QP — a lost/over-subscribed WQ slot
   there is exactly a hang whose probability grows with the number of concurrent producers.
3. **Throughput may be recoverable**: `num_sms=24` + `QPS_PER_PE=16` measured round-trip
   **30 612 µs** vs the clamped default's **53 667 µs** at 2048/7168 (dispatch iso 4 765 vs 7 562 µs)
   — a **1.75×** win, 6/6 clean. This is NOT yet shipped: it needs N≥16 at 2048 and a 4096 leg
   before the default changes, and `buffer.py` is out of scope for this workstream (the QP count is
   set there).

### 24.2.2 Falsified: the iSHMEM leader-gate chain is NOT the deadlock

The natural follow-on hypothesis was iSHMEM's **ordered-commit gate**: slots are claimed with
`fetch_add` on `nic_wq_cnt`, then the claimer spins `while (commit != base)` — **uncapped, strict
equality** (`ibgda_device_impl.h` put_nbi_warp ~:2250 and `emit_direct_wqe_skeleton` ~:905), plus an
uncapped SQ-wrap CQ-backpressure spin in `rdma_atomic64` (~:2481). With one shared QP the producer
holding `base-1` is a **different work-group**, so this is a cross-work-group forward-progress
dependency. Both mechanism and the fit to the data are real — but it is **not what hangs**:

| build (num_sms=24, QPS=1, 2048 tok / 7168) | PASS | HANG | N |
| --- | --- | --- | --- |
| production | 2 | 4 | 6 |
| + gate diagnostics (print after 20M spin iters) | 3 | 3 | 6 |
| + gate diagnostics **and BREAK** after 50M iters at all three spins | 3 | 3 | 6 |

Breaking every uncapped SQ spin leaves the hang rate unchanged. **Positive control:** rebuilt with
the threshold at 1 000 iterations, a single passing run emitted **360** `[IBGDA-GATE-STUCK]` lines
(`base=586 commit=582 nwqes=4` …) — so the instrumentation, the device `printf` and the gate spin
itself all work, and under normal contention the gate never exceeds ~50M iterations.
**Caveat, stated plainly:** device `printf` is flushed at kernel completion, so a hanging kernel's
messages are lost; the break arm is what carries the weight (if the gate held the deadlock,
releasing it would let the kernel finish). Residual possibility: an out-of-order publish caused by
the break could produce a *downstream* NIC-error hang that masks the rescue. To fully close that
would need hang-time host-visible state (USM marker) or `gdb-oneapi`, not `printf`.

The instrumentation was reverted; `libishmem.a` is back to the canonical archive
(`build/_install/lib/libishmem.a:1787231137000000000:29716510`, barrier `.cpp.o` md5
`13e1f807bddaaa3099d1d62bde881393`). So the *mechanism* of per-QP contention is established by the
QP A/B, but the precise stall site inside iSHMEM is **still open**.

### 24.2.3 THE FIX: one QP per channel — correctness AND throughput

`deep_ep/buffer.py` no longer pins `ISHMEM_IBGDA_QPS_PER_PE=1` on the **normal-internode branch**
(the low-latency branch is untouched — C=1 stays load-bearing there). It now uses the same
`clamp_pow2(num_qps_per_rank)` rule as the LL branch (24 → 16), still via `setdefault` so an
explicit user/harness value wins. `internode.cpp::fused_max_coresident_sms()` became QP-aware:
the grid cap is `max(kEmpiricalSafeSms=8, 2 * kSafeChannelsPerQp(=2) * qps_per_pe)`, still capped by
the driver's co-residency answer — so with 16 QPs the requested grid is no longer cut to 8, and with
QPS=1 the old safe 8 is preserved. Shipped default is now **num_sms=20 (10 channels), QPS=16**.

**Validation of the SHIPPED default** (reset + 4-GPU health gate before every launch):

| config | PASS | CORRUPT | HANG | N |
| --- | --- | --- | --- | --- |
| 2048 tok / hidden 7168 | **16** | 0 | 0 | 16 |
| 4096 tok / hidden 7168 | **8** | 0 | 0 | 8 |
| 64-config matrix (32 tok / 1024) | 64/64 | 0 | 0 | 1 |

Plus, before the default flip, at `FUSED_MAX_SMS=24` + `QPS=16`: **16/16 @2048, 8/8 @4096**.
Against the ~60% base failure rate, 16/16 has p ≈ 4e-7.

> ### ⚠️ 2026-08-22 — THE ABOVE VALIDATION IS OVERCLAIMED. DO NOT CITE IT AS "AS-SHIPPED".
>
> Every one of those 2048/4096 launches carried **`DEEP_EP_PERF_TOKENS=<tok>`**, which sets
> `DEEP_EP_MIN=1` and collapses `test_internode.py` to **2 configs + the perf bench** (2 `passed`
> lines, not 64) — *and* **`ISHMEM_IBGDA_DB_BATCH_SIZE=8`**, which is not a default. The 64-config
> matrix was only ever run at **32 tok / hidden 1024**, a regime §21 proves is BLIND to this bug
> class. So the shipped default was validated at the right *kernel configuration*
> (20 SMs / 10 channels / 16 QPs) but at the **wrong workload**.
>
> An independent run of the **plain default invocation** (full 64-config matrix, 2048 tok /
> hidden 7168, no `DB_BATCH`, no SMS/QPS overrides) **HANGS: 2/2**. Evidence it is a true hang and
> not a timeout: with `TIMEOUT_SEC=5400` the run sat **763 s with `passed_lines=0`** and the log
> **mtime did not advance for 12 minutes**, stalled on the *first* config — the same first config
> that completes well inside 300 s in perf mode.
>
> Therefore the one-QP-per-channel change is, on current evidence, **rate-reducing rather than
> curative** (consistent with §24.2.2 still being OPEN: the mechanism is established, the stall
> site is not). Two uncontrolled variables separate the passing and hanging runs — **test mode**
> (perf vs full matrix) and **`DB_BATCH_SIZE`** (8 vs unset). A 2×2 factorial is in flight to
> attribute it. Note `DB_BATCH_SIZE` must NOT be assumed inert here: the "inert because
> `force_db=true`" note is source-derived, and doorbell batching is mechanically adjacent to the
> per-QP contention this section establishes.
>
> **UPDATE — RESOLVED, see §25. The "rate-reducing rather than curative" inference above was
> WRONG.** The 2×2 factorial settled it: `DB_BATCH_SIZE` carries none of the effect (A1≡A2,
> A3≡A4 — the "inert" claim now has direct experimental support), and **test mode carries 100%**
> (perf 8/8 PASS vs full matrix 0/8, p ≈ 1e-4). It was never probabilistic: the full matrix runs
> the `without top-k` leg's **cached dispatch**, a code path the reduced perf mode structurally
> cannot reach, and that path fed combine a **padded** `x`. Deterministic bug, not a residual
> scheduling hazard — see §25. **This does not rehabilitate the tables above**: they were measured
> in a mode that could not exercise the cached dispatch, which is exactly why the bug escaped.
>
> Still genuinely open: whether `QPS_PER_PE=16` is *required* now that the pad-row bug is fixed,
> or whether the QP=1 hangs in §24.2.1 were a second, independent failure mode. The QP A/B stands
> on its own perf-mode evidence and QPS=16 is also 2.3x faster, so the default is justified either
> way — but the correctness *necessity* of QP=16 has not been re-tested post-§25.
>
> **Gate rule this cost us:** a correctness gate MUST run the **full matrix at 2048+/hidden 7168
> with no `DEEP_EP_PERF_TOKENS`**, because `DEEP_EP_PERF_TOKENS` silently sets `DEEP_EP_MIN=1` and
> makes the run near-blind to correctness. Never validate a fix in perf mode.

**Perf — the clamp regression is not just recovered, it is beaten** (hidden 7168, round-trip µs):

| tokens | clamped num_sms=8, QPS=1 (24.4) | **new default (20 / QPS=16)** | speedup |
| --- | --- | --- | --- |
| 32 | 1 626.9 | **1 487.3** | 1.09x |
| 512 | 14 934.7 | **8 083.9** | 1.85x |
| 2048 | 53 666.7 | **23 053.5** | 2.33x |
| 4096 | 103 368.1 | **42 753.7** | 2.42x |

At 2048 this is also **1.36x faster than the old UNSAFE `num_sms=24`, QPS=1 configuration**
(31 313.6 µs) that the §13 tables were measured on — so correctness and throughput moved the same
way. Dispatch iso at 4096 is 7 960 µs (7.38 GB/s RDMA send, 12.06 GB/s NVL recv).

**Rule for the playbook:** on this stack a fused/warp-specialized internode kernel needs **one
IBGDA QP per channel**. Channels sharing a QP is a correctness hazard whose onset is ~12 channels
per QP (6/QP and 4/QP measured clean), not merely a perf knob. Prefer raising `QPS_PER_PE` over
shrinking the grid — shrinking the grid costs 2.3x throughput to buy the same safety.

### 24.3 Validation (post-fix, reset + 4-GPU health gate before EVERY launch)

> ✅ **RESOLVED in §25:** the default-invocation hang was a *separate, deterministic* bug (padded
> `x` rows fed to combine by the cached dispatch), not a residual of the QP hazard. The default
> invocation now passes 9/9 at 2048/7168 and 4/4 at 4096/7168 — see §25.5.
>
> ⚠️ **Same overclaim applies to this table — see the boxed warning in §24.2.1.** These runs were
> all in reduced perf mode (`DEEP_EP_PERF_TOKENS` ⇒ `DEEP_EP_MIN=1`) with `DB_BATCH_SIZE=8`. The
> plain default full-matrix invocation at 2048/7168 hangs 2/2. Treat the numbers below as valid
> **only** for that reduced configuration.

| config | PASS | CORRUPT | HANG | N |
| --- | --- | --- | --- | --- |
| 2048 tok / hidden 7168 | **16** | 0 | 0 | 16 |
| 4096 tok / hidden 7168 | **10** | 0 | 0 | 10 |
| 64-config matrix (32 tok / hidden 1024) | 64/64 × 2 runs | 0 | 0 | 2 |

Against the pre-fix 60% failure rate, 16/16 clean has probability 0.4^16 ≈ 4e-7 under the null.
The 4096 leg previously failed 3/6 (dispatch-only bench) and is now 10/10.

### 24.4 Perf after the clamp — the §13 tables are SUPERSEDED

The §13 numbers were measured at `num_sms=24`, i.e. on the **unsafe, over-subscribed grid we no
longer use**. They are not merely provisional; they describe a configuration that fails 60% of the
time. Post-clamp (`num_sms` effective = 8), hidden 7168, `num_experts=8 topk=2`, 2 nodes × 2 ranks,
BF16, mean of the bench loop:

| tokens | round-trip (µs) | dispatch iso (µs) | combine iso (µs) |
| --- | --- | --- | --- |
| 32   | 1 626.9  | 953.9   | 1 166.2 |
| 512  | 14 934.7 | 2 520.5 | 12 606.6 |
| 2048 | 53 666.7 | 7 561.9 | 45 838.7 |
| 4096 | 103 368.1| 17 550.2| 85 719.8 |

Versus the (unsafe) `num_sms=24` measurements at the same sizes — 2048: round-trip 31 313.6,
dispatch 5 493.5, combine 25 837.1 — **the clamp costs ~1.7× on round-trip at 2048 and ~2.8× versus
the §13 fused table at 4096.** This is a real, substantial regression and is reported as such.
It is still the right trade: the faster number was wrong 60% of the time.

Post-clamp the fused path remains **faster than the legacy phase-split** at 4096 (§13 legacy
round-trip 414 178 µs vs 103 368 µs here, 4.0×), so the migration is still a net win — but the
headline "fused is 3.1× faster" from §13 was measured on an unshippable configuration.

### 24.5 Future work (recorded, NOT done)

The clamp ties the **algorithmic** channel count to the **hardware** co-residency limit, and that
coupling is what costs the throughput. The decoupled design: keep a co-resident number of
work-groups and have each one **loop over multiple channels sequentially** (persistent-kernel
style), recovering channel parallelism without ever depending on a non-resident peer. That is a real
design change with its own hazards (per-channel state must be reset between iterations; the SLM
window/lock/tail arrays are per-channel; the named-barrier participant counts are fixed per
work-group) and it should not be attempted until §24.2's open question is answered — a persistent
kernel that still needs N co-resident work-groups buys nothing.

Also still open (perf, paused): the 4.7× NVL-sender copy gap (combine ~1.9 GB/s vs dispatch
~8.9 GB/s for a structurally identical copy), §21.

### 24.6 New reproducer plumbing

- `DEEP_EP_FUSED_MAX_SMS` (C++ override of the derived limit) — added to the `run.sh`
  `_add_opt_genv` whitelist and to `/tmp/corrcamp.sh`.
- `_build_deepep_container.sh`: builds **inside** `deepep-v2-node0`. The host oneAPI is **2026.0**
  and has no torch; the container carries **2025.3** (the version the sims run) plus torch. The
  host-side `_build_deepep.sh` cannot build this repo any more. `ISHMEM_DIR` must be
  `/root/jiafuzha/ishmem_ibgda/build/_install` — the other in-container copy at
  `/root/jiafuzha/code-repo/ishmem_ibgda/build/_install` lacks `ishmemx_putmem_nbi_subgroup`,
  `ishmemx_fence_qp` and `ishmemx_long_atomic_add_qp` and fails the device compile.

## 25. THE REAL BUG BEHIND THE "SHIPPED DEFAULT HANGS" REPORT: padded `x` rows in combine

### 25.1 What was reported and what it really was

After §24.3 shipped, an independent run of the **plain default** invocation
(`tests/docker-2node-v2/run.sh` at 2048 tok / hidden 7168, no overrides) hung **2/2**, while every
validation run in §24.3 passed. The two configurations differed in exactly two variables — test
mode (`DEEP_EP_PERF_TOKENS` ⇒ `DEEP_EP_MIN=1`, one config, vs the full 64-config matrix) and
`ISHMEM_IBGDA_DB_BATCH_SIZE`. A 2×2 factorial settled it (2048 tok / hidden 7168, shipped default
kernel config, reset + 4-GPU health gate before EVERY launch, hang judged by **log-mtime stall**,
not rc):

| cell | test mode | `DB_BATCH` | PASS | HANG | N | wall |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | perf (`DEEP_EP_PERF_TOKENS`) | 8 | **4** | 0 | 4 | 80 s |
| A2 | perf | unset | **4** | 0 | 4 | 80 s |
| A3 | full 64-config matrix | 8 | 0 | **4** | 4 | ~465 s (stall-killed) |
| A4 | full 64-config matrix | unset | 0 | **4** | 4 | ~465–485 s (stall-killed) |

⇒ **`DB_BATCH_SIZE` carries none of the effect** (A1≡A2, A3≡A4) — the playbook's "inert" claim now
has direct experimental support. **The test mode carries 100% of it** (8/8 vs 0/8, Fisher p ≈ 1e-4).
So it was never a probabilistic scheduling/QP hazard at all: it is a **deterministic bug in a code
path the reduced mode never executes**.

### 25.2 Bisect: the trigger is the `without top-k` leg, specifically its CACHED dispatch

Single-cell selector added to the driver copy `tests/perf_combine_chunk.py`
(`DEEP_EP_SEL_X` ∈ {rand,x,rand8,x8}, `DEEP_EP_SEL_TOPK` ∈ {0,1}) so one matrix cell can be run in
the fast (80 s) reduced mode. All cells 2048 tok / hidden 7168, reset + gate per launch:

| cell | selector | PASS | HANG | N |
| --- | --- | --- | --- | --- |
| B1 (= A2) | `x`, top-k | 4 | 0 | 4 |
| B2 | `x`, **no** top-k | 0 | **3** | 3 |
| B3 | `x_pure_rand`, top-k | **3** | 0 | 3 |
| B4 | `x_pure_rand`, **no** top-k (the matrix's FIRST config) | 0 | **3** | 3 |
| C1 | `x`, no top-k, **cached dispatch skipped** (`DEEP_EP_SKIP_CACHED=1`) | **3** | 0 | 3 |
| C3 | `x`, top-k, **extra NON-cached second dispatch** (`DEEP_EP_EXTRA_DISPATCH=1`) | **3** | 0 | 3 |

`with_topk=False` carries it completely (`x_pure_rand` is irrelevant: B3 passes). Within that leg,
`test_internode.py` runs a **cached dispatch** (`buffer.dispatch(x, handle=handle)`) that no other
leg runs — C1 removes it and the hang vanishes; C3 shows a *second* dispatch is not the problem, so
it is **cached mode specifically**.

Per-rank phase instrumentation showed all 4 ranks *completing* the cached dispatch and then hanging
in **combine**, with the smoking gun in the shapes:

```
[PHASE rank=0] cached_dispatch end recv_x.shape=(4096, 7168)
[PHASE rank=0] pre_combine_elapsed=44.9s combine_x.shape=(4096, 7168) handle_recv=3386
```

### 25.3 ROOT CAUSE: `num_tokens = x.size(0)` is the PADDED row count, not the received-token count

- On XPU `test_internode.py` always passes `num_worst_tokens = num_tokens * num_topk`, so the
  non-cached dispatch returns a **padded** `recv_x` (4096 rows) and a **padded** `recv_src_meta`.
  The test truncates `recv_x` itself to `recv_gbl_rank_prefix_sum[-1]` (≈3386) — but the *handle*
  stays padded.
- `Buffer.internode_dispatch`'s cached branch takes `num_recv_tokens = recv_src_meta.size(0)` ⇒ the
  cached dispatch returns **4096** rows, and the test feeds that straight into `combine`.
- `internode_combine` sets `num_tokens = x.size(0)`, and the fused combine's **NVL sender** used it
  as the end of the LAST `(rank, channel)` block
  (`internode_combine_fused.inc`, `token_end_idx = (prefix_idx == num_channels*num_ranks-1) ?
  num_tokens : gbl_channel_prefix_matrix[prefix_idx+1]`, CUDA `internode.cu:1849-1855`).
- Result: the sender pushes ~710 **pad rows** the receiver never expects. The NVL queue never
  drains, the receiver never advances, both spin → **deterministic hang** (capped spins simply
  turn it into a silent drop instead).
- In CUDA this never fires because `x.size(0)` is always the exact received-token count
  (`num_worst_tokens` is not used on the cached→combine path there). The padding is an
  XPU-only accommodation, and the port inherited CUDA's assumption unchanged.

### 25.4 THE FIX (commit: "internode-normal: combine must use the real received-token count")

Pass the dispatch's `recv_gbl_rank_prefix_sum` (already in the handle, index 6) down to the combine
kernel and use `gbl_rank_prefix_sum[num_ranks-1]` — the **exact** received-token count, available
device-side with **no host sync** — instead of `num_tokens`:

- `deep_ep/buffer.py::internode_combine` — forward `gbl_rank_prefix_sum` (already unpacked).
- `csrc/xpu/deep_ep_xpu.cpp::internode_combine` — new `const torch::Tensor& gbl_rank_prefix_sum`.
- `csrc/xpu/xpu_runtime.hpp`, `csrc/xpu/internode.cpp::combine_nvl_rdma`,
  `internode_combine_fused.inc::launch_fused_combine` — new `const int* gbl_rank_prefix_sum`.
- NVL sender: `total_recv_tokens = gbl_rank_prefix_sum ? gbl_rank_prefix_sum[num_ranks-1]
  : num_tokens`, last block ends there, and both start/end are `min()`-clamped to it (defensive:
  a stale prefix matrix can no longer walk past the real data either).

**RULE for the playbook: in the fused kernels never use a tensor's row count as a logical token
count.** `x.size(0)` is an *allocation* size and may be padded (`num_worst_tokens`); the logical
count lives in the prefix-sum tensors. CUDA could conflate the two; XPU cannot.

### 25.5 Validation (post-fix, reset + 4-GPU health gate before EVERY launch, `k/N`)

| config | PASS | CORRUPT | HANG | N | pre-fix |
| --- | --- | --- | --- | --- | --- |
| B2/C2 cell (`x`, no top-k, 2048/7168, reduced mode) | **4** | 0 | 0 | 4 | 0/7 |
| **full 64-config matrix, 2048 tok / hidden 7168, DEFAULT invocation** | **9** | 0 | 0 | 9 | 0/8 |
| **full 64-config matrix, 4096 tok / hidden 7168, DEFAULT invocation** | **4** | 0 | 0 | 4 | n/a |

Every run reports `64 passed` lines (the full matrix, values checked). Fisher exact on the
full-matrix 2048 leg: 9/9 vs 0/8 ⇒ p ≈ 2e-5.

**INDEPENDENTLY RE-VERIFIED (separate operator, separate driver script, containers + `/dev/shm` +
IPC sockets cleaned before every run), using the byte-for-byte invocation that hung 2/2 pre-fix —
no SMS/QPS/`DB_BATCH`/`DEEP_EP_PERF_TOKENS` overrides of any kind:**

| config | result | wall |
| --- | --- | --- |
| full 64-config matrix, 2048 tok / hidden 7168 | **5/5 PASS**, `64 passed` lines each | 69–71 s |
| full 64-config matrix, 4096 tok / hidden 7168 | **2/2 PASS**, `64 passed` lines each | 72 s |

Combined with the table above: **20/20 post-fix vs 0/10 pre-fix.** This independent leg is the one
that matters — the §24.3 failure was precisely a fix validated only by the party that wrote it, in
a mode of its own choosing.

**Healthy full-matrix wall-clock at 2048/7168 (asked for and never previously measured): ~80 s**(≈40 s of that is the first-launch SPIR-V JIT of the fused kernels; the remaining 63 configs are
~0.5 s each). At 4096/7168 it is also ~80 s. The harness adds ~4 min/launch of reset + health gate
outside that. Anything that stalls >10 min at one config is a genuine hang, not slowness.

### 25.6 Methodological notes worth keeping

- **A reduced/"minimal" test mode is not a validation gate.** `DEEP_EP_MIN` skips the cached
  dispatch entirely, so the §24.3 "16/16" table could never have caught this. Always land the
  final `k/N` with the *as-shipped default invocation*.
- Hang-vs-slow must be judged by **log mtime advancing**, never by rc or by a timeout that is
  shorter than the workload.
- `run.sh` forwards env only via the explicit `_add_opt_genv` whitelist — a new debug var silently
  does nothing until it is added there (bitten 4×).

---

## 26. AUTHORITATIVE PERF RE-BASELINE (post-§25) — §13 / §24.4 are both SUPERSEDED

> **NOTE:** the combine column below is itself superseded by **§27** (`1a68632`), which recovers
> up to 3.58x of it. Keep this table as the *diagnostic baseline* — it is the measurement that
> identified the bottleneck — but quote §27 for current combine numbers.
> ⚠️ §27's win applies only when `num_nvl_ranks < 8`; on a full 8-GPU node it is a no-op and the
> combine numbers below still stand.

Measured 2026-08-22 on the shipped default (20 SMs / 10 channels / `QPS_PER_PE=16`) **after** the
padded-`x` fix (`5a7d53f`). Every earlier perf table in this document predates either the QP
default flip or that fix — and combine was previously pushing ~710 **pad rows** per block, so those
combine numbers measured inflated traffic. Use ONLY the table below.

Harness: `tests/docker-2node-v2`, perf mode, hidden 7168, `num_experts=8`, `topk=2`,
2 nodes × 2 ranks, BF16, reset + clean state between runs. Values are rank 0; ranks agree to <1%.

| tokens | dispatch iso (µs) | combine iso (µs) | **combine/dispatch** | disp nvl_recv | comb nvl_send | disp rdma_send | comb rdma_recv |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32   | 968   | 865    | **0.89** | 0.74 | 0.83 | 0.47 | 0.53 |
| 64   | 1 083 | 1 149  | 1.06 | 1.46 | 1.37 | 0.85 | 0.80 |
| 128  | 1 220 | 2 219  | 1.82 | 2.47 | 1.35 | 1.51 | 0.83 |
| 512  | 2 220 | 6 880  | 3.10 | 5.30 | 1.72 | 3.30 | 1.07 |
| 1024 | 3 050 | 12 850 | 4.21 | 7.95 | 1.89 | 4.81 | 1.14 |
| 2048 | 4 466 | 22 850 | 5.12 | 10.85 | 2.12 | 6.57 | 1.28 |
| 4096 | 7 968 | 41 900 | **5.26** | 12.31 | 2.34 | 7.37 | 1.40 |

(bandwidths in GB/s; round-trip at 4096 is 47 588 µs.)

### 26.1 What the shape of this curve rules OUT

**The framing "the NVL sender peer-IPC copy is slow" is WRONG and should be retired.** Combine is
degraded by the same ~5.3x on *both* legs at identical byte counts — `nvl_send` 2.34 vs dispatch
`nvl_recv` 12.31, **and** `rdma_recv` 1.40 vs dispatch `rdma_send` 7.37. A defective peer-IPC copy
loop cannot explain the RDMA leg. Look for a **common upstream serializer**.

**It is a scaling ceiling, not fixed overhead.** At 32 tokens combine is *faster* than dispatch
(0.89x); the ratio then climbs monotonically and saturates ~5.2x. Combine's bandwidth **plateaus
at ~2.3 GB/s NVL / ~1.4 GB/s RDMA** while dispatch keeps climbing. Something in combine stops
scaling once the pipeline is full — a per-token serialization, a fixed-width stage, or a queue
depth too shallow to hide latency.

Consistent with the existing telemetry at 4096: **`FwdWaitTail` 34.32% + 2nd barrier 29.52% = 64%**
of forwarder time is wait/barrier, while `FwdCopy` is only 25.93% and `FwdSend` 0.84%. The copy is
not where the time goes.

### 26.2 Copy-loop hypotheses already FALSIFIED — do not re-open without new evidence

- **Unroll factor.** `internode_combine_fused.inc:552` uses `UNROLLED_GROUP_COPY(2, ...)` while its
  own C9 comment claims x4 (stale comment, worth fixing for hygiene). Irrelevant to perf:
  **dispatch uses no unroll at all** — a plain lane-strided loop at
  `internode_dispatch_fused.inc:823-825` — and is 5x faster.
- **Per-token `sycl::group_barrier(sg)` inside the copy loop** (`:558`). **Dispatch has the
  identical barrier** at `:826`. Not the gap.
- **Cache hints.** Both use `ld_nc_global_v` / `st_na_global_v`. Closed earlier.

### 26.3 Where to look next — ✅ ANSWERED in §27
1. What `FwdWaitTail` actually waits on, and whether the producer is the true limiter.
   → **It is starvation, not the cause.** The NVL sender showed 97.8% copy / **0.0% credit wait**.
2. Which barrier the 29.5% is, and whether combine has a sync dispatch lacks.
   → Also downstream starvation behind the same sender.
3. Chunk/queue sizing limiting in-flight tokens (`num_max_nvl_chunked_send_tokens`,
   `num_max_rdma_chunked_*`) — a shallower effective pipeline plateaus exactly like this.
   → **FALSIFIED**: a queue-depth limit would show as sender *wait*, and wait was 0.0%.
4. Warp-role allocation: count warps doing useful work per role in each kernel.
   → **THIS WAS IT.** Dispatch partitions by *token* (70 live warps); combine partitioned by
   *destination* (20 live warps, 6 of 8 slots idle). See §27.
5. Diff against `csrc/cuda_kernels/internode.cu` combine (L1716) for a **lost pipelining/overlap
   stage** — same family as the §25 bug: a faithful-looking port that dropped a structural property.
   → Not a port-fidelity bug: CUDA has the same 1:1 mapping at `internode.cu:1849`.

## 27. COMBINE THROUGHPUT: the NVL-sender warp deficit (2026-08, FIXED, `1a68632`)

Question: combine was ~5.3x slower than dispatch at 4096 tokens and its bandwidth
plateaued (~2.3 GB/s NVL, ~1.4 GB/s RDMA) while dispatch kept scaling (12.3 / 7.4).

### 27.1 Attribution vs N (telemetry build, `-DDEEP_EP_COMBINE_TELEMETRY`)
Per-warp-role cycle attribution, hidden 7168, 2 nodes x 2 ranks, 20 SMs / 10 channels:

| ntok | combine snd copy% | snd wait% | snd cyc/tok | dispatch snd cyc/tok (70 warps) | dispatch fwd cyc/tok (20 warps) |
|---|---|---|---|---|---|
| 32 | 91.5 | 0.3 | 32k | 31k | 12.6k |
| 512 | 97.7 | 0.0 | 181k | 114k | 20.4k |
| 4096 | 97.8 | 0.0 | 158k | 77k | 23.6k |

All roles' `tot / live-warps` agree (~5.5e7 cyc at 4096) => the NVL sender runs for
the whole kernel and is **97.8% copy, 0.0% credit wait**.  It IS the critical path;
the forwarder's `FwdWaitTail` and the receiver's 80-90% wait are STARVATION, not the
cause.  This **falsifies the chunk/queue-sizing hypothesis** (a queue-depth limit
would show up as sender wait).

### 27.2 What is NOT the lever (all measured, all negative)
- **Unroll depth.** 2 -> 8 gave 158k -> 146k cyc/tok (-8%).  `UNROLLED_GROUP_COPY(2)`
  is fine; the C9 "x4" comment was stale (fixed in `8702f10`).
- **Compiler serialization by the `asm volatile` LSC helpers.** A plain C++ `dst4[k] =
  src[k]` loop (fully schedulable by IGC) was 226k cyc/tok, **43% WORSE**.  The LSC
  helpers are not the problem.
- **The peer-IPC (P2P MMIO) write.** Split the sender telemetry by
  `dst_nvl_rank == nvl_rank`: self 163k vs peer 155k cyc/tok in combine, 23.9k vs
  23.7k in the dispatch forwarder.  **Destination type is irrelevant in both kernels**
  -- this kills the "slow peer copy" story that survived earlier falsification rounds.
- Per-token `uc_store` of topk weights: not even executed in the perf bench
  (`num_topk == 0` there), so it cannot explain the perf numbers.

### 27.3 Root cause
Per-warp streaming throughput is pinned near **0.115 GB/s regardless of code form**.
The gap is therefore pure producer parallelism:
- dispatch RDMA sender: partitions **by token** across `DEEP_EP_FUSED_SENDER_WARPS = 7`
  warps/channel => **70 live warps**, and broadcasts one load to up to 2 destinations.
- combine NVL sender: partitioned **by destination** (`dst_nvl_rank = warp_id`, warps
  `>= num_nvl_ranks` return) => **20 live warps**, 6 of 8 slots idle at num_nvl_ranks=2.
3.5x fewer warps x ~1.5x less amortization ~= the observed 5.3x.
Per int4 element the two are identical (~87 cycles), which is the confirming detail.

### 27.4 Fix
`snd_split = min(kNumRDMARanks, NUM_MAX_NVL_PEERS / num_nvl_ranks)` sub-warps per
destination; sub-warp `s` owns RDMA lanes `l` with `l % snd_split == s`.  Safe with
**no new synchronisation and no extra named barrier** (the budget stays at 6 of 8)
because every queue resource is already per-RDMA-lane: token range
(`gbl_channel_prefix_matrix[(rdma, nvl, channel)]`), head, tail (`ch_tail + lane_id`)
and slot region (`current_rdma_idx * per_rdma + ...`).  Override:
`DEEP_EP_COMBINE_SND_SPLIT`.

Isolated combine (us): 865/1149/2219/6880/12850/22850/41900 ->
875/988/1688/2994/4801/6481/11713 for tokens 32/64/128/512/1024/2048/4096
(**3.58x at 4096**); combine/dispatch ratio 5.26 -> 1.44.
Gate: full 64-config matrix, no `DEEP_EP_PERF_TOKENS`, reset + 4-GPU health gate per
launch: **8/8 @2048/7168 and 3/3 @4096/7168**, 64 `passed` lines each.

**Caveat / future work:** this is a **no-op on a full 8-GPU node** (`snd_split == 1`
when `num_nvl_ranks == 8`); it recovers slots that are only idle when
`num_nvl_ranks < 8`.  Scaling combine's producer stage at N=8 needs token-range
splitting WITHIN one (dst, rdma) queue, which does require cooperative slot claiming
-- not attempted.  Note the same 1:1 mapping exists in CUDA `internode.cu:1849`, so
this is a small-NVL-deployment win rather than a port-fidelity bug.

### 27.5 Build trap (cost one full build+run cycle)
setuptools dependency-checks only the `.cpp` files, **not** the `.inc` files they
`#include`.  A pure-`.inc` edit is silently NOT recompiled and you get a stale `.so`
that looks freshly built.  `_build_deepep_container.sh` now `touch`es
`csrc/xpu/*.cpp` first; verify a new build with
`strings deep_ep_cpp*.so | grep <new-format-string>`.
Also: `dev_clock()` (`__spirv_ReadClockKHR`) is not volatile, so IGC may sink/hoist
the reads -- one telemetry build produced negative cycle deltas.  Cross-check any
suspicious attribution against the `[PERF ...]` wall-clock numbers.
