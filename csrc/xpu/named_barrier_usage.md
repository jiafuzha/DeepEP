# NamedBarrier usage in XPU

## The Goal: Fused kernels with sub-group-subset barriers

The CUDA low-latency kernels (`csrc/cuda_kernels/internode_ll.cu`) use **warp-group-scoped
named barriers** (`bar.sync warp_group_id+N, num_warps_per_group*32`) to synchronize
only a subset of warps within a thread block while other warps run ahead. This enables:
- **Fused dispatch**: cast+send WARP + recv+flag WARP in a SINGLE work-group, separated by a
  subset barrier (no kernel split, no grid barrier).
- **Fused combine**: reduce-send WARP + recv-notify WARP in a SINGLE work-group.
- **`num_warp_groups > 1` scaling**: multiple expert warp-groups per work-group, each with their
  own subset barrier.

On Intel Xe/BMG, the only SYCL primitive capable of a **sub-group-subset barrier** is the
SPIR-V `NamedBarrier` (`cl_khr_subgroup_named_barrier`), declared at global scope in
`csrc/xpu/xpu_kernels.hpp`:

```cpp
// GLOBAL scope — MUST NOT be inside `namespace deep_ep`
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
struct __namedBarrier;
extern SYCL_EXTERNAL __namedBarrier __attribute__((opencl_local)) *
named_barrier_init(int count);
extern SYCL_EXTERNAL void work_group_named_barrier(
    __namedBarrier __attribute__((opencl_local)) *, unsigned int);
#endif

class NamedBarrier {
    ::__namedBarrier __attribute__((opencl_local)) * handle_ = nullptr;
public:
    void init(int num_subgroups);  // participant count (sub-groups, NOT work-items)
    void sync(unsigned int flags = kNamedBarrierGlobalFence);  // subset barrier
};
```

## Verification

To verify the NamedBarrier works **before** integrating NamedBarriers into the real kernel:

```c++
// test_nbarrier_ishmem_repro.cpp — mimics the DeepEP + iSHMEM coexistence
#include <sycl/sycl.hpp>
#include <ishmem.h>

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
struct __namedBarrier;
extern SYCL_EXTERNAL __namedBarrier __attribute__((opencl_local)) *
named_barrier_init(int count);
extern SYCL_EXTERNAL void work_group_named_barrier(
    __namedBarrier __attribute__((opencl_local)) *, unsigned int);
#endif

int main() {
    // ... MPI/ishmem init ...
    q.submit([&](sycl::handler& h) {
        h.single_task([=] {
            ishmem_putmem_nbi(/*...*/);  // calls non-inlined iSHMEM function
            ::__namedBarrier *bar = ::named_barrier_init(2);  // named barrier
            ::work_group_named_barrier(bar, 0x2);
        });
    });
}
```

```bash
# Build with AOT (spir64) and RDC to reproduce at JIT time:
icpx -fsycl -fsycl-rdc -fsycl-targets=spir64 \
    $(pkg-config --cflags ishmem) repro.cpp $(pkg-config --libs ishmem) \
    -o repro

# Run
./repro

```

## Why this is critical for the fused-kernel port

The current XPU LL kernels are **phase-split** (separate dispatch-send and dispatch-recv
kernels, separate combine-reduce and combine-notify kernels) with a host-side barrier
between them. This is a workaround for the NamedBarrier incompatibility.

Fusing them into single-work-group kernels with warp specialization (matching CUDA) requires:
1. **Sub-group-subset barriers** (`NamedBarrier`) to sync casters+counters within a WG while
   non-participating warps run ahead
2. **Coexistence with iSHMEM**


### CUDA → XPU barrier mapping for fused kernels

| CUDA | XPU/SYCL | Notes |
|---|---|---|
| `bar.sync <id>, <count>` | `NamedBarrier::init(count/32)` + `.sync(flags)` | Sub-group-subset barrier |
| `__syncwarp()` | `sycl::group_barrier(sub_group)` | Intra-sub-group (32 WI) |
| `__syncthreads()` | `sycl::group_barrier(work_group)` | Whole-WG |
| `cg::this_grid().sync()` | `GridBarrier` (from `xpu_kernels.hpp`) | Grid-wide (may also conflict; use with Solution 2) |
| `__threadfence_block()` | `sycl::atomic_fence(..., device)` | Device-scope memory fence |

### Known limitations on BMG

- **SLM subset barriers deadlock** — only hardware barriers (NamedBarrier, group_barrier)
  provide guaranteed cross-sub-group forward progress. Do NOT hand-roll SLM arrival-counter
  spin barriers.
- **`num_warp_groups == 1` is forced** on BMG because `num_device_sms == 160 >= num_experts`.
  For `num_experts <= 160` there is only 1 warp group — the warp-specialization is
  `caster_warps + counter_warp` within a SINGLE warp group, not across multiple groups.
- **`num_warps_per_group == 32`** (all 32 sub-groups in one WG) with
  `caster_warps = num_warps - 1` and `counter_warp = 1`.

## Files and locations

| File | Purpose |
|---|---|
| `csrc/xpu/xpu_kernels.hpp` | `NamedBarrier` class + global SPIR-V declarations |
| `/root/jiafuzha/code-repo/ishmem_ibgda/src/ishmem.h` | `ISHMEM_DEVICE_ATTRIBUTES` (no per-fn annotation available) |
| `/root/jiafuzha/code-repo/ishmem_ibgda/src/ibgda_device_impl.h` | IBGDA inline device functions (bnxt `noinline` markers) |
| `/root/jiafuzha/code-repo/ishmem_ibgda/src/rma_impl.h` | RMA inline templates (put/get) |
| `csrc/xpu/internode_ll.cpp` | Current LL kernel impl with NamedBarrier + root_group |
| `csrc/cuda_kernels/internode_ll.cu` | CUDA reference with `bar.sync` warp specialization |
