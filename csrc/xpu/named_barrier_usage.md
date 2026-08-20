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

## STATUS (2026-08-20): RESOLVED — NamedBarrier + iSHMEM works, but the iSHMEM archive MUST be built with `ISHMEMI_IBGDA_BNXT_NOINLINE=OFF`

NamedBarrier works standalone on BMG **and** in kernels that transitively call iSHMEM IBGDA
device functions — *provided* `libishmem.a` was built with
`-DISHMEMI_IBGDA_BNXT_NOINLINE=OFF`. An archive built with the CMake **default (`ON`)** makes
any such kernel fail at runtime module finalization.

> An earlier revision of this document declared this permanently blocked. That conclusion was
> wrong: it was measured against an archive whose `CMakeCache.txt` said
> `ISHMEMI_IBGDA_BNXT_NOINLINE:BOOL=ON`, i.e. the flag had never actually been applied. Always
> verify the cache, not the build script.

### Symptom (when the archive is built with NOINLINE=ON)

The AOT device link SUCCEEDS (`Build succeeded for : pvc / bmg / ...`), but the kernel
fails at *runtime* module finalization:

```
error: parsing vISA inline assembly failed:
Found a total of 1 errors in vISA input.
error: backend compiler failed build.
```

A `warning: Stack call has been detected` at link time is the early tell.

### Root cause (exact)

Dump with `IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=<dir>` and read `*.errors.txt`:

```
/-------------------- !!!KERNEL HEADER ERRORS FOUND!!! --------------------\
Error in CISA routine with name: _ZTS11ReproKernel
              Error Message: More than 1 kernel attribute defined NBarrierCnt
\--------------------------------------------------------------------------/
```

`grep -n '^\.function\|^\.kernel_attr NBarrierCnt' *_entry_0001.inline.visaasm`:

```
4:.kernel  "_ZTS11ReproKernel"
41:.kernel_attr NBarrierCnt=2                    <-- from NamedBarriersResolution
1604:.function "..._Z38ishmemi_ibgda_device_emit_bnxt_wqe_nbi..._37"
1606:.kernel_attr NBarrierCnt=2                  <-- duplicate
1881:.function "..._Z34ishmemi_ibgda_bnxt_release_sq_lock..._72"
1883:.kernel_attr NBarrierCnt=2                  <-- duplicate
1898:.function "..._Z32ishmemi_ibgda_bnxt_ring_doorbell..._71"
1900:.kernel_attr NBarrierCnt=2                  <-- duplicate
```

`__attribute__((noinline))` on the bnxt helpers forces IGC to outline them as vISA stack-call
`.function`s. Once `named_barrier_init` is present anywhere in the kernel, IGC stamps
`.kernel_attr NBarrierCnt=N` on the kernel body **and again on every outlined `.function`**,
and vISA rejects the duplicate attribute on the enclosing CISA routine. Removing the
attribute lets IGC inline them, so no outlined routines exist to be double-stamped.

### The fix, and why DeepEP cannot apply it itself

Build iSHMEM with the helpers inlinable:

```bash
cd /root/jiafuzha/ishmem_ibgda && bash _build_ishmem.sh   # passes -DISHMEMI_IBGDA_BNXT_NOINLINE=OFF
# then, in DeepEP:
rm -rf build/ishmem-sycl-dlink && python3 setup.py build_ext --inplace
```

Verify it actually took effect — do **not** trust the script alone:

```bash
grep ISHMEMI_IBGDA_BNXT_NOINLINE /root/jiafuzha/ishmem_ibgda/build/CMakeCache.txt
# want: ISHMEMI_IBGDA_BNXT_NOINLINE:BOOL=OFF
```

Passing `-DISHMEMI_IBGDA_BNXT_NOINLINE=` in **DeepEP's** compile flags does nothing:
`src/CMakeLists.txt` attaches the macro as `target_compile_definitions(ishmem-objects
PRIVATE ...)`, and `src/ibgda_device_impl.h` is not part of the installed include tree
(`$ISHMEM_DIR/include` ships only `ishmem.h`, `ishmemx.h`, `ishmem/*.h`). DeepEP's own
translation units never see the bnxt code — it arrives as pre-compiled device bitcode in
`libishmem.a`, so the attribute is fixed at iSHMEM build time. `setup.py` therefore only
*detects* the bad configuration (`check_ishmem_bnxt_inlinable`) and prints a loud warning.

### Other workarounds (not needed once NOINLINE=OFF, recorded for reference)

| Attempt | Result |
|---|---|
| `IGC_FunctionControl=0` (default) | Fails (`NBarrierCnt` duplicate) |
| `IGC_FunctionControl=4` (force-inline everything) | Gets past vISA, then `Abort was called at 528 line in .../memory_manager.cpp` — private-memory explosion |
| `IGC_SelectiveFunctionControl=1` | Obsolete; not a fix, do not set |

### A/B evidence

`csrc/xpu/tools/test_nbarrier_ishmem_repro.cpp`, same source, same link, JIT (`spir64`):

| iSHMEM archive | Result |
|---|---|
| `NOINLINE=ON` | `parsing vISA inline assembly failed` ❌ |
| `NOINLINE=OFF` | `[pe 0] out=42,43` / `[pe 1] out=42,43` ✅ (no stack-call warning) |

With `NOINLINE=OFF`, DeepEP rebuilds clean and `tests/docker-2node-v2` reports
`===== PASS tests/test_internode.py =====`.

### Relationship to the phase-split XPU implementation

The current phase boundaries in `csrc/xpu/internode.cpp` (whole-grid NVL `nvl_barrier`,
`ishmemx_barrier_all_work_group`, and differing grid shapes per phase) are an **artifact of
the NamedBarrier workaround**, not a requirement of the algorithm.

> **Retraction (2026-08-20).** An earlier revision of this document claimed the XPU internode
> NORMAL phase boundaries were *inherently* grid-scope/cross-PE and therefore that NamedBarrier
> was structurally the wrong tool. **That was wrong.** It reasoned from the workaround's own
> structure back to the algorithm — circular. Verified fact: `csrc/cuda_kernels/internode.cu`
> contains **no** `cooperative_groups`, **no** `this_grid()`, and **no** grid sync of any kind.
> `dispatch` (line 447) and `combine` (line 1716) are each **one fused kernel** whose every
> internal synchronization is an intra-block `barrier.sync`/`bar.sync`. Only `notify_dispatch`
> and `cached_notify` are separate kernels in CUDA, and they are separate in the XPU port too.
> With the toolchain issue resolved, re-fusing to a single warp-specialized kernel per direction
> is the correct target.

## Verification

Repro source is checked in at `csrc/xpu/tools/test_nbarrier_ishmem_repro.cpp`. Build and run
it inside `deepep-v2-node0`:

```bash
source /opt/intel/oneapi/setvars.sh --force
export ISHMEM_DIR=/root/jiafuzha/ishmem_ibgda/build/_install
icpx -fsycl -fsycl-rdc -fsycl-targets=spir64 -std=c++20 -O3 \
     -I$ISHMEM_DIR/include csrc/xpu/tools/test_nbarrier_ishmem_repro.cpp \
     -L$ISHMEM_DIR/lib -lishmem -lze_loader -lhwloc -libverbs -lmlx5 -lpthread -lmpi \
     -o repro
# the two "Undefined function _Z18named_barrier_initi / _Z24work_group_named_barrier..."
# link warnings are EXPECTED (SPIR-V builtins resolved by IGC, not by the LLVM linker).

export LD_LIBRARY_PATH=$ISHMEM_DIR/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhwloc.so
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 ISHMEM_ENABLE_GPU_IPC=0
export ISHMEM_IB_ENABLE_IBGDA=1 ISHMEM_IBGDA_DIRECT_DOORBELL=1
export ISHMEM_IBGDA_BAR_BACKEND=igub ISHMEM_IBGDA_QPS_PER_PE=1
mpirun -n 2 bash -c 'export ZE_AFFINITY_MASK=$MPI_LOCALRANKID;
                     export ISHMEM_IBGDA_NIC=mlx5_$MPI_LOCALRANKID; ./repro'
```

Expected once the toolchain is fixed: `[pe 0] out=42,43` / `[pe 1] out=42,43`.
Today (with `ISHMEMI_IBGDA_BNXT_NOINLINE=OFF`): passes as expected. With an archive built
`NOINLINE=ON`: `error: parsing vISA inline assembly failed`.

Add `IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=$PWD/igcdump` to capture the
`*.errors.txt` / `*.inline.visaasm` evidence above.

### Original sketch (for reference)

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
