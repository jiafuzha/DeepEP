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

## STATUS (2026-08-20): BLOCKED on BMG + iSHMEM/IBGDA — do NOT attempt fusion yet

NamedBarrier works **standalone** on BMG (a 3-of-4 sub-group subset barrier was validated),
but it **cannot be used in any kernel that transitively calls an iSHMEM IBGDA device
function** on the current toolchain. This blocks the whole "re-fuse the phase-split
internode kernels into CUDA-parity warp-specialized kernels" effort.

### Symptom

The AOT device link SUCCEEDS (`Build succeeded for : pvc / bmg / ...`), but the kernel
fails at *runtime* module finalization:

```
error: parsing vISA inline assembly failed:
Found a total of 1 errors in vISA input.
error: backend compiler failed build.
```

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
1987:.function "..._Z27ishmemi_ibgda_bnxt_fill_msn..._70"
2155:.function "..._Z32ishmemi_ibgda_bnxt_claim_sq_slot..._69"
2363:.function "..._Z36ishmemi_ibgda_device_can_bnxt_direct..._15"
```

Once `named_barrier_init` is present anywhere in the kernel, IGC stamps
`.kernel_attr NBarrierCnt=N` on the kernel body **and again on every outlined vISA
stack-call `.function`** that IGC emitted for the large iSHMEM IBGDA bnxt helpers. vISA
rejects the duplicate attribute on the enclosing CISA routine.

### `ISHMEMI_IBGDA_BNXT_NOINLINE=OFF` does NOT fix this

The harness iSHMEM at `/root/jiafuzha/ishmem_ibgda` IS built with
`-DISHMEMI_IBGDA_BNXT_NOINLINE=OFF` (see its `_build_ishmem.sh`), i.e. the helpers carry no
`__attribute__((noinline))`. IGC **still** outlines them as vISA stack calls, because they
are far too large for its inliner (`warning: Stack call has been detected` at link time).
The `inline` keyword is only a hint; the CMake flag cannot force IGC's hand.

### Workarounds tried — none viable

| Attempt | Result |
|---|---|
| `-DISHMEMI_IBGDA_BNXT_NOINLINE=OFF` (already the default in `_build_ishmem.sh`) | Still outlined → still fails |
| `IGC_FunctionControl=0` (default) | Fails (`NBarrierCnt` duplicate) |
| `IGC_FunctionControl=4` (force-inline everything) | Gets past vISA, then `Abort was called at 528 line in .../memory_manager.cpp` — private-memory explosion from inlining the huge WQE emitters |
| Whole-WG-participant NamedBarrier in `FaithfulDispatchRdmaSendKernel` (real DeepEP module) | Builds, then `parsing vISA inline assembly failed` at runtime → `===== FAIL tests/test_internode.py =====` (reverted) |

### A/B evidence

`csrc/xpu/tools/test_nbarrier_ishmem_repro.cpp`, same source, same link, JIT (`spir64`):

- NamedBarrier block compiled **out** (`#if 0`) → `[pe 0] out=42,43 (expect 42,43)` ✅
- NamedBarrier block compiled **in** → `parsing vISA inline assembly failed` ❌

### Consequence for the internode NORMAL fusion

Until this IGC/iSHMEM interaction is fixed upstream, the `csrc/xpu/internode.cpp`
phase-split kernels **must stay phase-split**. Every kernel in that pipeline calls
`ishmemx_putmem_nbi_subgroup` / `ishmemx_fence_qp` / `ishmemx_long_atomic_add_qp` /
`ishmemx_barrier_all_work_group`, so all of them are in the blocked set.

**Separately**, note that the current XPU internode NORMAL phase boundaries are
**grid-scope or cross-PE** (whole-grid NVL `nvl_barrier`, `ishmemx_barrier_all_work_group`,
and different grid shapes per phase: 1 WG vs `num_use_channels*num_qp_ch` WGs vs
`num_rdma_ranks*num_qp_ch` WGs). NamedBarrier is a *within-work-group* sub-group-subset
barrier and cannot replace any of them even once the toolchain issue is resolved. A true
CUDA-parity re-fusion would additionally require re-introducing the CUDA sliding-window
credit transport (`rdma_send_channel_{lock,tail,window}`, `forward_channel_{head,retired}`),
which the XPU port deliberately replaced with an AMO-flag transport, and which depends on
the forwarder **pull**-reading peer NVL buffers over IPC — unstable on BMG.

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
Today: `error: parsing vISA inline assembly failed`.

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
