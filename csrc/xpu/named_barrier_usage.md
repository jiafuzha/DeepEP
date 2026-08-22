> # ✅ RESOLVED (2026-08-22) — cause is per-QP contention; fixed with ONE IBGDA QP PER CHANNEL
>
> **History (keep — this is a first-class rule for every fused/warp-specialized port).** The fused,
> NamedBarrier warp-specialized internode-normal kernels documented here were the default at HEAD and
> intermittently HUNG and SILENTLY DROPPED TOKENS (fused 6/10 fail vs legacy `b231f1c` 0/10 at
> 2048 tok / hidden 7168, Fisher p = 0.0054; one launch lost exactly 65 whole tokens).
>
> **RULE (corrected 2026-08-22): give every channel its OWN IBGDA QP.** The first fix clamped the
> grid for work-group co-residency, and it worked — but for the wrong reason. Three experiments
> (playbook §24.2.1) showed co-residency is NOT the mechanism: 24-27 work-groups at 4 channels is
> 13/13 clean even when residency is denied for 3x `kFusedSpinCap`; a standalone probe co-schedules
> 128/128 work-groups of 512 WI; and at the SAME grid of 24, `QPS_PER_PE=16` is 6/6 clean while the
> old `QPS_PER_PE=1` is 2/6 (Fisher p = 0.030). With one QP per PE every channel's RDMA sender drives
> the same send queue; the failure appears at ~12 channels/QP (6/QP and 4/QP are clean). The grid
> clamp only helped because fewer channels means fewer sharers.
>
> Fixed by `deep_ep/buffer.py` (normal-internode branch now sets `ISHMEM_IBGDA_QPS_PER_PE` =
> `clamp_pow2(num_qps_per_rank)` = 16, `setdefault`; the LL branch keeps C=1) plus a QP-aware
> `internode::fused_max_coresident_sms()`. Shipped default: num_sms=20 / 10 channels / 16 QPs —
> **16/16 at 2048/7168, 8/8 at 4096/7168, 64/64 matrix**, and 2.33x FASTER at 2048 tokens than the
> grid clamp it replaces (23.1 ms vs 53.7 ms round-trip).
>
> These kernels still split every channel across two work-groups (`channel_id = sm_id/2`,
> `is_forwarder = sm_id%2`) that spin on each other's queue counters, and `kFusedSpinCap` still
> `break`s rather than hangs — so a starved producer can still surface as silently dropped tokens.
> `csrc/xpu/internode_ll.cpp::ll_put_wgs()` documents the co-residency rule for the LL put grid; it
> is a real constraint there, it just was not this bug.
>
> The 64-config matrix at its default size (32 tokens / hidden 1024) is **blind** to this class of bug.
> Validate only at 2048+ tokens / hidden 7168 and report `k/N`.
>
> Full evidence: `.github/agents/cuda-to-xpu-internode-normal-migration.agent.md` §21-§24.

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

- **No `bfloat16` conversion operators in a NamedBarrier kernel.**
  `sycl::ext::oneapi::bfloat16`'s conversion operators lower to the *external* devicelib symbols
  `__devicelib_ConvertBF16ToFINTEL` / `__devicelib_ConvertFToBF16INTEL`. These are undefined in
  the module, so IGC materializes them as vISA **stack calls** — the exact
  `NBarrierCnt`-on-an-outlined-`.function` situation that `ISHMEMI_IBGDA_BNXT_NOINLINE=OFF`
  cures for iSHMEM. Symptom: `error: IGC: internal compiler error` at JIT time, accompanied by
  `warning: ... Stack call has been detected` for the same kernel.
  **Fix:** do the conversion inline with bit manipulation (RNE rounding). Generalized rule: *any*
  external/outlined device function in a NamedBarrier kernel will trigger this, not just iSHMEM's.
- **Named-barrier count is limited by kernel COMPLEXITY, not by a hard architectural cap.**
  Measured on BMG: a *trivial* standalone kernel compiles and runs with **10** named barriers,
  so there is no architectural limit at ~5-9. But inside the real fused internode combine
  kernel, **8 named barriers JIT fine (full matrix passes) and 9 raises
  `error: IGC: internal compiler error`** (dies between `push_analysis` and `codegen` in the
  shader dumps). Budget barriers conservatively in large kernels and re-measure by bisection
  after significant kernel growth — unrelated additions can push you over the cliff.
  This is what caps CUDA's per-destination `bar.sync (dst_rdma_rank + 2), ...`
  (`internode.cu:1966`): combine declares `2 + num_rdma_ranks` barriers, so
  `num_rdma_ranks <= 4` is supported and `R=8` (10 barriers) ICEs.
- **SLM subset barriers deadlock** — only hardware barriers (NamedBarrier, group_barrier)
  provide guaranteed cross-sub-group forward progress. Do NOT hand-roll SLM arrival-counter
  spin barriers.
- **No per-lane divergent spin/lock patterns.** A SYCL sub-group is a single lock-stepped EU
  thread, so CUDA idioms that rely on Volta+ independent thread scheduling — e.g.
  `acquire_lock(rdma_send_channel_lock + lane_id)` at `internode.cu`, where each lane holds a
  *different* lock and progress requires lanes to advance independently — deadlock on Xe.
  Restructure: serialize the critical section, or have one lane evaluate a sub-group-wide
  condition and broadcast it.
- **`num_warp_groups == 1` is forced** on BMG because `num_device_sms == 160 >= num_experts`.
  For `num_experts <= 160` there is only 1 warp group — the warp-specialization is
  `caster_warps + counter_warp` within a SINGLE warp group, not across multiple groups.
- **`num_warps_per_group == 32`** (all 32 sub-groups in one WG) with
  `caster_warps = num_warps - 1` and `counter_warp = 1`.

## Debugging workflow

- **Fast offline ICE repro (~3 min, no 2-node harness).** The runtime JIT-compiles from embedded
  SPIR-V — the `-device pvc,bmg,...` string is handed to the runtime, it is not a finished AOT
  image. So dump with `IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=$PWD/igcdump`, then replay:

  ```bash
  ocloc compile -file <dump>.spv -spirv_input -device bmg
  ```

- **Build trap: `setup.py` does NOT track `.inc` files as dependencies.** Editing only a `.inc`
  relinks the `.so` from *stale* objects and silently tests the OLD code. Always
  `touch csrc/xpu/internode.cpp` before rebuilding.

## Files and locations

| File | Purpose |
|---|---|
| `csrc/xpu/xpu_kernels.hpp` | `NamedBarrier` class + global SPIR-V declarations |
| `/root/jiafuzha/ishmem_ibgda/src/ishmem.h` | `ISHMEM_DEVICE_ATTRIBUTES` (no per-fn annotation available) |
| `/root/jiafuzha/ishmem_ibgda/src/ibgda_device_impl.h` | IBGDA inline device functions (bnxt `noinline` markers) |
| `/root/jiafuzha/ishmem_ibgda/src/rma_impl.h` | RMA inline templates (put/get) |
| `csrc/xpu/internode_ll.cpp` | Current LL kernel impl with NamedBarrier + root_group |
| `csrc/cuda_kernels/internode_ll.cu` | CUDA reference with `bar.sync` warp specialization |
