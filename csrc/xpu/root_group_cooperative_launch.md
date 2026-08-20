# SYCL Root Group (Cooperative Launch) for Fused LL Kernels

> **Reference:** `sycl_ext_oneapi_root_group` experimental extension
> (`https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_root_group.asciidoc`)
>
> **Purpose:** Capture key information about the SYCL `root_group` cooperative-kernel-launch
> primitive, its mapping to CUDA's `cudaLaunchCooperativeKernel` + `cg::this_grid().sync()`,
> and how it interacts with the DeepEP internode low-latency fused-kernel design
> (sub-group-subset `NamedBarrier` + iSHMEM coexistence).

---

## 1. What SYCL root_group provides

The `sycl_ext_oneapi_root_group` extension introduces a **root group**: a new SYCL group type
representing **all work-items across all work-groups** executing a given kernel. It exposes:

- `sycl::nd_item::ext_oneapi_get_root_group()` → a `root_group` handle
- `root_group::get_group_linear_id()` / `get_local_linear_id()` / `get_group_range()` /
  `get_local_range()` — global linear IDs and ranges
- **`sycl::group_barrier(root_group)`** — a **device-wide barrier** spanning ALL work-groups
  of a single kernel launch (hardware-managed, no global-memory spin-loop needed)

This is the direct SYCL equivalent of CUDA's `cg::this_grid().sync()`.

### Key difference from CUDA cooperative launch

| Feature | CUDA | SYCL root_group |
|---|---|---|
| Barrier primitive | `cg::this_grid().sync()` | `sycl::group_barrier(root_group)` |
| Launch API | `cudaLaunchCooperativeKernel` / `cudaLaunchKernelEx` with `cudaLaunchAttributeCooperative` | `sycl::ext::oneapi::experimental::nd_launch(queue, launch_config, kernel)` |
| Launch property | `cudaLaunchAttributeCooperative = 1` | `sycl::ext::oneapi::experimental::use_root_sync` property |
| Grid size query | `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | `get_kernel_info<Name, info::kernel::max_num_work_groups_sync>(queue, wg_size, props, local_mem)` |
| Occupancy guarantee | Full SM occupancy of launched blocks | Limited WGs ≤ device concurrency limit (prevents deadlock) |
| WG-local mem query | `cudaFuncSetAttribute` / launch config | Passed as last arg to `get_kernel_info` |

### API surface (from the extension spec)

```cpp
namespace syclex = sycl::ext::oneapi::experimental;

// 1. Query the MAX number of work-groups that can be launched cooperatively
syclex::properties props{syclex::use_root_sync};
auto maxWGs = syclex::get_kernel_info<MyKernel,
    syclex::info::kernel::max_num_work_groups_sync>(
    q, wg_size, props, local_mem_size);

// 2. Build an nd_range within that limit
auto ndr = sycl::nd_range<1>{maxWGs * wg_size, wg_size};

// 3. Construct a launch_config with the use_root_sync property
syclex::launch_config cfg{ndr, props};

// 4. Launch (replaces q.submit + h.parallel_for)
sycl::nd_range<1> effective_ndr = syclex::nd_launch(q, cfg, MyKernel{args...});
q.wait();
```

### `nd_launch` return value

`nd_launch` returns the **effective nd_range** that was actually launched (may differ from the
requested one if the runtime adjusted it for co-residency). The kernel receives this as
`item.get_nd_range()`.

---

## 2. CUDA cooperative launch in DeepEP's internode_ll.cu

The CUDA LL kernels use cooperative launches via the `SETUP_LAUNCH_CONFIG` / `LAUNCH_KERNEL`
macros in `csrc/cuda_kernels/launch.cuh`:

```cpp
// launch.cuh
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream)
    cudaLaunchConfig_t cfg = {(num_sms), (num_threads), 0, stream, nullptr, 0};
    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeCooperative;
    attr[0].val.cooperative = 1;                    // <-- cooperative launch
    attr[1].id = cudaLaunchAttributeClusterDimension;
    attr[1].val.clusterDim.x = (num_sms % 2 == 0 ? 2 : 1);  // SM cluster hint
    ...

#define LAUNCH_KERNEL(config, kernel, ...)
    CUDA_CHECK(cudaLaunchKernelEx(config, kernel, ##__VA_ARGS__))
```

In the LL dispatch kernel (`internode_ll.cu:552`), the launch uses `num_sms` work-groups with
`num_warps * 32` threads each. Inside the kernel, `cg::this_grid().sync()` (line 360, 977)
synchronizes all work-groups between the **send** and **recv** phases of the fused
dispatch/combine.

The CUDA kernels achieve further fusion through **warp-group-scoped barriers** (`bar.sync`):
within a single work-group, warp subsets synchronize independently while other warps run
ahead — this is the warp-specialization pattern that requires the `NamedBarrier` on XPU
(see `named_barrier_ishmem_conflict.md`).

---

## 3. Interaction with DeepEP's XPU fused-kernel design

### Current state: GridBarrier (software spin-loop)

The XPU LL kernels currently use the `GridBarrier` struct in `csrc/xpu/xpu_kernels.hpp`
(lines 285–366) — a **software global-memory counter + sense-reversing barrier** with UC
loads/stores. This is the de facto equivalent of `cg::this_grid().sync()` when no
hardware-managed grid barrier is available.

The `GridBarrier` requires:
- Two zero-initialized `uint32_t` scratch slots in global memory (`counter`, `sense`)
- Co-residency guarantee: total launched WGs ≤ max concurrent WGs (deadlock otherwise)
- A UC load spin-loop on the sense flag

### How root_group helps

With `root_group`, the `GridBarrier` software spin-loop can be replaced by a single
`sycl::group_barrier(root_group)` call. Benefits:

1. **Zero scratch memory**: No need for the global `counter`/`sense` buffer pair
2. **Hardware-managed**: The runtime/backend ensures all WGs arrive before releasing
   (likely using hardware barrier hardware on the GPU, not a spin-loop)
3. **No UC load spin**: Eliminates the UC-load polling overhead
4. **Safer launch**: `nd_launch` with `use_root_sync` validates the grid size against
   the device's concurrency limit at launch time — no user-managed co-residency check

### Critical caveat: root_group barrier scope is DEVICE, not SYSTEM

The `root_group` barrier synchronizes **only co-resident work-groups on the same device**.
It does NOT replace cross-PE synchronization (iSHMEM `ishmem_barrier_all`) between
different GPU ranks. The current design already separates these:

- **Cross-WG (on-device)**: `GridBarrier::arrive_and_wait()` (to become `group_barrier(root_group)`)
- **Cross-PE (inter-node)**: `ishmem_barrier_all()` (host-side, before kernel launch)

This distinction matches CUDA: `cg::this_grid().sync()` is device-local; cross-node sync
is handled separately (NVSHMEM `barrier_all`).

---

## 4. root_group + NamedBarrier coexistence

The XPU fused-kernel design requires TWO synchronization primitives in the SAME kernel:

| Primitive | Scope | Purpose |
|---|---|---|
| `NamedBarrier::sync()` | Sub-group-subset within a WG | Warp-specialization barrier (caster ↔ counter warps) |
| `group_barrier(root_group)` | All WGs of the kernel | Grid sync between send and recv phases |

The `NamedBarrier` / iSHMEM `NBarrierCnt` conflict (see `named_barrier_ishmem_conflict.md`)
arises from **IGC's incorrect stamping of `NBarrierCnt=1` on non-inlined iSHMEM subroutines**
in RDC-linked modules. The `root_group` barrier runs through a DIFFERENT mechanism (SPIR-V
`ControlBarrier` with `Device` scope, or possibly a backend-native cooperative-launch barrier)
and does NOT use the `NamedBarrier` SPIR-V instructions — so it does NOT contribute additional
`NBarrierCnt` attributes and does NOT exacerbate the conflict.

However, because `NamedBarrier` IS still needed for warp specialization, the
`NamedBarrier` + iSHMEM coexistence problem remains a **hard blocker** for any fused kernel
that uses BOTH `NamedBarrier` AND iSHMEM on this stack.

> **STATUS UPDATE (2026-08-20) — the old workaround is OBSOLETE.**
> `IGC_SelectiveFunctionControl=1` is **no longer the fix and must NOT be set**.
> Nor does building iSHMEM with `-DISHMEMI_IBGDA_BNXT_NOINLINE=OFF` resolve it: that flag
> only drops `__attribute__((noinline))`, and `inline` is a hint IGC ignores once the large
> bnxt IBGDA helpers exceed its inline budget (link emits `warning: Stack call has been
> detected`). IGC then stamps `.kernel_attr NBarrierCnt=N` on the kernel body **and** on every
> outlined vISA stack-call `.function`, and vISA rejects the module:
> `Error Message: More than 1 kernel attribute defined NBarrierCnt`, surfacing at runtime as
> `error: parsing vISA inline assembly failed`.
> See `csrc/xpu/named_barrier_usage.md` for the captured `*.inline.visaasm` evidence and the
> checked-in repro at `csrc/xpu/tools/test_nbarrier_ishmem_repro.cpp`.

---

## 5. Porting strategy

### Phase 1: Replace GridBarrier with root_group (without NamedBarrier)

For the current XPU LL kernels (which are split into separate dispatch-send / dispatch-recv
kernels, combine-reduce / combine-notify kernels with a host barrier), replace
`GridBarrier::arrive_and_wait()` with `sycl::group_barrier(root_group)`:

```cpp
// BEFORE (software grid barrier)
auto grid_barrier = GridBarrier(counter_ptr, sense_ptr, num_groups);
// ... send phase ...
grid_barrier.arrive_and_wait(item);
// ... recv phase ...

// AFTER (hardware grid barrier via root_group)
auto root = item.ext_oneapi_get_root_group();
// ... send phase ...
sycl::group_barrier(root);
// ... recv phase ...
```

Launch changes:
```cpp
// BEFORE
q.submit([&](sycl::handler& h) {
    h.parallel_for(ndr, kernel);
});

// AFTER
syclex::properties props{syclex::use_root_sync};
auto maxWGs = syclex::get_kernel_info<Kernel, syclex::info::kernel::max_num_work_groups_sync>(
    q, wg_size, props, 0);
syclex::launch_config cfg{sycl::nd_range<1>{maxWGs * wg_size, wg_size}, props};
syclex::nd_launch(q, cfg, kernel);
```

### Phase 2: Add NamedBarrier for warp specialization

Once root_group is in place, add `NamedBarrier` for the fused-kernel warp specialization
pattern (matching CUDA's `bar.sync`).

> **BLOCKED (2026-08-20).** This phase cannot currently land: `NamedBarrier` + iSHMEM in one
> kernel fails vISA finalization with `More than 1 kernel attribute defined NBarrierCnt`.
> `IGC_SelectiveFunctionControl=1` is OBSOLETE and does not help, and neither does
> `-DISHMEMI_IBGDA_BNXT_NOINLINE=OFF`. See `csrc/xpu/named_barrier_usage.md`.

### CUDA → SYCL mapping summary

| CUDA | SYCL/XPU | Notes |
|---|---|---|
| `cudaLaunchAttributeCooperative` | `use_root_sync` property | Enables grid-scope barriers |
| `cudaLaunchKernelEx(&cfg, kernel, ...)` | `syclex::nd_launch(q, cfg, kernel)` | Different launch API |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | `get_kernel_info<..., max_num_work_groups_sync>(...)` | Query max concurrent WGs |
| `cg::this_grid().sync()` | `sycl::group_barrier(root_group)` | Device-wide WG barrier |
| `bar.sync <id>, <count>` | `NamedBarrier::init(count/32)` + `.sync()` | Sub-group-subset barrier |
| `__syncwarp()` | `sycl::group_barrier(sub_group)` | Intra-sub-group (32 WI) |
| `__syncthreads()` | `sycl::group_barrier(work_group)` | Whole-WG barrier |

### Limitations on BMG

- `num_warp_groups == 1` is forced on BMG (160 Xe-cores ≥ num_experts for typical configs).
  Warp specialization is `caster_warps + counter_warp` within a SINGLE warp group.
- `num_warps_per_group == 32` (full 1024-WI work-group) for LL kernels.
- `root_group` requires the `use_root_sync` property at launch time — the kernel MUST be
  launched via `nd_launch`, NOT plain `q.submit()` + `parallel_for`.
- `get_kernel_info<..., max_num_work_groups_sync>` is a **host-side query**; the result
  depends on the kernel, device, WG size, and dynamic local memory. Call it once per
  kernel configuration and cache the result.

### Build requirement

The `root_group` extension requires:
```bash
# oneAPI 2025.0+ with the experimental extensions enabled
icpx -fsycl -fsycl-targets=spir64 \
    -Xsycl-target-backend "-cl-ext=+cl_khr_subgroups" \
    ...
```

The extension is **experimental** — API may change before stabilization. Check the
Intel LLVM release notes for the target oneAPI version.

---

## 6. Related files

| File | Role |
|---|---|
| `csrc/xpu/xpu_kernels.hpp:285-366` | `GridBarrier` (current software grid sync; to be replaced) |
| `csrc/xpu/xpu_kernels.hpp:1-80` | `NamedBarrier` class (warp specialization) |
| `csrc/xpu/named_barrier_ishmem_conflict.md` | NamedBarrier + iSHMEM coexistence fix |
| `csrc/cuda_kernels/launch.cuh:8-13` | CUDA `SETUP_LAUNCH_CONFIG` (cooperative launch attr) |
| `csrc/cuda_kernels/internode_ll.cu:360,977` | CUDA `cg::this_grid().sync()` usage |
| `csrc/cuda_kernels/internode_ll.cu:420,918,1030` | CUDA `bar.sync` warp-specialization usage |
| `csrc/xpu/internode_ll.cpp` | XPU LL kernel (current split-kernel; target for fusion) |
| `csrc/xpu/internode_ll_design.md` | XPU LL kernel design document |

---

*Created: 2026-08-04*