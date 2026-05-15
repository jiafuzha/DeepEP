---
name: memory-semantics-and-ptx-assembly-converter
description: 'Convert CUDA inline PTX (bar.sync, barrier.sync, mbarrier, cp.async, fences, atomics, load/store, special registers) and memory semantics to SYCL-equivalent implementations for DeepEP XPU migration. Use when removing PTX or CUDA memory ordering from kernels and preserving synchronization/memory correctness.'
argument-hint: 'Provide PTX sites or target files, required portability level (portable SYCL vs Intel-specific), and expected validation commands.'
user-invocable: true
---

# PTX Assembly Converter

## Outcome
This skill produces:
- A file-by-file PTX inventory for the requested migration scope.
- A PTX-to-SYCL mapping plan with portability notes.
- Concrete replacement code patterns, including named-barrier style migration for `bar.sync`.
- Validation checks proving semantic and build correctness after PTX removal.

## Default Policy
- Prefer Intel-specific `nbarrier` style implementation first when toolchain/target support is available.
- Always rewrite TMA/SM90 PTX (`mbarrier.*`, `cp.async.*`) into non-TMA algorithms before final SYCL conversion.
- Keep a portable SYCL fallback for barriers and synchronization when Intel-specific paths are unavailable.

## Required Reading Before Conversion
Read these first and apply them during conversion:
- Intel vISA docs index: https://github.com/intel/intel-graphics-compiler/tree/master/documentation/visa
- vISA execution model: https://raw.githubusercontent.com/intel/intel-graphics-compiler/master/documentation/visa/3_execution_model.md
- vISA BARRIER: https://raw.githubusercontent.com/intel/intel-graphics-compiler/master/documentation/visa/instructions/BARRIER.md
- vISA NBARRIER: https://raw.githubusercontent.com/intel/intel-graphics-compiler/master/documentation/visa/instructions/NBARRIER.md
- TVISA reference repo: https://github.com/CaoZhongZ/tvisa
- TVISA named barrier helpers: https://raw.githubusercontent.com/CaoZhongZ/tvisa/main/include/gateway.hpp
- TVISA nbarrier usage example: https://raw.githubusercontent.com/CaoZhongZ/tvisa/main/include/unit_test_nbarrier.hpp

## DeepEP PTX Inventory (Current Repository)
Use this as the initial baseline inventory for migration planning.

### Files with inline PTX/asm
- `csrc/kernels/utils.cuh`
- `csrc/kernels/ibgda_device.cuh`
- `csrc/kernels/intranode.cu`
- `csrc/kernels/internode.cu`
- `csrc/kernels/internode_ll.cu`
- `csrc/kernels/exception.cuh`

### PTX instruction families observed
- Barrier/synchronization:
  - `bar.sync`, `barrier.sync`
  - `mbarrier.init`, `mbarrier.inval`, `mbarrier.try_wait.parity`, `mbarrier.arrive`, `mbarrier.arrive.expect_tx`
- Async/TMA transfer path:
  - `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes...`
  - `cp.async.bulk.global.shared::cta.bulk_group...`
  - `cp.async.bulk.commit_group`, `cp.async.bulk.wait_group.read`
- Fences:
  - `fence.acq_rel.sys`, `fence.acq_rel.gpu`, `fence.acq_rel.cta`
  - `fence.proxy.async.shared::cta`, `fence.mbarrier_init.release.cluster`
- Memory ops with ordering/cache qualifiers:
  - `ld.acquire.*`, `ld.relaxed.*`, `ld.volatile.*`, `ld.global.nc.*`
  - `st.relaxed.*`, `st.release.*`, `st.global.L1::no_allocate.*`
- Atomics:
  - `atom.add.release.*`
  - `atom.acquire.cta.shared::cta.cas`, `atom.release.cta.shared::cta.exch`
- Special register/utility/math:
  - `mov.s32 ..., %laneid`
  - `elect.sync`
  - `prmt.b32`
  - `lg2.approx.f32`, `ex2.approx.f32`
  - `trap`

## Mapping Matrix: PTX to SYCL-Equivalent
Prefer Intel-specific mappings first where supported on the migration target, and provide portable SYCL fallbacks for compatibility.

| PTX Pattern | Primary SYCL Equivalent | Intel-Specific Option | Notes |
| --- | --- | --- | --- |
| `bar.sync`, `barrier.sync` (full work-group) | `sycl::group_barrier(item.get_group())` | vISA `BARRIER` lowering via compiler | Equivalent to full thread-group barrier with memory ordering. |
| `bar.sync id, count` (named/subset barrier semantics) | vISA `NBARRIER.signal/wait` style flow as in TVISA | Software named-barrier emulation in local memory + `group_barrier` + atomics | Default to Intel-specific named barrier when available; use software fallback otherwise. |
| `__syncwarp` style behavior | `sycl::group_barrier(item.get_sub_group())` | N/A | For subgroup/warp-local sync. |
| `fence.acq_rel.sys/gpu/cta` | `sycl::atomic_fence(sycl::memory_order::acq_rel, scope)` | vISA fence if explicitly required | Scope map: `sys->system`, `gpu->device`, `cta->work_group`. |
| `ld.acquire.*`, `st.release.*` | `sycl::atomic_ref` load/store with acquire/release ordering | N/A | Use matching `memory_scope` and type-safe atomics. |
| `atom.*.cas`, `atom.*.exch`, `atom.add.*` | `sycl::atomic_ref` compare_exchange / exchange / fetch_add | N/A | Preserve ordering/scope from PTX suffixes. |
| `ld.global.nc.*`, `L1::no_allocate` cache hints | Plain loads/stores or local prefetch strategy | Optional ext APIs if supported | No guaranteed 1:1 portable cache-hint mapping. Preserve correctness first. |
| `mbarrier.*` + `cp.async.bulk.*` (SM90/TMA) | Rewrite to non-TMA pipeline: staged local-memory copies + barriers + explicit double/triple buffering | Intel async copy primitives if available in target stack | Current XPU migration should remove TMA dependency first. |
| `mov ... %laneid` | `item.get_sub_group().get_local_id()[0]` | N/A | Preserve subgroup assumptions. |
| `elect.sync` | `sycl::group_ballot`/`reduce`-based election (single elected lane) | Vendor subgroup helpers | Ensure exactly one elected participant in active subgroup. |
| `prmt.b32` byte permute | Bit ops / byte shuffle utility | N/A | For endian conversion, prefer byte-swap helpers. |
| `lg2.approx`, `ex2.approx` | `sycl::native::log2`, `sycl::native::exp2` | N/A | If approximation error is unsafe, use precise `sycl::log2/exp2`. |
| `trap` | `assert(false)` plus controlled termination path | N/A | Keep diagnostics, avoid silent hangs. |

## TVISA-Informed `bar.sync` to `nbarrier` Migration Pattern
VISA semantics:
- `BARRIER` is full thread-group barrier.
- `NBARRIER` enables named producer-consumer synchronization for a subset.

TVISA pattern:
- Initialize named barriers (`named_barrier_init<N>()`).
- Signal (`nbarrier_signal(id, n_threads)`).
- Wait (`nbarrier_wait(id)`).

### Example Transformation

#### Source CUDA PTX
```cpp
// Subset barrier: only selected threads/warps participate.
asm volatile("bar.sync %0, %1;" :: "r"(barrier_id), "r"(num_threads));
```

#### Intel-Specific Named Barrier Style (default when available)
```cpp
// Mirror TVISA usage where target toolchain supports nbarrier-style lowering.
// Keep a portable fallback for unsupported environments.
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
named_barrier_init<4>();
uint8_t id = static_cast<uint8_t>(barrier_id);
uint8_t n_threads = static_cast<uint8_t>(num_threads_for_barrier_id);
nbarrier_signal(id, n_threads);
nbarrier_wait(id);
#else
sycl::group_barrier(item.get_group());
#endif
```

#### Portable SYCL Fallback
```cpp
// Named-barrier emulation in local memory for subset synchronization.
// This keeps the kernel fully SYCL and avoids target-specific inline assembly.

auto group = item.get_group();
auto lid = item.get_local_linear_id();

// local_counter and local_phase are local memory scalars shared by the group.
if (lid == 0) {
  local_counter[barrier_id] = 0;
}
sycl::group_barrier(group);

bool participates = (lid < num_threads_for_barrier_id);
if (participates) {
  sycl::atomic_ref<int,
                   sycl::memory_order::acq_rel,
                   sycl::memory_scope::work_group,
                   sycl::access::address_space::local_space>
      counter(local_counter[barrier_id]);
  counter.fetch_add(1);
}

// Group-wide polling point with forward progress via barriers.
while (true) {
  sycl::group_barrier(group);
  int seen = local_counter[barrier_id];
  sycl::group_barrier(group);
  if (seen >= num_threads_for_barrier_id) {
    break;
  }
}
```

## Procedure
1. Inventory PTX usage.
- Scan target files for `asm`, `asm volatile`, and PTX mnemonics.
- Group sites by semantic category: barrier, fence, memory ordering, atomics, async copy, special registers, math.

2. Classify each PTX site.
- Decide if it is full-group sync, subset sync, memory-order primitive, data movement primitive, or micro-optimization hint.
- Mark whether there is a portable SYCL mapping or only an Intel-specific optimization path.

3. Apply conversion in this order.
- Replace synchronization and ordering first (`bar.sync`, fences, atomics).
- Replace memory operations next (`ld/st` ordered variants).
- Rewrite SM90/TMA (`mbarrier`/`cp.async`) into non-TMA algorithm (mandatory).
- Replace utility instructions (`laneid`, `prmt`, approx math).

4. Resolve cross-file integration.
- Update helper wrappers and callsites together.
- Ensure converted code paths compile without CUDA inline PTX.

5. Validate.
- Build in SYCL/XPU mode.
- Run correctness checks for race-sensitive kernels.
- Confirm no remaining PTX in migrated files.

## Decision Points
- If semantics require only full work-group synchronization: use `group_barrier`.
- If synchronization is named/subset (`bar.sync id,count`): use Intel-specific nbarrier path first when available; otherwise use named-barrier emulation.
- If PTX site is SM90/TMA-specific: always rewrite algorithm to non-TMA first, then port.
- If PTX cache hints have no portable equivalent: keep correctness, then performance-tune with target-specific extensions.

## Completion Checks
- No inline PTX remains in migrated SYCL kernel files.
- Converted kernels compile for XPU.
- Barrier/fence/atomic semantics remain correct under stress tests.
- Any Intel-specific path is guarded with a portable fallback.

## Notes
- SYCLomatic (`c2s`/`dpct`) can accelerate baseline migration, but PTX-heavy sections require manual conversion and semantic review.
- For this repository, prioritize removing TMA/PTX coupling in `csrc/kernels/utils.cuh` and all `bar.sync`/`barrier.sync` sites in intranode/internode kernels.

# Additional Migration Skills Learned from CUDA→SYCL Comparison (DeepEP SYCL Fork )

## 1. PTX Barrier/Synchronization
- **CUDA PTX:** `bar.sync`, `barrier.sync`, `mbarrier.*`, `cp.async.*`
- **SYCL Intel-Specific:** Use `nbarrier_signal`/`nbarrier_wait` (TVISA style) for named/subset barriers; use `sycl::group_barrier` for full-group sync.
- **SYCL Portable:** Emulate named barriers in local memory with `sycl::atomic_ref` and polling; always provide a fallback for non-Intel targets.

## 2. Memory Fences and Ordering
- **CUDA PTX:** `fence.acq_rel.sys`, `fence.acq_rel.gpu`, `fence.acq_rel.cta`
- **SYCL:** Use `sycl::atomic_fence` with correct `memory_order` and `memory_scope`. Map `sys`→`system`, `gpu`→`device`, `cta`→`work_group`.

## 3. Atomics and Memory Operations
- **CUDA PTX:** `atom.add.release.*`, `atom.acquire.cta.shared::cta.cas`, `ld.acquire.*`, `st.release.*`
- **SYCL:** Use `sycl::atomic_ref` with explicit ordering and scope. For loads/stores, use acquire/release orderings.

## 4. SM90/TMA/Async Copy
- **CUDA PTX:** `mbarrier.*`, `cp.async.bulk.*`
- **SYCL:** Rewrite to explicit staged local-memory copies, double/triple buffering, and group barriers. Remove TMA dependency before porting.

## 5. Special Registers and Utility
- **CUDA PTX:** `mov.s32 ..., %laneid`, `elect.sync`, `prmt.b32`
- **SYCL:** Use `item.get_sub_group().get_local_id()[0]` for lane id, `sycl::group_ballot`/`reduce` for election, and bitwise ops for permute.

## 6. General Principles
- Always remove all inline PTX from migrated SYCL files.
- Validate correctness with stress/race tests after migration.
- Guard Intel-specific code with portable SYCL fallbacks.

---
These patterns are distilled from direct comparison of CUDA and SYCL kernels in the DeepEP migration fork https://github.com/leizhenyuan/DeepEP/tree/zhenyuan_enable_intel_intranode/csrc/sycl, and should be applied to all future PTX-to-SYCL conversions in this repository.
