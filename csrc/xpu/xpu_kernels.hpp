#pragma once

#include <cassert>
#include <limits>
#include <type_traits>

#include "xpu_runtime.hpp"

#define EP_HOST_ASSERT(condition) TORCH_CHECK((condition), "DeepEP XPU assertion failed: " #condition)

namespace deep_ep {

template <typename dtype_t>
inline dtype_t align_down(dtype_t a, dtype_t b) {
    return a / b * b;
}

template <typename T>
SYCL_EXTERNAL inline T plain_load(const T* ptr) {
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
    return *ptr;
}

template <typename T>
SYCL_EXTERNAL inline void plain_store(T* ptr, T value) {
    *ptr = value;
    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
}

SYCL_EXTERNAL inline void visa_spin_hint() {}

// ---------------------------------------------------------------------------
// SPMD named barrier (parity for CUDA `bar.sync <id>, <count>`).
//
// CUDA warp specialization uses `bar.sync <id>, <count>` to synchronize a NAMED
// SUBSET of the thread block (e.g. only the worker warps, or one warp group)
// while other warps run ahead. Plain SYCL only offers whole-work-group
// (group_barrier(work_group)) or single-sub-group (group_barrier(sub_group))
// barriers -- neither can sync an arbitrary subset of sub-groups within a
// work-group. The SPIR-V NamedBarrier builtins (cl_khr_subgroup_named_barrier)
// provide exactly this and, unlike ESIMD named_barrier, are usable directly in
// an ordinary SPMD nd_range kernel alongside sub_group shuffles, atomic_ref and
// iSHMEM device calls.
//
// named_barrier_init(count): `count` is the number of PARTICIPATING SUB-GROUPS
//   (NOT work-items); returns a per-work-group named-barrier handle.
// work_group_named_barrier(handle, flags): subset barrier; must be entered
//   UNIFORMLY by every work-item of each participating sub-group. Sub-groups that
//   do not call it proceed freely (a true subset barrier).
//
// Validated on Arc Pro B60 (BMG): a 3-of-4 sub-group subset barrier synchronizes
// the 3 participants while the 4th bypasses. The link-time "undefined function"
// warnings for these two symbols are EXPECTED -- they are SPIR-V builtins IGC
// resolves at JIT, not at LLVM link time.
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
struct __namedBarrier;
extern SYCL_EXTERNAL __namedBarrier __attribute__((opencl_local)) *
named_barrier_init(int count);
extern SYCL_EXTERNAL void work_group_named_barrier(__namedBarrier __attribute__((opencl_local)) *, unsigned int);
#endif

// Memory-fence flags accepted by work_group_named_barrier (OpenCL semantics).
constexpr unsigned int kNamedBarrierLocalFence = 0x1;   // CLK_LOCAL_MEM_FENCE
constexpr unsigned int kNamedBarrierGlobalFence = 0x2;  // CLK_GLOBAL_MEM_FENCE

class NamedBarrier {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    __namedBarrier __attribute__((opencl_local)) * handle_ = nullptr;
#endif

public:
    // `num_subgroups` = number of participating sub-groups (matches the CUDA
    // arrive-count / 32). Must be called uniformly by all participants.
    SYCL_EXTERNAL inline void init([[maybe_unused]] int num_subgroups) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        handle_ = named_barrier_init(num_subgroups);
#endif
    }

    SYCL_EXTERNAL inline void sync([[maybe_unused]] unsigned int flags = kNamedBarrierGlobalFence) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        work_group_named_barrier(handle_, flags);
#endif
    }
};

// Uncacheable (UC) load that bypasses GPU L1/L2/L3 caches, mirroring iSHMEM's
// ishmemi_ibgda_uc_load*. NIC RDMA DMA writes land in VRAM/host memory but the
// GPU caches are NOT coherent with external PCIe-P2P writes; since the iSHMEM
// symmetric heap addresses are reused every iteration, a plain (cached) load
// can return stale lines (even the zero-init bytes) instead of the freshly
// NIC-delivered data. Reading the RDMA receive region through this UC load
// forces the GPU to fetch the current bytes from memory.
template <typename T>
SYCL_EXTERNAL inline T uc_load(const T* ptr) {
#ifdef __SYCL_DEVICE_ONLY__
    T out;
    if constexpr (sizeof(T) == 1) {
        uint8_t v = *reinterpret_cast<const volatile uint8_t*>(__builtin_intel_sycl_ptr_annotation(
            reinterpret_cast<uint8_t*>(const_cast<T*>(ptr)), "sycl-cache-read-hint", 0x7));
        __builtin_memcpy(&out, &v, 1);
    } else if constexpr (sizeof(T) == 2) {
        uint16_t v = *reinterpret_cast<const volatile uint16_t*>(__builtin_intel_sycl_ptr_annotation(
            reinterpret_cast<uint16_t*>(const_cast<T*>(ptr)), "sycl-cache-read-hint", 0x7));
        __builtin_memcpy(&out, &v, 2);
    } else if constexpr (sizeof(T) == 4) {
        uint32_t v = *reinterpret_cast<const volatile uint32_t*>(__builtin_intel_sycl_ptr_annotation(
            reinterpret_cast<uint32_t*>(const_cast<T*>(ptr)), "sycl-cache-read-hint", 0x7));
        __builtin_memcpy(&out, &v, 4);
    } else if constexpr (sizeof(T) == 8) {
        uint64_t v = *reinterpret_cast<const volatile uint64_t*>(__builtin_intel_sycl_ptr_annotation(
            reinterpret_cast<uint64_t*>(const_cast<T*>(ptr)), "sycl-cache-read-hint", 0x7));
        __builtin_memcpy(&out, &v, 8);
    } else {
        out = *ptr;
    }
    return out;
#else
    return *ptr;
#endif
}

// LSC (Load-Store-Cache) explicit cache-control uncached 32-bit load. Unlike
// uc_load (which annotates the pointer with the sycl-cache-read-hint and lets
// IGC lower it), this emits the GenISA `lsc_load.ugm.uc.uc` directly, forcing
// BOTH L1 and L3 uncached at the message descriptor level. Used to A/B whether
// the flag-path 2-node staleness/correctness differs from the hint-based load
// (toggle DEEP_EP_LL_FLAG_LSC=1). GenISA syntax mirrors the proven form in
// ishmem_ibgda test/diagnostic/uar_doorbell_method_test.cpp (the `.d32x1`/"=r"
// form is rejected by IGC; `(M1, 1) %0:d32 flat[%1]:a64` with "=rw"/"rw" works).
SYCL_EXTERNAL inline int lsc_uc_load_i32(const int* ptr) {
#ifdef __SYCL_DEVICE_ONLY__
    uint32_t out;
    asm volatile(
        "lsc_load.ugm.uc.uc (M1, 1) %0:d32 flat[%1]:a64\n"
        : "=rw"(out)
        : "rw"(ptr)
        : "memory");
    int r;
    __builtin_memcpy(&r, &out, 4);
    return r;
#else
    return *ptr;
#endif
}

// Acquire/invalidate counterpart of the LSC release fence: invalidates the GPU
// data cache so a subsequent load observes externally-written (NIC RDMA) data.
SYCL_EXTERNAL inline void lsc_fence_sysacq() {
#ifdef __SYCL_DEVICE_ONLY__
    asm volatile("lsc_fence.ugm.invalidate.sysacq\n" ::: "memory");
#endif
}

// Release/flush LSC fence: evicts/flushes the GPU data cache to the system memory
// domain so an external agent (NIC DMA reading the symmetric send staging) sees
// the freshly-stored bytes. Pairs with lsc_fence_sysacq.
SYCL_EXTERNAL inline void lsc_fence_sysrel() {
#ifdef __SYCL_DEVICE_ONLY__
    asm volatile("lsc_fence.ugm.evict.sysrel\n" ::: "memory");
#endif
}

// Uncached (write-through) store, paired with uc_load. Writing the RDMA
// receive-region sentinel through this (instead of a cached store) guarantees
// the UC-load reader observes the sentinel, not a stale cached line from a
// previous iteration's count value.
template <typename T>
SYCL_EXTERNAL inline void uc_store(T* ptr, T value) {
#ifdef __SYCL_DEVICE_ONLY__
    if constexpr (sizeof(T) == 1) {
        uint8_t v;
        __builtin_memcpy(&v, &value, 1);
        *reinterpret_cast<volatile uint8_t*>(__builtin_intel_sycl_ptr_annotation(
            reinterpret_cast<uint8_t*>(ptr), "sycl-cache-write-hint", 0x7)) = v;
    } else if constexpr (sizeof(T) == 2) {
        uint16_t v;
        __builtin_memcpy(&v, &value, 2);
        *reinterpret_cast<volatile uint16_t*>(__builtin_intel_sycl_ptr_annotation(
            reinterpret_cast<uint16_t*>(ptr), "sycl-cache-write-hint", 0x7)) = v;
    } else if constexpr (sizeof(T) == 4) {
        uint32_t v;
        __builtin_memcpy(&v, &value, 4);
        *reinterpret_cast<volatile uint32_t*>(__builtin_intel_sycl_ptr_annotation(
            reinterpret_cast<uint32_t*>(ptr), "sycl-cache-write-hint", 0x7)) = v;
    } else if constexpr (sizeof(T) == 8) {
        uint64_t v;
        __builtin_memcpy(&v, &value, 8);
        *reinterpret_cast<volatile uint64_t*>(__builtin_intel_sycl_ptr_annotation(
            reinterpret_cast<uint64_t*>(ptr), "sycl-cache-write-hint", 0x7)) = v;
    } else {
        *ptr = value;
    }
#else
    *ptr = value;
#endif
}

SYCL_EXTERNAL inline void get_channel_task_range(
    int num_tokens, int num_channels, int channel_id, int& token_start_idx, int& token_end_idx) {
    int num_tokens_per_channel = (num_tokens + num_channels - 1) / num_channels;
    token_start_idx = sycl::min(num_tokens_per_channel * channel_id, num_tokens);
    token_end_idx = sycl::min(token_start_idx + num_tokens_per_channel, num_tokens);
}

template <typename T>
SYCL_EXTERNAL inline T subgroup_reduce_sum(T value, sycl::nd_item<1> item) {
    return sycl::reduce_over_group(item.get_sub_group(), value, sycl::plus<T>());
}

SYCL_EXTERNAL inline bool elect_one(sycl::nd_item<1> item) {
    return item.get_sub_group().get_local_linear_id() == 0;
}

SYCL_EXTERNAL inline int lane_id(sycl::nd_item<1> item) {
    return static_cast<int>(item.get_sub_group().get_local_linear_id());
}

template <typename T>
SYCL_EXTERNAL inline T subgroup_broadcast(T value, int src_lane, sycl::nd_item<1> item) {
    return sycl::select_from_group(item.get_sub_group(), value, src_lane);
}

template <typename T>
SYCL_EXTERNAL inline T ld_global(const T* ptr) {
    return plain_load(ptr);
}

template <typename T>
SYCL_EXTERNAL inline void st_global(T* ptr, T value) {
    plain_store(ptr, value);
}

template <typename T>
SYCL_EXTERNAL inline T atomic_add_global(T* ptr, T value) {
    sycl::atomic_ref<T, sycl::memory_order::acq_rel, sycl::memory_scope::system, sycl::access::address_space::global_space> ref(*ptr);
    return ref.fetch_add(value);
}

// ---------------------------------------------------------------------------
// GridBarrier -- device-wide (all-work-groups) barrier for a SINGLE kernel
// launch. This is the XPU/SYCL parity for CUDA cooperative-groups
// `cg::this_grid().sync()` and is intended to replace it in the low-latency
// internode kernel port.
//
// WHY A MEMORY-BASED BARRIER (NOT A HARDWARE BARRIER):
//   Intel Xe barrier hardware (`sycl::group_barrier(work_group)`) only spans a
//   SINGLE work-group. Empirically on Arc Pro B60 / BMG neither
//   `ishmemx_barrier_all_work_group` (wrong scope) nor the SPIR-V
//   `__spirv_ControlBarrierArriveINTEL` split barrier with `Device` execution
//   scope produce any cross-work-group effect. So, exactly like CUDA's
//   `this_grid().sync()`, a grid-wide barrier must be built from a
//   GLOBAL-MEMORY counter + spin.
//
// WHY UNCACHED (UC) ACCESS IS MANDATORY:
//   GPU L1/L2/L3 caches are NOT coherent for a plain spin-wait: a work-group
//   spinning on a cached global flag may never observe another work-group's
//   update. Therefore the published `sense` flag is written with `uc_store`
//   and read with `uc_load`/`lsc_uc_load_i32` (both bypass L1/L2/L3), and
//   system-scope acquire/release fences order the surrounding global writes.
//
// SENSE-REVERSING (PHASE) BARRIER:
//   A naive "increment to N then reset to 0" barrier has a reset race (a fast
//   work-group can re-enter the next barrier and see the not-yet-reset value).
//   Instead each work-group keeps its OWN `local_sense` register that toggles
//   0<->1 every call; the last arriver publishes the new sense to the global
//   flag, and every other work-group spins until the global flag matches its
//   freshly toggled local sense. This makes repeated barrier calls within one
//   kernel correct without any reset window.
//
// *** CRITICAL CO-RESIDENCY CONSTRAINT -- READ THIS ***
//   A global-memory grid barrier DEADLOCKS if the participating work-groups are
//   not all CONCURRENTLY RESIDENT on the GPU: a work-group that is never
//   scheduled never arrives, so the already-resident ones spin forever. It is
//   therefore ONLY safe when
//        (total launched work-groups) <= (max concurrently-resident work-groups
//                                          for this kernel + WG size on the device)
//   This mirrors CUDA requiring a *cooperative launch* + occupancy check for
//   `this_grid().sync()`. To bound the grid: query the device
//   (`max_compute_units`, threads-per-EU, WG size) or simply launch a fixed
//   modest grid (e.g. a few work-groups per Xe-core) and have each work-group
//   loop over the logical work. When in doubt, keep the grid small.
//
// SCRATCH REQUIREMENT:
//   `counter` and `sense` must point to zero-initialized global memory (two
//   uint32_t). Zero them (memset / a tiny init kernel) BEFORE the kernel that
//   uses the barrier is launched. They are reusable across many barrier calls
//   within that launch (counter self-resets; sense flips).
struct GridBarrier {
    uint32_t* counter;      // global, zero-initialized: arrival counter
    uint32_t* sense;        // global, zero-initialized: published phase flag
    uint32_t  num_groups;   // number of participating work-groups (grid size)
    uint32_t  local_sense;  // per-work-group phase, starts at 0

    // Construct once per work-item at kernel start. `n_groups` is the launch's
    // work-group count, e.g. item.get_group_range(0).
    GridBarrier() = default;
    SYCL_EXTERNAL inline GridBarrier(uint32_t* counter_, uint32_t* sense_, uint32_t n_groups)
        : counter(counter_), sense(sense_), num_groups(n_groups), local_sense(0) {}

    // Grid-wide barrier: returns only after EVERY participating work-group has
    // reached this call. Must be entered uniformly by all work-items of every
    // participating work-group.
    template <int Dim>
    SYCL_EXTERNAL inline void arrive_and_wait(const sycl::nd_item<Dim>& item) {
        auto wg = item.get_group();

        // (1) Local barrier first: all work-items in this WG finish their
        // pre-barrier work and their global writes are ordered before the
        // leader arrives.
        sycl::group_barrier(wg);

        // Toggle this work-group's phase. Every work-item computes the same
        // value; only the leader touches global state, but all work-items must
        // agree on `new_sense` so the second group_barrier releases coherently.
        const uint32_t new_sense = local_sense ^ 1u;
        local_sense = new_sense;

        if (item.get_local_linear_id() == 0) {
            // (2) Release fence so this WG's prior global writes are visible to
            // other WGs before we announce arrival, then atomically increment.
            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
            sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel,
                             sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                arrived(*counter);
            uint32_t prev = arrived.fetch_add(1u);

            if (prev == num_groups - 1) {
                // (3a) Last arriver: reset the counter for the next phase (UC
                // store, so no stale cached value survives) then publish the new
                // sense with a release fence so waiters observe it.
                uc_store<uint32_t>(counter, 0u);
                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                uc_store<uint32_t>(sense, new_sense);
                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
            } else {
                // (3b) Not last: spin on the UC-loaded global sense flag until it
                // matches our new phase, then acquire so post-barrier reads see
                // the freshly published global state.
                while (uc_load<uint32_t>(sense) != new_sense) {
                    visa_spin_hint();
                }
                sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
            }
        }

        // (4) Local barrier: release all work-items of this WG together so they
        // all observe the post-barrier global state.
        sycl::group_barrier(wg);
    }
};

template <int kNumRanks, bool kSyncOnly = false>
SYCL_EXTERNAL inline void barrier_block(int** barrier_signal_ptrs, int rank, int barrier_signal, sycl::nd_item<1> item) {
    int thread_id = static_cast<int>(item.get_local_id(0));
    if constexpr (kNumRanks == 1) {
        if constexpr (!kSyncOnly) {
            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        }
        item.barrier(sycl::access::fence_space::local_space);
        return;
    }
    if constexpr (!kSyncOnly) {
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        item.barrier(sycl::access::fence_space::local_space);
    }

    if (thread_id < kNumRanks) {
        plain_store(barrier_signal_ptrs[thread_id] + rank, barrier_signal);
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (thread_id < kNumRanks) {
        int* local_slot = barrier_signal_ptrs[rank] + thread_id;
        while (true) {
            if (plain_load(local_slot) >= barrier_signal) {
                break;
            }
            visa_spin_hint();
        }
    }
    item.barrier(sycl::access::fence_space::local_space);
}

template <typename dtype_t>
struct DeviceBuffer {
    uint8_t* ptr = nullptr;
    int64_t total_bytes = 0;

    DeviceBuffer() = default;

    DeviceBuffer(void*& gbl_ptr, int64_t num_elems, int64_t offset = 0) {
        total_bytes = num_elems * static_cast<int64_t>(sizeof(dtype_t));
        ptr = static_cast<uint8_t*>(gbl_ptr) + offset * static_cast<int64_t>(sizeof(dtype_t));
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;
    }

    SYCL_EXTERNAL dtype_t* buffer() const { return reinterpret_cast<dtype_t*>(ptr); }

    SYCL_EXTERNAL dtype_t& operator[](int64_t idx) const { return buffer()[idx]; }
};

}  // namespace deep_ep
