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
