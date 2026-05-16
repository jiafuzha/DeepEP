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
