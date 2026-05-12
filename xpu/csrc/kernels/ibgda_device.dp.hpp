// Minimal iSHMEM-backed compatibility layer for migrated internode kernels.
// The original CUDA path used NVSHMEM IBGDA internals; on XPU we route the
// reachable high-throughput operations through public iSHMEM device APIs.
#pragma once

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

#include "configs.dp.hpp"
#include "utils.dp.hpp"

namespace deep_ep {

struct ibgda_device_state_t {
    int num_rc_per_pe;
    int num_devices_initialized;
    bool use_async_postsend;
};

inline constexpr ibgda_device_state_t kIshmemIbgdaCompatState = {
    256,  // Large enough for current num_channels/num_sms assertions in the translated kernels.
    1,
    false,
};

__device__ static __forceinline__ const ibgda_device_state_t* ibgda_get_state() {
    return &kIshmemIbgdaCompatState;
}

__device__ __forceinline__ void nvshmemi_ibgda_rma_p(void* rptr, const int& value, int pe, int qp_id) {
    (void)qp_id;
    ishmem_int_p(static_cast<int*>(rptr), value, pe);
}

template <bool kAlwaysDoPostSend = true>
__device__ static __forceinline__ void nvshmemi_ibgda_put_nbi_warp(
    uint64_t rptr, uint64_t lptr, size_t bytes, int pe, int qp_id, int lane_id, int message_idx) {
    (void)kAlwaysDoPostSend;
    (void)qp_id;
    (void)message_idx;
    __syncwarp();
    if (lane_id == 0)
        ishmem_putmem_nbi(reinterpret_cast<void*>(rptr), reinterpret_cast<const void*>(lptr), bytes, pe);
    __syncwarp();
}

__device__ __forceinline__ void nvshmemi_ibgda_amo_nonfetch_add(
    void* rptr, const int& value, int pe, int qp_id, bool is_local_copy = false) {
    (void)qp_id;
    if (is_local_copy) {
        atomicAdd_system(static_cast<int*>(rptr), value);
    } else {
        ishmem_int_atomic_add(static_cast<int*>(rptr), value, pe);
    }
}

__device__ __forceinline__ uint64_t nvshmemi_get_p2p_ptr(const uint64_t& ptr, const int& rank, const int& dst_rank) {
    if (rank == dst_rank)
        return ptr;
    return reinterpret_cast<uint64_t>(ishmem_ptr(reinterpret_cast<const void*>(ptr), dst_rank));
}

__device__ static __forceinline__ void nvshmemi_ibgda_quiet(int dst_pe, int qp_id) {
    (void)dst_pe;
    (void)qp_id;
    ishmem_quiet();
}

}  // namespace deep_ep
