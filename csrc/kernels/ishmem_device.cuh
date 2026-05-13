#pragma once

#include <limits>

#include <ishmem.h>
#if __has_include(<ishmemx.h>)
#include <ishmemx.h>
#endif

#include "configs.cuh"
#include "exception.cuh"
#include "transport.hpp"
#include "utils.cuh"

namespace deep_ep {

namespace low_latency_transport {

__device__ static __forceinline__ int get_num_queue_pairs_per_rank() {
    return 1;
}

__device__ static __forceinline__ int get_num_queue_pairs_per_pe() {
    return std::numeric_limits<int>::max() / 2;
}

template <bool kLowLatencyMode>
__device__ static __forceinline__ void sync_with_same_gpu_idx(const transport::TeamHandle& rdma_team) {
    if constexpr (kLowLatencyMode) {
        ishmem_team_sync(rdma_team);
    } else {
        ishmem_sync_all();
    }
}

__device__ static __forceinline__ void quiet(int, int) {
    ishmem_quiet();
}

__device__ static __forceinline__ uint64_t get_p2p_ptr(const uint64_t& ptr, const int&, const int& dst_rank) {
    return reinterpret_cast<uint64_t>(ishmem_ptr(reinterpret_cast<const void*>(ptr), dst_rank));
}

__device__ static __forceinline__ void store_remote_int(int* dst_ptr, int value, int dst_rank, int) {
    ishmem_int_p(dst_ptr, value, dst_rank);
}

__device__ static __forceinline__ void store_or_write_p2p_int(uint64_t dst_ptr, uint64_t dst_p2p_ptr, int value, int dst_rank) {
    if (dst_p2p_ptr == 0) {
        store_remote_int(reinterpret_cast<int*>(dst_ptr), value, dst_rank, 0);
    } else {
        st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), value);
    }
}

__device__ static __forceinline__ void put_warp(void* dst_ptr,
                                                const void* src_ptr,
                                                size_t size,
                                                int dst_rank,
                                                int,
                                                int lane_id,
                                                int) {
    if (lane_id == 0)
        ishmem_putmem_nbi(dst_ptr, src_ptr, size, dst_rank);
    __syncwarp();
}

__device__ static __forceinline__ bool put_if_remote(uint64_t dst_ptr,
                                                     uint64_t dst_p2p_ptr,
                                                     uint64_t src_ptr,
                                                     size_t size,
                                                     int dst_rank,
                                                     int qp_id,
                                                     int lane_id,
                                                     int slot_idx) {
    if (dst_p2p_ptr != 0)
        return false;
    put_warp(reinterpret_cast<void*>(dst_ptr), reinterpret_cast<const void*>(src_ptr), size, dst_rank, qp_id, lane_id, slot_idx);
    return true;
}

__device__ static __forceinline__ void atomic_add(int* dst_ptr, int value, int dst_rank, int) {
    ishmem_int_atomic_add(dst_ptr, value, dst_rank);
}

__device__ static __forceinline__ void atomic_add_or_write_p2p_int(uint64_t dst_ptr,
                                                                   uint64_t dst_p2p_ptr,
                                                                   int value,
                                                                   int dst_rank,
                                                                   int qp_id) {
    if (dst_p2p_ptr == 0) {
        atomic_add(reinterpret_cast<int*>(dst_ptr), value, dst_rank, qp_id);
    } else {
        st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), value);
    }
}

__device__ static __forceinline__ void barrier_all_block() {
    __syncthreads();
    if (threadIdx.x == 0)
        ishmem_barrier_all();
    __syncthreads();
}

}  // namespace low_latency_transport

}  // namespace deep_ep
