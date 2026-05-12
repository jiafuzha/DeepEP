#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <sycl/ext/oneapi/experimental/clock.hpp>
#include <sycl/sycl.hpp>
#include <type_traits>

#define NUM_MAX_NVL_PEERS 8
#define NUM_MAX_RDMA_PEERS 20
#define NUM_WORKSPACE_BYTES (32 * 1024 * 1024)
#define NUM_MAX_LOCAL_EXPERTS 1024
#define NUM_BUFFER_ALIGNMENT_BYTES 128

#define FINISHED_SUM_TAG 1024
#define NUM_WAIT_NANOSECONDS 500

#ifndef ENABLE_FAST_DEBUG
#define NUM_CPU_TIMEOUT_SECS 100
#define NUM_TIMEOUT_CYCLES 200000000000ull  // 200G cycles ~= 100s
#else
#define NUM_CPU_TIMEOUT_SECS 10
#define NUM_TIMEOUT_CYCLES 20000000000ull  // 20G cycles ~= 10s
#endif

#define LOW_LATENCY_SEND_PHASE 1
#define LOW_LATENCY_RECV_PHASE 2

#if !defined(__NVPTX__) && !defined(__CUDACC__)
#ifndef DISABLE_SM90_FEATURES
#define DISABLE_SM90_FEATURES
#endif
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define DISABLE_AGGRESSIVE_PTX_INSTRS
#endif
#endif

// Make CLion CUDA indexing work
#ifdef __CLION_IDE__
#define __CUDA_ARCH__ 900  // NOLINT(*-reserved-identifier)
#define __CUDACC_RDC__     // NOLINT(*-reserved-identifier)
#endif

// Define __CUDACC_RDC__ to ensure proper extern declarations for NVSHMEM device symbols
#ifndef DISABLE_NVSHMEM
#ifndef __CUDACC_RDC__
#define __CUDACC_RDC__  // NOLINT(*-reserved-identifier)
#endif
#endif

// Remove Torch restrictions
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
#ifdef __CUDA_NO_BFLOAT162_OPERATORS__
#undef __CUDA_NO_BFLOAT162_OPERATORS__
#endif

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __shared__
#define __shared__
#endif
#ifndef __forceinline__
#if defined(__clang__) || defined(__GNUC__)
#define __forceinline__ inline __attribute__((always_inline))
#else
#define __forceinline__ inline
#endif
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(...)
#endif
#ifndef __align__
#define __align__(n)
#endif

#ifndef DISABLE_SM90_FEATURES
#else
// Ampere does not support FP8 features
#define __NV_E4M3 0
#define __NV_E5M2 1
typedef int __nv_fp8_interpretation_t;
typedef int __nv_fp8x4_e4m3;
typedef uint8_t __nv_fp8_storage_t;
#endif

namespace deep_ep {

template <typename A, typename B, typename C = std::common_type_t<A, B>>
constexpr C min(A a, B b) {
    return std::min<C>(static_cast<C>(a), static_cast<C>(b));
}

template <typename A, typename B, typename C = std::common_type_t<A, B>>
constexpr C max(A a, B b) {
    return std::max<C>(static_cast<C>(a), static_cast<C>(b));
}

using int2 = sycl::int2;
using int4 = sycl::int4;
using nv_bfloat16 = sycl::ext::oneapi::bfloat16;
using nv_bfloat162 = sycl::vec<sycl::ext::oneapi::bfloat16, 2>;

#ifndef TOPK_IDX_BITS
#define TOPK_IDX_BITS 64
#endif

#define INT_BITS_T2(bits) int##bits##_t
#define INT_BITS_T(bits) INT_BITS_T2(bits)
typedef INT_BITS_T(TOPK_IDX_BITS) topk_idx_t;  // int32_t or int64_t
#undef INT_BITS_T
#undef INT_BITS_T2

namespace cuda_compat {

struct dim3 {
    uint32_t x = 1, y = 1, z = 1;
};

inline dim3 thread_idx() {
#ifdef __SYCL_DEVICE_ONLY__
    const auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    return {static_cast<uint32_t>(item.get_local_id(2)),
            static_cast<uint32_t>(item.get_local_id(1)),
            static_cast<uint32_t>(item.get_local_id(0))};
#else
    return {};
#endif
}

inline dim3 block_idx() {
#ifdef __SYCL_DEVICE_ONLY__
    const auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    return {static_cast<uint32_t>(item.get_group(2)), static_cast<uint32_t>(item.get_group(1)), static_cast<uint32_t>(item.get_group(0))};
#else
    return {};
#endif
}

inline dim3 block_dim() {
#ifdef __SYCL_DEVICE_ONLY__
    const auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    return {static_cast<uint32_t>(item.get_local_range(2)),
            static_cast<uint32_t>(item.get_local_range(1)),
            static_cast<uint32_t>(item.get_local_range(0))};
#else
    return {};
#endif
}

inline dim3 grid_dim() {
#ifdef __SYCL_DEVICE_ONLY__
    const auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    return {static_cast<uint32_t>(item.get_group_range(2)),
            static_cast<uint32_t>(item.get_group_range(1)),
            static_cast<uint32_t>(item.get_group_range(0))};
#else
    return {};
#endif
}

inline void syncthreads() {
#ifdef __SYCL_DEVICE_ONLY__
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group());
#endif
}

inline void syncwarp() {
#ifdef __SYCL_DEVICE_ONLY__
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
#endif
}

inline uint64_t clock64() {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::ext::oneapi::experimental::clock<sycl::ext::oneapi::experimental::clock_scope::sub_group>();
#else
    return 0;
#endif
}

template <typename T>
inline T shfl_sync(uint32_t, const T& value, int src_lane_idx) {
#ifdef __SYCL_DEVICE_ONLY__
    const auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    return sycl::group_broadcast(sg, value, static_cast<typename sycl::sub_group::linear_id_type>(src_lane_idx));
#else
    return value;
#endif
}

template <typename T>
inline T shfl_xor_sync(uint32_t, const T& value, int lane_mask) {
#ifdef __SYCL_DEVICE_ONLY__
    const auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    return sycl::permute_group_by_xor(sg, value, lane_mask);
#else
    return value;
#endif
}

inline bool all_sync(uint32_t, bool pred) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::all_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), pred);
#else
    return pred;
#endif
}

inline bool any_sync(uint32_t, bool pred) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), pred);
#else
    return pred;
#endif
}

template <typename T>
inline T ldg(const T* ptr) {
    return *ptr;
}

inline int atomic_add_system(int* ptr, int value) {
    sycl::atomic_ref<int, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::generic_space> ref(*ptr);
    return ref.fetch_add(value);
}

inline int atomic_sub_system(int* ptr, int value) {
    sycl::atomic_ref<int, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::generic_space> ref(*ptr);
    return ref.fetch_sub(value);
}

inline int4 make_int4(int x, int y, int z, int w) {
    return int4{x, y, z, w};
}

}  // namespace cuda_compat

}  // namespace deep_ep

namespace dpct {

using queue_ptr = sycl::queue*;
using err0 = int;

enum class library_data_t {
    real_bfloat16,
    real_half,
    real_float,
};

inline const char* get_error_string_dummy(int error) {
    switch (error) {
        case 0:
            return "success";
        default:
            return "dpct compatibility error";
    }
}

template <sycl::access::address_space AddressSpace = sycl::access::address_space::generic_space, typename T>
inline bool atomic_compare_exchange_strong(T* addr, T& expected, T desired) {
    sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::device, AddressSpace> ref(*addr);
    return ref.compare_exchange_strong(expected, desired);
}

template <typename T>
inline T select_from_sub_group(const sycl::sub_group& sg, T value, uint32_t idx) {
    return sycl::group_broadcast(sg, value, static_cast<typename sycl::sub_group::linear_id_type>(idx));
}

}  // namespace dpct

#ifndef threadIdx
#define threadIdx (::deep_ep::cuda_compat::thread_idx())
#endif
#ifndef blockIdx
#define blockIdx (::deep_ep::cuda_compat::block_idx())
#endif
#ifndef blockDim
#define blockDim (::deep_ep::cuda_compat::block_dim())
#endif
#ifndef gridDim
#define gridDim (::deep_ep::cuda_compat::grid_dim())
#endif
#ifndef __syncthreads
#define __syncthreads() (::deep_ep::cuda_compat::syncthreads())
#endif
#ifndef __syncwarp
#define __syncwarp() (::deep_ep::cuda_compat::syncwarp())
#endif
#ifndef clock64
#define clock64() (::deep_ep::cuda_compat::clock64())
#endif
#ifndef __nanosleep
#define __nanosleep(ns) ((void)(ns))
#endif
#ifndef __ffs
#define __ffs(value) (__builtin_ffs(value))
#endif
#ifndef __shfl_sync
#define __shfl_sync(mask, value, src_lane_idx) (::deep_ep::cuda_compat::shfl_sync((mask), (value), (src_lane_idx)))
#endif
#ifndef __shfl_xor_sync
#define __shfl_xor_sync(mask, value, lane_mask) (::deep_ep::cuda_compat::shfl_xor_sync((mask), (value), (lane_mask)))
#endif
#ifndef __all_sync
#define __all_sync(mask, pred) (::deep_ep::cuda_compat::all_sync((mask), (pred)))
#endif
#ifndef __any_sync
#define __any_sync(mask, pred) (::deep_ep::cuda_compat::any_sync((mask), (pred)))
#endif
#ifndef atomicAdd_system
#define atomicAdd_system(ptr, value) (::deep_ep::cuda_compat::atomic_add_system((ptr), (value)))
#endif
#ifndef atomicSub_system
#define atomicSub_system(ptr, value) (::deep_ep::cuda_compat::atomic_sub_system((ptr), (value)))
#endif
#ifndef __ldg
#define __ldg(ptr) (::deep_ep::cuda_compat::ldg((ptr)))
#endif
#ifndef make_int4
#define make_int4(x, y, z, w) (::deep_ep::cuda_compat::make_int4((x), (y), (z), (w)))
#endif

#ifndef DISABLE_NVSHMEM
#include <ishmem.h>
#include <ishmemx.h>

using nvshmem_team_t = ishmem_team_t;
using nvshmem_team_config_t = ishmem_team_config_t;
using nvshmemx_uniqueid_t = ishmemx_uniqueid_t;
using nvshmemx_init_attr_t = ishmemx_attr_t;

constexpr nvshmem_team_t NVSHMEM_TEAM_INVALID = ISHMEM_TEAM_INVALID;
constexpr nvshmem_team_t NVSHMEM_TEAM_WORLD = ISHMEM_TEAM_WORLD;
constexpr int NVSHMEMX_INIT_WITH_UNIQUEID = 1;

inline int nvshmemx_get_uniqueid(nvshmemx_uniqueid_t* unique_id) {
    return ishmemx_get_uniqueid(unique_id);
}

inline void nvshmemx_set_attr_uniqueid_args(int rank, int num_ranks, nvshmemx_uniqueid_t* root_unique_id, nvshmemx_init_attr_t* attr) {
    *attr = {};
    attr->rank = rank;
    attr->nranks = num_ranks;
    attr->uid = root_unique_id;
    attr->use_uid = true;
    attr->gpu = true;
}

inline void nvshmemx_init_attr(int, nvshmemx_init_attr_t* attr) {
    ishmemx_init_attr(attr);
}

inline void nvshmem_barrier_all() {
    ishmem_barrier_all();
}

inline void nvshmem_sync_all() {
    ishmem_sync_all();
}

inline int nvshmem_sync(nvshmem_team_t team) {
    return ishmem_team_sync(team);
}

inline int nvshmem_my_pe() {
    return ishmem_my_pe();
}

inline void* nvshmem_align(size_t alignment, size_t size) {
    return ishmem_align(alignment, size);
}

inline void nvshmem_free(void* ptr) {
    ishmem_free(ptr);
}

inline int nvshmem_team_split_strided(nvshmem_team_t parent_team,
                                      int start,
                                      int stride,
                                      int size,
                                      const nvshmem_team_config_t* config,
                                      long config_mask,
                                      nvshmem_team_t* new_team) {
    return ishmem_team_split_strided(parent_team, start, stride, size, config, config_mask, new_team);
}

inline void nvshmem_team_destroy(nvshmem_team_t team) {
    ishmem_team_destroy(team);
}

inline void nvshmem_finalize() {
    ishmem_finalize();
}

template <typename Group>
inline void ishmem_barrier_all_block_compat(const Group& group) {
    ishmemx_barrier_all_work_group(group);
}

inline void nvshmemx_barrier_all_block() {
#ifdef __SYCL_DEVICE_ONLY__
    ishmem_barrier_all_block_compat(sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group());
#endif
}
#endif
