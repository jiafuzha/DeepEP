#pragma once

#include "exception.cuh"

#include <algorithm>
#include <type_traits>
#include <utility>

#if defined(DEEPEP_XPU_NATIVE)
#include <sycl/ext/oneapi/experimental/clock.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#endif

#define UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC)                                                     \
    {                                                                                                                                 \
        constexpr int kLoopStride = 32 * (UNROLL_FACTOR);                                                                             \
        typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)];                          \
        auto __src = (SRC);                                                                                                           \
        auto __dst = (DST);                                                                                                           \
        for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) {                                      \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32); \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]);  \
        }                                                                                                                             \
        {                                                                                                                             \
            int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID);                                                                  \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                                                       \
                if (__i + __j * 32 < (N)) {                                                                                           \
                    unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32);                                                           \
                }                                                                                                                     \
            }                                                                                                                         \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                                                       \
                if (__i + __j * 32 < (N)) {                                                                                           \
                    ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]);                                                            \
                }                                                                                                                     \
            }                                                                                                                         \
        }                                                                                                                             \
    }

namespace deep_ep {

#if defined(DEEPEP_XPU_NATIVE)
namespace xpu_named_barrier {

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
extern SYCL_EXTERNAL void named_barrier_init(int id);

template <int N>
EP_DEVICE EP_FORCEINLINE void init() {
    if constexpr (N > 0) {
        init<N - 1>();
        named_barrier_init(N - 1);
    }
}

EP_DEVICE EP_FORCEINLINE void wait(uint8_t id) {
    asm volatile("nbarrier.wait %0(0,0)<0;1,0>\n" ::"rw"(id));
}

EP_DEVICE EP_FORCEINLINE void signal(uint8_t id, uint8_t num_threads) {
    asm volatile("nbarrier.signal %0(0,0)<0;1,0> %1(0,0)<0;1,0>\n" ::"rw"(id), "rw"(num_threads));
}

EP_DEVICE EP_FORCEINLINE void sync(uint8_t id, uint8_t num_threads) {
    signal(id, num_threads);
    wait(id);
}
#else
template <int N>
EP_DEVICE EP_FORCEINLINE void init() {}
EP_DEVICE EP_FORCEINLINE void wait(uint8_t) {}
EP_DEVICE EP_FORCEINLINE void signal(uint8_t, uint8_t) {}
EP_DEVICE EP_FORCEINLINE void sync(uint8_t, uint8_t) {}
#endif

}  // namespace xpu_named_barrier

EP_DEVICE EP_FORCEINLINE void visa_spin_hint() {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    asm volatile("pause\n");
#endif
}
#endif

namespace backend_primitives {

#if defined(DEEPEP_XPU_NATIVE)
template <typename T, sycl::memory_scope kScope>
using atomic_ref_t = sycl::atomic_ref<T, sycl::memory_order::acq_rel, kScope, sycl::access::address_space::global_space>;

EP_DEVICE EP_FORCEINLINE void system_fence() {
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}

EP_DEVICE EP_FORCEINLINE void gpu_fence() {
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
}

EP_DEVICE EP_FORCEINLINE void cta_fence() {
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::work_group);
}

EP_DEVICE EP_FORCEINLINE void store_relaxed_system(const int* ptr, int val) {
    atomic_ref_t<int, sycl::memory_scope::system>(*const_cast<int*>(ptr)).store(val, sycl::memory_order::relaxed);
}

EP_DEVICE EP_FORCEINLINE void store_release_system(const int* ptr, int val) {
    atomic_ref_t<int, sycl::memory_scope::system>(*const_cast<int*>(ptr)).store(val, sycl::memory_order::release);
}

EP_DEVICE EP_FORCEINLINE void store_release_cta(const int* ptr, int val) {
    atomic_ref_t<int, sycl::memory_scope::work_group>(*const_cast<int*>(ptr)).store(val, sycl::memory_order::release);
}

EP_DEVICE EP_FORCEINLINE int load_acquire_system(const int* ptr) {
    return atomic_ref_t<int, sycl::memory_scope::system>(*const_cast<int*>(ptr)).load(sycl::memory_order::acquire);
}

EP_DEVICE EP_FORCEINLINE uint64_t load_acquire_system(const uint64_t* ptr) {
    return atomic_ref_t<uint64_t, sycl::memory_scope::system>(*const_cast<uint64_t*>(ptr)).load(sycl::memory_order::acquire);
}

EP_DEVICE EP_FORCEINLINE int load_acquire_gpu(const int* ptr) {
    return atomic_ref_t<int, sycl::memory_scope::device>(*const_cast<int*>(ptr)).load(sycl::memory_order::acquire);
}

EP_DEVICE EP_FORCEINLINE int load_acquire_cta(const int* ptr) {
    return atomic_ref_t<int, sycl::memory_scope::work_group>(*const_cast<int*>(ptr)).load(sycl::memory_order::acquire);
}

EP_DEVICE EP_FORCEINLINE int atomic_add_release_system(const int* ptr, int value) {
    return atomic_ref_t<int, sycl::memory_scope::system>(*const_cast<int*>(ptr)).fetch_add(value, sycl::memory_order::acq_rel);
}

EP_DEVICE EP_FORCEINLINE int atomic_add_release_gpu(const int* ptr, int value) {
    return atomic_ref_t<int, sycl::memory_scope::device>(*const_cast<int*>(ptr)).fetch_add(value, sycl::memory_order::acq_rel);
}
#else
__device__ __forceinline__ void system_fence() {
    asm volatile("fence.acq_rel.sys;" ::: "memory");
}

__device__ __forceinline__ void gpu_fence() {
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
}

__device__ __forceinline__ void cta_fence() {
    asm volatile("fence.acq_rel.cta;" ::: "memory");
}

__device__ __forceinline__ void store_relaxed_system(const int* ptr, int val) {
    asm volatile("st.relaxed.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ void store_release_system(const int* ptr, int val) {
    asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ void store_release_cta(const int* ptr, int val) {
    asm volatile("st.release.cta.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ int load_acquire_system(const int* ptr) {
    int ret;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ uint64_t load_acquire_system(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int load_acquire_gpu(const int* ptr) {
    int ret;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int load_acquire_cta(const int* ptr) {
    int ret;
    asm volatile("ld.acquire.cta.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int atomic_add_release_system(const int* ptr, int value) {
    int ret;
    asm volatile("atom.add.release.sys.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
}

__device__ __forceinline__ int atomic_add_release_gpu(const int* ptr, int value) {
    int ret;
    asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
}
#endif

}  // namespace backend_primitives

template <int kBytes>
struct VecInt {};
template <>
struct VecInt<1> {
    using vec_t = int8_t;
};
template <>
struct VecInt<2> {
    using vec_t = int16_t;
};
template <>
struct VecInt<4> {
    using vec_t = int;
};
template <>
struct VecInt<8> {
    using vec_t = int64_t;
};
template <>
struct VecInt<16> {
    using vec_t = int4;
};

template <typename FuncT>
struct PatternVisitor {
    FuncT func;

    EP_HOST_DEVICE explicit PatternVisitor(FuncT&& func) : func(std::forward<FuncT>(func)) {}

    EP_HOST_DEVICE auto operator[](const uint32_t& i) { return func(i); }
};

EP_DEVICE EP_FORCEINLINE void trap() {
#if defined(DEEPEP_XPU_NATIVE)
    __builtin_trap();
#else
    asm("trap;");
#endif
}

EP_DEVICE EP_FORCEINLINE void memory_fence() {
    backend_primitives::system_fence();
}

EP_DEVICE EP_FORCEINLINE void memory_fence_gpu() {
    backend_primitives::gpu_fence();
}

EP_DEVICE EP_FORCEINLINE void memory_fence_cta() {
    backend_primitives::cta_fence();
}

EP_DEVICE EP_FORCEINLINE void st_relaxed_sys_global(const int* ptr, int val) {
    backend_primitives::store_relaxed_system(ptr, val);
}

EP_DEVICE EP_FORCEINLINE void st_release_sys_global(const int* ptr, int val) {
    backend_primitives::store_release_system(ptr, val);
}

EP_DEVICE EP_FORCEINLINE void st_release_cta(const int* ptr, int val) {
    backend_primitives::store_release_cta(ptr, val);
}

EP_DEVICE EP_FORCEINLINE int ld_acquire_sys_global(const int* ptr) {
    return backend_primitives::load_acquire_system(ptr);
}

EP_DEVICE EP_FORCEINLINE uint64_t ld_acquire_sys_global(const uint64_t* ptr) {
    return backend_primitives::load_acquire_system(ptr);
}

EP_DEVICE EP_FORCEINLINE int ld_acquire_global(const int* ptr) {
    return backend_primitives::load_acquire_gpu(ptr);
}

EP_DEVICE EP_FORCEINLINE int atomic_add_release_sys_global(const int* ptr, int value) {
    return backend_primitives::atomic_add_release_system(ptr, value);
}

EP_DEVICE EP_FORCEINLINE int atomic_add_release_global(const int* ptr, int value) {
    return backend_primitives::atomic_add_release_gpu(ptr, value);
}

EP_DEVICE EP_FORCEINLINE int ld_acquire_cta(const int* ptr) {
    return backend_primitives::load_acquire_cta(ptr);
}

#if defined(DEEPEP_XPU_NATIVE)
template <typename dtype_t>
EP_DEVICE EP_FORCEINLINE dtype_t ld_nc_global(const dtype_t* ptr) {
    return *ptr;
}

EP_DEVICE EP_FORCEINLINE uint8_t ld_na_relaxed(const uint8_t* ptr) { return *ptr; }
EP_DEVICE EP_FORCEINLINE uint16_t ld_na_relaxed(const uint16_t* ptr) { return *ptr; }
EP_DEVICE EP_FORCEINLINE uint32_t ld_na_relaxed(const uint32_t* ptr) { return *ptr; }
EP_DEVICE EP_FORCEINLINE uint64_t ld_na_relaxed(const uint64_t* ptr) { return *ptr; }
EP_DEVICE EP_FORCEINLINE int ld_volatile_global(const int* ptr) { return *ptr; }
EP_DEVICE EP_FORCEINLINE float ld_volatile_global(const float* ptr) { return *ptr; }
EP_DEVICE EP_FORCEINLINE int64_t ld_volatile_global(const int64_t* ptr) { return *ptr; }
EP_DEVICE EP_FORCEINLINE int64_t ld_volatile_global(const uint64_t* ptr) { return static_cast<int64_t>(*ptr); }
EP_DEVICE EP_FORCEINLINE void st_na_relaxed(const uint8_t* ptr, uint8_t val) { *const_cast<uint8_t*>(ptr) = val; }
EP_DEVICE EP_FORCEINLINE void st_na_relaxed(const uint16_t* ptr, uint16_t val) { *const_cast<uint16_t*>(ptr) = val; }
EP_DEVICE EP_FORCEINLINE void st_na_relaxed(const uint32_t* ptr, uint32_t val) { *const_cast<uint32_t*>(ptr) = val; }
EP_DEVICE EP_FORCEINLINE void st_na_relaxed(const int* ptr, int val) { *const_cast<int*>(ptr) = val; }
EP_DEVICE EP_FORCEINLINE void st_na_relaxed(const int4* ptr, int4 val) { *const_cast<int4*>(ptr) = val; }
EP_DEVICE EP_FORCEINLINE void st_na_release(const int* ptr, int val) {
    backend_primitives::store_release_system(ptr, val);
}
EP_DEVICE EP_FORCEINLINE void st_na_release(const uint32_t* ptr, uint32_t val) {
    backend_primitives::atomic_ref_t<uint32_t, sycl::memory_scope::system>(*const_cast<uint32_t*>(ptr)).store(val, sycl::memory_order::release);
}
EP_DEVICE EP_FORCEINLINE void st_na_release(const uint64_t* ptr, uint64_t val) {
    backend_primitives::atomic_ref_t<uint64_t, sycl::memory_scope::system>(*const_cast<uint64_t*>(ptr)).store(val, sycl::memory_order::release);
}
template <typename dtype_t>
EP_DEVICE EP_FORCEINLINE void st_na_global(const dtype_t* ptr, const dtype_t& value) {
    *const_cast<dtype_t*>(ptr) = value;
}
EP_DEVICE EP_FORCEINLINE float log2f_approx(const float& x) { return sycl::log2(x); }
EP_DEVICE EP_FORCEINLINE float exp2f_approx(const float& x) { return sycl::exp2(x); }
#else
__device__ __forceinline__ uint8_t ld_na_relaxed(const uint8_t* ptr) {
    uint16_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b8 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return static_cast<uint8_t>(ret);
}

__device__ __forceinline__ uint16_t ld_na_relaxed(const uint16_t* ptr) {
    uint16_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b16 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ uint32_t ld_na_relaxed(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ uint64_t ld_na_relaxed(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int ld_volatile_global(const int* ptr) {
    int ret;
    asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ float ld_volatile_global(const float* ptr) {
    float ret;
    asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int64_t ld_volatile_global(const int64_t* ptr) {
    int64_t ret;
    asm volatile("ld.volatile.global.s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int64_t ld_volatile_global(const uint64_t* ptr) {
    int64_t ret;
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define LD_NC_FUNC "ld.global.nc.L1::no_allocate.L2::256B"
#else
#define LD_NC_FUNC "ld.volatile.global"
#endif

// `ld.global.nc.L1::no_allocate` will be translated into `LDG.E.NA.[width].CONSTANT` in SASS
template <typename dtype_t>
__device__ __forceinline__ dtype_t ld_nc_global(const dtype_t* ptr) {
    auto ret = ld_nc_global(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(ptr));
    return *reinterpret_cast<dtype_t*>(&ret);
}

template <>
__device__ __forceinline__ uint8_t ld_nc_global(const uint8_t* ptr) {
    uint16_t ret;
    // NOTES: we must use `uint16_t` as inline ASM does not support 8-bit constraint letter (`h` below means unsigned 16-bit)
    asm volatile(LD_NC_FUNC ".u8 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return static_cast<uint8_t>(ret);
}

template <>
__device__ __forceinline__ int ld_nc_global(const int* ptr) {
    int ret;
    asm volatile(LD_NC_FUNC ".s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

template <>
__device__ __forceinline__ int64_t ld_nc_global(const int64_t* ptr) {
    int64_t ret;
    asm volatile(LD_NC_FUNC ".s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

template <>
__device__ __forceinline__ float ld_nc_global(const float* ptr) {
    float ret;
    asm volatile(LD_NC_FUNC ".f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
}

template <>
__device__ __forceinline__ int2 ld_nc_global(const int2* ptr) {
    int2 ret;
    asm volatile(LD_NC_FUNC ".v2.s32 {%0, %1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l"(ptr));
    return ret;
}

template <>
__device__ __forceinline__ int4 ld_nc_global(const int4* ptr) {
    int4 ret;
    asm volatile(LD_NC_FUNC ".v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ void st_na_relaxed(const uint8_t* ptr, uint8_t val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b8 [%0], %1;" : : "l"(ptr), "h"(static_cast<uint16_t>(val)));
}

__device__ __forceinline__ void st_na_relaxed(const uint16_t* ptr, uint16_t val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b16 [%0], %1;" : : "l"(ptr), "h"(val));
}

__device__ __forceinline__ void st_na_relaxed(const uint32_t* ptr, uint32_t val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
}

__device__ __forceinline__ void st_na_relaxed(const int* ptr, int val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
}

__device__ __forceinline__ void st_na_relaxed(const int4* ptr, int4 val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
}

__device__ __forceinline__ void st_na_release(const int* ptr, int val) {
    asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
}

__device__ __forceinline__ void st_na_release(const uint32_t* ptr, uint32_t val) {
    asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
}

__device__ __forceinline__ void st_na_release(const uint64_t* ptr, uint64_t val) {
    asm volatile("st.release.gpu.global.L1::no_allocate.b64 [%0], %1;" : : "l"(ptr), "l"(val));
}

// `st.global.L1::no_allocate` will be translated into `ST.E.NA.[width]` in SASS
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define ST_NA_FUNC "st.global.L1::no_allocate"
#else
#define ST_NA_FUNC "st.global"
#endif

template <typename dtype_t>
__device__ __forceinline__ void st_na_global(const dtype_t* ptr, const dtype_t& value) {
    st_na_global(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(ptr),
                 *reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(&value));
}

template <>
__device__ __forceinline__ void st_na_global(const int* ptr, const int& value) {
    asm volatile(ST_NA_FUNC ".s32 [%0], %1;" ::"l"(ptr), "r"(value));
}

template <>
__device__ __forceinline__ void st_na_global(const int64_t* ptr, const int64_t& value) {
    asm volatile(ST_NA_FUNC ".s64 [%0], %1;" ::"l"(ptr), "l"(value));
}

template <>
__device__ __forceinline__ void st_na_global(const float* ptr, const float& value) {
    asm volatile(ST_NA_FUNC ".f32 [%0], %1;" ::"l"(ptr), "f"(value));
}

template <>
__device__ __forceinline__ void st_na_global(const int4* ptr, const int4& value) {
    asm volatile(ST_NA_FUNC ".v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
}

__device__ __forceinline__ float log2f_approx(const float& x) {
    float ret;
    asm volatile("lg2.approx.f32 %0, %1;" : "=f"(ret) : "f"(x));
    return ret;
}

__device__ __forceinline__ float exp2f_approx(const float& x) {
    float ret;
    asm volatile("ex2.approx.f32 %0, %1;" : "=f"(ret) : "f"(x));
    return ret;
}
#endif

EP_DEVICE EP_FORCEINLINE int get_lane_id() {
#if defined(DEEPEP_XPU_NATIVE)
    return static_cast<int>(sycl::ext::oneapi::this_work_item::get_sub_group().get_local_linear_id());
#else
    int lane_id;
    asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
#endif
}

template <typename T>
EP_DEVICE EP_FORCEINLINE T warp_shuffle(T value, int src_lane_idx) {
#if defined(DEEPEP_XPU_NATIVE)
    return sycl::select_from_group(sycl::ext::oneapi::this_work_item::get_sub_group(), value, src_lane_idx);
#else
    return __shfl_sync(0xffffffff, value, src_lane_idx);
#endif
}

template <typename T>
EP_DEVICE EP_FORCEINLINE T warp_shuffle_xor(T value, int lane_mask) {
#if defined(DEEPEP_XPU_NATIVE)
    return sycl::permute_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), value, lane_mask);
#else
    return __shfl_xor_sync(0xffffffff, value, lane_mask);
#endif
}

EP_DEVICE EP_FORCEINLINE void warp_sync() {
#if defined(DEEPEP_XPU_NATIVE)
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
#else
    __syncwarp();
#endif
}

EP_DEVICE EP_FORCEINLINE bool warp_all(bool predicate) {
#if defined(DEEPEP_XPU_NATIVE)
    return sycl::all_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), predicate);
#else
    return __all_sync(0xffffffff, predicate);
#endif
}

EP_DEVICE EP_FORCEINLINE bool warp_any(bool predicate) {
#if defined(DEEPEP_XPU_NATIVE)
    return sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), predicate);
#else
    return __any_sync(0xffffffff, predicate);
#endif
}

EP_DEVICE EP_FORCEINLINE uint64_t device_clock64() {
#if defined(DEEPEP_XPU_NATIVE)
    using clock_scope = sycl::ext::oneapi::experimental::clock_scope;
    return sycl::ext::oneapi::experimental::clock<clock_scope::device>();
#else
    return clock64();
#endif
}

EP_DEVICE EP_FORCEINLINE uint32_t elect_one_sync() {
#if defined(DEEPEP_XPU_NATIVE)
    return get_lane_id() == 0;
#else
#ifndef DISABLE_SM90_FEATURES
    uint32_t pred = 0;
    asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        "      elect.sync %%rx|%%px, %1;\n"
        "@%%px mov.s32 %0, 1;\n"
        "}\n"
        : "+r"(pred)
        : "r"(0xffffffff));
    return pred;
#else
    return get_lane_id() == 0;
#endif
#endif
}

// TMA PTX instructions
#ifndef DISABLE_SM90_FEATURES

__device__ __forceinline__ void fence_barrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster; \n" ::);
}

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar_ptr, uint32_t arrive_count) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" ::"r"(arrive_count), "r"(mbar_int_ptr));
}

__device__ __forceinline__ void mbarrier_inval(uint64_t* mbar_ptr) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(mbar_int_ptr));
}

template <bool kWithMultiStages = false>
__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar_ptr, uint32_t& phase, int stage_idx = 0) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    const auto& wait = kWithMultiStages ? (phase >> stage_idx) & 1 : phase;
    asm volatile(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra DONE; \n\t"
        "bra     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}" ::"r"(mbar_int_ptr),
        "r"(wait),
        "r"(0x989680));
    phase ^= kWithMultiStages ? (1 << stage_idx) : 1;
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx(uint64_t* mbar_ptr, int num_bytes) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t" ::"r"(num_bytes), "r"(mbar_int_ptr));
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar_ptr) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0]; \n\t" ::"r"(mbar_int_ptr));
}

__device__ __forceinline__ void tma_store_fence() {
    asm volatile("fence.proxy.async.shared::cta;");
}

constexpr uint64_t kEvictFirst = 0x12f0000000000000;
constexpr uint64_t kEvictNormal = 0x1000000000000000;

__device__ __forceinline__ void tma_load_1d(
    const void* smem_ptr, const void* gmem_ptr, uint64_t* mbar_ptr, int num_bytes, bool evict_first = true) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    auto smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    const auto cache_hint = evict_first ? kEvictFirst : kEvictNormal;
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::"r"(smem_int_ptr),
        "l"(gmem_ptr),
        "r"(num_bytes),
        "r"(mbar_int_ptr),
        "l"(cache_hint)
        : "memory");
}

__device__ __forceinline__ void tma_store_1d(const void* smem_ptr, const void* gmem_ptr, int num_bytes, bool evict_first = true) {
    auto smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    const auto cache_hint = evict_first ? kEvictFirst : kEvictNormal;
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;\n" ::"l"(gmem_ptr),
                 "r"(smem_int_ptr),
                 "r"(num_bytes),
                 "l"(cache_hint)
                 : "memory");
    asm volatile("cp.async.bulk.commit_group;");
}

template <int N>
__device__ __forceinline__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}

#endif

template <typename dtype_t>
EP_HOST_DEVICE constexpr dtype_t ceil_div(dtype_t a, dtype_t b) {
    return (a + b - 1) / b;
}

template <typename dtype_t>
EP_HOST_DEVICE constexpr dtype_t align_up(dtype_t a, dtype_t b) {
    return ceil_div<dtype_t>(a, b) * b;
}

template <typename dtype_t>
EP_HOST_DEVICE constexpr dtype_t align_down(dtype_t a, dtype_t b) {
    return a / b * b;
}

EP_DEVICE EP_FORCEINLINE void get_channel_task_range(int num_tokens, int num_sms, int sm_id, int& token_start_idx, int& token_end_idx) {
    int num_tokens_per_sm = ceil_div(num_tokens, num_sms);
    token_start_idx = std::min(num_tokens_per_sm * sm_id, num_tokens);
    token_end_idx = std::min(token_start_idx + num_tokens_per_sm, num_tokens);
}

template <typename dtype_a_t, typename dtype_b_t>
EP_DEVICE EP_FORCEINLINE dtype_b_t pack2(const dtype_a_t& x, const dtype_a_t& y) {
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    dtype_b_t packed;
    auto unpacked_ptr = reinterpret_cast<dtype_a_t*>(&packed);
    unpacked_ptr[0] = x, unpacked_ptr[1] = y;
    return packed;
}

template <typename dtype_a_t, typename dtype_b_t>
EP_DEVICE EP_FORCEINLINE void unpack2(const dtype_b_t& packed, dtype_a_t& x, dtype_a_t& y) {
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    auto unpacked_ptr = reinterpret_cast<const dtype_a_t*>(&packed);
    x = unpacked_ptr[0], y = unpacked_ptr[1];
}

template <typename dtype_t>
EP_DEVICE EP_FORCEINLINE dtype_t broadcast(dtype_t& ptr, int src_lane_idx) {
    EP_STATIC_ASSERT(sizeof(dtype_t) % sizeof(int) == 0, "");
    auto send_int_values = reinterpret_cast<int*>(&ptr);
    int recv_int_values[sizeof(dtype_t) / sizeof(int)];
    #pragma unroll
    for (int i = 0; i < sizeof(dtype_t) / sizeof(int); ++i)
        recv_int_values[i] = warp_shuffle(send_int_values[i], src_lane_idx);
    return *reinterpret_cast<dtype_t*>(recv_int_values);
}

constexpr float kFP8Margin = 1e-4;
constexpr float kFinfoAmaxE4M3 = 448.0f;
constexpr float kFinfoAmaxInvE4M3 = 1 / 448.0f;

EP_DEVICE EP_FORCEINLINE float fast_pow2(int x) {
    // We can ensure `-126 <= x and x <= 127`
    uint32_t bits_x = (x + 127) << 23;
    return *reinterpret_cast<float*>(&bits_x);
}

EP_DEVICE EP_FORCEINLINE int fast_log2_ceil(float x) {
    auto bits_x = *reinterpret_cast<uint32_t*>(&x);
    auto exp_x = (bits_x >> 23) & 0xff;
    auto man_bits = bits_x & ((1 << 23) - 1);
    return exp_x - 127 + (man_bits != 0);
}

EP_DEVICE EP_FORCEINLINE void calculate_fp8_scales(float amax, float& scale, float& scale_inv, bool round_scale) {
    if (round_scale) {
        auto exp_scale_inv = fast_log2_ceil(amax * kFinfoAmaxInvE4M3);
        scale = fast_pow2(-exp_scale_inv);
        scale_inv = fast_pow2(exp_scale_inv);
    } else {
        scale_inv = amax * kFinfoAmaxInvE4M3;
        scale = kFinfoAmaxE4M3 / amax;
    }
}

template <bool kIsUE8M0, typename out_dtype_t = std::conditional_t<kIsUE8M0, uint8_t, float>>
EP_DEVICE EP_FORCEINLINE out_dtype_t extract_required_scale_format(float value) {
    if constexpr (kIsUE8M0) {
        return static_cast<uint8_t>((*reinterpret_cast<uint32_t*>(&value)) >> 23);
    } else {
        return value;
    }
}

#if defined(DEEPEP_XPU_NATIVE)
template <int kNumRanks, bool kSyncOnly = false>
EP_DEVICE EP_FORCEINLINE void barrier_block(int** barrier_signal_ptrs, int rank) {
    if constexpr (not kSyncOnly) {
        memory_fence();
    }

    const auto lane_id = get_lane_id();
    const int epoch = backend_primitives::load_acquire_system(barrier_signal_ptrs[rank] + rank) + 1;
    if (lane_id < kNumRanks) {
        backend_primitives::store_release_system(barrier_signal_ptrs[lane_id] + rank, epoch);
    }

    if (lane_id < kNumRanks) {
        int32_t* local_slot = barrier_signal_ptrs[rank] + lane_id;
        const auto start_time = device_clock64();
        while (true) {
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
            if (*local_slot == epoch) {
                break;
            }
            visa_spin_hint();
            if (device_clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                trap();
            }
        }
    }
    xpu_named_barrier::sync(0, 32);
}

EP_DEVICE EP_FORCEINLINE void named_bar_sync(int barrier_id, int num_threads) {
    if (num_threads > 255) {
        trap();
    }
    xpu_named_barrier::sync(static_cast<uint8_t>(barrier_id), static_cast<uint8_t>(num_threads));
}

EP_DEVICE EP_FORCEINLINE int atomic_cas_cta_acquire(int* addr, int x, int y) {
    auto ref = backend_primitives::atomic_ref_t<int, sycl::memory_scope::work_group>(*addr);
    int expected = x;
    ref.compare_exchange_strong(expected, y, sycl::memory_order::acq_rel, sycl::memory_order::acquire);
    return expected;
}

EP_DEVICE EP_FORCEINLINE int atomic_exch_cta_release(int* addr, int x) {
    return backend_primitives::atomic_ref_t<int, sycl::memory_scope::work_group>(*addr).exchange(x, sycl::memory_order::release);
}

EP_DEVICE EP_FORCEINLINE void acquire_lock(int* mutex) {
    while (atomic_cas_cta_acquire(mutex, 0, 1) != 0) {
    }
}

EP_DEVICE EP_FORCEINLINE void release_lock(int* mutex) {
    atomic_exch_cta_release(mutex, 0);
}

template <typename T>
struct ReduceSum {
    EP_DEVICE T operator()(T a, T b) const { return a + b; }
};
template <typename T>
struct ReduceMax {
    EP_DEVICE T operator()(T a, T b) const { return a > b ? a : b; }
};
template <typename T>
struct ReduceMin {
    EP_DEVICE T operator()(T a, T b) const { return a < b ? a : b; }
};
template <typename T>
struct ReduceAnd {
    EP_DEVICE T operator()(T a, T b) const { return a & b; }
};
template <typename T>
struct ReduceOr {
    EP_DEVICE T operator()(T a, T b) const { return a | b; }
};

template <int kNumLanesPerGroup, bool kIntergroupReduce, typename T, typename Op>
EP_DEVICE EP_FORCEINLINE T warp_reduce(T value, Op) {
    static_assert(kNumLanesPerGroup >= 1, "Invalid number of lanes");
    auto op = Op{};
    const auto lane_id = get_lane_id();
    if constexpr (kIntergroupReduce) {
        if constexpr (kNumLanesPerGroup <= 1)
            value = op(value, warp_shuffle_xor(value, 1));
        if constexpr (kNumLanesPerGroup <= 2)
            value = op(value, warp_shuffle_xor(value, 2));
        if constexpr (kNumLanesPerGroup <= 4)
            value = op(value, warp_shuffle_xor(value, 4));
        if constexpr (kNumLanesPerGroup <= 8)
            value = op(value, warp_shuffle_xor(value, 8));
        if constexpr (kNumLanesPerGroup <= 16)
            value = op(value, warp_shuffle_xor(value, 16));
    } else {
        auto grouped_shuffle_xor = [&](int mask) {
            const auto group_base = (lane_id / kNumLanesPerGroup) * kNumLanesPerGroup;
            const auto intra_group_lane = lane_id - group_base;
            return warp_shuffle(value, group_base + (intra_group_lane ^ mask));
        };
        if constexpr (kNumLanesPerGroup >= 32)
            value = op(value, grouped_shuffle_xor(16));
        if constexpr (kNumLanesPerGroup >= 16)
            value = op(value, grouped_shuffle_xor(8));
        if constexpr (kNumLanesPerGroup >= 8)
            value = op(value, grouped_shuffle_xor(4));
        if constexpr (kNumLanesPerGroup >= 4)
            value = op(value, grouped_shuffle_xor(2));
        if constexpr (kNumLanesPerGroup >= 2)
            value = op(value, grouped_shuffle_xor(1));
    }
    return value;
}
#else

__forceinline__ __device__ void named_bar_sync(int barrier_id, int num_threads) {
    asm volatile("bar.sync %0, %1;" ::"r"(barrier_id), "r"(num_threads));
}

template <int kNumRanks, bool kSyncOnly = false>
__forceinline__ __device__ void barrier_block(int** barrier_signal_ptrs, int rank) {
    auto thread_id = static_cast<int>(threadIdx.x);

    // For non-sync-only cases, the memory operations by other threads in the block must be visible to the `sys` scope
    if constexpr (not kSyncOnly) {
        memory_fence();
        __syncthreads();
    }

    // Add self-ranks, sub other ranks
    if (thread_id < kNumRanks) {
        atomicAdd_system(barrier_signal_ptrs[rank] + thread_id, FINISHED_SUM_TAG);
        atomicSub_system(barrier_signal_ptrs[thread_id] + rank, FINISHED_SUM_TAG);
    }
    EP_DEVICE_ASSERT(kNumRanks <= blockDim.x);

    // Check timeout
    auto start_time = device_clock64();
    while (true) {
        auto value = thread_id < kNumRanks ? ld_volatile_global(barrier_signal_ptrs[rank] + thread_id) : 0;
        if (warp_all(value <= 0))
            break;

        if (device_clock64() - start_time > NUM_TIMEOUT_CYCLES and thread_id < kNumRanks) {
            printf("DeepEP timeout check failed: rank = %d, thread = %d, value = %d)\n", rank, thread_id, value);
            trap();
        }
    }
    __syncthreads();
}

__forceinline__ __device__ int atomic_cas_cta_acquire(int* addr, int x, int y) {
    int ret;
    asm volatile("atom.acquire.cta.shared::cta.cas.b32 %0, [%1], %2, %3;" : "=r"(ret) : "l"(addr), "r"(x), "r"(y) : "memory");
    return ret;
}

__forceinline__ __device__ int atomic_exch_cta_release(int* addr, int x) {
    int ret;
    asm volatile("atom.release.cta.shared::cta.exch.b32 %0, [%1], %2;" : "=r"(ret) : "l"(addr), "r"(x) : "memory");
    return ret;
}

__forceinline__ __device__ void acquire_lock(int* mutex) {
    // To make later memory operations valid, we must use `acquire` for memory semantics
    while (atomic_cas_cta_acquire(mutex, 0, 1) != 0)
        ;
}

__forceinline__ __device__ void release_lock(int* mutex) {
    // To make previous memory operations visible to other threads, we must use `release` for memory semantics
    atomic_exch_cta_release(mutex, 0);
}

// Operation functors
template <typename T>
struct ReduceSum {
    __device__ T operator()(T a, T b) const { return a + b; }
};
template <typename T>
struct ReduceMax {
    __device__ T operator()(T a, T b) const { return a > b ? a : b; }
};
template <typename T>
struct ReduceMin {
    __device__ T operator()(T a, T b) const { return a < b ? a : b; }
};
template <typename T>
struct ReduceAnd {
    __device__ T operator()(T a, T b) const { return a & b; }
};
template <typename T>
struct ReduceOr {
    __device__ T operator()(T a, T b) const { return a | b; }
};

// Unified reduction function
template <int kNumLanesPerGroup, bool kIntergroupReduce, typename T, typename Op>
__forceinline__ __device__ T warp_reduce(T value, Op op) {
    EP_STATIC_ASSERT(kNumLanesPerGroup == 32 or kNumLanesPerGroup == 16 or kNumLanesPerGroup == 8 or kNumLanesPerGroup == 4 or
                         kNumLanesPerGroup == 2 or kNumLanesPerGroup == 1,
                     "Invalid number of lanes");
    constexpr uint32_t mask = 0xffffffff;
    if constexpr (kIntergroupReduce) {
        if constexpr (kNumLanesPerGroup <= 1)
            value = op(value, __shfl_xor_sync(mask, value, 1));
        if constexpr (kNumLanesPerGroup <= 2)
            value = op(value, __shfl_xor_sync(mask, value, 2));
        if constexpr (kNumLanesPerGroup <= 4)
            value = op(value, __shfl_xor_sync(mask, value, 4));
        if constexpr (kNumLanesPerGroup <= 8)
            value = op(value, __shfl_xor_sync(mask, value, 8));
        if constexpr (kNumLanesPerGroup <= 16)
            value = op(value, __shfl_xor_sync(mask, value, 16));
    } else {
        if constexpr (kNumLanesPerGroup >= 32)
            value = op(value, __shfl_xor_sync(mask, value, 16));
        if constexpr (kNumLanesPerGroup >= 16)
            value = op(value, __shfl_xor_sync(mask, value, 8));
        if constexpr (kNumLanesPerGroup >= 8)
            value = op(value, __shfl_xor_sync(mask, value, 4));
        if constexpr (kNumLanesPerGroup >= 4)
            value = op(value, __shfl_xor_sync(mask, value, 2));
        if constexpr (kNumLanesPerGroup >= 2)
            value = op(value, __shfl_xor_sync(mask, value, 1));
    }
    return value;
}
#endif

// Convenience aliases
template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
EP_DEVICE EP_FORCEINLINE T warp_reduce_sum(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceSum<T>{});
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
EP_DEVICE EP_FORCEINLINE T warp_reduce_max(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceMax<T>{});
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
EP_DEVICE EP_FORCEINLINE T warp_reduce_min(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceMin<T>{});
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
EP_DEVICE EP_FORCEINLINE T warp_reduce_and(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceAnd<T>{});
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
EP_DEVICE EP_FORCEINLINE T warp_reduce_or(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceOr<T>{});
}

}  // namespace deep_ep
