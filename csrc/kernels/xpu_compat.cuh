#pragma once

#include <c10/core/Stream.h>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <limits>

#define EP_DEVICE inline
#define EP_HOST_DEVICE inline
#define EP_FORCEINLINE __attribute__((always_inline))

struct alignas(8) int2 {
    int x;
    int y;
};

struct alignas(16) int4 {
    int x;
    int y;
    int z;
    int w;
};

struct float2 {
    float x;
    float y;
};

using nv_bfloat16 = sycl::ext::oneapi::bfloat16;

struct alignas(4) __nv_bfloat162 {
    nv_bfloat16 x;
    nv_bfloat16 y;

    __nv_bfloat162() = default;
    __nv_bfloat162(nv_bfloat16 x_val, nv_bfloat16 y_val) : x(x_val), y(y_val) {}
};

using nv_bfloat162 = __nv_bfloat162;

inline float2 __bfloat1622float2(const __nv_bfloat162& value) {
    return {static_cast<float>(value.x), static_cast<float>(value.y)};
}

using __nv_fp8_interpretation_t = int;
using __nv_fp8_storage_t = uint8_t;
struct alignas(4) __nv_fp8x4_e4m3 {
    uint8_t raw[4];
};

#define __NV_E4M3 0
#define __NV_E5M2 1
#define CUDART_ZERO_BF16 ::sycl::ext::oneapi::bfloat16(0.0f)
#define CUDART_INF_BF16 ::sycl::ext::oneapi::bfloat16(::std::numeric_limits<float>::infinity())

using cudaStream_t = c10::Stream;

enum cudaDataType_t : int {
    CUDA_R_16BF = 0,
    CUDA_R_8F_E4M3 = 1,
};

using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;

inline const char* cudaGetErrorString(cudaError_t) {
    return "CUDA runtime is unavailable in the XPU native build";
}

struct cudaIpcMemHandle_t {
    alignas(64) unsigned char reserved[64];
};

struct CUmemFabricHandle {
    alignas(64) unsigned char reserved[64];
};

using CUresult = int;
constexpr CUresult CUDA_SUCCESS = 0;

inline void cuGetErrorString(CUresult, const char** error_str) {
    if (error_str != nullptr) {
        *error_str = "CUDA driver is unavailable in the XPU native build";
    }
}
