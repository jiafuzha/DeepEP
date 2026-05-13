#pragma once

#include "configs.cuh"
#include "exception.cuh"

namespace deep_ep::launch {

#if defined(DEEPEP_XPU_NATIVE)
struct LaunchConfig {
    int num_sms;
    int num_threads;
    cudaStream_t stream;
    int dynamic_smem_bytes;
};

inline void configure_launch_config(LaunchConfig& cfg, int num_sms, int num_threads, cudaStream_t stream) {
    cfg = {num_sms, num_threads, stream, 0};
}

template <typename KernelT, typename... Args>
inline void launch_kernel(LaunchConfig*, KernelT, Args...) {
    throw EPException("XPU", __FILE__, __LINE__, "Generic kernel launch is not implemented for the XPU native build");
}
#elif !defined(DISABLE_SM90_FEATURES)
inline void configure_launch_config(cudaLaunchConfig_t& cfg, cudaLaunchAttribute (&attr)[2], int num_sms, int num_threads, cudaStream_t stream) {
    cfg = {(num_sms), (num_threads), 0, stream, nullptr, 0};
    attr[0].id = cudaLaunchAttributeCooperative;
    attr[0].val.cooperative = 1;
    attr[1].id = cudaLaunchAttributeClusterDimension;
    attr[1].val.clusterDim.x = (num_sms % 2 == 0 ? 2 : 1);
    attr[1].val.clusterDim.y = 1;
    attr[1].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 2;
}

template <typename KernelT, typename... Args>
inline void launch_kernel(cudaLaunchConfig_t* config, KernelT kernel, Args... args) {
    CUDA_CHECK(cudaLaunchKernelEx(config, kernel, args...));
}
#else
template <typename KernelT, typename... Args>
inline void launch_kernel(int num_sms, int num_threads, cudaStream_t stream, KernelT kernel, Args... args) {
    kernel<<<num_sms, num_threads, 0, stream>>>(args...);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        EPException cuda_exception("CUDA", __FILE__, __LINE__, cudaGetErrorString(e));
        fprintf(stderr, "%s\n", cuda_exception.what());
        throw cuda_exception;
    }
}
#endif

}  // namespace deep_ep::launch

#ifndef SETUP_LAUNCH_CONFIG
#if defined(DEEPEP_XPU_NATIVE)
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream) \
    deep_ep::launch::LaunchConfig cfg;                    \
    deep_ep::launch::configure_launch_config(cfg, (num_sms), (num_threads), (stream))
#elif !defined(DISABLE_SM90_FEATURES)
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream)                       \
    cudaLaunchConfig_t cfg;                                                     \
    cudaLaunchAttribute attr[2];                                                \
    deep_ep::launch::configure_launch_config(cfg, attr, (num_sms), (num_threads), (stream))
#else
#define SETUP_LAUNCH_CONFIG(sms, threads, stream) \
    int __num_sms = (sms);                        \
    int __num_threads = (threads);                \
    auto __stream = (stream)
#endif
#endif

#ifndef LAUNCH_KERNEL
#if defined(DEEPEP_XPU_NATIVE)
#define LAUNCH_KERNEL(config, kernel, ...) deep_ep::launch::launch_kernel(config, kernel, ##__VA_ARGS__)
#elif !defined(DISABLE_SM90_FEATURES)
#define LAUNCH_KERNEL(config, kernel, ...) deep_ep::launch::launch_kernel(config, kernel, ##__VA_ARGS__)
#else
#define LAUNCH_KERNEL(config, kernel, ...) deep_ep::launch::launch_kernel(__num_sms, __num_threads, __stream, kernel, ##__VA_ARGS__)
#endif
#endif

#ifndef SET_SHARED_MEMORY_FOR_TMA
#if defined(DEEPEP_XPU_NATIVE)
#define SET_SHARED_MEMORY_FOR_TMA(kernel) void()
#elif !defined(DISABLE_SM90_FEATURES)
#define SET_SHARED_MEMORY_FOR_TMA(kernel)                                                                                \
    EP_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess); \
    cfg.dynamicSmemBytes = smem_size;
#else
#define SET_SHARED_MEMORY_FOR_TMA(kernel) void()
#endif
#endif

#define SWITCH_RANKS(case_macro)                           \
    switch (num_ranks) {                                   \
        case 2:                                            \
            case_macro(2);                                 \
        case 4:                                            \
            case_macro(4);                                 \
        case 8:                                            \
            case_macro(8);                                 \
        default:                                           \
            EP_HOST_ASSERT(false and "Unsupported ranks"); \
    }                                                      \
    while (false)

#define SWITCH_RDMA_RANKS(case_macro)                           \
    switch (num_ranks / NUM_MAX_NVL_PEERS) {                    \
        case 2:                                                 \
            case_macro(2);                                      \
        case 3:                                                 \
            case_macro(3);                                      \
        case 4:                                                 \
            case_macro(4);                                      \
        case 6:                                                 \
            case_macro(6);                                      \
        case 8:                                                 \
            case_macro(8);                                      \
        case 12:                                                \
            case_macro(12);                                     \
        case 16:                                                \
            case_macro(16);                                     \
        case 18:                                                \
            case_macro(18);                                     \
        case 20:                                                \
            case_macro(20);                                     \
        default:                                                \
            EP_HOST_ASSERT(false and "Unsupported RDMA ranks"); \
    }                                                           \
    while (false)

#define SWITCH_RANKS_WITH_DTYPE(dtype, case_macro)         \
    switch (num_ranks) {                                   \
        case 2:                                            \
            case_macro(dtype, 2);                          \
        case 4:                                            \
            case_macro(dtype, 4);                          \
        case 8:                                            \
            case_macro(dtype, 8);                          \
        default:                                           \
            EP_HOST_ASSERT(false and "Unsupported ranks"); \
    }                                                      \
    while (false)

#define SWITCH_TYPES(case_macro)                          \
    switch (type) {                                       \
        case CUDA_R_16BF:                                 \
            case_macro(nv_bfloat16);                      \
        default:                                          \
            EP_HOST_ASSERT(false and "Unsupported type"); \
    }                                                     \
    while (false)

#define SWITCH_HIDDEN(case_macro)                           \
    switch (hidden) {                                       \
        case 2048:                                          \
            case_macro(2048);                               \
        case 2560:                                          \
            case_macro(2560);                               \
        case 3072:                                          \
            case_macro(3072); /* for gpt-oss */             \
        case 4096:                                          \
            case_macro(4096);                               \
        case 5120:                                          \
            case_macro(5120);                               \
        case 6144:                                          \
            case_macro(6144); /* For qwen3 coder */         \
        case 7168:                                          \
            case_macro(7168);                               \
        case 8192:                                          \
            case_macro(8192);                               \
        default:                                            \
            EP_HOST_ASSERT(false and "Unsupported hidden"); \
    }                                                       \
    while (false)
