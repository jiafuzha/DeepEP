#pragma once

#include <sycl/sycl.hpp>
#include "configs.dp.hpp"
#include "exception.dp.hpp"

#ifndef DEEP_EP_LAUNCH_CONFIG_T
#define DEEP_EP_LAUNCH_CONFIG_T
namespace deep_ep {

struct LaunchConfig {
    int num_sms = 1;
    int num_threads = 1;
    std::size_t shared_bytes = 0;
    dpct::queue_ptr stream = nullptr;
};

[[noreturn]] inline void throw_untranslated_kernel_launch(const char* kernel_name, const char* file, int line) {
    throw EPException("SYCL", file, line, std::string("Untranslated CUDA-style kernel launch: ") + kernel_name);
}

}  // namespace deep_ep
#endif

#ifndef SETUP_LAUNCH_CONFIG
#define SETUP_LAUNCH_CONFIG(sms, threads, stream) \
    ::deep_ep::LaunchConfig cfg{(sms), (threads), 0u, (stream)}
#endif

#ifndef LAUNCH_KERNEL
#define LAUNCH_KERNEL(config, kernel, ...)                            \
    do {                                                              \
        (void)(config);                                               \
        ::deep_ep::throw_untranslated_kernel_launch(#kernel, __FILE__, __LINE__); \
    } while (0)
#endif

#ifndef SET_SHARED_MEMORY_FOR_TMA
#define SET_SHARED_MEMORY_FOR_TMA(kernel) \
    do {                                  \
        (void)(kernel);                   \
    } while (0)
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
        case dpct::library_data_t::real_bfloat16:         \
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
