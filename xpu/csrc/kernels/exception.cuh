#pragma once

#include <cstdlib>
#include <exception>
#include <string>

#include "configs.cuh"

#ifndef EP_STATIC_ASSERT
#define EP_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

class EPException : public std::exception {
private:
    std::string message = {};

public:
    explicit EPException(const char* name, const char* file, const int line, const std::string& error) {
        message = std::string("Failed: ") + name + " error " + file + ":" + std::to_string(line) + " '" + error + "'";
    }

    const char* what() const noexcept override { return message.c_str(); }
};

#ifndef CUDA_CHECK
#if defined(DEEPEP_USE_XPU)
#define CUDA_CHECK(cmd)                                                                \
    do {                                                                               \
        (void)(cmd);                                                                   \
        throw EPException("XPU", __FILE__, __LINE__, "CUDA runtime call is unavailable in XPU build"); \
    } while (0)
#else
#define CUDA_CHECK(cmd)                                                           \
    do {                                                                          \
        cudaError_t e = (cmd);                                                    \
        if (e != cudaSuccess) {                                                   \
            throw EPException("CUDA", __FILE__, __LINE__, cudaGetErrorString(e)); \
        }                                                                         \
    } while (0)
#endif
#endif

#ifndef CU_CHECK
#if defined(DEEPEP_USE_XPU)
#define CU_CHECK(cmd)                                                                  \
    do {                                                                               \
        (void)(cmd);                                                                   \
        throw EPException("XPU", __FILE__, __LINE__, "CUDA driver call is unavailable in XPU build"); \
    } while (0)
#else
#define CU_CHECK(cmd)                                                            \
    do {                                                                         \
        CUresult e = (cmd);                                                      \
        if (e != CUDA_SUCCESS) {                                                 \
            const char* error_str = NULL;                                        \
            cuGetErrorString(e, &error_str);                                     \
            throw EPException("CU", __FILE__, __LINE__, std::string(error_str)); \
        }                                                                        \
    } while (0)
#endif
#endif

#ifndef EP_HOST_ASSERT
#define EP_HOST_ASSERT(cond)                                           \
    do {                                                               \
        if (not(cond)) {                                               \
            throw EPException("Assertion", __FILE__, __LINE__, #cond); \
        }                                                              \
    } while (0)
#endif

#ifndef EP_UNSUPPORTED_XPU
#define EP_UNSUPPORTED_XPU(feature)                                                                  \
    do {                                                                                             \
        throw EPException("XPU", __FILE__, __LINE__, std::string(feature) + " is not implemented for XPU yet"); \
    } while (0)
#endif

#ifndef EP_DEVICE_ASSERT
#if defined(DEEPEP_USE_XPU)
#define EP_DEVICE_ASSERT(cond)                                                           \
    do {                                                                                 \
        if (not(cond)) {                                                                 \
            printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
            abort();                                                                      \
        }                                                                                \
    } while (0)
#else
#define EP_DEVICE_ASSERT(cond)                                                             \
    do {                                                                                   \
        if (not(cond)) {                                                                   \
            printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
            asm("trap;");                                                                  \
        }                                                                                  \
    } while (0)
#endif
#endif
