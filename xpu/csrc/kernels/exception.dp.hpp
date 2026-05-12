#pragma once

#include <sycl/sycl.hpp>
#include <cstdio>
#include <string>
#include <exception>

#include "configs.dp.hpp"

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
#define CUDA_CHECK(cmd)                                                        \
    do {                                                                       \
        try {                                                                  \
            (void)(cmd);                                                       \
        } catch (const sycl::exception& e) {                                   \
            throw EPException("SYCL", __FILE__, __LINE__, std::string(e.what())); \
        }                                                                      \
    } while (0)
#endif

#ifndef CU_CHECK
/*
DPCT1001:7: The statement could not be removed.
*/
/*
DPCT1000:8: Error handling if-stmt was detected but could not be rewritten.
*/
/*
DPCT1009:10: SYCL reports errors using exceptions and does not use error codes. Please replace the "get_error_string_dummy(...)" with a real
error-handling function.
*/
#define CU_CHECK(cmd)                                                             \
    do {                                                                          \
        int e = (cmd);                                                            \
        if (e != 0) {                                                             \
            throw EPException("SYCL", __FILE__, __LINE__, std::string(dpct::get_error_string_dummy(e))); \
        }                                                                         \
    } while (0)
#endif

#ifndef EP_HOST_ASSERT
#define EP_HOST_ASSERT(cond)                                           \
    do {                                                               \
        if (not(cond)) {                                               \
            throw EPException("Assertion", __FILE__, __LINE__, #cond); \
        }                                                              \
    } while (0)
#endif

#ifndef EP_DEVICE_ASSERT
#define EP_DEVICE_ASSERT(cond)                      \
    do {                                            \
        if (not(cond)) {                            \
            __builtin_trap();                       \
        }                                           \
    } while (0)
#endif
