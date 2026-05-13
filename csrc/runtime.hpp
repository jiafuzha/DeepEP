#pragma once

#include <c10/core/Event.h>
#include <c10/core/Stream.h>
#include <torch/types.h>

#if defined(DEEPEP_XPU_NATIVE)
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#else
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#endif

#include "kernels/exception.cuh"

namespace deep_ep {

#if defined(DEEPEP_XPU_NATIVE)
using RuntimeStream = c10::Stream;
using RuntimeEvent = c10::Event;

inline RuntimeStream unwrap_stream(const RuntimeStream& stream) {
    return stream;
}

inline RuntimeStream get_current_stream() {
    return c10::xpu::getCurrentXPUStream().unwrap();
}

inline RuntimeStream get_stream_from_pool(bool high_priority = true) {
    return c10::xpu::getStreamFromPool(high_priority).unwrap();
}

inline void set_current_stream(const RuntimeStream& stream) {
    c10::xpu::setCurrentXPUStream(c10::xpu::XPUStream(stream));
}

inline c10::DeviceType runtime_device_type() {
    return c10::kXPU;
}

inline torch::Device runtime_device() {
    return torch::Device(torch::kXPU, c10::xpu::current_device());
}

inline cudaDataType_t runtime_scalar_type_to_data_type(torch::ScalarType scalar_type) {
    switch (scalar_type) {
        case torch::kBFloat16:
            return CUDA_R_16BF;
        case torch::kFloat8_e4m3fn:
            return CUDA_R_8F_E4M3;
        default:
            throw std::runtime_error("Unsupported XPU scalar type for native kernel dispatch");
    }
}
#else
using RuntimeStream = at::cuda::CUDAStream;
using RuntimeEvent = c10::Event;

inline c10::Stream unwrap_stream(const RuntimeStream& stream) {
    return stream.unwrap();
}

inline RuntimeStream get_current_stream() {
    return at::cuda::getCurrentCUDAStream();
}

inline RuntimeStream get_stream_from_pool(bool high_priority = true) {
    return at::cuda::getStreamFromPool(high_priority);
}

inline void set_current_stream(const RuntimeStream& stream) {
    at::cuda::setCurrentCUDAStream(stream);
}

inline c10::DeviceType runtime_device_type() {
    return c10::kCUDA;
}

inline torch::Device runtime_device() {
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    return torch::Device(torch::kCUDA, device_id);
}

inline cudaDataType_t runtime_scalar_type_to_data_type(torch::ScalarType scalar_type) {
    return at::cuda::ScalarTypeToCudaDataType(scalar_type);
}
#endif

inline torch::TensorOptions runtime_tensor_options(torch::ScalarType dtype) {
    return torch::TensorOptions().dtype(dtype).device(runtime_device());
}

}  // namespace deep_ep
