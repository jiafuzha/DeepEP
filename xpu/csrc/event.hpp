#if defined(DEEPEP_USE_XPU)
#include <ATen/xpu/XPUContext.h>
#else
#include <ATen/cuda/CUDAContext.h>
#endif

#include <memory>

#include "kernels/exception.cuh"

namespace deep_ep {

namespace backend {

#if defined(DEEPEP_USE_XPU)
using Stream = at::xpu::XPUStream;
constexpr auto kDeviceType = torch::kXPU;

inline Stream get_current_stream() {
    return at::xpu::getCurrentXPUStream();
}

inline Stream get_stream_from_pool(const bool high_priority) {
    return at::xpu::getStreamFromPool(high_priority);
}

inline void set_current_stream(const Stream& stream) {
    at::xpu::setCurrentXPUStream(stream);
}

#else
using Stream = at::cuda::CUDAStream;
constexpr auto kDeviceType = torch::kCUDA;

inline Stream get_current_stream() {
    return at::cuda::getCurrentCUDAStream();
}

inline Stream get_stream_from_pool(const bool high_priority) {
    return at::cuda::getStreamFromPool(high_priority);
}

inline void set_current_stream(const Stream& stream) {
    at::cuda::setCurrentCUDAStream(stream);
}

#endif

}  // namespace backend

struct EventHandle {
    std::shared_ptr<torch::Event> event;

    EventHandle() {
        event = std::make_shared<torch::Event>(backend::kDeviceType);
        event->record(backend::get_current_stream());
    }

    explicit EventHandle(const backend::Stream& stream) {
        event = std::make_shared<torch::Event>(backend::kDeviceType);
        event->record(stream);
    }

    EventHandle(const EventHandle& other) = default;

    void current_stream_wait() const { backend::get_current_stream().unwrap().wait(*event); }
};

torch::Event create_event(const backend::Stream& s) {
    auto event = torch::Event(backend::kDeviceType);
    event.record(s);
    return event;
}

void stream_wait(const backend::Stream& s_0, const backend::Stream& s_1) {
    EP_HOST_ASSERT(s_0.id() != s_1.id());
    s_0.unwrap().wait(create_event(s_1));
}

void stream_wait(const backend::Stream& s, const EventHandle& event) {
    s.unwrap().wait(*event.event);
}

}  // namespace deep_ep
