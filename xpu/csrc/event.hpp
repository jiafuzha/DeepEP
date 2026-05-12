#include <ATen/xpu/XPUContext.h>
#include <c10/core/Event.h>
#include <c10/xpu/XPUStream.h>

#include <memory>
#include <sycl/sycl.hpp>

#include "kernels/exception.dp.hpp"

namespace deep_ep {

struct EventHandle {
    std::shared_ptr<c10::Event> event;

    EventHandle() {
        event = std::make_shared<c10::Event>(c10::kXPU);
        event->record(c10::xpu::getCurrentXPUStream());
    }

    explicit EventHandle(const c10::xpu::XPUStream& stream) {
        event = std::make_shared<c10::Event>(c10::kXPU);
        event->record(stream);
    }

    EventHandle(const EventHandle& other) = default;

    void current_stream_wait() const { c10::xpu::getCurrentXPUStream().unwrap().wait(*event); }
};

inline c10::Event create_event(const c10::xpu::XPUStream& s) {
    auto event = c10::Event(c10::kXPU);
    event.record(s);
    return event;
}

inline void stream_wait(const c10::xpu::XPUStream& s_0, const c10::xpu::XPUStream& s_1) {
    EP_HOST_ASSERT(s_0.id() != s_1.id());
    s_0.unwrap().wait(create_event(s_1));
}

inline void stream_wait(const c10::xpu::XPUStream& s, const EventHandle& event) {
    s.unwrap().wait(*event.event);
}

}  // namespace deep_ep
