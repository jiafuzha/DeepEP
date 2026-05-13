#include <memory>

#include "kernels/exception.cuh"
#include "runtime.hpp"

namespace deep_ep {

struct EventHandle {
    std::shared_ptr<RuntimeEvent> event;

    EventHandle() {
        event = std::make_shared<RuntimeEvent>(runtime_device_type());
        event->record(unwrap_stream(get_current_stream()));
    }

    explicit EventHandle(const RuntimeStream& stream) {
        event = std::make_shared<RuntimeEvent>(runtime_device_type());
        event->record(unwrap_stream(stream));
    }

    EventHandle(const EventHandle& other) = default;

    void current_stream_wait() const { unwrap_stream(get_current_stream()).wait(*event); }
};

inline RuntimeEvent create_event(const RuntimeStream& s) {
    auto event = RuntimeEvent(runtime_device_type());
    event.record(unwrap_stream(s));
    return event;
}

inline void stream_wait(const RuntimeStream& s_0, const RuntimeStream& s_1) {
    EP_HOST_ASSERT(s_0.id() != s_1.id());
    unwrap_stream(s_0).wait(create_event(s_1));
}

inline void stream_wait(const RuntimeStream& s, const EventHandle& event) {
    unwrap_stream(s).wait(*event.event);
}

}  // namespace deep_ep
