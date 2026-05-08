#include <stdexcept>

#include <torch/extension.h>

struct Config {
    int num_sms;
    int ctas_per_sm;
    int nvl_chunk_size;
    int rdma_channels;
    int bytes_per_element;

    Config(int num_sms_, int ctas_per_sm_, int nvl_chunk_size_, int rdma_channels_, int bytes_per_element_)
        : num_sms(num_sms_),
          ctas_per_sm(ctas_per_sm_),
          nvl_chunk_size(nvl_chunk_size_),
          rdma_channels(rdma_channels_),
          bytes_per_element(bytes_per_element_) {}
};

struct EventHandle {
    void current_stream_wait() const {}
};

struct Buffer {
    Buffer(int, int, int64_t, int64_t, bool, bool, bool, bool) {
        throw std::runtime_error("XPU native runtime is not implemented yet");
    }
};

bool is_sm90_compiled() {
    return false;
}

int64_t get_low_latency_rdma_size_hint(int num_max_dispatch_tokens_per_rank, int hidden, int, int) {
    return static_cast<int64_t>(num_max_dispatch_tokens_per_rank) * hidden * 2;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("is_sm90_compiled", &is_sm90_compiled);
    m.def("get_low_latency_rdma_size_hint", &get_low_latency_rdma_size_hint);
    m.attr("topk_idx_t") = pybind11::module::import("torch").attr("int64");

    pybind11::class_<Config>(m, "Config")
        .def(pybind11::init<int, int, int, int, int>());

    pybind11::class_<EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &EventHandle::current_stream_wait);

    pybind11::class_<Buffer>(m, "Buffer")
        .def(pybind11::init<int, int, int64_t, int64_t, bool, bool, bool, bool>());
}
