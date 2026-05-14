#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>

#include "kernels/configs.cuh"
#include "xpu_native_runtime.hpp"

namespace py = pybind11;

namespace deep_ep {
size_t get_low_latency_rdma_size_hint(int num_max_dispatch_tokens_per_rank, int hidden, int num_ranks, int num_experts);
}

namespace {

bool is_sm90_compiled() {
    return false;
}

bool supports_xpu_native_intranode() {
    return false;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<deep_ep::Config>(m, "Config")
        .def(py::init<int, int, int, int, int>(),
              py::arg("num_sms"),
              py::arg("num_max_nvl_chunked_send_tokens"),
              py::arg("num_max_nvl_chunked_recv_tokens"),
              py::arg("num_max_rdma_chunked_send_tokens") = 6,
              py::arg("num_max_rdma_chunked_recv_tokens") = 128)
        .def_readonly("num_sms", &deep_ep::Config::num_sms)
        .def_readonly("num_max_nvl_chunked_send_tokens", &deep_ep::Config::num_max_nvl_chunked_send_tokens)
        .def_readonly("num_max_nvl_chunked_recv_tokens", &deep_ep::Config::num_max_nvl_chunked_recv_tokens)
        .def_readonly("num_max_rdma_chunked_send_tokens", &deep_ep::Config::num_max_rdma_chunked_send_tokens)
        .def_readonly("num_max_rdma_chunked_recv_tokens", &deep_ep::Config::num_max_rdma_chunked_recv_tokens)
        .def("get_nvl_buffer_size_hint", &deep_ep::Config::get_nvl_buffer_size_hint)
        .def("get_rdma_buffer_size_hint", &deep_ep::Config::get_rdma_buffer_size_hint);

    py::class_<deep_ep::EventHandle>(m, "EventHandle")
        .def(py::init<>())
        .def("current_stream_wait", &deep_ep::EventHandle::current_stream_wait);

    py::class_<deep_ep::Buffer>(m, "Buffer")
        .def(py::init<int, int, int64_t, int64_t, bool, bool, bool, bool>())
        .def("is_available", &deep_ep::Buffer::is_available)
        .def("get_num_rdma_ranks", &deep_ep::Buffer::get_num_rdma_ranks)
        .def("get_rdma_rank", &deep_ep::Buffer::get_rdma_rank)
        .def("get_root_rdma_rank", &deep_ep::Buffer::get_root_rdma_rank)
        .def("get_local_device_id", &deep_ep::Buffer::get_local_device_id)
        .def("get_local_ipc_handle", &deep_ep::Buffer::get_local_ipc_handle)
        .def("get_local_nvshmem_unique_id", &deep_ep::Buffer::get_local_nvshmem_unique_id)
        .def("get_local_buffer_tensor", &deep_ep::Buffer::get_local_buffer_tensor)
        .def("get_comm_stream", &deep_ep::Buffer::get_comm_stream)
        .def("sync", &deep_ep::Buffer::sync)
        .def("destroy", &deep_ep::Buffer::destroy)
        .def("get_dispatch_layout", &deep_ep::Buffer::get_dispatch_layout)
        .def("intranode_dispatch", &deep_ep::Buffer::intranode_dispatch)
        .def("intranode_combine", &deep_ep::Buffer::intranode_combine)
        .def("low_latency_update_mask_buffer", &deep_ep::Buffer::low_latency_update_mask_buffer)
        .def("low_latency_query_mask_buffer", &deep_ep::Buffer::low_latency_query_mask_buffer)
        .def("low_latency_clean_mask_buffer", &deep_ep::Buffer::low_latency_clean_mask_buffer)
        .def("clean_low_latency_buffer", &deep_ep::Buffer::clean_low_latency_buffer)
        .def("get_next_low_latency_combine_buffer", &deep_ep::Buffer::get_next_low_latency_combine_buffer);

    m.def("is_sm90_compiled", &is_sm90_compiled);
    m.def("supports_xpu_native_intranode", &supports_xpu_native_intranode);
    m.def("get_low_latency_rdma_size_hint", &deep_ep::get_low_latency_rdma_size_hint);
    m.attr("topk_idx_t") =
        py::reinterpret_borrow<py::object>((PyObject*)torch::getTHPDtype(c10::CppTypeToScalarType<deep_ep::topk_idx_t>::value));
}
