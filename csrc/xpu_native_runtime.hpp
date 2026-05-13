#pragma once

#include <pybind11/pybind11.h>
#include <torch/types.h>

#include <cstdint>
#include <vector>

#include "event.hpp"
#include "runtime.hpp"

namespace deep_ep {

struct Buffer {
private:
    bool low_latency_mode = false;
    int64_t num_nvl_bytes = 0;
    int64_t num_rdma_bytes = 0;
    bool enable_shrink = false;

    void* rdma_buffer_ptr = nullptr;
    int* mask_buffer_ptr = nullptr;
    int* sync_buffer_ptr = nullptr;

    int device_id = 0;
    int num_device_sms = 0;
    int rank = 0;
    int rdma_rank = 0;
    int nvl_rank = 0;
    int num_ranks = 0;
    int num_rdma_ranks = 1;
    int num_nvl_ranks = 1;

    RuntimeStream comm_stream;
    bool available = false;
    bool explicitly_destroy = false;
    bool destroyed = false;
    bool transport_initialized = false;

    void sync_internode_runtime(const pybind11::object& root_unique_id_obj);
    void destroy_internode_resources();

public:
    Buffer(int rank,
           int num_ranks,
           int64_t num_nvl_bytes,
           int64_t num_rdma_bytes,
           bool low_latency_mode,
           bool explicitly_destroy,
           bool enable_shrink,
           bool use_fabric);

    ~Buffer() noexcept(false);

    bool is_available() const;
    int get_num_rdma_ranks() const;
    int get_rdma_rank() const;
    int get_root_rdma_rank(bool global) const;
    int get_local_device_id() const;
    pybind11::object get_local_ipc_handle() const;
    pybind11::bytearray get_local_nvshmem_unique_id() const;
    torch::Tensor get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset, bool use_rdma_buffer) const;
    torch::Stream get_comm_stream() const;
    void sync(const std::vector<int>& device_ids, const std::vector<pybind11::object>& all_gathered_handles, const pybind11::object& root_unique_id_obj);
    void destroy();

    void low_latency_update_mask_buffer(int rank_to_mask, bool mask);
    void low_latency_query_mask_buffer(const torch::Tensor& mask_status);
    void low_latency_clean_mask_buffer();
    torch::Tensor get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) const;
};

}  // namespace deep_ep
