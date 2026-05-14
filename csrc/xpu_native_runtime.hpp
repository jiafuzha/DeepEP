#pragma once

#include <pybind11/pybind11.h>
#include <torch/types.h>

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include <level_zero/ze_api.h>

#include "config.hpp"
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

    void* buffer_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
    int* barrier_signal_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
    void** buffer_ptrs_gpu = nullptr;
    int** barrier_signal_ptrs_gpu = nullptr;

    ze_context_handle_t ze_context = nullptr;
    ze_device_handle_t ze_device = nullptr;
    ze_ipc_mem_handle_t nvl_ipc_handle = {};
    bool nvl_ipc_handle_ready = false;
    size_t nvl_allocation_bytes = 0;

    int device_id = 0;
    int num_device_sms = 0;
    int rank = 0;
    int rdma_rank = 0;
    int nvl_rank = 0;
    int num_ranks = 0;
    int num_rdma_ranks = 1;
    int num_nvl_ranks = 1;

    std::optional<RuntimeStream> comm_stream;
    bool available = false;
    bool explicitly_destroy = false;
    bool destroyed = false;
    bool transport_initialized = false;

    void sync_internode_runtime(const pybind11::object& root_unique_id_obj);
    void sync_intranode_handles(const std::vector<int>& device_ids, const std::vector<pybind11::object>& all_gathered_handles);
    void destroy_intranode_resources();
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
    void clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts);
    torch::Tensor get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) const;

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>> get_dispatch_layout(
        const torch::Tensor& topk_idx,
        int num_experts,
        std::optional<EventHandle>& previous_event,
        bool async,
        bool allocate_on_comm_stream);

    std::tuple<torch::Tensor,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::vector<int>,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               std::optional<EventHandle>>
    intranode_dispatch(const torch::Tensor& x,
                       const std::optional<torch::Tensor>& x_scales,
                       const std::optional<torch::Tensor>& topk_idx,
                       const std::optional<torch::Tensor>& topk_weights,
                       const std::optional<torch::Tensor>& num_tokens_per_rank,
                       const torch::Tensor& is_token_in_rank,
                       const std::optional<torch::Tensor>& num_tokens_per_expert,
                       int cached_num_recv_tokens,
                       const std::optional<torch::Tensor>& cached_rank_prefix_matrix,
                       const std::optional<torch::Tensor>& cached_channel_prefix_matrix,
                       int expert_alignment,
                       int num_worst_tokens,
                       const Config& config,
                       std::optional<EventHandle>& previous_event,
                       bool async,
                       bool allocate_on_comm_stream);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> intranode_combine(
        const torch::Tensor& x,
        const std::optional<torch::Tensor>& topk_weights,
        const std::optional<torch::Tensor>& bias_0,
        const std::optional<torch::Tensor>& bias_1,
        const torch::Tensor& src_idx,
        const torch::Tensor& rank_prefix_matrix,
        const torch::Tensor& channel_prefix_matrix,
        const torch::Tensor& send_head,
        const Config& config,
        std::optional<EventHandle>& previous_event,
        bool async,
        bool allocate_on_comm_stream);
};

}  // namespace deep_ep
