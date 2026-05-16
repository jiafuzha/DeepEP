#pragma once

#include <c10/xpu/XPUEvent.h>
#include <c10/xpu/XPUStream.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <sycl/sycl.hpp>
#include <tuple>
#include <vector>

namespace deep_ep {

#ifndef TOPK_IDX_BITS
#define TOPK_IDX_BITS 64
#endif

#define INT_BITS_T2(bits) int##bits##_t
#define INT_BITS_T(bits) INT_BITS_T2(bits)
using topk_idx_t = INT_BITS_T(TOPK_IDX_BITS);
#undef INT_BITS_T
#undef INT_BITS_T2

constexpr int NUM_MAX_NVL_PEERS = 8;
constexpr int NUM_BUFFER_ALIGNMENT_BYTES = 128;
constexpr int NUM_MAX_LOCAL_EXPERTS = 1024;
constexpr int NUM_CPU_TIMEOUT_SECS = 100;

struct int4 {
    int x;
    int y;
    int z;
    int w;
};

enum class DataType {
    kBFloat16,
    kInt32,
};

template <typename dtype_t>
dtype_t ceil_div(dtype_t a, dtype_t b) {
    return (a + b - 1) / b;
}

template <typename dtype_t>
dtype_t align_up(dtype_t a, dtype_t b) {
    return ceil_div<dtype_t>(a, b) * b;
}

struct Config {
    int num_sms;
    int num_max_nvl_chunked_send_tokens;
    int num_max_nvl_chunked_recv_tokens;
    int num_max_rdma_chunked_send_tokens;
    int num_max_rdma_chunked_recv_tokens;

    Config(int num_sms,
           int num_max_nvl_chunked_send_tokens,
           int num_max_nvl_chunked_recv_tokens,
           int num_max_rdma_chunked_send_tokens,
           int num_max_rdma_chunked_recv_tokens)
        : num_sms(num_sms),
          num_max_nvl_chunked_send_tokens(num_max_nvl_chunked_send_tokens),
          num_max_nvl_chunked_recv_tokens(num_max_nvl_chunked_recv_tokens),
          num_max_rdma_chunked_send_tokens(num_max_rdma_chunked_send_tokens),
          num_max_rdma_chunked_recv_tokens(num_max_rdma_chunked_recv_tokens) {
        TORCH_CHECK(num_sms >= 0, "num_sms must be non-negative");
        TORCH_CHECK(num_max_nvl_chunked_send_tokens > 0, "num_max_nvl_chunked_send_tokens must be positive");
        TORCH_CHECK(num_max_nvl_chunked_recv_tokens > 0, "num_max_nvl_chunked_recv_tokens must be positive");
        TORCH_CHECK(num_max_nvl_chunked_send_tokens < num_max_nvl_chunked_recv_tokens,
                    "num_max_nvl_chunked_send_tokens must be smaller than num_max_nvl_chunked_recv_tokens");
        TORCH_CHECK(num_max_rdma_chunked_send_tokens > 0, "num_max_rdma_chunked_send_tokens must be positive");
        TORCH_CHECK(num_max_rdma_chunked_recv_tokens > 0, "num_max_rdma_chunked_recv_tokens must be positive");
        this->num_max_rdma_chunked_recv_tokens = align_up<int>(num_max_rdma_chunked_recv_tokens, num_max_rdma_chunked_send_tokens);
        TORCH_CHECK(num_max_rdma_chunked_send_tokens < this->num_max_rdma_chunked_recv_tokens,
                    "num_max_rdma_chunked_send_tokens must be smaller than aligned RDMA recv tokens");
        TORCH_CHECK(num_max_rdma_chunked_send_tokens <= this->num_max_rdma_chunked_recv_tokens / 2,
                    "num_max_rdma_chunked_send_tokens must be at most half of RDMA recv tokens");
    }

    size_t get_nvl_buffer_size_hint(size_t hidden_bytes, int num_ranks) const;
    size_t get_rdma_buffer_size_hint(int64_t hidden_bytes, int num_ranks) const;
};

struct EventHandle {
    std::shared_ptr<c10::xpu::XPUEvent> event;

    EventHandle();
    explicit EventHandle(const c10::xpu::XPUStream& stream);
    void current_stream_wait() const;
};

void stream_wait(const c10::xpu::XPUStream& dst, const c10::xpu::XPUStream& src);
void stream_wait(const c10::xpu::XPUStream& dst, const EventHandle& event);

void launch_get_dispatch_layout(const topk_idx_t* topk_idx,
                                int* num_tokens_per_rank,
                                int* num_tokens_per_rdma_rank,
                                int* num_tokens_per_expert,
                                bool* is_token_in_rank,
                                int num_tokens,
                                int num_topk,
                                int num_ranks,
                                int num_experts,
                                sycl::queue& queue);

size_t get_low_latency_rdma_size_hint(int num_max_dispatch_tokens_per_rank, int hidden, int num_ranks, int num_experts);

namespace internode {

struct SourceMeta {
    int src_rdma_rank;
    int is_token_in_nvl_rank_bits;
};

std::vector<uint8_t> get_unique_id();

int init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool low_latency_mode);

void* alloc(size_t size, size_t alignment);

void free(void* ptr);

void barrier();

void finalize();

int get_source_meta_bytes();

void dispatch(void* recv_x,
              float* recv_x_scales,
              topk_idx_t* recv_topk_idx,
              float* recv_topk_weights,
              void* recv_src_meta,
              const void* x,
              const float* x_scales,
              const topk_idx_t* topk_idx,
              const float* topk_weights,
              int* send_rdma_head,
              int* send_nvl_head,
              int* recv_rdma_channel_prefix_matrix,
              int* recv_gbl_channel_prefix_matrix,
              const int* rdma_channel_prefix_matrix,
              const int* recv_rdma_rank_prefix_sum,
              const int* gbl_channel_prefix_matrix,
              const int* recv_gbl_rank_prefix_sum,
              const bool* is_token_in_rank,
              int num_tokens,
              int num_recv_tokens,
              int hidden,
              int num_topk,
              int rank,
              int num_ranks,
              sycl::queue& queue);

void combine(DataType type,
             void* combined_x,
             float* combined_topk_weights,
             const bool* is_combined_token_in_rank,
             const void* x,
             const float* topk_weights,
             const void* bias_0,
             const void* bias_1,
             const int* combined_rdma_head,
             const int* combined_nvl_head,
             const void* src_meta,
             const int* rdma_channel_prefix_matrix,
             const int* rdma_rank_prefix_sum,
             const int* gbl_channel_prefix_matrix,
             int num_tokens,
             int num_combined_tokens,
             int hidden,
             int num_topk,
             int rank,
             int num_ranks,
             sycl::queue& queue);

}  // namespace internode

namespace internode_ll {

void clean_low_latency_buffer(int* clean_0,
                              int num_clean_int_0,
                              int* clean_1,
                              int num_clean_int_1,
                              int rank,
                              int num_ranks,
                              int* mask_buffer_ptr,
                              int* sync_buffer_ptr,
                              sycl::queue& queue);

void update_mask_buffer(int* mask_buffer_ptr, int rank_to_mask, bool mask, sycl::queue& queue);

void query_mask_buffer(int* mask_buffer_ptr, int num_ranks, int* output_mask_tensor, sycl::queue& queue);

void clean_mask_buffer(int* mask_buffer_ptr, int num_ranks, sycl::queue& queue);

void dispatch_bf16(void* packed_recv_x,
                   int* packed_recv_src_info,
                   int64_t* packed_recv_layout_range,
                   int* packed_recv_count,
                   int* cumulative_local_expert_recv_stats,
                   int64_t* dispatch_wait_recv_cost_stats,
                   void* rdma_buffer,
                   int* mask_buffer_ptr,
                   const void* x,
                   const topk_idx_t* topk_idx,
                   int num_tokens,
                   int hidden,
                   int num_max_dispatch_tokens_per_rank,
                   int num_topk,
                   int num_experts,
                   int rank,
                   int num_ranks,
                   sycl::queue& queue);

void combine_bf16(void* combined_x,
                  void* rdma_buffer,
                  int* mask_buffer_ptr,
                  const void* x,
                  const topk_idx_t* topk_idx,
                  const float* topk_weights,
                  const int* src_info,
                  const int64_t* layout_range,
                  int64_t* combine_wait_recv_cost_stats,
                  int num_combined_tokens,
                  int hidden,
                  int num_max_dispatch_tokens_per_rank,
                  int num_topk,
                  int num_experts,
                  int rank,
                  int num_ranks,
                  sycl::queue& queue,
                  bool zero_copy);

}  // namespace internode_ll

namespace intranode {

void notify_dispatch(const int* num_tokens_per_rank,
                     int* moe_recv_counter_mapped,
                     int num_ranks,
                     const int* num_tokens_per_expert,
                     int* moe_recv_expert_counter_mapped,
                     int num_experts,
                     int num_tokens,
                     const bool* is_token_in_rank,
                     int* channel_prefix_matrix,
                     int* rank_prefix_matrix_copy,
                     int num_memset_int,
                     int expert_alignment,
                     void** buffer_ptrs,
                     int** barrier_signal_ptrs,
                     int rank,
                     int barrier_signal_base,
                     sycl::queue& queue,
                     int num_channels);

void cached_notify_dispatch(const int* rank_prefix_matrix,
                            int num_memset_int,
                            void** buffer_ptrs,
                            int** barrier_signal_ptrs,
                            int rank,
                            int num_ranks,
                            int barrier_signal_base,
                            sycl::queue& queue);

void dispatch(void* recv_x,
              float* recv_x_scales,
              int* recv_src_idx,
              topk_idx_t* recv_topk_idx,
              float* recv_topk_weights,
              int* recv_channel_offset,
              int* send_head,
              const void* x,
              const float* x_scales,
              const topk_idx_t* topk_idx,
              const float* topk_weights,
              const bool* is_token_in_rank,
              const int* channel_prefix_matrix,
              int num_tokens,
              int num_worst_tokens,
              int hidden_int4,
              int num_topk,
              int num_experts,
              int num_scales,
              int scale_token_stride,
              int scale_hidden_stride,
              void** buffer_ptrs,
              int rank,
              int num_ranks,
              sycl::queue& queue,
              int num_sms,
              int num_max_send_tokens,
              int num_recv_buffer_tokens);

void cached_notify_combine(void** buffer_ptrs,
                           int* send_head,
                           int num_channels,
                           int num_recv_tokens,
                           int num_memset_int,
                           int** barrier_signal_ptrs,
                           int rank,
                           int num_ranks,
                           int barrier_signal_base,
                           sycl::queue& queue);

void combine(DataType type,
             void* recv_x,
             float* recv_topk_weights,
             const void* x,
             const float* topk_weights,
             const void* bias_0,
             const void* bias_1,
             const int* src_idx,
             const int* rank_prefix_matrix,
             const int* channel_prefix_matrix,
             int* send_head,
             int num_tokens,
             int num_recv_tokens,
             int hidden,
             int num_topk,
             void** buffer_ptrs,
             int rank,
             int num_ranks,
             sycl::queue& queue,
             int num_sms,
             int num_max_send_tokens,
             int num_recv_buffer_tokens);

void barrier(int** barrier_signal_ptrs, int rank, int num_ranks, int barrier_signal_base, sycl::queue& queue);

}  // namespace intranode

}  // namespace deep_ep
