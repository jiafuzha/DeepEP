#include <sycl/sycl.hpp>

#include "configs.dp.hpp"
#include "exception.dp.hpp"

namespace deep_ep {

namespace internode_ll {

namespace {

[[noreturn]] void throw_unsupported() {
    throw EPException(
        "XPU",
        __FILE__,
        __LINE__,
        "DeepEP XPU low-latency kernels are intentionally unsupported: the mirrored API surface is kept for compatibility, "
        "but a portable SYCL/iSHMEM implementation is not available yet.");
}

}  // namespace

void clean_low_latency_buffer(int* clean_0,
                              int num_clean_int_0,
                              int* clean_1,
                              int num_clean_int_1,
                              int rank,
                              int num_ranks,
                              int* mask_buffer_ptr,
                              int* sync_buffer_ptr,
                              dpct::queue_ptr stream) {
    (void)clean_0;
    (void)num_clean_int_0;
    (void)clean_1;
    (void)num_clean_int_1;
    (void)rank;
    (void)num_ranks;
    (void)mask_buffer_ptr;
    (void)sync_buffer_ptr;
    (void)stream;
    throw_unsupported();
}

void dispatch(void* packed_recv_x,
              void* packed_recv_x_scales,
              int* packed_recv_src_info,
              int64_t* packed_recv_layout_range,
              int* packed_recv_count,
              int* mask_buffer_ptr,
              int* cumulative_local_expert_recv_stats,
              int64_t* dispatch_wait_recv_cost_stats,
              void* rdma_recv_x,
              int* rdma_recv_count,
              void* rdma_x,
              const void* x,
              const topk_idx_t* topk_idx,
              int* next_clean,
              int num_next_clean_int,
              int num_tokens,
              int hidden,
              int num_max_dispatch_tokens_per_rank,
              int num_topk,
              int num_experts,
              int rank,
              int num_ranks,
              bool use_fp8,
              bool round_scale,
              bool use_ue8m0,
              void* workspace,
              int num_device_sms,
              dpct::queue_ptr stream,
              int phases) {
    (void)packed_recv_x;
    (void)packed_recv_x_scales;
    (void)packed_recv_src_info;
    (void)packed_recv_layout_range;
    (void)packed_recv_count;
    (void)mask_buffer_ptr;
    (void)cumulative_local_expert_recv_stats;
    (void)dispatch_wait_recv_cost_stats;
    (void)rdma_recv_x;
    (void)rdma_recv_count;
    (void)rdma_x;
    (void)x;
    (void)topk_idx;
    (void)next_clean;
    (void)num_next_clean_int;
    (void)num_tokens;
    (void)hidden;
    (void)num_max_dispatch_tokens_per_rank;
    (void)num_topk;
    (void)num_experts;
    (void)rank;
    (void)num_ranks;
    (void)use_fp8;
    (void)round_scale;
    (void)use_ue8m0;
    (void)workspace;
    (void)num_device_sms;
    (void)stream;
    (void)phases;
    throw_unsupported();
}

void combine(void* combined_x,
             void* rdma_recv_x,
             int* rdma_recv_flag,
             void* rdma_send_x,
             const void* x,
             const topk_idx_t* topk_idx,
             const float* topk_weights,
             const int* src_info,
             const int64_t* layout_range,
             int* mask_buffer_ptr,
             int64_t* combine_wait_recv_cost_stats,
             int* next_clean,
             int num_next_clean_int,
             int num_combined_tokens,
             int hidden,
             int num_max_dispatch_tokens_per_rank,
             int num_topk,
             int num_experts,
             int rank,
             int num_ranks,
             bool use_logfmt,
             void* workspace,
             int num_device_sms,
             dpct::queue_ptr stream,
             int phases,
             bool zero_copy) {
    (void)combined_x;
    (void)rdma_recv_x;
    (void)rdma_recv_flag;
    (void)rdma_send_x;
    (void)x;
    (void)topk_idx;
    (void)topk_weights;
    (void)src_info;
    (void)layout_range;
    (void)mask_buffer_ptr;
    (void)combine_wait_recv_cost_stats;
    (void)next_clean;
    (void)num_next_clean_int;
    (void)num_combined_tokens;
    (void)hidden;
    (void)num_max_dispatch_tokens_per_rank;
    (void)num_topk;
    (void)num_experts;
    (void)rank;
    (void)num_ranks;
    (void)use_logfmt;
    (void)workspace;
    (void)num_device_sms;
    (void)stream;
    (void)phases;
    (void)zero_copy;
    throw_unsupported();
}

void query_mask_buffer(int* mask_buffer_ptr, int num_ranks, int* output_mask_tensor, dpct::queue_ptr stream) {
    (void)mask_buffer_ptr;
    (void)num_ranks;
    (void)output_mask_tensor;
    (void)stream;
    throw_unsupported();
}

void update_mask_buffer(int* mask_buffer_ptr, int rank_to_mask, bool mask, dpct::queue_ptr stream) {
    (void)mask_buffer_ptr;
    (void)rank_to_mask;
    (void)mask;
    (void)stream;
    throw_unsupported();
}

void clean_mask_buffer(int* mask_buffer_ptr, int num_ranks, dpct::queue_ptr stream) {
    (void)mask_buffer_ptr;
    (void)num_ranks;
    (void)stream;
    throw_unsupported();
}

}  // namespace internode_ll

}  // namespace deep_ep
