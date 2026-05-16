#include "xpu_kernels.hpp"

namespace deep_ep {
namespace internode {
namespace {

class DispatchMetadataKernel;

template <typename dtype_t>
class CombineCopyKernel;

}  // namespace

void dispatch(void*,
              float* recv_x_scales,
              topk_idx_t* recv_topk_idx,
              float* recv_topk_weights,
              void* recv_src_meta,
              const void*,
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
              const bool*,
              int num_tokens,
              int,
              int,
              int,
              int,
              int num_ranks,
              sycl::queue& queue) {
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<DispatchMetadataKernel>(sycl::range<1>(1), [=](sycl::id<1>) {
            if (send_rdma_head != nullptr) {
                for (int i = 0; i < num_tokens * sycl::max(num_ranks / NUM_MAX_NVL_PEERS, 1); ++i) {
                    send_rdma_head[i] = -1;
                }
            }
            if (send_nvl_head != nullptr) {
                for (int i = 0; i < num_tokens * NUM_MAX_NVL_PEERS; ++i) {
                    send_nvl_head[i] = -1;
                }
            }
            if (recv_src_meta != nullptr) {
                auto meta = static_cast<int*>(recv_src_meta);
                for (int i = 0; i < num_tokens * static_cast<int>(sizeof(SourceMeta) / sizeof(int)); ++i) {
                    meta[i] = 0;
                }
            }
            if (recv_rdma_channel_prefix_matrix != nullptr && rdma_channel_prefix_matrix != nullptr) {
                int n = sycl::max(num_ranks / NUM_MAX_NVL_PEERS, 1);
                for (int i = 0; i < n; ++i) {
                    recv_rdma_channel_prefix_matrix[i] = rdma_channel_prefix_matrix[i];
                }
            }
            if (recv_gbl_channel_prefix_matrix != nullptr && gbl_channel_prefix_matrix != nullptr) {
                for (int i = 0; i < num_ranks; ++i) {
                    recv_gbl_channel_prefix_matrix[i] = gbl_channel_prefix_matrix[i];
                }
            }
            (void)recv_x_scales;
            (void)recv_topk_idx;
            (void)recv_topk_weights;
            (void)x_scales;
            (void)topk_idx;
            (void)topk_weights;
            (void)recv_rdma_rank_prefix_sum;
            (void)recv_gbl_rank_prefix_sum;
        });
    });
}

template <typename dtype_t>
void launch_combine_copy(void* combined_x,
                         float* combined_topk_weights,
                         const void* x,
                         const float* topk_weights,
                         const void* bias_0,
                         const void* bias_1,
                         int num_tokens,
                         int num_combined_tokens,
                         int hidden,
                         int num_topk,
                         sycl::queue& queue) {
    auto dst = static_cast<dtype_t*>(combined_x);
    auto src = static_cast<const dtype_t*>(x);
    auto b0 = static_cast<const dtype_t*>(bias_0);
    auto b1 = static_cast<const dtype_t*>(bias_1);
    const int64_t total = static_cast<int64_t>(num_combined_tokens) * hidden;
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombineCopyKernel<dtype_t>>(sycl::range<1>(total), [=](sycl::id<1> id) {
            int64_t i = static_cast<int64_t>(id[0]);
            int token = static_cast<int>(i / hidden);
            dtype_t value = token < num_tokens ? src[i] : dtype_t{};
            if (b0 != nullptr) {
                value += b0[i];
            }
            if (b1 != nullptr) {
                value += b1[i];
            }
            dst[i] = value;
            if (combined_topk_weights != nullptr && topk_weights != nullptr && hidden > 0 && (i % hidden) < num_topk) {
                int topk = static_cast<int>(i % hidden);
                combined_topk_weights[static_cast<int64_t>(token) * num_topk + topk] =
                    token < num_tokens ? topk_weights[static_cast<int64_t>(token) * num_topk + topk] : 0.0f;
            }
        });
    });
}

void combine(DataType type,
             void* combined_x,
             float* combined_topk_weights,
             const bool*,
             const void* x,
             const float* topk_weights,
             const void* bias_0,
             const void* bias_1,
             const int*,
             const int*,
             const void*,
             const int*,
             const int*,
             const int*,
             int num_tokens,
             int num_combined_tokens,
             int hidden,
             int num_topk,
             int,
             int,
             sycl::queue& queue) {
    if (type == DataType::kBFloat16) {
        launch_combine_copy<sycl::ext::oneapi::bfloat16>(
            combined_x, combined_topk_weights, x, topk_weights, bias_0, bias_1, num_tokens, num_combined_tokens, hidden, num_topk, queue);
    } else {
        launch_combine_copy<int32_t>(
            combined_x, combined_topk_weights, x, topk_weights, bias_0, bias_1, num_tokens, num_combined_tokens, hidden, num_topk, queue);
    }
}

}  // namespace internode
}  // namespace deep_ep
