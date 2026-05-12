#include <sycl/sycl.hpp>

#include "configs.dp.hpp"
#include "exception.dp.hpp"
#include "launch.dp.hpp"

namespace deep_ep {

namespace layout {

template <int kNumThreads, int kNumExpertsPerSM, int kNumRanksPerSM>
void get_dispatch_layout_kernel(const topk_idx_t* topk_idx,
                                int* num_tokens_per_rank,
                                int* num_tokens_per_rdma_rank,
                                int* num_tokens_per_expert,
                                bool* is_token_in_rank,
                                int num_tokens,
                                int num_topk,
                                int num_ranks,
                                int num_experts,
                                dpct::queue_ptr stream) {
    constexpr int kNumRDMARanksPerSM = kNumRanksPerSM / NUM_MAX_NVL_PEERS;
    const int num_sms =
        ((num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM) + (num_ranks + kNumRanksPerSM - 1) / kNumRanksPerSM;

    stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 2> num_tokens_per_expert_per_thread(sycl::range<2>(kNumThreads, kNumExpertsPerSM), cgh);
        sycl::local_accessor<int, 2> num_tokens_per_rank_per_thread(sycl::range<2>(kNumThreads, kNumRanksPerSM), cgh);
        sycl::local_accessor<int, 2> num_tokens_per_rdma_rank_per_thread(sycl::range<2>(kNumThreads, kNumRDMARanksPerSM), cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(num_sms * kNumThreads), sycl::range<1>(kNumThreads)),
                         [=](sycl::nd_item<1> item) {
                             const int sm_id = static_cast<int>(item.get_group_linear_id());
                             const int thread_id = static_cast<int>(item.get_local_linear_id());

                             const int expert_begin_idx = sm_id * kNumExpertsPerSM;
                             const int expert_end_idx = std::min(expert_begin_idx + kNumExpertsPerSM, num_experts);
                             if (expert_begin_idx < expert_end_idx) {
                                 #pragma unroll
                                 for (int i = 0; i < kNumExpertsPerSM; ++i)
                                     num_tokens_per_expert_per_thread[thread_id][i] = 0;

                                 for (int token_idx = thread_id; token_idx < num_tokens; token_idx += kNumThreads) {
                                     const auto shifted_topk_idx = topk_idx + token_idx * num_topk;
                                     #pragma unroll
                                     for (int j = 0; j < num_topk; ++j) {
                                         const int expert_idx = static_cast<int>(shifted_topk_idx[j]);
                                         if (expert_begin_idx <= expert_idx and expert_idx < expert_end_idx)
                                             ++num_tokens_per_expert_per_thread[thread_id][expert_idx - expert_begin_idx];
                                     }
                                 }
                                 item.barrier(sycl::access::fence_space::local_space);

                                 EP_STATIC_ASSERT(kNumExpertsPerSM <= kNumThreads, "Too many experts per SM");
                                 if (expert_begin_idx + thread_id < expert_end_idx) {
                                     int sum = 0;
                                     #pragma unroll
                                     for (int i = 0; i < kNumThreads; ++i)
                                         sum += num_tokens_per_expert_per_thread[i][thread_id];
                                     num_tokens_per_expert[expert_begin_idx + thread_id] = sum;
                                 }
                                 return;
                             }

                             if (num_tokens_per_rdma_rank != nullptr)
                                 EP_DEVICE_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0 and num_ranks > NUM_MAX_NVL_PEERS);

                             const int sm_begin = (num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM;
                             const int rank_begin_idx = (sm_id - sm_begin) * kNumRanksPerSM;
                             const int rank_end_idx = std::min(rank_begin_idx + kNumRanksPerSM, num_ranks);
                             const int rdma_rank_begin_idx = rank_begin_idx / NUM_MAX_NVL_PEERS;
                             const int rdma_rank_end_idx = rank_end_idx / NUM_MAX_NVL_PEERS;

                             if (rank_begin_idx < rank_end_idx) {
                                 const int num_expert_per_rank = num_experts / num_ranks;
                                 const int expert_begin = rank_begin_idx * num_expert_per_rank;
                                 const int expert_end = rank_end_idx * num_expert_per_rank;

                                 #pragma unroll
                                 for (int i = 0; i < kNumRanksPerSM; ++i)
                                     num_tokens_per_rank_per_thread[thread_id][i] = 0;
                                 #pragma unroll
                                 for (int i = 0; i < kNumRDMARanksPerSM; ++i)
                                     num_tokens_per_rdma_rank_per_thread[thread_id][i] = 0;

                                 for (int token_idx = thread_id; token_idx < num_tokens; token_idx += kNumThreads) {
                                     const auto shifted_topk_idx = topk_idx + token_idx * num_topk;
                                     int is_in_rank[kNumRanksPerSM] = {0};
                                     int is_in_rdma_rank[kNumRDMARanksPerSM] = {0};

                                     #pragma unroll
                                     for (int j = 0; j < num_topk; ++j) {
                                         const int expert_idx = static_cast<int>(shifted_topk_idx[j]);
                                         if (expert_begin <= expert_idx and expert_idx < expert_end) {
                                             const int rank_idx = expert_idx / num_expert_per_rank - rank_begin_idx;
                                             is_in_rank[rank_idx]++;
                                             is_in_rdma_rank[rank_idx / NUM_MAX_NVL_PEERS]++;
                                         }
                                     }

                                     auto shifted_is_token_in_rank = is_token_in_rank + token_idx * num_ranks;
                                     #pragma unroll
                                     for (int j = 0; j + rank_begin_idx < rank_end_idx; ++j) {
                                         shifted_is_token_in_rank[j + rank_begin_idx] = (is_in_rank[j] > 0);
                                         num_tokens_per_rank_per_thread[thread_id][j] += (is_in_rank[j] > 0);
                                     }

                                     #pragma unroll
                                     for (int j = 0; j + rdma_rank_begin_idx < rdma_rank_end_idx; ++j)
                                         num_tokens_per_rdma_rank_per_thread[thread_id][j] += (is_in_rdma_rank[j] > 0);
                                 }
                                 item.barrier(sycl::access::fence_space::local_space);

                                 EP_STATIC_ASSERT(kNumRanksPerSM <= kNumThreads, "Too many ranks per SM");
                                 if (rank_begin_idx + thread_id < rank_end_idx) {
                                     int sum = 0;
                                     #pragma unroll
                                     for (int i = 0; i < kNumThreads; ++i)
                                         sum += num_tokens_per_rank_per_thread[i][thread_id];
                                     num_tokens_per_rank[rank_begin_idx + thread_id] = sum;
                                 }

                                 if (num_tokens_per_rdma_rank != nullptr and rdma_rank_begin_idx + thread_id < rdma_rank_end_idx) {
                                     int sum = 0;
                                     #pragma unroll
                                     for (int i = 0; i < kNumThreads; ++i)
                                         sum += num_tokens_per_rdma_rank_per_thread[i][thread_id];
                                     num_tokens_per_rdma_rank[rdma_rank_begin_idx + thread_id] = sum;
                                 }
                             }
                         });
    });
}

void get_dispatch_layout(const topk_idx_t* topk_idx,
                         int* num_tokens_per_rank,
                         int* num_tokens_per_rdma_rank,
                         int* num_tokens_per_expert,
                         bool* is_token_in_rank,
                         int num_tokens,
                         int num_topk,
                         int num_ranks,
                         int num_experts,
                         dpct::queue_ptr stream) {
    constexpr int kNumThreads = 256;
    constexpr int kNumExpertsPerSM = 4;
    constexpr int kNumRanksPerSM = 8;
    EP_STATIC_ASSERT(kNumRanksPerSM % NUM_MAX_NVL_PEERS == 0, "Invalid number of ranks per SM");

    get_dispatch_layout_kernel<kNumThreads, kNumExpertsPerSM, kNumRanksPerSM>(topk_idx,
                                                                               num_tokens_per_rank,
                                                                               num_tokens_per_rdma_rank,
                                                                               num_tokens_per_expert,
                                                                               is_token_in_rank,
                                                                               num_tokens,
                                                                               num_topk,
                                                                               num_ranks,
                                                                               num_experts,
                                                                               stream);
}

}  // namespace layout

}  // namespace deep_ep
